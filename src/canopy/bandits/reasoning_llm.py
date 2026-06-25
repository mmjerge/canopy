"""Real-LLM reasoning-tree search harness: value-guided tree search vs. best-of-N.

This is the generator-agnostic core for testing -- on a real benchmark (e.g. GSM8K) -- the
claim from ``canopy.bandits.reasoning``: that value-guided (edge-following) test-time search
finds correct answers at lower compute than best-of-N. It is deliberately decoupled from any
specific model: every method takes a ``generate`` callable
``(prompt, max_tokens, seed) -> text``, so it runs against a mock LLM in tests and against a
real model (e.g. ``canopy.bandits.bedrock.BedrockClient``) in
``examples/gsm8k_reasoning_search.py``.

Two strategies, compared at an equal budget of generation calls:

* :func:`best_of_n` -- sample ``n`` full chains of thought and take the majority-vote answer
  (self-consistency); and
* :func:`value_guided_search` -- a beam/tree search over reasoning steps: at each step sample
  ``branching`` candidate continuations, score each by a few cheap rollouts to an answer (the
  multi-fidelity probe), keep the best, and continue.

Honest scope: this is a search / compute-allocation method. Whether it beats best-of-N on
real models is an empirical question about whether the rollout value is informative at the
pivotal steps -- which is exactly what this harness is built to measure.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Protocol

GenerateFn = Callable[[str, int, int], str]
"""A generator: ``(prompt, max_tokens, seed) -> completion text``."""


class Generator(Protocol):
    def __call__(self, prompt: str, max_tokens: int, seed: int) -> str: ...


@dataclass
class Budget:
    """Counts generation calls and output tokens (the test-time compute spent)."""

    calls: int = 0
    tokens: int = 0

    def charge(self, text: str) -> None:
        self.calls += 1
        self.tokens += len(text.split())


# --- GSM8K-style answer extraction and grading ---------------------------------

_NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def extract_answer(text: str) -> str | None:
    """Extract the final numeric answer (GSM8K convention: after ``####``, else last number)."""
    if "####" in text:
        tail = text.split("####")[-1]
        nums = _NUM.findall(tail)
        if nums:
            return _normalize(nums[0])
    nums = _NUM.findall(text)
    return _normalize(nums[-1]) if nums else None


def _normalize(s: str) -> str:
    s = s.replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


def is_correct(text: str, gold: str) -> bool:
    a = extract_answer(text)
    return a is not None and a == _normalize(gold)


# --- prompts -------------------------------------------------------------------

_SOLVE = (
    "Solve the problem step by step and end with '#### <answer>'.\n\n" "Problem: {q}\n\nSolution:"
)
_CONTINUE = (
    "Solve the problem step by step and end with '#### <answer>'.\n\n"
    "Problem: {q}\n\nSolution so far:\n{prefix}\nNext step:"
)
_ROLLOUT = (
    "Solve the problem step by step and end with '#### <answer>'.\n\n"
    "Problem: {q}\n\nSolution so far:\n{prefix}\nFinish the solution:"
)


# --- strategies ----------------------------------------------------------------


@dataclass
class SearchResult:
    answer: str | None
    correct: bool
    budget: Budget = field(default_factory=Budget)


def best_of_n(
    question: str,
    gold: str,
    generate: GenerateFn,
    n: int,
    max_tokens: int = 512,
) -> SearchResult:
    """Sample ``n`` full solutions and return the majority-vote answer (self-consistency)."""
    budget = Budget()
    votes: Counter[str] = Counter()
    for i in range(n):
        text = generate(_SOLVE.format(q=question), max_tokens, i)
        budget.charge(text)
        a = extract_answer(text)
        if a is not None:
            votes[a] += 1
    answer = votes.most_common(1)[0][0] if votes else None
    return SearchResult(answer=answer, correct=(answer == _normalize(gold)), budget=budget)


def value_guided_search(
    question: str,
    gold: str,
    generate: GenerateFn,
    branching: int = 3,
    n_steps: int = 4,
    rollouts: int = 2,
    max_tokens: int = 512,
    value_fn: Callable[[list[str]], float] | None = None,
    final_rollouts: int = 1,
) -> SearchResult:
    """Beam/tree search over reasoning steps with cheap-rollout value (multi-fidelity probe).

    Maintains a single partial solution. At each of ``n_steps`` steps it samples ``branching``
    candidate next steps; each candidate is scored by ``rollouts`` cheap completions whose
    texts are passed to ``value_fn``. The highest-value candidate is kept; finally
    ``final_rollouts`` completions of the chosen prefix are generated and their answers are
    **majority-voted** (so value-guided has the same aggregation strength as best-of-N, on top
    of the guided prefix).

    ``value_fn(rollout_texts) -> float`` is the value signal, defaulting to **self-consistency**
    (no ground truth). A verifier/PRM-style value isolates whether the *search machinery* helps
    given an informative value.
    """
    if value_fn is None:
        value_fn = _self_consistency
    budget = Budget()
    prefix = ""
    for step in range(n_steps):
        best_step, best_val = None, -1.0
        for c in range(branching):
            cand = generate(
                _CONTINUE.format(q=question, prefix=prefix), max_tokens // 2, 1000 * step + c
            )
            budget.charge(cand)
            roll_texts: list[str] = []
            for r in range(rollouts):
                roll = generate(
                    _ROLLOUT.format(q=question, prefix=prefix + "\n" + cand),
                    max_tokens,
                    50_000 + 1000 * (step * branching + c) + r,
                )
                budget.charge(roll)
                roll_texts.append(roll)
            val = value_fn(roll_texts)
            if val > best_val:
                best_val, best_step = val, cand
        prefix = prefix + "\n" + (best_step or "")
    votes: Counter[str] = Counter()
    for f in range(final_rollouts):
        final = generate(_ROLLOUT.format(q=question, prefix=prefix), max_tokens, 999_000 + f)
        budget.charge(final)
        a = extract_answer(final)
        if a is not None:
            votes[a] += 1
    answer = votes.most_common(1)[0][0] if votes else None
    return SearchResult(answer=answer, correct=(answer == _normalize(gold)), budget=budget)


def _self_consistency(rollout_texts: list[str]) -> float:
    """Default value: size of the largest agreeing answer set over the rollouts (in [0,1])."""
    answers: Counter[str] = Counter()
    for t in rollout_texts:
        a = extract_answer(t)
        if a is not None:
            answers[a] += 1
    if not answers or not rollout_texts:
        return 0.0
    return answers.most_common(1)[0][1] / len(rollout_texts)
