"""Real-LLM reasoning-tree search harness: value-guided tree search vs. best-of-N.

This is the generator-agnostic core for testing -- on a real benchmark (e.g. GSM8K) -- the
claim from ``canopy.bandits.reasoning``: that value-guided (edge-following) test-time search
finds correct answers at lower compute than best-of-N. It is deliberately decoupled from any
specific model: every method takes a ``generate`` callable
``(prompt, max_tokens, seed) -> text``, so it runs against a mock LLM in tests and against a
real model (e.g. ``canopy.llm.BedrockClient`` via ``canopy.llm.as_generate_fn``) in
``examples/reasoning/gsm8k_reasoning_search.py``.

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


def extract_boxed(text: str) -> str | None:
    """Extract the last ``\\boxed{...}`` answer (MATH convention), else fall back to
    the GSM8K-style extraction so prose answers still grade when parseable."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return extract_answer(text)
    i, depth = start + len("\\boxed{"), 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        i += 1
    ans = "".join(out).strip().replace(" ", "").replace("\\!", "").replace("\\,", "")
    return _normalize(ans) if ans else None


def extract_letter(text: str, letters: str = "ABCD") -> str | None:
    """Extract the final multiple-choice letter (GPQA/MMLU convention): scan from the
    end for a standalone option letter."""
    for i in range(len(text) - 1, -1, -1):
        ch = text[i].upper()
        if ch in letters:
            prev_ok = i == 0 or not text[i - 1].isalnum()
            next_ok = i == len(text) - 1 or not text[i + 1].isalnum()
            if prev_ok and next_ok:
                return ch
    return None


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


ExtractFn = Callable[[str], "str | None"]


def best_of_n(
    question: str,
    gold: str,
    generate: GenerateFn,
    n: int,
    max_tokens: int = 512,
    extract: ExtractFn = extract_answer,
    solve_template: str | None = None,
) -> SearchResult:
    """Sample ``n`` full solutions and return the majority-vote answer (self-consistency).

    ``extract`` parses a completion into a final answer (numeric ``####`` by default;
    pass :func:`extract_boxed` for MATH or :func:`extract_letter` for multiple choice),
    and ``solve_template`` overrides the benchmark prompt (must contain ``{q}``).
    """
    solve = solve_template or _SOLVE
    budget = Budget()
    votes: Counter[str] = Counter()
    for i in range(n):
        text = generate(solve.format(q=question), max_tokens, i)
        budget.charge(text)
        a = extract(text)
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
    extract: ExtractFn = extract_answer,
    continue_template: str | None = None,
    rollout_template: str | None = None,
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
        value_fn = lambda texts: _self_consistency(texts, extract)  # noqa: E731
    cont = continue_template or _CONTINUE
    roll_t = rollout_template or _ROLLOUT
    budget = Budget()
    prefix = ""
    for step in range(n_steps):
        best_step, best_val = None, -1.0
        for c in range(branching):
            cand = generate(
                cont.format(q=question, prefix=prefix), max_tokens // 2, 1000 * step + c
            )
            budget.charge(cand)
            roll_texts: list[str] = []
            for r in range(rollouts):
                roll = generate(
                    roll_t.format(q=question, prefix=prefix + "\n" + cand),
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
        final = generate(roll_t.format(q=question, prefix=prefix), max_tokens, 999_000 + f)
        budget.charge(final)
        a = extract(final)
        if a is not None:
            votes[a] += 1
    answer = votes.most_common(1)[0][0] if votes else None
    return SearchResult(answer=answer, correct=(answer == _normalize(gold)), budget=budget)


@dataclass
class StepLog:
    """Per-step instrumentation of the value-guided descent (one entry per step).

    ``cheap[i]`` is the value the search follows for candidate ``i`` (e.g.
    self-consistency of its rollouts); ``true[i]`` is the fraction of the same
    rollouts that actually grade correct against gold -- the ground-truth node value
    the cheap probe is supposed to track. ``chosen`` is the candidate followed
    (argmax of cheap).
    """

    step: int
    cheap: list[float]
    true: list[float]
    chosen: int

    @property
    def spread(self) -> float:
        """Sibling true-value spread (max - min): large means a pivotal step."""
        return max(self.true) - min(self.true)

    @property
    def hit(self) -> bool:
        """Did the cheap edge follow the truly-best continuation?"""
        return self.true[self.chosen] == max(self.true)

    @property
    def gap(self) -> float:
        """True value forgone by following the cheap edge instead of the best."""
        return max(self.true) - self.true[self.chosen]


def instrumented_value_guided_search(
    question: str,
    gold: str,
    generate: GenerateFn,
    branching: int = 3,
    n_steps: int = 4,
    rollouts: int = 2,
    max_tokens: int = 512,
    value_fn: Callable[[list[str]], float] | None = None,
    final_rollouts: int = 1,
    extract: ExtractFn = extract_answer,
    continue_template: str | None = None,
    rollout_template: str | None = None,
) -> tuple[SearchResult, list[StepLog]]:
    """:func:`value_guided_search` with per-node instrumentation.

    Identical descent (same prompts, same seeds, same cheap-value choices), but each
    candidate's rollouts are additionally graded against ``gold`` to log the *true*
    node value next to the *cheap* one -- the measurement behind the almost
    tree-K-Lipschitz characterization (cheap-vs-true correlation = the backbone;
    sibling true-value spread = the violations). Grading uses the gold answer, so this
    is an offline analysis tool, not a deployable search.
    """
    if value_fn is None:
        value_fn = lambda texts: _self_consistency(texts, extract)  # noqa: E731
    cont = continue_template or _CONTINUE
    roll_t = rollout_template or _ROLLOUT
    budget = Budget()
    prefix = ""
    logs: list[StepLog] = []
    gold_norm = _normalize(gold)
    for step in range(n_steps):
        cheap_vals: list[float] = []
        true_vals: list[float] = []
        cands: list[str] = []
        for c in range(branching):
            cand = generate(
                cont.format(q=question, prefix=prefix), max_tokens // 2, 1000 * step + c
            )
            budget.charge(cand)
            roll_texts: list[str] = []
            for r in range(rollouts):
                roll = generate(
                    roll_t.format(q=question, prefix=prefix + "\n" + cand),
                    max_tokens,
                    50_000 + 1000 * (step * branching + c) + r,
                )
                budget.charge(roll)
                roll_texts.append(roll)
            cheap_vals.append(value_fn(roll_texts))
            true_vals.append(
                float(sum(extract(t) == gold_norm for t in roll_texts)) / max(1, len(roll_texts))
            )
            cands.append(cand)
        chosen = int(max(range(branching), key=lambda i: cheap_vals[i]))
        logs.append(StepLog(step=step, cheap=cheap_vals, true=true_vals, chosen=chosen))
        prefix = prefix + "\n" + cands[chosen]
    votes: Counter[str] = Counter()
    for f in range(final_rollouts):
        final = generate(roll_t.format(q=question, prefix=prefix), max_tokens, 999_000 + f)
        budget.charge(final)
        a = extract(final)
        if a is not None:
            votes[a] += 1
    answer = votes.most_common(1)[0][0] if votes else None
    return (
        SearchResult(answer=answer, correct=(answer == gold_norm), budget=budget),
        logs,
    )


def _self_consistency(rollout_texts: list[str], extract: ExtractFn = extract_answer) -> float:
    """Default value: size of the largest agreeing answer set over the rollouts (in [0,1])."""
    answers: Counter[str] = Counter()
    for t in rollout_texts:
        a = extract(t)
        if a is not None:
            answers[a] += 1
    if not answers or not rollout_texts:
        return 0.0
    return answers.most_common(1)[0][1] / len(rollout_texts)
