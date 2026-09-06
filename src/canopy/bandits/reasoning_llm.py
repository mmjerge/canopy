"""Real-LLM reasoning-tree search harness: value-guided tree search vs. best-of-N.

This is the generator-agnostic core for testing -- on a real benchmark (e.g. GSM8K) -- the
claim from ``canopy.bandits.reasoning``: that value-guided (edge-following) test-time search
finds correct answers at lower compute than best-of-N. It is deliberately decoupled from any
specific model: every method takes a ``generate`` callable
``(prompt, max_tokens, seed) -> text``, so it runs against a mock LLM in tests and against a
real model (e.g. ``canopy.llm.BedrockClient`` via ``canopy.llm.as_generate_fn``) in
``examples/reasoning/reasoning_search.py`` (GSM8K baseline and MATH headline).

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


def _grade_numeric(answer: str | None, gold: str) -> bool:
    """Default (GSM8K-style) grade: normalized string / numeric equality."""
    return answer is not None and answer == _normalize(gold)


# --- MATH-style extraction and grading (boxed answers, symbolic equivalence) -----

_BOXED_MARK = "\\boxed"


def _extract_boxed(text: str) -> str | None:
    """Return the content of the last balanced ``\\boxed{...}`` in ``text``, else ``None``."""
    idx = text.rfind(_BOXED_MARK)
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def normalize_math(s: str) -> str:
    """Canonicalize a LaTeX-ish math answer to a comparable string (light MATH-eval rules)."""
    s = s.strip()
    for tok in ("$", "\\(", "\\)", "\\left", "\\right", "\\!", "\\,", "\\;", "\\ ", "%", "\\%"):
        s = s.replace(tok, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = s.replace(" ", "").replace("{", "").replace("}", "").rstrip(".")
    return s


def extract_boxed_answer(text: str) -> str | None:
    """Extract a MATH answer: prefer ``\\boxed{...}``, else after ``####``, else last number."""
    boxed = _extract_boxed(text)
    if boxed is not None:
        return normalize_math(boxed)
    if "####" in text:
        tail = text.split("####")[-1]
        # Some models echo the literal ``<answer>`` placeholder from the prompt; drop it, then
        # take the first NON-EMPTY line (models may put a blank line before the actual answer).
        tail = re.sub(r"<\s*answer\s*>", " ", tail, flags=re.IGNORECASE)
        first = next((ln.strip() for ln in tail.splitlines() if ln.strip()), "")
        return normalize_math(first) if first else None
    nums = _NUM.findall(text)
    return _normalize(nums[-1]) if nums else None


def _to_float(s: str) -> float | None:
    """Best-effort float for a simple rational/decimal answer (``(3)/(2)``, ``3/2``, ``0.5``)."""
    t = s.replace("(", "").replace(")", "").replace(",", "")
    try:
        if "/" in t:
            num, den = t.split("/", 1)
            return float(num) / float(den)
        return float(t)
    except (ValueError, ZeroDivisionError):
        return None


def grade_math(answer: str | None, gold: str) -> bool:
    """Grade a MATH answer against gold: normalized string, numeric, then symbolic equivalence."""
    if answer is None:
        return False
    g = normalize_math(str(gold))
    if answer == g:
        return True
    fa, fg = _to_float(answer), _to_float(g)
    if fa is not None and fg is not None:
        return abs(fa - fg) < 1e-6
    try:  # optional symbolic check (e.g. surds); never hard-fails the run
        import sympy

        if sympy.simplify(sympy.sympify(answer) - sympy.sympify(g)) == 0:
            return True
    except Exception:  # noqa: BLE001 -- sympy missing or unparseable answer
        pass
    return False


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

# The three templates (solve, continue, rollout) each strategy uses, defaulting to the math ones
# above. A benchmark can supply its own (e.g. code) via the ``prompts`` argument.
MATH_PROMPTS = (_SOLVE, _CONTINUE, _ROLLOUT)
CODE_PROMPTS = (
    "Write a complete, correct Python solution to the following problem. Respond with only the "
    "solution as a single ```python code block.\n\nProblem:\n{q}\n",
    "You are writing a Python solution to the problem below.\n\nProblem:\n{q}\n\nSolution so "
    "far:\n{prefix}\nContinue with the next part of the code:",
    "Complete the Python solution to the problem below. Respond with only the full solution as a "
    "single ```python code block.\n\nProblem:\n{q}\n\nSolution so far:\n{prefix}\n",
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
    extract_fn: Callable[[str], str | None] = extract_answer,
    grade_fn: Callable[[str | None, str], bool] = _grade_numeric,
    prompts: tuple = MATH_PROMPTS,
    select_fn: Callable | None = None,
) -> SearchResult:
    """Sample ``n`` full solutions and select one (self-consistency by default).

    ``select_fn(answers, gold) -> answer`` chooses among the extracted answers using only cheap /
    public information; the default is a majority vote (self-consistency). Code benchmarks pass a
    public-test selector so best-of-N and value-guided are selected on the same (public) signal
    and only the final choice is graded on the hidden suite.
    """
    solve_prompt = prompts[0]
    budget = Budget()
    answers: list = []
    for i in range(n):
        text = generate(solve_prompt.format(q=question), max_tokens, i)
        budget.charge(text)
        a = extract_fn(text)
        if a is not None:
            answers.append(a)
    if select_fn is not None:
        answer = select_fn(answers, gold) if answers else None
    else:
        answer = Counter(answers).most_common(1)[0][0] if answers else None
    return SearchResult(answer=answer, correct=grade_fn(answer, gold), budget=budget)


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
    extract_fn: Callable[[str], str | None] = extract_answer,
    grade_fn: Callable[[str | None, str], bool] = _grade_numeric,
    trace_log: list | None = None,
    prompts: tuple = MATH_PROMPTS,
    select_fn: Callable | None = None,
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

    If ``trace_log`` (a list) is provided, one record per candidate node is appended:
    ``{step, candidate, cheap_value, true_value, n_rollouts, chosen}`` where ``cheap_value`` is
    the biased probe (``value_fn``) and ``true_value`` is the *unbiased* node value -- the
    fraction of that candidate's rollouts that actually grade correct against ``gold``. This is
    the data for characterizing the reasoning value function as (almost) tree-$K$-Lipschitz:
    whether the cheap probe predicts the true node value, and how many steps are pivotal
    (violations). Logging grades every rollout but changes neither the search nor the budget.
    """
    if value_fn is None:
        value_fn = lambda texts: _self_consistency(texts, extract_fn)  # noqa: E731
    _, continue_prompt, rollout_prompt = prompts
    budget = Budget()
    prefix = ""
    for step in range(n_steps):
        best_step, best_val, best_c = None, -1.0, 0
        step_records = []
        for c in range(branching):
            cand = generate(
                continue_prompt.format(q=question, prefix=prefix), max_tokens // 2, 1000 * step + c
            )
            budget.charge(cand)
            roll_texts: list[str] = []
            for r in range(rollouts):
                roll = generate(
                    rollout_prompt.format(q=question, prefix=prefix + "\n" + cand),
                    max_tokens,
                    50_000 + 1000 * (step * branching + c) + r,
                )
                budget.charge(roll)
                roll_texts.append(roll)
            val = value_fn(roll_texts)
            if trace_log is not None:
                n_correct = sum(1 for t in roll_texts if grade_fn(extract_fn(t), gold))
                step_records.append(
                    {
                        "step": step,
                        "candidate": c,
                        "cheap_value": float(val),
                        "true_value": n_correct / max(1, len(roll_texts)),
                        "n_rollouts": len(roll_texts),
                        "chosen": False,
                    }
                )
            if val > best_val:
                best_val, best_step = val, cand
                best_c = c
        if trace_log is not None and step_records:
            step_records[best_c]["chosen"] = True
            trace_log.extend(step_records)
        prefix = prefix + "\n" + (best_step or "")
    answers: list = []
    for f in range(final_rollouts):
        final = generate(rollout_prompt.format(q=question, prefix=prefix), max_tokens, 999_000 + f)
        budget.charge(final)
        a = extract_fn(final)
        if a is not None:
            answers.append(a)
    if select_fn is not None:
        answer = select_fn(answers, gold) if answers else None
    else:
        answer = Counter(answers).most_common(1)[0][0] if answers else None
    return SearchResult(answer=answer, correct=grade_fn(answer, gold), budget=budget)


def _self_consistency(rollout_texts: list[str], extract_fn=extract_answer) -> float:
    """Default value: size of the largest agreeing answer set over the rollouts (in [0,1])."""
    answers: Counter[str] = Counter()
    for t in rollout_texts:
        a = extract_fn(t)
        if a is not None:
            answers[a] += 1
    if not answers or not rollout_texts:
        return 0.0
    return answers.most_common(1)[0][1] / len(rollout_texts)
