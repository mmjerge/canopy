"""Execution-graded value probes for code-generation benchmarks.

The pre-registered HumanEval/MBPP null (paper 4.3) traced the failure to the probe: a
*single* public assert is a one-bit, high-variance signal that plausible-but-wrong
completions often pass. This module supplies the repaired, execution-graded probe and
the measurement harness for the Stage-1 gate of docs/code_benchmarks.md:

* :func:`run_tests` -- fraction of test snippets a candidate passes, each executed in
  an isolated subprocess with a timeout (the standard benchmark-harness pattern).
* :func:`extract_code` -- pull the code block out of a model completion.
* :func:`exec_value_fn` -- a ``value_fn`` for the search machinery: the mean public-test
  pass fraction over rollout completions (continuous in [0, 1], not one bit).
* :func:`best_of_n_exec` -- structure-blind baseline with the same probe: sample ``n``
  completions, submit the one with the highest public-test score, grade on the hidden
  suite.
* :func:`probe_truth_pairs` -- the Stage-1 gate measurement: per candidate, (cheap
  public-test score, true hidden-suite pass) pairs, whose correlation is the
  tree-Lipschitz-backbone statistic for code (the analog of the cheap-vs-true node
  value correlation on MATH).

Note on execution: candidates are model-generated code and are executed. Each runs in
a separate subprocess with a wall-clock timeout, which contains crashes and hangs but
is NOT a security sandbox; run benchmark evaluation in an environment you trust (or a
container), as with any HumanEval/MBPP harness.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field

from canopy.bandits.reasoning_llm import Budget, GenerateFn

_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Return the last fenced code block, else the raw text (models often emit bare code)."""
    blocks = _FENCE.findall(text)
    return (blocks[-1] if blocks else text).strip()


def run_tests(code: str, tests: list[str], timeout: float = 5.0) -> float:
    """Fraction of ``tests`` (assert snippets) that pass against ``code``.

    Each test runs as ``code + test`` in its own subprocess: a crash, exception,
    failed assert, or timeout counts as a fail for that test only.
    """
    if not tests:
        return 0.0
    passed = 0
    for test in tests:
        script = f"{code}\n\n{test}\n"
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                timeout=timeout,
            )
            passed += int(proc.returncode == 0)
        except subprocess.TimeoutExpired:
            pass
    return passed / len(tests)


def exec_value_fn(public_tests: list[str], timeout: float = 5.0):
    """Value function for the search machinery: mean public-test pass fraction of the
    rollout completions. Continuous in [0, 1] -- the repaired cheap probe."""

    def value(rollout_texts: list[str]) -> float:
        if not rollout_texts:
            return 0.0
        scores = [run_tests(extract_code(t), public_tests, timeout) for t in rollout_texts]
        return float(sum(scores) / len(scores))

    return value


@dataclass
class CodeResult:
    """Outcome of one code-generation attempt set."""

    passed: bool  # hidden suite fully passed by the submitted candidate
    probe_score: float  # public-test score of the submitted candidate
    budget: Budget = field(default_factory=Budget)


def best_of_n_exec(
    prompt: str,
    public_tests: list[str],
    hidden_tests: list[str],
    generate: GenerateFn,
    n: int,
    max_tokens: int = 512,
    timeout: float = 5.0,
) -> CodeResult:
    """Sample ``n`` completions, submit the best by public-test score, grade on hidden.

    This is the structure-blind baseline with the execution probe: the probe picks the
    submission, the hidden suite decides correctness. Ties go to the earlier sample.
    """
    budget = Budget()
    best_code, best_score = "", -1.0
    for i in range(n):
        text = generate(prompt, max_tokens, i)
        budget.charge(text)
        code = extract_code(text)
        score = run_tests(code, public_tests, timeout)
        if score > best_score:
            best_score, best_code = score, code
    ok = run_tests(best_code, hidden_tests, timeout) == 1.0 if hidden_tests else False
    return CodeResult(passed=ok, probe_score=max(best_score, 0.0), budget=budget)


def probe_truth_pairs(
    prompt: str,
    public_tests: list[str],
    hidden_tests: list[str],
    generate: GenerateFn,
    n: int,
    max_tokens: int = 512,
    timeout: float = 5.0,
) -> list[tuple[float, float]]:
    """The Stage-1 gate measurement: (cheap probe score, true hidden pass) per candidate.

    Sample ``n`` candidates and, for each, record the public-test fraction (the cheap
    value the search would follow) and the hidden-suite pass fraction (the truth). The
    correlation across candidates -- pooled over problems -- is the code analog of the
    cheap-vs-true node-value correlation measured on MATH: if it is near zero, the
    theory predicts value-guided search cannot beat best-of-N here, and the race is
    not worth running.
    """
    pairs = []
    for i in range(n):
        code = extract_code(generate(prompt, max_tokens, i))
        pairs.append(
            (run_tests(code, public_tests, timeout), run_tests(code, hidden_tests, timeout))
        )
    return pairs
