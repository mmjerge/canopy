"""Tests for the execution-graded code probe (canopy.bandits.code_llm), no LLM needed."""

from __future__ import annotations

import numpy as np

from canopy.bandits.code_llm import (
    best_of_n_exec,
    exec_value_fn,
    extract_code,
    probe_truth_pairs,
    run_tests,
)

GOOD = "def add(a, b):\n    return a + b"
BAD = "def add(a, b):\n    return a - b"
SUBTLE = "def add(a, b):\n    return a + b if a >= 0 else a - b"  # passes positive-only tests
PUBLIC = ["assert add(1, 2) == 3", "assert add(2, 2) == 4"]
HIDDEN = PUBLIC + ["assert add(-1, 1) == 0", "assert add(-2, -3) == -5"]


def test_run_tests_fractions():
    assert run_tests(GOOD, HIDDEN) == 1.0
    assert run_tests(BAD, HIDDEN) == 0.0
    assert run_tests(SUBTLE, PUBLIC) == 1.0  # the plausible-but-wrong candidate
    assert run_tests(SUBTLE, HIDDEN) == 0.5
    assert run_tests("def add(a, b): raise ValueError", PUBLIC) == 0.0
    assert run_tests(GOOD, []) == 0.0


def test_run_tests_contains_hangs():
    assert run_tests("import time\ndef add(a, b):\n    time.sleep(60)", PUBLIC, timeout=1.0) == 0.0


def test_extract_code():
    assert extract_code(f"Here you go:\n```python\n{GOOD}\n```") == GOOD
    assert extract_code(GOOD) == GOOD
    assert extract_code("```\nx = 1\n```\ntext\n```python\ny = 2\n```") == "y = 2"


def test_exec_value_fn_is_continuous():
    value = exec_value_fn(PUBLIC)
    assert value([GOOD, BAD]) == 0.5  # mean of 1.0 and 0.0 -- not a single bit
    assert value([]) == 0.0


def test_best_of_n_exec_submits_by_probe():
    completions = [BAD, GOOD, SUBTLE]

    def gen(prompt, max_tokens, seed):
        return completions[seed % len(completions)]

    r = best_of_n_exec("p", PUBLIC, HIDDEN, gen, n=3)
    assert r.passed  # GOOD scores 1.0 on public first (earlier tie beats SUBTLE) and passes hidden
    assert r.budget.calls == 3


def test_probe_truth_pairs_expose_weak_probe():
    """The Stage-1 gate: an all-positive public suite cannot separate SUBTLE from GOOD."""
    completions = [GOOD, SUBTLE, BAD]

    def gen(prompt, max_tokens, seed):
        return completions[seed % len(completions)]

    pairs = np.array(probe_truth_pairs("p", PUBLIC, HIDDEN, gen, n=3))
    probe, truth = pairs[:, 0], pairs[:, 1]
    # GOOD and SUBTLE both max the probe but differ on truth -- the probe cannot rank them
    assert probe[0] == probe[1] == 1.0
    assert truth[0] == 1.0 and truth[1] == 0.5
    # while BAD is separated by both
    assert probe[2] == 0.0 and truth[2] == 0.0