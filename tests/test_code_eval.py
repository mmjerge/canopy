"""Tests for the execution-graded code probes (canopy.bandits.code_eval), no LLM needed."""

from __future__ import annotations

import numpy as np

from canopy.bandits.code_eval import (
    extract_code,
    grade_code,
    probe_truth_pairs,
    public_fraction,
)

GOOD = "def add(a, b):\n    return a + b"
BAD = "def add(a, b):\n    return a - b"
SUBTLE = "def add(a, b):\n    return a + b if a >= 0 else a - b"  # passes positive-only tests
GOLD = {
    "code_prefix": "",
    "entry_point": "add",
    "public": ["assert add(1, 2) == 3", "assert add(2, 2) == 4"],
    "hidden_suffix": (
        "assert add(1, 2) == 3\nassert add(2, 2) == 4\n"
        "assert add(-1, 1) == 0\nassert add(-2, -3) == -5"
    ),
}


def test_public_fraction_is_continuous():
    assert public_fraction(GOOD, GOLD) == 1.0
    assert public_fraction(BAD, GOLD) == 0.0
    assert public_fraction(SUBTLE, GOLD) == 1.0  # the plausible-but-wrong candidate
    half = {**GOLD, "public": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]}
    assert public_fraction(SUBTLE, half) == 0.5  # partial pass -> fractional score
    assert public_fraction("", GOLD) == 0.0


def test_probe_truth_pairs_expose_weak_probe():
    """The Stage-1 gate: an all-positive public suite cannot separate SUBTLE from GOOD."""
    texts = [f"```python\n{c}\n```" for c in (GOOD, SUBTLE, BAD)]
    pairs = np.array(probe_truth_pairs(texts, GOLD))
    probe, truth = pairs[:, 0], pairs[:, 1]
    # GOOD and SUBTLE both max the probe but differ on truth -- the probe cannot rank them
    assert probe[0] == probe[1] == 1.0
    assert truth[0] == 1.0 and truth[1] == 0.0
    # while BAD is separated by both
    assert probe[2] == 0.0 and truth[2] == 0.0


def test_grade_and_extract_roundtrip():
    assert grade_code(GOOD, GOLD)
    assert not grade_code(SUBTLE, GOLD)
    assert extract_code(f"Here:\n```python\n{GOOD}\n```") == GOOD
