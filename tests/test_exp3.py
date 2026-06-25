"""Tests for EXP3 (Chapter 6)."""

from __future__ import annotations

import numpy as np
import pytest

from canopy.bandits import EXP3


def test_distribution_valid():
    e = EXP3(n_arms=4, learning_rate=0.1)
    dist = e.distribution()
    assert dist.sum() == pytest.approx(1.0)
    assert np.all(dist >= 0)


def test_select_returns_valid_arm_and_distribution():
    rng = np.random.default_rng(0)
    e = EXP3(n_arms=3, learning_rate=0.1)
    arm, dist = e.select(rng)
    assert arm in (0, 1, 2)
    assert dist.sum() == pytest.approx(1.0)


def test_mass_concentrates_on_best_arm():
    rng = np.random.default_rng(0)
    n, t_total = 4, 5000
    loss_means = np.array([0.1, 0.4, 0.6, 0.9])  # arm 0 best
    e = EXP3(n, learning_rate=np.sqrt(np.log(n) / (t_total * n)))
    for _ in range(t_total):
        losses = np.clip(loss_means + 0.05 * rng.standard_normal(n), 0, 1)
        arm, dist = e.select(rng)
        e.update(arm, losses[arm], dist)
    assert e.distribution().argmax() == 0
