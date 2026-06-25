"""Tests for Hedge / Randomized Weighted Majority (Chapter 1)."""

from __future__ import annotations

import numpy as np
import pytest

from oco.experts import Hedge, RandomizedWeightedMajority


def test_distribution_is_valid_probability():
    h = Hedge(n_experts=4, learning_rate=0.1)
    dist = h.distribution()
    assert dist.shape == (4,)
    assert dist.sum() == pytest.approx(1.0)
    assert np.all(dist >= 0)


def test_negative_loss_rejected():
    h = Hedge(n_experts=3, learning_rate=0.1)
    with pytest.raises(ValueError):
        h.update(np.array([0.1, -0.2, 0.3]))


def test_weight_shifts_toward_lower_loss_expert():
    h = Hedge(n_experts=2, learning_rate=1.0)
    # Expert 0 always better (lower loss).
    for _ in range(10):
        h.update(np.array([0.0, 1.0]))
    dist = h.distribution()
    assert dist[0] > dist[1]


def test_regret_vanishes_against_best_expert():
    rng = np.random.default_rng(0)
    n, t_total = 5, 3000
    loss_means = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    h = Hedge(n, learning_rate=np.sqrt(np.log(n) / t_total))
    alg_loss, cum = 0.0, np.zeros(n)
    for _ in range(t_total):
        losses = np.clip(loss_means + 0.05 * rng.standard_normal(n), 0, 1)
        alg_loss += h.expected_loss(losses)
        cum += losses
        h.update(losses)
    avg_regret = (alg_loss - cum.min()) / t_total
    assert avg_regret < 0.05  # O(sqrt(log N / T))


def test_rwm_sample_in_range():
    rng = np.random.default_rng(1)
    rwm = RandomizedWeightedMajority(n_experts=3, learning_rate=0.5)
    for _ in range(50):
        assert rwm.sample(rng) in (0, 1, 2)
