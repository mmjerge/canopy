"""Tests for reasoning-tree search (value-guided vs best-of-N)."""

from __future__ import annotations

import numpy as np

from canopy.bandits import (
    best_of_n,
    reasoning_tree_rewards,
    success_rate,
    value_guided_search,
)


def test_reward_structure():
    reward, decisions = reasoning_tree_rewards(depth=8, n_decisions=5, rng=np.random.default_rng(0))
    assert reward.shape == (2**8,)
    assert decisions.size == 5
    assert np.isclose(reward.max(), 1.0)        # a fully-correct trace exists
    assert reward.min() >= 0.0
    # exactly one in 2^5 leaves is fully correct (per distinct decision pattern)
    assert np.isclose(np.mean(np.isclose(reward, 1.0)), 2.0 ** (-5), atol=0.01)


def test_value_guided_beats_best_of_n_at_fixed_budget():
    k, budget = 8, 1024
    vg = success_rate("value_guided", depth=k, n_decisions=k, budget=budget, sigma=0.3, seeds=120)
    bo = success_rate("best_of_n", depth=k, n_decisions=k, budget=budget, sigma=0.3, seeds=120)
    assert vg > 0.6
    assert vg > bo + 0.3      # large, decisive gap


def test_best_of_n_is_exponential_value_guided_polynomial():
    # at K=10, best-of-N essentially fails at a budget where value-guided largely succeeds
    bo = success_rate("best_of_n", depth=10, n_decisions=10, budget=4096, sigma=0.3, seeds=120)
    vg = success_rate("value_guided", depth=10, n_decisions=10, budget=4096, sigma=0.3, seeds=120)
    assert bo < 0.2
    assert vg > 0.6


def test_noiseless_value_guided_is_near_perfect():
    # the probe is a random rollout, so success needs enough rollouts to resolve the 1/K gap;
    # with no observation noise and enough probes it is essentially perfect.
    out = []
    for seed in range(60):
        reward, _ = reasoning_tree_rewards(depth=8, n_decisions=8, rng=np.random.default_rng(seed))
        ok, _ = value_guided_search(reward, depth=8, probes_per_child=64, sigma=0.0,
                                    rng=np.random.default_rng(100 + seed))
        out.append(ok)
    assert np.mean(out) > 0.95


def test_best_of_n_returns_a_sampled_reward():
    reward, _ = reasoning_tree_rewards(depth=6, n_decisions=6, rng=np.random.default_rng(3))
    ok, r = best_of_n(reward, n_samples=20, sigma=0.1, rng=np.random.default_rng(4))
    assert 0.0 <= r <= 1.0
    assert isinstance(ok, bool)
