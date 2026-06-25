"""Tests for top-k leaf identification (UniformTopK and HierarchicalTopK)."""

from __future__ import annotations

import numpy as np
import pytest

from canopy.bandits import (
    HierarchicalTopK,
    TreeBandit,
    UniformTopK,
    geometric_sigma,
    hierarchical_spread,
)


def _easy_env(noise_std=0.05, seed=0):
    # Top-2 leaves (indices 6, 7) clearly separated from the rest.
    leaf_means = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.50, 0.90, 0.95])
    return TreeBandit(branching=2, depth=3, leaf_means=leaf_means,
                      noise_std=noise_std, rng=np.random.default_rng(seed))


def test_uniform_returns_k_leaves_and_respects_budget():
    env = _easy_env()
    res = UniformTopK(budget=800).run(env, k=2)
    assert len(res.leaves) == 2
    assert res.cost <= 800


def test_uniform_respects_budget_below_n_leaves():
    # 1024 leaves but only 200 cost (leaf_cost=1): must not exceed budget, most unseen.
    env = TreeBandit.from_hierarchical_gaussian(
        branching=4, depth=5, rng=np.random.default_rng(0)
    )
    res = UniformTopK(budget=200).run(env, k=5)
    assert res.cost <= 200
    unseen = sum(1 for v in res.estimates.values() if v == float("-inf"))
    assert unseen >= env.n_leaves - 200


def test_hierarchical_returns_k_leaves_and_respects_budget():
    env = _easy_env()
    res = HierarchicalTopK(budget=2000).run(env, k=2)
    assert len(res.leaves) == 2
    assert res.cost <= 2000 + 4 * env.leaf_cost  # small per-round overshoot allowed


def test_invalid_k_rejected():
    env = _easy_env()
    with pytest.raises(ValueError):
        HierarchicalTopK(budget=500).run(env, k=0)
    with pytest.raises(ValueError):
        HierarchicalTopK(budget=500).run(env, k=99)


def test_recovers_top_k_on_easy_instance():
    # With clear gaps, both methods should find the true top-2.
    env_u = _easy_env(seed=1)
    res_u = UniformTopK(budget=2000).run(env_u, k=2)
    assert res_u.evaluate(env_u, k=2) == 1.0

    env_h = _easy_env(seed=1)
    res_h = HierarchicalTopK(budget=4000).run(env_h, k=2)
    assert res_h.evaluate(env_h, k=2) == 1.0


def test_certification_implies_correct_topk():
    # When the algorithm certifies, the certified leaves must be the true top-k.
    # Use a well-separated, low-noise instance and a generous budget.
    certified_runs = 0
    for seed in range(10):
        env = _easy_env(noise_std=0.03, seed=seed)
        res = HierarchicalTopK(budget=20000, confidence=0.05).run(env, k=2)
        if res.certified:
            certified_runs += 1
            assert set(res.leaves) == set(env.top_k_leaves(2))
    assert certified_runs >= 1  # certification should fire on an easy instance


def test_hierarchical_beats_uniform_in_tight_budget_regime():
    # Many leaves, tight budget, cheap biased probes, best-first beam: the tree should
    # clearly beat the structure-blind uniform baseline.
    branching, depth, k = 4, 5, 5  # 1024 leaves
    budget, n_seeds = 1500, 12
    sigma = geometric_sigma(base=0.5, decay=0.55)
    spread = hierarchical_spread(sigma, depth, branching, z=3.0)
    hier, uni = [], []
    for seed in range(n_seeds):
        env_h = TreeBandit.from_hierarchical_gaussian(
            branching, depth, sigma=sigma, noise_std=0.1, probe_cost=0.05,
            rng=np.random.default_rng(seed)
        )
        hier.append(
            HierarchicalTopK(budget=budget, spread=spread, beam_width=20).run(env_h, k).evaluate(env_h, k)
        )
        env_u = TreeBandit.from_hierarchical_gaussian(
            branching, depth, sigma=sigma, noise_std=0.1, probe_cost=0.05,
            rng=np.random.default_rng(seed)
        )
        uni.append(UniformTopK(budget=budget).run(env_u, k).evaluate(env_u, k))
    assert np.mean(hier) > np.mean(uni)
