"""Tests for the structure-blind SuccessiveEliminationTopK baseline."""

from __future__ import annotations

import numpy as np
import pytest

from canopy.bandits import (
    HierarchicalTopK,
    SuccessiveEliminationTopK,
    TreeBandit,
    geometric_sigma,
    hierarchical_spread,
)


def _easy_env(noise_std=0.05, seed=0):
    leaf_means = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.50, 0.90, 0.95])
    return TreeBandit(
        branching=2,
        depth=3,
        leaf_means=leaf_means,
        noise_std=noise_std,
        rng=np.random.default_rng(seed),
    )


def test_returns_k_leaves_and_respects_budget():
    env = _easy_env()
    res = SuccessiveEliminationTopK(budget=1500).run(env, k=2)
    assert len(res.leaves) == 2
    assert res.cost <= 1500 + env.leaf_cost


def test_recovers_top_k_on_easy_instance():
    env = _easy_env(seed=3)
    res = SuccessiveEliminationTopK(budget=4000).run(env, k=2)
    assert res.evaluate(env, k=2) == 1.0


def test_certification_implies_correct():
    certified = 0
    for seed in range(10):
        env = _easy_env(noise_std=0.03, seed=seed)
        res = SuccessiveEliminationTopK(budget=20000).run(env, k=2)
        if res.certified:
            certified += 1
            assert set(res.leaves) == set(env.top_k_leaves(2))
    assert certified >= 1


def test_invalid_k_rejected():
    env = _easy_env()
    with pytest.raises(ValueError):
        SuccessiveEliminationTopK(budget=500).run(env, k=0)


def test_tree_beats_strong_baseline_when_probes_are_cheap():
    # The central multi-fidelity claim: when internal probes are much cheaper than leaf
    # evaluations, the tree localizes good leaves for nearly free and beats the strong
    # structure-blind baseline -- especially under a tight budget.
    branching, depth, k = 4, 5, 5  # 1024 leaves
    budget, n_seeds = 400, 12  # tight
    sigma = geometric_sigma(base=0.5, decay=0.55)
    spread = hierarchical_spread(sigma, depth, branching, z=3.0)
    hier, se = [], []
    for seed in range(n_seeds):
        env_h = TreeBandit.from_hierarchical_gaussian(
            branching,
            depth,
            sigma=sigma,
            noise_std=0.1,
            probe_cost=0.05,
            rng=np.random.default_rng(seed),
        )
        hier.append(
            HierarchicalTopK(budget=budget, spread=spread, beam_width=20)
            .run(env_h, k)
            .evaluate(env_h, k)
        )
        env_s = TreeBandit.from_hierarchical_gaussian(
            branching,
            depth,
            sigma=sigma,
            noise_std=0.1,
            probe_cost=0.05,
            rng=np.random.default_rng(seed),
        )
        se.append(SuccessiveEliminationTopK(budget=budget).run(env_s, k).evaluate(env_s, k))
    assert np.mean(hier) > np.mean(se) + 0.1  # large, robust margin in this regime
