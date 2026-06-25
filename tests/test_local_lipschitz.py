"""Tests for locally-adaptive Lipschitz expansion."""

from __future__ import annotations

import numpy as np

from canopy.bandits import TreeBandit, lipschitz_spread, run_hoo, run_local_lipschitz
from canopy.bandits.rewards import heterogeneous_smoothness_leaf_means


def test_runs_and_bounded():
    env = TreeBandit.from_hierarchical_gaussian(4, 4, rng=np.random.default_rng(0))
    res = run_local_lipschitz(env, horizon=4000, rng=np.random.default_rng(1))
    assert res.cum_regret.shape == (4000,)
    assert np.all(np.diff(res.cum_regret) >= -1e-9)
    assert res.memory <= sum(4**level for level in range(5))


def test_local_adaptive_beats_global_constants_on_heterogeneous_tree():
    branching, depth, horizon = 4, 5, 12000
    rho = 1.0 / branching
    tight = lipschitz_spread(0.15, rho)
    loose = lipschitz_spread(1.5, rho)
    local, g_tight, g_loose = [], [], []
    for seed in range(6):
        lm = heterogeneous_smoothness_leaf_means(branching, depth, np.random.default_rng(seed))
        e1 = TreeBandit(branching, depth, leaf_means=lm, noise_std=0.1, rng=np.random.default_rng(100 + seed))
        local.append(run_local_lipschitz(e1, horizon, np.random.default_rng(100 + seed)).final_regret)
        e2 = TreeBandit(branching, depth, leaf_means=lm, noise_std=0.1, rng=np.random.default_rng(100 + seed))
        g_tight.append(run_hoo(e2, horizon, tight, np.random.default_rng(100 + seed), memory_bounded=True).final_regret)
        e3 = TreeBandit(branching, depth, leaf_means=lm, noise_std=0.1, rng=np.random.default_rng(100 + seed))
        g_loose.append(run_hoo(e3, horizon, loose, np.random.default_rng(100 + seed), memory_bounded=True).final_regret)
    assert np.mean(local) < np.mean(g_tight)
    assert np.mean(local) < np.mean(g_loose)
