"""Tests for the online / regret-minimization mode."""

from __future__ import annotations

import numpy as np

from oco.bandits import (
    TreeBandit,
    geometric_sigma,
    hierarchical_spread,
    run_adaptive,
    run_adaptive_mgf,
    run_adaptive_variance,
    run_fixed_depth,
    run_hoo,
    run_hybrid,
)
from oco.bandits.tree import Node


def test_play_is_unbiased_for_subtree_average():
    env = TreeBandit(branching=2, depth=2, leaf_means=np.array([0.0, 0.2, 0.8, 1.0]),
                     noise_std=0.05, rng=np.random.default_rng(0))
    node = Node(1, 0)  # covers leaves [0.0, 0.2], true value 0.1
    rng = np.random.default_rng(1)
    obs = [env.play(node, rng) for _ in range(20000)]
    assert abs(np.mean(obs) - 0.1) < 0.02  # play() is unbiased for the subtree average


def test_fixed_depth_shapes_and_memory():
    env = TreeBandit.from_hierarchical_gaussian(4, 3, rng=np.random.default_rng(0))
    res = run_fixed_depth(env, depth=2, horizon=2000, rng=np.random.default_rng(1))
    assert res.cum_regret.shape == (2000,)
    assert res.memory == 4**2
    # cumulative regret is non-decreasing (per-round regret >= 0)
    assert np.all(np.diff(res.cum_regret) >= -1e-9)


def test_adaptive_runs_and_is_bounded():
    env = TreeBandit.from_hierarchical_gaussian(4, 4, rng=np.random.default_rng(0))
    spread = hierarchical_spread(geometric_sigma(0.5, 0.55), 4, 4, z=3.0)
    res = run_adaptive(env, horizon=4000, spread=spread, rng=np.random.default_rng(1))
    assert res.cum_regret.shape == (4000,)
    assert np.all(np.diff(res.cum_regret) >= -1e-9)
    total_nodes = sum(4**level for level in range(5))
    assert 1 <= res.memory <= total_nodes


def test_adaptive_beats_root_and_saves_memory_vs_full():
    branching, depth, horizon = 4, 4, 8000
    spread = hierarchical_spread(geometric_sigma(0.5, 0.55), depth, branching, z=3.0)
    adaptive, root, full = [], [], []
    adaptive_mem = []
    for seed in range(5):
        env = TreeBandit.from_hierarchical_gaussian(
            branching, depth, sigma=geometric_sigma(0.5, 0.55), noise_std=0.1,
            rng=np.random.default_rng(seed)
        )
        a = run_adaptive(env, horizon, spread, np.random.default_rng(100 + seed))
        adaptive.append(a.final_regret)
        adaptive_mem.append(a.memory)
        env0 = TreeBandit.from_hierarchical_gaussian(
            branching, depth, sigma=geometric_sigma(0.5, 0.55), noise_std=0.1,
            rng=np.random.default_rng(seed)
        )
        root.append(run_fixed_depth(env0, 0, horizon, np.random.default_rng(100 + seed)).final_regret)
        envD = TreeBandit.from_hierarchical_gaussian(
            branching, depth, sigma=geometric_sigma(0.5, 0.55), noise_std=0.1,
            rng=np.random.default_rng(seed)
        )
        full.append(run_fixed_depth(envD, depth, horizon, np.random.default_rng(100 + seed)).final_regret)
    # Adaptive crushes the root-only (pure bias) baseline ...
    assert np.mean(adaptive) < np.mean(root)
    # ... and tracks full-resolution regret while using far less memory than 256.
    assert np.mean(adaptive_mem) < branching**depth
    assert np.mean(adaptive) < 2.0 * np.mean(full)


def test_variance_aware_is_competitive_and_memory_light():
    # The novel variant should match full-resolution regret without assuming spread,
    # and use far less memory than full resolution.
    branching, depth, horizon = 4, 4, 8000
    var_regret, var_mem, full = [], [], []
    for seed in range(5):
        env = TreeBandit.from_hierarchical_gaussian(
            branching, depth, sigma=geometric_sigma(0.5, 0.55), noise_std=0.1,
            rng=np.random.default_rng(seed)
        )
        r = run_adaptive_variance(env, horizon, np.random.default_rng(200 + seed))
        var_regret.append(r.final_regret)
        var_mem.append(r.memory)
        envD = TreeBandit.from_hierarchical_gaussian(
            branching, depth, sigma=geometric_sigma(0.5, 0.55), noise_std=0.1,
            rng=np.random.default_rng(seed)
        )
        full.append(run_fixed_depth(envD, depth, horizon, np.random.default_rng(200 + seed)).final_regret)
    assert np.all(np.array(var_regret) >= 0)
    assert np.mean(var_mem) < branching**depth
    assert np.mean(var_regret) < 1.5 * np.mean(full)


def test_adaptive_mgf_runs_and_is_bounded():
    # The MGF-certified variant is conservative (worse regret) but must run cleanly and
    # produce a valid, non-decreasing regret curve within the node budget.
    env = TreeBandit.from_hierarchical_gaussian(4, 4, rng=np.random.default_rng(0))
    res = run_adaptive_mgf(env, horizon=3000, rng=np.random.default_rng(1))
    assert res.cum_regret.shape == (3000,)
    assert np.all(np.diff(res.cum_regret) >= -1e-9)
    assert res.memory <= sum(4**level for level in range(5))


def test_hoo_runs_and_beats_root():
    branching, depth, horizon = 4, 4, 6000
    spread = hierarchical_spread(geometric_sigma(0.5, 0.55), depth, branching, z=3.0)
    hoo, root = [], []
    for seed in range(4):
        env = TreeBandit.from_hierarchical_gaussian(
            branching, depth, sigma=geometric_sigma(0.5, 0.55), noise_std=0.1,
            rng=np.random.default_rng(seed)
        )
        r = run_hoo(env, horizon, spread, np.random.default_rng(300 + seed))
        assert r.cum_regret.shape == (horizon,)
        assert np.all(np.diff(r.cum_regret) >= -1e-9)
        hoo.append(r.final_regret)
        env0 = TreeBandit.from_hierarchical_gaussian(
            branching, depth, sigma=geometric_sigma(0.5, 0.55), noise_std=0.1,
            rng=np.random.default_rng(seed)
        )
        root.append(run_fixed_depth(env0, 0, horizon, np.random.default_rng(300 + seed)).final_regret)
    assert np.mean(hoo) < np.mean(root)


def test_hoo_memory_bounded_compresses_on_deep_tree():
    # On a deep tree, memory-bounded HOO explores far fewer nodes than the tree size.
    branching, depth, horizon = 3, 8, 8000
    spread = hierarchical_spread(geometric_sigma(0.5, 0.6), depth, branching, z=3.0)
    env = TreeBandit.from_hierarchical_gaussian(
        branching, depth, sigma=geometric_sigma(0.5, 0.6), noise_std=0.1,
        rng=np.random.default_rng(0)
    )
    res = run_hoo(env, horizon, spread, np.random.default_rng(1), memory_bounded=True)
    total_nodes = sum(branching**level for level in range(depth + 1))
    assert res.memory < total_nodes / 10  # finite-state compression



def test_data_driven_robust_to_hidden_jumps():
    # With narrow (hidden) jump discontinuities, the data-driven variance-aware bias
    # beats the assumed-smooth spread schedule, which gets stuck refining jump cells.
    branching, depth, horizon = 4, 5, 12000
    spread_smooth = hierarchical_spread(geometric_sigma(0.12, 0.6), depth, branching, z=3.0)
    assumed, datadriven = [], []
    for seed in range(6):
        env = TreeBandit.from_piecewise_smooth(
            branching, depth, n_jumps=4, jump_width=4, noise_std=0.1,
            rng=np.random.default_rng(seed)
        )
        assumed.append(run_hoo(env, horizon, spread_smooth, np.random.default_rng(100 + seed),
                               memory_bounded=True).final_regret)
        env2 = TreeBandit.from_piecewise_smooth(
            branching, depth, n_jumps=4, jump_width=4, noise_std=0.1,
            rng=np.random.default_rng(seed)
        )
        datadriven.append(run_adaptive_variance(env2, horizon, np.random.default_rng(100 + seed)).final_regret)
    assert np.mean(datadriven) < np.mean(assumed)



def test_hybrid_beats_assumed_smooth_on_hidden_jumps():
    # The Lipschitz-floor + jump-detection hybrid avoids the assumed-smooth blow-up on
    # narrow/hidden jumps and stays competitive with the data-driven method.
    branching, depth, horizon = 4, 5, 12000
    spread_smooth = hierarchical_spread(geometric_sigma(0.12, 0.6), depth, branching, z=3.0)
    assumed, hybrid = [], []
    for seed in range(6):
        env = TreeBandit.from_piecewise_smooth(
            branching, depth, n_jumps=4, jump_width=4, noise_std=0.1,
            rng=np.random.default_rng(seed)
        )
        assumed.append(run_hoo(env, horizon, spread_smooth, np.random.default_rng(100 + seed),
                               memory_bounded=True).final_regret)
        env2 = TreeBandit.from_piecewise_smooth(
            branching, depth, n_jumps=4, jump_width=4, noise_std=0.1,
            rng=np.random.default_rng(seed)
        )
        hybrid.append(run_hybrid(env2, horizon, spread_smooth, np.random.default_rng(100 + seed),
                                 jump_factor=0.5, n_min=8).final_regret)
    assert np.mean(hybrid) < np.mean(assumed)
