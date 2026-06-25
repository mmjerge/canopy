"""Tests for the prefix-tree LLM routing instantiation."""

from __future__ import annotations

import numpy as np

from canopy.bandits import PrefixTreeRouting, make_routing_scenario, run_router


def _env(seed, lam=0.3):
    q, costs = make_routing_scenario(4, 5, np.random.default_rng(seed))
    return PrefixTreeRouting(
        4, 5, q, costs, lam=lam, noise_std=0.1, rng=np.random.default_rng(100 + seed)
    )


def test_oracle_has_zero_regret():
    env = _env(0)
    res = run_router(env, 5000, np.random.default_rng(1), strategy="oracle")
    assert res.final_regret == 0.0
    assert res.cum_regret.shape == (5000,)


def test_region_partitions_leaves():
    env = _env(0)
    for resolution in range(env.depth + 1):
        regions = [env.region(leaf, resolution) for leaf in range(env.n_leaves)]
        assert set(regions) == set(range(env.branching**resolution))


def test_router_beats_fixed_policies_and_saves_cost():
    horizon = 15000
    router, best_single, all_largest = [], [], []
    router_cost, big_cost = [], []
    for seed in range(5):
        env = _env(seed)
        r = run_router(
            env, horizon, np.random.default_rng(200 + seed), strategy="hierarchical", resolution=2
        )
        env2 = _env(seed)
        bs = run_router(env2, horizon, np.random.default_rng(200 + seed), strategy="best_single")
        env3 = _env(seed)
        al = run_router(env3, horizon, np.random.default_rng(200 + seed), strategy="all_largest")
        router.append(r.final_regret)
        best_single.append(bs.final_regret)
        all_largest.append(al.final_regret)
        router_cost.append(r.total_cost)
        big_cost.append(al.total_cost)
    # learns regional routing -> much lower regret than the best fixed policy ...
    assert np.mean(router) < 0.5 * np.mean(best_single)
    # ... and spends less than always routing to the big model.
    assert np.mean(router_cost) < np.mean(big_cost)
