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


def test_flat_baseline_learns_and_beats_random():
    horizon = 15000
    flat, rand = [], []
    for seed in range(5):
        env = _env(seed)
        f = run_router(env, horizon, np.random.default_rng(300 + seed), strategy="flat")
        env2 = _env(seed)
        rd = run_router(env2, horizon, np.random.default_rng(300 + seed), strategy="random")
        flat.append(f.final_regret)
        rand.append(rd.final_regret)
    # the structure-blind online learner still learns: lower regret than routing at random.
    assert np.mean(flat) < np.mean(rand)


def test_hierarchical_beats_flat_when_strengths_are_regional():
    # With genuinely complementary regional strengths, the tree-structured learner should
    # outperform the structure-blind one (which commits to a single global model).
    horizon = 15000
    hier, flat = [], []
    for seed in range(5):
        env = _env(seed)
        h = run_router(
            env, horizon, np.random.default_rng(400 + seed), strategy="hierarchical", resolution=2
        )
        env2 = _env(seed)
        f = run_router(env2, horizon, np.random.default_rng(400 + seed), strategy="flat")
        hier.append(h.final_regret)
        flat.append(f.final_regret)
    assert np.mean(hier) < np.mean(flat)


# --- ContextualUCBRouter: episodic credit assignment + identifiability --------------


def _run_episodes(router, n_episodes: int, reward_rng: np.random.Generator) -> float:
    """Episodic environment where the best arm depends on the region.

    Turns alternate easy (region 0) / hard (region 1); arm 0 ('cheap') succeeds on easy
    turns, arm 1 ('strong') on hard ones. The terminal episode reward -- fraction of
    turns routed correctly -- is credited to every (region, arm) pull of the episode
    (Monte-Carlo credit assignment, as in the tau-bench agent loop).
    """
    from canopy.bandits import ContextualUCBRouter

    assert isinstance(router, ContextualUCBRouter)
    total = 0.0
    for _ in range(n_episodes):
        pulls = []
        correct = 0
        for step in range(6):
            region = step % 2  # easy, hard, easy, ...
            arm = router.select(region)
            pulls.append((region, arm))
            correct += int(arm == region)  # best arm index == region index
        reward = correct / 6 + reward_rng.normal(0, 0.02)
        for region, arm in pulls:
            router.update(region, arm, reward)
        total += correct / 6
    return total / n_episodes


def test_contextual_router_learns_per_region_policy_under_episodic_credit():
    from canopy.bandits import ContextualUCBRouter

    router = ContextualUCBRouter(
        2, np.zeros(2), n_regions=2, lam=0.0, c=0.3, rng=np.random.default_rng(7)
    )
    _run_episodes(router, 300, np.random.default_rng(0))
    means = np.divide(router.sums, np.maximum(router.counts, 1))
    assert means[:, 0].argmax() == 0  # easy region -> 'cheap'
    assert means[:, 1].argmax() == 1  # hard region -> 'strong'


def test_jitter_makes_episodic_credit_identifiable():
    """Without the epsilon jitter, regions explore in lockstep and the shared episodic
    reward cannot separate per-region arm means -- the failure the jitter exists to fix."""
    from canopy.bandits import ContextualUCBRouter

    with_jitter = ContextualUCBRouter(
        2, np.zeros(2), n_regions=2, lam=0.0, c=0.3, rng=np.random.default_rng(8)
    )
    late_with = _run_episodes(with_jitter, 300, np.random.default_rng(1))
    late_with = _run_episodes(with_jitter, 100, np.random.default_rng(2))

    no_jitter = ContextualUCBRouter(2, np.zeros(2), n_regions=2, lam=0.0, c=0.3, explore_eps=0.0)
    _run_episodes(no_jitter, 300, np.random.default_rng(1))
    late_without = _run_episodes(no_jitter, 100, np.random.default_rng(2))

    assert late_with > 0.9  # learned the per-region policy
    assert late_with > late_without + 0.2  # lockstep variant stays near chance (~0.5)


def test_flat_router_cannot_express_region_policy():
    from canopy.bandits import ContextualUCBRouter

    flat = ContextualUCBRouter(
        2, np.zeros(2), n_regions=1, lam=0.0, c=0.3, rng=np.random.default_rng(9)
    )
    _run_episodes(flat, 300, np.random.default_rng(3))
    late = _run_episodes(flat, 100, np.random.default_rng(4))
    assert late < 0.75  # a single-region policy caps at ~0.5 correct routing (+ jitter noise)


def test_explore_eps_zero_is_deterministic_per_pull():
    from canopy.bandits import ContextualUCBRouter

    r = ContextualUCBRouter(3, np.array([0.1, 0.5, 1.0]), n_regions=1, explore_eps=0.0)
    first = []
    for _ in range(3):  # select+update cycles try every untried arm once, deterministically
        arm = r.select(0)
        first.append(arm)
        r.update(0, arm, 0.5)
    assert sorted(first) == [0, 1, 2]
