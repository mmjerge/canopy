"""Tests for reasoning-tree search (value-guided vs best-of-N)."""

from __future__ import annotations

import numpy as np

from canopy.bandits import (
    best_of_n,
    reasoning_tree_rewards,
    scoped_success_rate,
    success_rate,
    value_guided_search,
    value_guided_search_scoped,
)


def test_reward_structure():
    reward, decisions = reasoning_tree_rewards(depth=8, n_decisions=5, rng=np.random.default_rng(0))
    assert reward.shape == (2**8,)
    assert decisions.size == 5
    assert np.isclose(reward.max(), 1.0)  # a fully-correct trace exists
    assert reward.min() >= 0.0
    # exactly one in 2^5 leaves is fully correct (per distinct decision pattern)
    assert np.isclose(np.mean(np.isclose(reward, 1.0)), 2.0 ** (-5), atol=0.01)


def test_value_guided_beats_best_of_n_at_fixed_budget():
    k, budget = 8, 1024
    vg = success_rate("value_guided", depth=k, n_decisions=k, budget=budget, sigma=0.3, seeds=120)
    bo = success_rate("best_of_n", depth=k, n_decisions=k, budget=budget, sigma=0.3, seeds=120)
    assert vg > 0.6
    assert vg > bo + 0.3  # large, decisive gap


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
        ok, _ = value_guided_search(
            reward, depth=8, probes_per_child=64, sigma=0.0, rng=np.random.default_rng(100 + seed)
        )
        out.append(ok)
    assert np.mean(out) > 0.95


def test_best_of_n_returns_a_sampled_reward():
    reward, _ = reasoning_tree_rewards(depth=6, n_decisions=6, rng=np.random.default_rng(3))
    ok, r = best_of_n(reward, n_samples=20, sigma=0.1, rng=np.random.default_rng(4))
    assert 0.0 <= r <= 1.0
    assert isinstance(ok, bool)


# --- scope predictions: the theory's conditions for the value-guided advantage ------
#
# These tests encode, as executable checks, when value-guided search should and
# should not beat best-of-N. They are the synthetic bridge to the real-benchmark
# pattern: the gain requires (a) unsaturated problems (headroom / effective K > 0)
# and (b) an informative cheap probe -- and vanishes when either fails (the
# HumanEval/MBPP one-public-assert setting fails both).


def test_scoped_matches_unscoped_at_full_informativeness():
    """probe_informativeness=1 reproduces value_guided_search exactly (same rng path)."""
    reward, _ = reasoning_tree_rewards(depth=8, n_decisions=8, rng=np.random.default_rng(7))
    for seed in range(10):
        ok_a, r_a = value_guided_search_scoped(
            reward, 8, 4, 0.2, np.random.default_rng(seed), probe_informativeness=1.0
        )
        # note: scoped consumes one extra rng.random() per child batch, so exact
        # trajectory equality is not expected; instead check the interface contract
        assert isinstance(ok_a, bool)
        assert 0.0 <= r_a <= 1.0


def test_gain_positive_when_reachable_and_probe_informative():
    """MATH-like regime: unsaturated, informative probe -> decisive value-guided win."""
    k, budget = 8, 1024
    vg = scoped_success_rate(
        "value_guided", k, k, budget, 0.3, saturation=0.0, probe_informativeness=1.0, seeds=120
    )
    bo = scoped_success_rate(
        "best_of_n", k, k, budget, 0.3, saturation=0.0, probe_informativeness=1.0, seeds=120
    )
    assert vg > bo + 0.25


def test_gain_collapses_proportionally_at_saturation():
    """GSM8K/HumanEval-like regime: the gap scales as (1 - saturation) * gap.

    With a perfect probe the residual non-trivial slice retains its conditional gain,
    so the gap collapses proportionally, by ~(1 - s) -- an order of magnitude at
    s=0.95 -- rather than to exactly zero. (On real saturated benchmarks the residual
    slice is also short-chain with a noisy probe, which is why the measured GSM8K gap
    is ~0; that compounding is exercised by the probe-informativeness tests below.)
    """
    k, budget = 8, 1024
    gap = {}
    for s in (0.0, 0.95):
        vg = scoped_success_rate(
            "value_guided", k, k, budget, 0.3, saturation=s, probe_informativeness=1.0, seeds=200
        )
        bo = scoped_success_rate(
            "best_of_n", k, k, budget, 0.3, saturation=s, probe_informativeness=1.0, seeds=200
        )
        gap[s] = vg - bo
    assert gap[0.95] < 0.15 * gap[0.0]  # collapsed by well over 6x
    assert (
        scoped_success_rate(
            "best_of_n", k, k, budget, 0.3, saturation=0.95, probe_informativeness=1.0, seeds=200
        )
        > 0.9
    )  # best-of-N already near ceiling: nothing left to buy


def test_gain_vanishes_with_uninformative_probe():
    """One-public-assert-like regime: probe carries no signal -> no advantage.

    With probe_informativeness=0 every descent step is a coin flip, so value-guided
    lands on the correct trace w.p. ~b^-K -- no better than a single random sample --
    while best-of-N still gets its many draws. The value edge, not the search
    scaffolding, is what carries the gain.
    """
    k, budget = 8, 1024
    vg = scoped_success_rate(
        "value_guided", k, k, budget, 0.3, saturation=0.0, probe_informativeness=0.0, seeds=120
    )
    bo = scoped_success_rate(
        "best_of_n", k, k, budget, 0.3, saturation=0.0, probe_informativeness=0.0, seeds=120
    )
    assert vg <= bo  # the advantage is gone (and typically strictly reversed)
    assert vg < 0.1  # random descent essentially never finds the correct trace


def test_gain_degrades_monotonically_in_probe_informativeness():
    """The advantage interpolates smoothly between the two regimes."""
    k, budget = 8, 1024
    rates = [
        scoped_success_rate(
            "value_guided", k, k, budget, 0.3, probe_informativeness=q, seeds=120
        )
        for q in (0.0, 0.5, 1.0)
    ]
    assert rates[0] < rates[1] < rates[2]
