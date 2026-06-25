"""Tests for the violation family, data-driven violation detection, and graceful cost."""

from __future__ import annotations

import numpy as np
import pytest

from canopy.bandits import (
    HierarchicalTopK,
    SuccessiveEliminationTopK,
    TreeBandit,
    detect_violations,
    violation_family_leaf_means,
)
from canopy.bandits.rewards import (
    adversarial_spike_leaf_means,
    hierarchical_gaussian_leaf_means,
)


def test_violation_family_has_unique_optimum_and_decoys():
    lm = violation_family_leaf_means(
        4, 5, n_violations=8, violation_level=3, rng=np.random.default_rng(0)
    )
    assert lm.shape == (4**5,)
    assert lm.max() == pytest.approx(0.95)
    assert np.sum(np.isclose(lm, 0.95)) == 1  # the optimum is a single, unique leaf
    assert np.sum(np.isclose(lm, 0.85)) > 0  # decoys present


def test_violation_family_validates_count():
    with pytest.raises(ValueError):
        violation_family_leaf_means(4, 5, n_violations=0, violation_level=3)
    with pytest.raises(ValueError):
        violation_family_leaf_means(4, 5, n_violations=65, violation_level=3)  # > 4**3


def test_detect_violations_recovers_k_from_data():
    floor = lambda _l: 0.04  # noqa: E731
    for k in (2, 8, 16):
        counts = []
        for seed in range(6):
            lm = violation_family_leaf_means(
                4, 5, n_violations=k, violation_level=3, rng=np.random.default_rng(seed)
            )
            env = TreeBandit(
                4, 5, leaf_means=lm, noise_std=0.1, rng=np.random.default_rng(900 + seed)
            )
            counts.append(
                detect_violations(
                    env,
                    3,
                    floor,
                    np.random.default_rng(900 + seed),
                    n_samples_per_cell=250,
                    jump_factor=1.5,
                ).count
            )
        # recovered close to the true K (small bias from rare base false positives)
        assert abs(np.mean(counts) - k) <= 1.5


def test_detect_violations_quiet_on_smooth_tree():
    floor = lambda _l: 0.04  # noqa: E731
    lm = hierarchical_gaussian_leaf_means(
        4, 5, lambda lvl: 0.02 * 0.6**lvl, rng=np.random.default_rng(1)
    )
    env = TreeBandit(4, 5, leaf_means=lm, noise_std=0.1, rng=np.random.default_rng(2))
    report = detect_violations(env, 3, floor, np.random.default_rng(2), n_samples_per_cell=200)
    assert report.level == 3
    assert report.within_std.shape == (4**3,)
    assert report.count <= 3  # essentially no false jumps on a smooth tree


def test_detect_violations_validates_level():
    lm = violation_family_leaf_means(4, 5, n_violations=2, rng=np.random.default_rng(0))
    env = TreeBandit(4, 5, leaf_means=lm, noise_std=0.1)
    with pytest.raises(ValueError):
        detect_violations(env, 6, lambda _l: 0.04, np.random.default_rng(0))


def test_identification_cost_grows_with_violations():
    def cost_for(k: int) -> float:
        costs = []
        for seed in range(6):
            lm = violation_family_leaf_means(
                4, 5, n_violations=k, violation_level=3, rng=np.random.default_rng(seed)
            )
            env = TreeBandit(
                4, 5, leaf_means=lm, noise_std=0.1, rng=np.random.default_rng(200 + seed)
            )
            res = SuccessiveEliminationTopK(budget=400_000, confidence=0.1).run(env, 1)
            assert res.certified  # generous budget -> always certifies
            costs.append(res.cost)
        return float(np.mean(costs))

    assert cost_for(16) > cost_for(1)  # more near-optimal violations -> costlier to certify


def test_detect_and_relax_hybrid_beats_assume_smooth_on_spikes():
    # Multi-fidelity regime: cheap probes + tight budget. Detecting the spike cells from
    # data and relaxing the smooth bound there recovers accuracy that assume-smooth loses.
    b, d, level = 4, 5, 3
    cell = b ** (d - level)
    hi, hy = [], []
    for s in range(12):
        lm = adversarial_spike_leaf_means(b, d, 4, rng=np.random.default_rng(s))

        def env(seed):
            return TreeBandit(
                b,
                d,
                leaf_means=lm,
                noise_std=0.05,
                probe_cost=0.05,
                rng=np.random.default_rng(seed),
            )

        e = env(40 + s)
        hi.append(HierarchicalTopK(400, 0.1, spread=0.3, beam_width=8).run(e, 1).evaluate(e, 1))
        det = detect_violations(
            env(80 + s),
            level,
            lambda _l: 0.08,
            np.random.default_rng(80 + s),
            n_samples_per_cell=40,
        ).detected
        ranges = [(c * cell, (c + 1) * cell) for c in det]
        e = env(40 + s)
        hy.append(
            HierarchicalTopK(400, 0.1, spread=0.3, beam_width=8, relaxed_ranges=ranges)
            .run(e, 1)
            .evaluate(e, 1)
        )
    assert np.mean(hy) > np.mean(hi) + 0.1


def test_edge_targeted_beats_blind_at_tight_budget():
    # Sample efficiency: at a tight budget, isolating the edge cells from cheap probes and
    # sampling around them finds the hidden optimum far more often than structure-blind.
    b, d, level, budget = 4, 5, 3, 400
    cell = b ** (d - level)
    blind, edge = [], []
    for s in range(16):
        lm = adversarial_spike_leaf_means(b, d, 8, rng=np.random.default_rng(s))

        def env(seed):
            return TreeBandit(
                b,
                d,
                leaf_means=lm,
                noise_std=0.05,
                probe_cost=0.05,
                rng=np.random.default_rng(seed),
            )

        e = env(40 + s)
        blind.append(SuccessiveEliminationTopK(budget, 0.1).run(e, 1).evaluate(e, 1))
        det = detect_violations(
            env(80 + s),
            level,
            lambda _l: 0.08,
            np.random.default_rng(80 + s),
            n_samples_per_cell=30,
        ).detected
        ranges = [(c * cell, (c + 1) * cell) for c in det]
        e = env(40 + s)
        edge.append(
            HierarchicalTopK(budget, 0.1, spread=0.3, beam_width=8, relaxed_ranges=ranges)
            .run(e, 1)
            .evaluate(e, 1)
        )
    assert np.mean(edge) > np.mean(blind) + 0.2
