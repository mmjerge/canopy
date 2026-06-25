"""Tests for the multiscale (scale-adaptive) tree edge map."""

from __future__ import annotations

import numpy as np

from canopy.bandits import TreeBandit, detect_violations, multiscale_edge_map
from canopy.bandits.rewards import geometric_sigma, hierarchical_gaussian_leaf_means

B, D = 4, 5
N = B**D


def mixed_width_family(seed: int):
    rng = np.random.default_rng(seed)
    m = np.clip(
        hierarchical_gaussian_leaf_means(B, D, geometric_sigma(0.04, 0.6), root_value=0.4, rng=rng),
        0,
        1,
    )
    regions = []
    for _ in range(2):
        s = int(rng.integers(0, N - 64))
        m[s : s + 64] = 0.8
        regions.append((s, s + 64))
    for _ in range(2):
        s = int(rng.integers(0, N))
        m[s] = 0.95
        regions.append((s, s + 1))
    return m, regions


def test_structure_and_bounded_score():
    lm, _ = mixed_width_family(0)
    env = TreeBandit(B, D, leaf_means=lm, noise_std=0.08, rng=np.random.default_rng(1))
    em = multiscale_edge_map(env, np.random.default_rng(1), levels=[2, 3, 4], n_samples_per_cell=40)
    assert em.levels == [2, 3, 4]
    assert em.leaf_score.shape == (N,)
    assert np.all(np.isfinite(em.leaf_score))
    assert em.leaf_score.max() < 1e3  # robust floor -> no blow-up when cells are smooth
    for lvl in em.levels:
        assert em.within_std[lvl].shape == (B**lvl,)


def test_finest_ranges_valid():
    lm, _ = mixed_width_family(2)
    env = TreeBandit(B, D, leaf_means=lm, noise_std=0.08, rng=np.random.default_rng(3))
    em = multiscale_edge_map(env, np.random.default_rng(3), levels=[2, 3, 4])
    for s, e in em.finest_ranges():
        assert 0 <= s < e <= N


def test_multiscale_recall_beats_single_level_on_mixed_widths():
    def covers(ranges, regions):
        return np.mean([any(s < b and a < e for s, e in ranges) for a, b in regions])

    ms, single4 = [], []
    for seed in range(15):
        lm, regions = mixed_width_family(seed)
        env = TreeBandit(B, D, leaf_means=lm, noise_std=0.08, rng=np.random.default_rng(20 + seed))
        em = multiscale_edge_map(
            env, np.random.default_rng(20 + seed), levels=[2, 3, 4], n_samples_per_cell=40
        )
        ms.append(covers(em.finest_ranges(), regions))
        det = detect_violations(
            env, 4, lambda _l: 0.06, np.random.default_rng(20 + seed), n_samples_per_cell=40
        ).detected
        cs = B ** (D - 4)
        single4.append(covers([(c * cs, (c + 1) * cs) for c in det], regions))
    assert np.mean(ms) >= 0.95  # catches violations at every scale
    assert np.mean(ms) > np.mean(single4)  # and beats a single fine level


def test_edge_score_high_on_violations():
    lm, regions = mixed_width_family(5)
    env = TreeBandit(B, D, leaf_means=lm, noise_std=0.08, rng=np.random.default_rng(6))
    em = multiscale_edge_map(env, np.random.default_rng(6), levels=[2, 3, 4], n_samples_per_cell=60)
    # the edge score spikes at edges (boundaries/spikes), not block interiors, so most
    # violation regions contain an above-floor (score > 1) edge leaf...
    flagged_regions = sum(em.leaf_score[a:b].max() > 1.0 for a, b in regions)
    assert flagged_regions >= len(regions) - 1
    # ...and the strongest edges are clearly elevated over the smooth background.
    on = np.concatenate([np.arange(a, b) for a, b in regions])
    off = np.setdiff1d(np.arange(N), on)
    assert em.leaf_score[on].max() > 3.0 * np.median(em.leaf_score[off])
