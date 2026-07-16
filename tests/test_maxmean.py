"""Tests for the data-driven max-mean bounds (canopy.bandits.maxmean)."""

from __future__ import annotations

import numpy as np

from canopy.bandits.maxmean import (
    mgf_bound,
    mgf_bound_from_moments,
    samuelson_bound,
    subgaussian_bound,
)


def _sample(leaf_means, n, noise_std, seed):
    rng = np.random.default_rng(seed)
    m = len(leaf_means)
    idx = rng.integers(0, m, size=n)
    return np.asarray(leaf_means)[idx] + rng.normal(0, noise_std, size=n)


def test_mgf_bound_is_valid_on_spread_instance():
    leaf_means = 0.5 + 0.2 * np.random.default_rng(0).standard_normal(256)
    leaf_means = np.clip(leaf_means, 0, 1)
    true = leaf_means.max() - leaf_means.mean()
    x = _sample(leaf_means, 5000, 0.1, seed=1)
    assert mgf_bound(x, len(leaf_means), 0.1, delta=0.05) >= true


def test_mgf_bound_valid_on_spike_where_subgaussian_fails():
    leaf_means = np.full(256, 0.2)
    leaf_means[[3, 50, 200, 201, 202]] = 0.9
    true = leaf_means.max() - leaf_means.mean()
    x = _sample(leaf_means, 5000, 0.1, seed=2)
    # the sub-Gaussian heuristic under-estimates (not a real bound) ...
    assert subgaussian_bound(x, len(leaf_means), 0.1) < true
    # ... but the MGF bound stays valid.
    assert mgf_bound(x, len(leaf_means), 0.1, delta=0.05) >= true


def test_mgf_beats_samuelson_on_spread_instance():
    leaf_means = np.clip(0.5 + 0.2 * np.random.default_rng(3).standard_normal(256), 0, 1)
    x = _sample(leaf_means, 5000, 0.1, seed=4)
    assert mgf_bound(x, 256, 0.1) < samuelson_bound(x, 256, 0.1)


def test_from_moments_matches_direct():
    leaf_means = np.clip(0.4 + 0.15 * np.random.default_rng(5).standard_normal(64), 0, 1)
    x = _sample(leaf_means, 3000, 0.1, seed=6)
    lambdas = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    direct = mgf_bound(x, 64, 0.1, delta=0.05, lambdas=lambdas)
    se = np.array([np.sum(np.exp(lam * x)) for lam in lambdas])
    se2 = np.array([np.sum(np.exp(2 * lam * x)) for lam in lambdas])
    from_mom = mgf_bound_from_moments(
        len(x),
        float(x.mean()),
        se,
        se2,
        lambdas,
        64,
        0.1,
        delta=0.05,
        var_x=float(x.var(ddof=1)),
    )
    assert abs(direct - from_mom) < 1e-6


def test_from_moments_without_var_is_conservative():
    """Omitting var_x falls back to a Hoeffding mean radius, which can only enlarge it."""
    leaf_means = np.clip(0.4 + 0.15 * np.random.default_rng(7).standard_normal(64), 0, 1)
    x = _sample(leaf_means, 3000, 0.1, seed=8)
    lambdas = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    se = np.array([np.sum(np.exp(lam * x)) for lam in lambdas])
    se2 = np.array([np.sum(np.exp(2 * lam * x)) for lam in lambdas])
    with_var = mgf_bound_from_moments(
        len(x), float(x.mean()), se, se2, lambdas, 64, 0.1, var_x=float(x.var(ddof=1))
    )
    without_var = mgf_bound_from_moments(len(x), float(x.mean()), se, se2, lambdas, 64, 0.1)
    assert without_var >= with_var


def test_mgf_bound_nominal_coverage():
    """Calibration: across many independent probe draws, the 1-delta bound fails at most
    ~delta of the time (plus Monte-Carlo slack). Guards the confidence accounting -- e.g.
    dropping the probe-mean confidence radius makes the bound under-cover."""
    inst_rng = np.random.default_rng(11)
    leaf_means = np.clip(0.5 + 0.2 * inst_rng.standard_normal(64), 0, 1)
    true_b = float(leaf_means.max() - leaf_means.mean())
    delta, trials, n = 0.1, 400, 300
    violations = sum(
        mgf_bound(_sample(leaf_means, n, 0.1, seed=1000 + t), 64, 0.1, delta=delta) < true_b
        for t in range(trials)
    )
    # binomial(400, 0.1) has sd ~ 6, so 0.05 slack (20 trials) is ~3 sigma
    assert violations / trials <= delta + 0.05
