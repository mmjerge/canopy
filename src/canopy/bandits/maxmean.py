"""Data-driven high-probability bounds on (max leaf - subtree mean).

The hierarchical bandit needs an upper bound on the bias term

    B(v) = max_{l in L(v)} mu(l) - f(v),     f(v) = mean of leaf means under v,

computed only from random-path plays of v, where each play returns X = mu(L) + noise
for a uniformly random leaf L. We never observe individual leaf means.

Three estimators of B(v), in increasing sophistication:

* ``samuelson_bound``  -- hard, distribution-free: sigma_within * sqrt(m - 1)
  (Samuelson / Bhatia-Davis). Always valid, very loose for large m.
* ``subgaussian_bound`` -- light-tail heuristic: sigma_within * sqrt(2 log m).
  Tight when leaf means are sub-Gaussian, but not a guarantee.
* ``mgf_bound``        -- the proposed bound: a noise-deconvolved empirical-MGF
  (log-sum-exp) upper bound. For any lambda > 0,
        max_l mu(l) <= (1/lambda) [ log m + log E exp(lambda mu(L)) ],
  and since X = mu(L) + noise with known noise MGF exp(lambda^2 sigma^2 / 2),
        E exp(lambda mu(L)) = E exp(lambda X) * exp(-lambda^2 sigma^2 / 2).
  We estimate E exp(lambda X) from samples with a high-probability (empirical-Bernstein)
  upper confidence, deconvolve the noise, lower-bound the unobserved subtree mean f(v)
  by its own confidence radius, and minimize over a grid of lambda. This
  recovers sigma*sqrt(2 log m) when the tail is sub-Gaussian and stays valid (up to the
  Samuelson worst case) when it is heavier -- a strictly data-adaptive interpolation.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def _within_std(rewards: NDArray[np.float64], noise_std: float) -> float:
    if len(rewards) < 2:
        return float("inf")
    total_var = float(np.var(rewards, ddof=1))
    return math.sqrt(max(0.0, total_var - noise_std**2))


def samuelson_bound(rewards: NDArray[np.float64], n_leaves: int, noise_std: float) -> float:
    """Distribution-free hard bound: sigma_within * sqrt(m - 1) (Samuelson)."""
    return _within_std(rewards, noise_std) * math.sqrt(max(1, n_leaves - 1))


def subgaussian_bound(rewards: NDArray[np.float64], n_leaves: int, noise_std: float) -> float:
    """Light-tail heuristic: sigma_within * sqrt(2 log m)."""
    return _within_std(rewards, noise_std) * math.sqrt(2.0 * math.log(max(2, n_leaves)))


def _bernstein_radius(var: float, value_range: float, n: int, log_term: float) -> float:
    """Empirical-Bernstein confidence radius for a mean of n bounded observations."""
    return math.sqrt(2.0 * var * log_term / n) + 7.0 * value_range * log_term / (3.0 * (n - 1))


def mgf_bound(
    rewards: NDArray[np.float64],
    n_leaves: int,
    noise_std: float,
    delta: float = 0.05,
    lambdas: NDArray[np.float64] | None = None,
    reward_lo: float = 0.0,
    reward_hi: float = 1.0,
    trunc_z: float = 3.0,
) -> float:
    """Noise-deconvolved empirical-MGF upper bound on (max leaf mean - subtree mean).

    Since ``B(v) = max mu - f(v)`` and only ``X`` (not ``f(v)``) is observed, the bound
    combines (i) an empirical-Bernstein *upper* confidence on ``E exp(lam X)`` for each
    ``lam`` in the grid, and (ii) an empirical-Bernstein *lower* confidence on the probe
    mean ``f(v) = E[X]``; ``delta`` is split across these ``len(lambdas) + 1`` events.
    Returns a non-negative upper bound on ``B(v)``.

    Validity caveat (truncation): the Bernstein step needs bounded observations, so the
    range of ``exp(lam X)`` is taken over ``[reward_lo - trunc_z*sigma,
    reward_hi + trunc_z*sigma]``. With unbounded (e.g. Gaussian) noise each sample escapes
    that range with probability ``2*Phi(-trunc_z)`` (~0.27% at ``trunc_z=3``), so the
    guarantee is ``1 - delta - 2n*Phi(-trunc_z)`` rather than ``1 - delta``; increase
    ``trunc_z`` to tighten the escape term at the cost of a wider Bernstein range. An
    exact treatment for unbounded noise needs sub-exponential concentration (open; see
    docs/maxmean_bound.md).
    """
    x = np.asarray(rewards, dtype=np.float64)
    n = len(x)
    if n < 2:
        return float("inf")
    xbar = float(x.mean())
    m = max(2, n_leaves)
    if lambdas is None:
        lambdas = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    # noise can push observations a few sigma outside the mean range
    hi = reward_hi + trunc_z * noise_std
    lo = reward_lo - trunc_z * noise_std
    # union bound over the |lambdas| MGF confidences plus the mean lower confidence
    log_k_delta = math.log(2.0 * (len(lambdas) + 1) / delta)
    # lower confidence bound on f(v) = E[X]: subtracting xbar alone under-covers
    mean_rad = _bernstein_radius(float(np.var(x, ddof=1)), hi - lo, n, log_k_delta)

    best = float("inf")
    for lam in lambdas:
        y = np.exp(lam * x)  # exp(lambda X_i) in [exp(lam*lo), exp(lam*hi)]
        g_hat = float(y.mean())  # (1/n) sum exp(lam x) estimates E exp(lam X)
        var_y = float(np.var(y, ddof=1))
        rng_lam = math.exp(lam * hi) - math.exp(lam * lo)  # range of exp(lam X)
        # empirical-Bernstein upper confidence on E exp(lam X)
        g_upper = g_hat + _bernstein_radius(var_y, rng_lam, n, log_k_delta)
        if g_upper <= 0:
            continue
        log_mgf_mu = math.log(g_upper) - 0.5 * lam**2 * noise_std**2  # deconvolve noise
        bound_on_max = (math.log(m) + log_mgf_mu) / lam
        best = min(best, bound_on_max - (xbar - mean_rad))
    return max(0.0, best)


def mgf_bound_from_moments(
    n: int,
    mean_x: float,
    sum_exp: NDArray[np.float64],
    sum_exp2: NDArray[np.float64],
    lambdas: NDArray[np.float64],
    n_leaves: int,
    noise_std: float,
    delta: float = 0.05,
    reward_lo: float = 0.0,
    reward_hi: float = 1.0,
    trunc_z: float = 3.0,
    var_x: float | None = None,
) -> float:
    """Same bound as :func:`mgf_bound`, computed from running sufficient statistics.

    ``sum_exp[k] = sum_i exp(lambda_k X_i)`` and ``sum_exp2[k] = sum_i exp(2 lambda_k X_i)``
    are O(len(lambdas)) per node, so no raw samples need to be stored. This is what the
    online self-certifying algorithm maintains.

    ``var_x`` is the sample variance of the probes, used for the empirical-Bernstein
    *lower* confidence on the probe mean ``f(v)`` (see :func:`mgf_bound`); if the caller
    does not track it, a conservative Hoeffding radius over the truncated range is used.
    """
    if n < 2:
        return float("inf")
    m = max(2, n_leaves)
    hi = reward_hi + trunc_z * noise_std
    lo = reward_lo - trunc_z * noise_std
    log_k_delta = math.log(2.0 * (len(lambdas) + 1) / delta)
    if var_x is not None:
        mean_rad = _bernstein_radius(var_x, hi - lo, n, log_k_delta)
    else:
        mean_rad = (hi - lo) * math.sqrt(log_k_delta / (2.0 * n))
    best = float("inf")
    for k, lam in enumerate(lambdas):
        g_hat = sum_exp[k] / n
        e_y2 = sum_exp2[k] / n
        var_y = max(0.0, (e_y2 - g_hat**2)) * n / (n - 1)
        rng_lam = math.exp(lam * hi) - math.exp(lam * lo)
        g_upper = g_hat + _bernstein_radius(var_y, rng_lam, n, log_k_delta)
        if g_upper <= 0:
            continue
        log_mgf_mu = math.log(g_upper) - 0.5 * lam**2 * noise_std**2
        bound_on_max = (math.log(m) + log_mgf_mu) / lam
        best = min(best, bound_on_max - (mean_x - mean_rad))
    return max(0.0, best)
