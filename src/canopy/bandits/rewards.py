"""Reward models for the tree bandit.

The hierarchical Gaussian model generates node values top-down: each internal node
splits its value among its children via zero-sum Gaussian perturbations whose scale
shrinks with depth. Two consequences make it the natural model for this problem:

1. Every parent equals the exact average of its subtree leaves (the project's spec),
   because the children of any node average back to that node by construction.
2. The per-level perturbation scale ``sigma(level)`` directly yields a high-probability
   ``spread`` bound -- the maximum deviation of a leaf from an ancestor's average --
   which is precisely what :class:`~canopy.bandits.topk.HierarchicalTopK` needs to prune
   soundly. When ``sigma`` decays with depth, subtree averages are predictive and the
   hierarchical search wins; when ``sigma`` is large/flat, the structure carries little
   information and the method degrades gracefully.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

SigmaSchedule = Callable[[int], float]


def geometric_sigma(base: float = 0.4, decay: float = 0.5) -> SigmaSchedule:
    """Per-level perturbation scale ``sigma(level) = base * decay ** level``."""
    if base <= 0:
        raise ValueError("base must be > 0")
    if not 0 < decay <= 1:
        raise ValueError("decay must be in (0, 1]")
    return lambda level: base * decay**level


def lipschitz_spread(lipschitz_const: float, rho: float) -> SigmaSchedule:
    """Spread schedule derived from an ultrametric-Lipschitz assumption.

    If f is L-Lipschitz w.r.t. the tree ultrametric ``d(x, y) = rho ** level(LCA(x, y))``
    (equivalently: the oscillation of f within any level-``ell`` subtree is at most
    ``L * rho ** ell``), then ``max - mean <= L * rho ** ell``. So a single Lipschitz
    constant ``L`` and discount ``rho in (0, 1)`` *derive* the whole bias schedule
    ``spread(ell) = L * rho ** ell`` instead of assuming it.
    """
    if lipschitz_const <= 0:
        raise ValueError("lipschitz_const must be > 0")
    if not 0 < rho < 1:
        raise ValueError("rho must be in (0, 1)")
    return lambda level: lipschitz_const * rho**level


def hierarchical_gaussian_leaf_means(
    branching: int,
    depth: int,
    sigma: SigmaSchedule,
    root_value: float = 0.5,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Generate leaf means via a top-down zero-sum Gaussian diffusion.

    At each level, every node's value is split among its ``branching`` children by
    adding zero-mean Gaussian perturbations (re-centered to sum to zero), so the
    children average exactly to the parent. Returns the ``branching ** depth`` leaf
    means (ordered left to right).
    """
    rng = rng or np.random.default_rng()
    values = np.array([float(root_value)], dtype=np.float64)
    for level in range(depth):
        s = sigma(level)
        deltas = rng.normal(0.0, s, size=(values.size, branching))
        deltas -= deltas.mean(axis=1, keepdims=True)  # zero-sum per parent -> exact avg
        values = (values[:, None] + deltas).reshape(-1)
    return values


def hierarchical_spread(
    sigma: SigmaSchedule,
    depth: int,
    branching: int,
    z: float = 3.0,
) -> SigmaSchedule:
    """High-probability ``spread`` bound matched to a hierarchical Gaussian instance.

    A leaf's deviation from its level-``l`` ancestor's value accumulates independent
    zero-sum perturbations over levels ``l..depth-1``. Each level-``m`` perturbation has
    variance ``sigma(m)^2 * (branching - 1) / branching`` after re-centering, so the
    deviation's standard deviation is ``sqrt(sum_m sigma(m)^2 * (b-1)/b)``. We scale it
    by ``z`` (e.g. 3) for a high-probability bound.

    Note: Gaussian tails are unbounded, so this is a high-probability bound, consistent
    with the algorithm's overall ``1 - confidence`` guarantee rather than a worst-case
    one. Increase ``z`` for a more conservative (safer pruning) bound.
    """
    factor = (branching - 1) / branching

    def spread(level: int) -> float:
        var = sum(sigma(m) ** 2 for m in range(level, depth)) * factor
        return z * math.sqrt(var)

    return spread


def adversarial_spike_leaf_means(
    branching: int,
    depth: int,
    n_spikes: int,
    low: float = 0.2,
    high: float = 0.95,
    jitter: float = 0.02,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Leaf means with a flat low baseline and a few randomly-placed high "spikes".

    Because spikes are scattered uniformly at random, a subtree containing one spike
    has only a slightly elevated average (one high leaf diluted by many low ones), so
    internal-node averages are nearly uninformative -- the worst case for any method
    that trusts subtree averages. This is the adversarial counterpart to the smooth
    hierarchical Gaussian model.
    """
    rng = rng or np.random.default_rng()
    n_leaves = branching**depth
    if not 1 <= n_spikes <= n_leaves:
        raise ValueError("n_spikes must satisfy 1 <= n_spikes <= n_leaves")
    means = low + jitter * rng.standard_normal(n_leaves)
    spikes = rng.choice(n_leaves, size=n_spikes, replace=False)
    means[spikes] = high + jitter * rng.standard_normal(n_spikes)
    return means


def piecewise_smooth_leaf_means(
    branching: int,
    depth: int,
    n_jumps: int,
    base_sigma: float = 0.12,
    base_decay: float = 0.6,
    root_value: float = 0.3,
    jump_height: float = 0.6,
    jump_width: int | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Mostly-smooth leaf means with a finite number of sharp jump discontinuities.

    A smooth hierarchical-Gaussian base plus ``n_jumps`` localized high-reward blocks
    with sharp edges (cliffs). The global optimum lands inside a jump, so a method that
    assumes global smoothness (a small ``spread(level)``) under-estimates the bias at the
    cell straddling a jump and can miss the optimum; a data-driven bias term sees the
    elevated within-cell variance there and keeps exploring. This is the tree form of the
    piecewise-Lipschitz / dispersion setting.
    """
    rng = rng or np.random.default_rng()
    n_leaves = branching**depth
    base = hierarchical_gaussian_leaf_means(
        branching, depth, geometric_sigma(base_sigma, base_decay),
        root_value=root_value, rng=rng,
    )
    means = base.copy()
    if jump_width is None:
        jump_width = max(1, n_leaves // (branching**2))  # ~ a level-2 subtree
    for _ in range(n_jumps):
        start = int(rng.integers(0, max(1, n_leaves - jump_width)))
        means[start : start + jump_width] += jump_height * (0.5 + rng.random())
    return np.clip(means, 0.0, 1.0)


def heterogeneous_smoothness_leaf_means(
    branching: int,
    depth: int,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Leaf means whose local Lipschitz constant varies across the tree.

    The left half is very smooth (a gentle bump, tiny within-subtree variation -> small
    local L); the right half is rough (high within-subtree variation -> large local L) and
    hides the global optimum as a spike. A single global Lipschitz constant is wrong
    somewhere: too tight under-explores the rough half (misses the spike), too loose
    over-explores the smooth half (wastes budget). A locally-adaptive constant uses the
    right L per region.
    """
    rng = rng or np.random.default_rng()
    n = branching**depth
    half = n // 2
    x = np.linspace(0.0, 1.0, n)
    means = np.empty(n)
    # left: smooth gentle bump (peak ~0.7)
    means[:half] = 0.5 + 0.2 * np.exp(-((x[:half] - 0.25) ** 2) / 0.02)
    # right: rough, high within-subtree variation, low baseline
    means[half:] = 0.3 + 0.22 * rng.standard_normal(n - half)
    # the global optimum is a spike inside the rough region
    means[half + (n - half) // 2] = 0.97
    return np.clip(means, 0.0, 1.0)


def violation_family_leaf_means(
    branching: int,
    depth: int,
    n_violations: int,
    optimum: float = 0.95,
    decoy: float = 0.85,
    base: float = 0.5,
    base_jitter: float = 0.02,
    block_width: int = 4,
    violation_level: int = 3,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """A smooth base with ``n_violations`` controlled, *disjoint* Lipschitz violations.

    Exactly ``n_violations`` distinct level-``violation_level`` cells each receive one sharp
    block (a jump): the first is the unique global optimum (height ``optimum``), the rest are
    near-optimal *decoys* (height ``decoy < optimum``). Everything else is a flat,
    Lipschitz-smooth base. This decouples the *number of violations* ``K`` from problem
    difficulty in a controlled way: each added violation is a disjoint obstacle that an
    identifier must resolve, so the cost to certify the optimum grows gracefully with ``K``
    while the optimum's value stays fixed (unlike :func:`piecewise_smooth_leaf_means`, where
    extra high-reward jumps make the problem *easier*). Used to study regret/identification
    cost as a function of the number of Lipschitz violations.

    ``n_violations`` must satisfy ``1 <= n_violations <= branching**violation_level``.
    """
    rng = rng or np.random.default_rng()
    n_leaves = branching**depth
    n_cells = branching**violation_level
    if not 1 <= n_violations <= n_cells:
        raise ValueError(f"n_violations must satisfy 1 <= n <= {n_cells}")
    cell_size = branching ** (depth - violation_level)
    if not 1 <= block_width <= cell_size:
        raise ValueError("block_width must satisfy 1 <= block_width <= cell size")
    means = base + base_jitter * rng.standard_normal(n_leaves)
    chosen = rng.choice(n_cells, size=n_violations, replace=False)
    for i, cell in enumerate(chosen):
        start = cell * cell_size + int(rng.integers(0, cell_size - block_width + 1))
        if i == 0:
            # the unique global optimum: a single distinguished leaf (no ties to certify)
            means[start] = optimum
        else:
            # a near-optimal decoy: a detectable block just below the optimum
            means[start : start + block_width] = decoy
    return np.clip(means, 0.0, 1.0)
