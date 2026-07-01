"""LLM routing over a token prefix tree (the applied instantiation).

A prompt is a leaf of a ``branching``-ary prefix tree (branching = token vocab, depth =
prompt length). For each model m we have a quality ``q_m(leaf)`` in [0, 1] and a per-query
cost ``cost_m`` (big model = high quality, high cost; small model = cheaper, strong only in
some prefix regions). The net utility of routing a prompt to model m is

    u_m(leaf) = q_m(leaf) - lam * cost_m.

Routing exploits the tree's smoothness: prompts sharing a long prefix get similar quality
(Lipschitz in the LCA / longest-common-prefix metric), so a router can learn *which model
to use for which prefix region* without trying every prompt -- and the region boundaries
are where quality jumps (one token flips behavior), exactly the piecewise-Lipschitz case.

This module is a faithful simulator; to use real data, build a ``PrefixTreeRouting`` from
measured per-model quality on a prompt set (e.g. a router benchmark) -- the algorithms are
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from canopy.bandits.rewards import geometric_sigma, hierarchical_gaussian_leaf_means


@dataclass
class RouteResult:
    """Outcome of a routing run."""

    cum_regret: NDArray[np.float64]  # cumulative regret vs the per-prompt oracle
    total_cost: float
    avg_quality: float
    label: str

    @property
    def final_regret(self) -> float:
        return float(self.cum_regret[-1])


class PrefixTreeRouting:
    """Multi-model routing environment over a prefix tree.

    Args:
        branching: Token vocabulary size (>= 2).
        depth: Prompt length (number of tokens).
        quality: Array ``(n_models, n_leaves)`` of true per-model leaf qualities in [0, 1].
        costs: Per-model query cost, length ``n_models``.
        lam: Cost weight in the net utility ``q - lam * cost``.
        noise_std: Observation noise on a routed query's quality.
        rng: Optional generator.
    """

    def __init__(
        self,
        branching: int,
        depth: int,
        quality: NDArray[np.float64],
        costs: NDArray[np.float64],
        lam: float = 0.3,
        noise_std: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        if branching < 2 or depth < 1:
            raise ValueError("branching >= 2 and depth >= 1 required")
        self.branching = branching
        self.depth = depth
        self.n_leaves = branching**depth
        self.quality = np.asarray(quality, dtype=np.float64)
        if self.quality.shape[1] != self.n_leaves:
            raise ValueError("quality must have shape (n_models, branching**depth)")
        self.n_models = self.quality.shape[0]
        self.costs = np.asarray(costs, dtype=np.float64)
        self.lam = float(lam)
        self.noise_std = float(noise_std)
        self.rng = rng or np.random.default_rng()

        # net utility u_m(leaf) and the per-prompt oracle (best model per leaf)
        self.utility = self.quality - self.lam * self.costs[:, None]
        self.oracle_model = np.argmax(self.utility, axis=0)
        self.oracle_utility = self.utility[self.oracle_model, np.arange(self.n_leaves)]

    def region(self, leaf: int, resolution: int) -> int:
        """Index of the level-``resolution`` prefix region containing ``leaf``."""
        return leaf // (self.branching ** (self.depth - resolution))

    def sample_prompt(self) -> int:
        return int(self.rng.integers(0, self.n_leaves))

    def observe_quality(self, model: int, leaf: int) -> float:
        return float(self.quality[model, leaf] + self.rng.normal(0.0, self.noise_std))


def make_routing_scenario(
    branching: int,
    depth: int,
    rng: np.random.Generator,
    n_strong_regions: int = 4,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build a 2-model routing scenario where regional routing genuinely helps.

    Model 0 ("big"): high quality everywhere, smooth, expensive (cost 1.0).
    Model 1 ("small"): cheap (cost 0.1), strong only in a few prefix regions with sharp
    boundaries (jumps) and weak elsewhere. The oracle routes the small model in its strong
    regions (to save cost) and the big model elsewhere.

    Returns ``(quality, costs)`` with ``quality`` of shape (2, branching**depth).
    """
    n_leaves = branching**depth
    big = np.clip(
        0.85
        + hierarchical_gaussian_leaf_means(
            branching, depth, geometric_sigma(0.06, 0.6), root_value=0.0, rng=rng
        ),
        0.0,
        1.0,
    )
    small = np.full(n_leaves, 0.35) + 0.05 * rng.standard_normal(n_leaves)
    region_size = n_leaves // (branching**2)
    starts = rng.choice(branching**2, size=n_strong_regions, replace=False)
    for s in starts:
        small[s * region_size : (s + 1) * region_size] = 0.92 + 0.03 * rng.standard_normal(
            region_size
        )
    small = np.clip(small, 0.0, 1.0)
    quality = np.vstack([big, small])
    costs = np.array([1.0, 0.1])
    return quality, costs


class ContextualUCBRouter:
    """Online cost-aware contextual UCB router -- the learned routing policy, driven per call.

    This is the same learner as ``run_router``'s ``"hierarchical"`` strategy, factored out so it
    can be used *online* in a real system (e.g. a live agent loop) rather than only in the
    ``PrefixTreeRouting`` simulator: call :meth:`select` to pick a model for a context, then
    :meth:`update` once the reward (quality signal) for that pull is known. It maintains a
    per-``(model, region)`` mean-quality estimate and picks the arm maximizing the optimistic
    net utility ``qhat - lam*cost + c*sqrt(2 ln t / n)``.

    Set ``n_regions=1`` to recover the structure-blind (flat) learner -- the honest baseline that
    sees the same information but no context, exactly as in the routing experiments.

    Args:
        n_models: Size of the model pool (arms).
        costs: Per-model relative cost, length ``n_models`` (known a priori, e.g. from pricing).
        n_regions: Number of discrete contexts (1 = flat / structure-blind).
        lam: Cost weight in the net utility ``quality - lam*cost``.
        c: UCB exploration constant.
    """

    def __init__(
        self,
        n_models: int,
        costs: NDArray[np.float64],
        n_regions: int = 1,
        lam: float = 0.3,
        c: float = 0.4,
    ) -> None:
        self.n_models = int(n_models)
        self.costs = np.asarray(costs, dtype=np.float64)
        self.n_regions = int(n_regions)
        self.lam = float(lam)
        self.c = float(c)
        self.counts = np.zeros((self.n_models, self.n_regions))
        self.sums = np.zeros((self.n_models, self.n_regions))
        self.t = 0

    def select(self, region: int = 0) -> int:
        """Return the arm (model index) maximizing the optimistic net utility in ``region``."""
        region = min(max(region, 0), self.n_regions - 1)
        self.t += 1
        best_m, best_val = 0, -np.inf
        for m in range(self.n_models):
            n = self.counts[m, region]
            if n == 0:
                val = np.inf  # try every arm once per region
            else:
                qhat = self.sums[m, region] / n
                val = (qhat - self.lam * self.costs[m]) + self.c * np.sqrt(
                    2.0 * np.log(self.t + 2) / n
                )
            if val > best_val:
                best_val, best_m = val, m
        return best_m

    def update(self, region: int, model: int, quality: float) -> None:
        """Record a realized quality signal in ``[0,1]`` for a ``(region, model)`` pull."""
        region = min(max(region, 0), self.n_regions - 1)
        self.counts[model, region] += 1.0
        self.sums[model, region] += float(quality)


def run_router(
    env: PrefixTreeRouting,
    horizon: int,
    rng: np.random.Generator,
    strategy: str = "hierarchical",
    resolution: int = 2,
    c: float = 0.4,
) -> RouteResult:
    """Route a stream of prompts and measure regret vs the per-prompt oracle.

    Strategies:
        "hierarchical" -- learn per-region per-model quality with UCB and route by the
            optimistic net utility (uses the prefix-tree generalization at ``resolution``).
        "flat"         -- structure-blind online UCB over models with a single global estimate
            per model (the honest learning baseline: same information, no tree structure).
        "best_single"  -- always use the single model with the best average net utility
            (a truth-based reference: computed from the ground-truth utility, no exploration).
        "all_largest"  -- always use the highest-quality model, truth-based, ignores cost.
        "oracle"       -- per-prompt best model, truth-based upper bound.
        "random"       -- route uniformly at random.

    Only "hierarchical" and "flat" actually learn online; the others are references computed
    from the ground truth and pay no exploration cost.
    """
    n_regions = env.branching**resolution
    counts = np.zeros((env.n_models, n_regions))
    sums = np.zeros((env.n_models, n_regions))
    g_counts = np.zeros(env.n_models)  # structure-blind (flat) per-model counts
    g_sums = np.zeros(env.n_models)

    best_single = int(np.argmax(env.utility.mean(axis=1)))
    all_largest = int(np.argmax(env.quality.mean(axis=1)))

    cum = 0.0
    cum_regret = np.empty(horizon)
    total_cost = 0.0
    total_quality = 0.0
    for t in range(horizon):
        leaf = env.sample_prompt()
        if strategy == "hierarchical":
            r = env.region(leaf, resolution)
            best_m, best_val = 0, -np.inf
            for m in range(env.n_models):
                n = counts[m, r]
                if n == 0:
                    val = np.inf
                else:
                    qhat = sums[m, r] / n
                    val = (qhat - env.lam * env.costs[m]) + c * np.sqrt(2.0 * np.log(t + 2) / n)
                if val > best_val:
                    best_val, best_m = val, m
            model = best_m
        elif strategy == "flat":
            best_m, best_val = 0, -np.inf
            for m in range(env.n_models):
                n = g_counts[m]
                if n == 0:
                    val = np.inf
                else:
                    qhat = g_sums[m] / n
                    val = (qhat - env.lam * env.costs[m]) + c * np.sqrt(2.0 * np.log(t + 2) / n)
                if val > best_val:
                    best_val, best_m = val, m
            model = best_m
        elif strategy == "best_single":
            model = best_single
        elif strategy == "all_largest":
            model = all_largest
        elif strategy == "oracle":
            model = int(env.oracle_model[leaf])
        else:  # random
            model = int(rng.integers(0, env.n_models))

        q = env.observe_quality(model, leaf)
        if strategy == "hierarchical":
            r = env.region(leaf, resolution)
            counts[model, r] += 1
            sums[model, r] += q
        elif strategy == "flat":
            g_counts[model] += 1
            g_sums[model] += q

        cum += float(env.oracle_utility[leaf] - env.utility[model, leaf])
        cum_regret[t] = cum
        total_cost += float(env.costs[model])
        total_quality += float(env.quality[model, leaf])

    return RouteResult(
        cum_regret=cum_regret,
        total_cost=total_cost,
        avg_quality=total_quality / horizon,
        label=strategy if strategy != "hierarchical" else f"hierarchical(r={resolution})",
    )
