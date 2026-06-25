"""Locally-adaptive Lipschitz: tighter constants in smoother subtrees.

The landscape's smoothness varies across the tree: the left half is very smooth (small
local Lipschitz constant), the right half is rough (large constant) and hides the global
optimum. A single global constant is wrong somewhere -- too tight under-explores the rough
half (misses the optimum), too loose over-explores the smooth half. The locally-adaptive
algorithm estimates a per-subtree constant (propagated down the tree) and uses the right
one per region.

Panels:
  (A) cumulative regret over time for global-tight, global-loose, and local-adaptive.
  (B) the true within-subtree spread by region (left low, right high) -- the heterogeneous
      smoothness the adaptive method exploits.

Run with:  uv run --extra plot python examples/local_lipschitz_demo.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from oco.bandits import (
    TreeBandit,
    lipschitz_spread,
    run_hoo,
    run_local_lipschitz,
)
from oco.bandits.rewards import heterogeneous_smoothness_leaf_means
from oco.bandits.tree import Node

BRANCHING, DEPTH, HORIZON = 4, 5, 15000
N_SEEDS = 8
RHO = 1.0 / BRANCHING
TIGHT = lipschitz_spread(0.15, RHO)
LOOSE = lipschitz_spread(1.5, RHO)


def main() -> None:
    curves = {"global-tight L": [], "global-loose L": [], "local-adaptive L": []}
    for seed in range(N_SEEDS):
        lm = heterogeneous_smoothness_leaf_means(BRANCHING, DEPTH, np.random.default_rng(seed))
        e1 = TreeBandit(BRANCHING, DEPTH, leaf_means=lm, noise_std=0.1, rng=np.random.default_rng(100 + seed))
        curves["global-tight L"].append(run_hoo(e1, HORIZON, TIGHT, np.random.default_rng(100 + seed), memory_bounded=True).cum_regret)
        e2 = TreeBandit(BRANCHING, DEPTH, leaf_means=lm, noise_std=0.1, rng=np.random.default_rng(100 + seed))
        curves["global-loose L"].append(run_hoo(e2, HORIZON, LOOSE, np.random.default_rng(100 + seed), memory_bounded=True).cum_regret)
        e3 = TreeBandit(BRANCHING, DEPTH, leaf_means=lm, noise_std=0.1, rng=np.random.default_rng(100 + seed))
        curves["local-adaptive L"].append(run_local_lipschitz(e3, HORIZON, np.random.default_rng(100 + seed)).cum_regret)

    means = {k: np.mean(v, axis=0) for k, v in curves.items()}
    print(f"heterogeneous-smoothness tree: {BRANCHING**DEPTH} leaves, horizon {HORIZON}, "
          f"{N_SEEDS} seeds")
    for k, m in means.items():
        print(f"  {k:18s} final regret = {m[-1]:7.1f}")

    # true within-subtree spread by region (level 2) to show the heterogeneity
    resolution = 2
    lm0 = heterogeneous_smoothness_leaf_means(BRANCHING, DEPTH, np.random.default_rng(0))
    env0 = TreeBandit(BRANCHING, DEPTH, leaf_means=lm0, noise_std=0.0)
    region_std = []
    for idx in range(BRANCHING**resolution):
        start, end = env0.leaf_range(Node(resolution, idx))
        region_std.append(float(np.std(lm0[start:end])))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = {"global-tight L": "#1f77b4", "global-loose L": "#ff7f0e", "local-adaptive L": "#d62728"}
    rounds = np.arange(1, HORIZON + 1)
    for k, m in means.items():
        axA.plot(rounds, m, color=colors[k], lw=2, label=k)
    axA.set_xlabel("round")
    axA.set_ylabel("cumulative regret")
    axA.set_title("Local-adaptive L beats any single global constant", fontsize=10)
    axA.grid(True, ls=":", alpha=0.5)
    axA.legend(loc="upper left", fontsize=9)

    regions = np.arange(len(region_std))
    axB.bar(regions, region_std, color=["#2ca02c"] * (len(regions) // 2) + ["#8c564b"] * (len(regions) - len(regions) // 2))
    axB.set_xlabel(f"prefix region (level {resolution})")
    axB.set_ylabel("true within-region spread (local smoothness)")
    axB.set_title("Heterogeneous smoothness: left smooth, right rough", fontsize=10)
    axB.grid(True, axis="y", ls=":", alpha=0.5)

    fig.suptitle("Tighter Lipschitz localization: adapt the constant per subtree", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(__file__).parent / "tree_local_lipschitz.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved chart to {out}")


if __name__ == "__main__":
    main()
