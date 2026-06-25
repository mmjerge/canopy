"""Multi-fidelity benchmark: when does exploiting the tree beat a strong baseline?

The tree environment is now multi-fidelity: an internal-node probe is CHEAP but biased
(cost ``probe_cost``), a leaf evaluation is EXPENSIVE but unbiased (cost ``leaf_cost``).
Budgets are measured in cost. We compare, at equal cost budget:

  * HierarchicalTopK         -- spends cheap probes to localize, then leaf evals to confirm
  * SuccessiveEliminationTopK -- STRONG structure-blind baseline (leaf evals only)
  * UniformTopK              -- weak structure-blind baseline

Produces a single side-by-side chart (tree_topk_benchmark.png) with two panels:
  1. recall vs the probe/leaf cost ratio (fixed budget) -- the tree overtakes the strong
     baseline only once probes are cheap enough.
  2. recall vs cost budget at a cheap probe cost.

Run with:  uv run --extra plot python examples/benchmark.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from oco.bandits import (
    HierarchicalTopK,
    SuccessiveEliminationTopK,
    TreeBandit,
    UniformTopK,
    geometric_sigma,
    hierarchical_spread,
)

BRANCHING, DEPTH, K = 4, 5, 5  # 1024 leaves
LEAVES = BRANCHING**DEPTH
N_SEEDS = 30
SIGMA = geometric_sigma(base=0.5, decay=0.55)
SPREAD = hierarchical_spread(SIGMA, DEPTH, BRANCHING, z=3.0)
BEAM = 20


def make_env(seed: int, probe_cost: float) -> TreeBandit:
    return TreeBandit.from_hierarchical_gaussian(
        BRANCHING, DEPTH, sigma=SIGMA, noise_std=0.1, probe_cost=probe_cost,
        rng=np.random.default_rng(seed),
    )


def recall(method: str, budget: float, probe_cost: float) -> float:
    out = []
    for seed in range(N_SEEDS):
        env = make_env(seed, probe_cost)
        if method == "Hierarchical":
            res = HierarchicalTopK(budget=budget, spread=SPREAD, beam_width=BEAM).run(env, K)
        elif method == "SuccessiveElim":
            res = SuccessiveEliminationTopK(budget=budget).run(env, K)
        else:
            res = UniformTopK(budget=budget).run(env, K)
        out.append(res.evaluate(env, K))
    return float(np.mean(out))


COLORS = {"Hierarchical": "#1f77b4", "SuccessiveElim": "#2ca02c", "Uniform": "#d62728"}
MARKERS = {"Hierarchical": "o", "SuccessiveElim": "^", "Uniform": "s"}


def fidelity_panel(ax) -> None:
    budget = 1500.0
    ratios = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
    print(f"\n[fidelity sweep @ budget={budget}]  probe/leaf cost -> recall")
    for method in ("Hierarchical", "SuccessiveElim"):
        ys = [recall(method, budget, r) for r in ratios]  # SE ignores probe cost
        ax.plot(ratios, ys, marker=MARKERS[method], color=COLORS[method], lw=2, label=method)
        print(f"  {method:15s} " + " ".join(f"{y:.2f}" for y in ys))
    ax.set_xscale("log")
    ax.invert_xaxis()  # cheaper probes to the right
    ax.set_xlabel("probe cost / leaf cost  (cheaper probes ->)")
    ax.set_ylabel(f"mean top-{K} recall ({N_SEEDS} seeds)")
    ax.set_title(f"The tree wins only when probes are cheap enough\n"
                 f"budget={budget:.0f} cost, beam={BEAM}", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="upper left")


def cost_budget_panel(ax) -> None:
    probe_cost = 0.05
    budgets = [400, 800, 1500, 3000, 6000, 12000]
    print(f"\n[cost-budget sweep @ probe_cost={probe_cost}]  budget -> recall")
    for method in ("Hierarchical", "SuccessiveElim", "Uniform"):
        ys = [recall(method, b, probe_cost) for b in budgets]
        ax.plot(budgets, ys, marker=MARKERS[method], color=COLORS[method], lw=2, label=method)
        print(f"  {method:15s} " + " ".join(f"{y:.2f}" for y in ys))
    ax.set_xscale("log")
    ax.set_xlabel("cost budget")
    ax.set_title(f"Recall vs budget with cheap biased probes (probe/leaf = {probe_cost})\n"
                 f"beam={BEAM}", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="lower right")


def main() -> None:
    print(f"tree: {LEAVES} leaves, branching={BRANCHING}, depth={DEPTH}, top-{K}, "
          f"{N_SEEDS} seeds")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    fidelity_panel(axes[0])
    cost_budget_panel(axes[1])
    fig.suptitle(f"Multi-fidelity tree top-k: {LEAVES} leaves, branching={BRANCHING}, "
                 f"depth={DEPTH}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(__file__).parent / "tree_topk_benchmark.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved chart to {out}")


if __name__ == "__main__":
    main()
