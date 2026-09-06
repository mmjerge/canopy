"""Multi-fidelity benchmark: when does exploiting the tree beat a strong baseline?

The tree environment is multi-fidelity: an internal-node probe is CHEAP but biased (cost
``probe_cost``), a leaf evaluation is EXPENSIVE but unbiased (cost ``leaf_cost``). Budgets
are measured in cost. We compare, at equal cost budget:

  * HierarchicalTopK          -- cheap probes to localize, then leaf evals to confirm
  * SuccessiveEliminationTopK -- STRONG structure-blind baseline (leaf evals only)
  * UniformTopK               -- weak structure-blind baseline

Produces images/tree_topk_benchmark.{pdf,png} with two panels: recall vs the probe/leaf
cost ratio (fixed budget), and recall vs cost budget at a cheap probe cost. Lines show the
mean over seeds with a shaded 95% confidence band.

Run with:  uv run --extra plot python examples/tree_bandits/benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402

from canopy.bandits import (  # noqa: E402
    HierarchicalTopK,
    SuccessiveEliminationTopK,
    TreeBandit,
    UniformTopK,
    geometric_sigma,
    hierarchical_spread,
)
from _plotstyle import PALETTE, ci_band, progress, save_figure, set_style  # noqa: E402

BRANCHING, DEPTH, K = 4, 5, 5  # 1024 leaves
LEAVES = BRANCHING**DEPTH
N_SEEDS = 60
SIGMA = geometric_sigma(base=0.5, decay=0.55)
SPREAD = hierarchical_spread(SIGMA, DEPTH, BRANCHING, z=3.0)
BEAM = 20

COLORS = {
    "Hierarchical": PALETTE["blue"],
    "HierarchicalSound": PALETTE["sky"],
    "SuccessiveElim": PALETTE["green"],
    "Uniform": PALETTE["red"],
}
MARKERS = {"Hierarchical": "o", "HierarchicalSound": "D", "SuccessiveElim": "^", "Uniform": "s"}
LABELS = {
    "Hierarchical": "Hierarchical, beam (ours)",
    "HierarchicalSound": "Hierarchical, sound (analyzed)",
    "SuccessiveElim": "Successive elimination",
    "Uniform": "Uniform",
}


MIXTURE = "--mixture" in sys.argv  # route internal probes through the spec-faithful mixture channel


def make_env(seed: int, probe_cost: float) -> TreeBandit:
    return TreeBandit.from_hierarchical_gaussian(
        BRANCHING,
        DEPTH,
        sigma=SIGMA,
        noise_std=0.1,
        probe_cost=probe_cost,
        mixture_probes=MIXTURE,
        rng=np.random.default_rng(seed),
    )


def recall_samples(method: str, budget: float, probe_cost: float) -> np.ndarray:
    """Per-seed top-K recall (shape (N_SEEDS,))."""
    out = np.empty(N_SEEDS)
    for seed in range(N_SEEDS):
        env = make_env(seed, probe_cost)
        if method == "Hierarchical":
            res = HierarchicalTopK(budget=budget, spread=SPREAD, beam_width=BEAM).run(env, K)
        elif method == "HierarchicalSound":
            # the sound algorithm Theorem 2 analyzes: no beam focusing (worst-case valid pruning)
            res = HierarchicalTopK(budget=budget, spread=SPREAD, beam_width=None).run(env, K)
        elif method == "SuccessiveElim":
            res = SuccessiveEliminationTopK(budget=budget).run(env, K)
        else:
            res = UniformTopK(budget=budget).run(env, K)
        out[seed] = res.evaluate(env, K)
    return out


def fidelity_panel(ax) -> None:
    budget = 1500.0
    ratios = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
    print(f"\n[fidelity sweep @ budget={budget}]  probe/leaf cost -> mean recall")
    for method in ("Hierarchical", "HierarchicalSound", "SuccessiveElim"):
        samples = np.array(
            [recall_samples(method, budget, r) for r in progress(ratios, f"fidelity {method}")]
        )
        ci_band(ax, ratios, samples, COLORS[method], LABELS[method], MARKERS[method])
        print(f"  {method:15s} " + " ".join(f"{s.mean():.2f}" for s in samples))
    ax.set_xscale("log")
    ax.invert_xaxis()  # cheaper probes to the right
    ax.set_xlabel("probe cost / leaf cost  (cheaper probes $\\rightarrow$)")
    ax.set_ylabel(f"mean top-{K} recall")
    ax.set_title("The tree wins only when probes are cheap")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left")


def cost_budget_panel(ax) -> None:
    probe_cost = 0.05
    budgets = [400, 800, 1500, 3000, 6000, 12000]
    print(f"\n[cost-budget sweep @ probe_cost={probe_cost}]  budget -> mean recall")
    for method in ("Hierarchical", "HierarchicalSound", "SuccessiveElim", "Uniform"):
        samples = np.array(
            [recall_samples(method, b, probe_cost) for b in progress(budgets, f"budget {method}")]
        )
        ci_band(ax, budgets, samples, COLORS[method], LABELS[method], MARKERS[method])
        print(f"  {method:15s} " + " ".join(f"{s.mean():.2f}" for s in samples))
    ax.set_xscale("log")
    ax.set_xlabel("cost budget")
    ax.set_title(f"Recall vs budget (probe/leaf $= {probe_cost}$)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")


def main() -> None:
    set_style()
    print(f"tree: {LEAVES} leaves, branching={BRANCHING}, depth={DEPTH}, top-{K}, {N_SEEDS} seeds")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    fidelity_panel(axes[0])
    cost_budget_panel(axes[1])
    fig.tight_layout()
    out = save_figure(fig, "tree_topk_benchmark_mixture" if MIXTURE else "tree_topk_benchmark")
    print(f"\nsaved chart to {out} (+ .png)")


if __name__ == "__main__":
    main()
