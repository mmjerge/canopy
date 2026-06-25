"""Applied: cost-aware LLM routing over a prefix tree.

A big model (high quality, costly) and a cheap model (strong only in some prefix regions,
weak elsewhere, with sharp region boundaries). Routing every prompt to the big model is
the best *fixed* policy but wastes cost where the cheap model suffices. A hierarchical
router learns the per-prefix-region best model from the prompt stream, exploiting the
tree's smoothness to generalize across prompts that share a prefix.

Two panels:
  (A) cumulative routing regret vs the per-prompt oracle, for the router (good resolution)
      and the fixed-policy baselines.
  (B) cost-vs-quality Pareto: the router lands near the oracle (high quality, low cost),
      while always-big is high-quality but high-cost.

Run with:  uv run --extra plot python examples/llm_routing_demo.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from oco.bandits import PrefixTreeRouting, make_routing_scenario, run_router

BRANCHING, DEPTH, HORIZON = 4, 5, 20000  # 1024 prompts
N_SEEDS = 8
LAM = 0.3


def run(strategy: str, seed: int, **kw) -> "object":
    q, costs = make_routing_scenario(BRANCHING, DEPTH, np.random.default_rng(seed))
    env = PrefixTreeRouting(BRANCHING, DEPTH, q, costs, lam=LAM, noise_std=0.1,
                            rng=np.random.default_rng(100 + seed))
    return run_router(env, HORIZON, np.random.default_rng(100 + seed), strategy=strategy, **kw)


def main() -> None:
    methods = [
        ("hierarchical", {"resolution": 2}, "#d62728"),
        ("hierarchical", {"resolution": 0}, "#9467bd"),
        ("best_single", {}, "#1f77b4"),
        ("all_largest", {}, "#ff7f0e"),
        ("oracle", {}, "#2ca02c"),
    ]
    curves, points = {}, {}
    print(f"prefix-tree routing: {BRANCHING**DEPTH} prompts, horizon {HORIZON}, "
          f"{N_SEEDS} seeds, lam={LAM}")
    for strat, kw, _ in methods:
        regs, costs, quals = [], [], []
        for seed in range(N_SEEDS):
            r = run(strat, seed, **kw)
            regs.append(r.cum_regret)
            costs.append(r.total_cost)
            quals.append(r.avg_quality)
        label = run(strat, 0, **kw).label
        curves[label] = (np.mean(regs, axis=0), methods)
        points[label] = (float(np.mean(costs)) / HORIZON, float(np.mean(quals)))
        print(f"  {label:20s} regret={np.mean(regs, axis=0)[-1]:8.1f}  "
              f"cost/query={points[label][0]:.3f}  avg_quality={points[label][1]:.3f}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))
    rounds = np.arange(1, HORIZON + 1)
    color_of = {}
    for strat, kw, col in methods:
        label = run(strat, 0, **kw).label
        color_of[label] = col
    for label, (curve, _) in curves.items():
        axA.plot(rounds, curve, color=color_of[label], lw=2, label=label)
    axA.set_xlabel("prompts seen")
    axA.set_ylabel("cumulative routing regret")
    axA.set_title("Router learns per-region routing (sublinear regret)", fontsize=10)
    axA.grid(True, ls=":", alpha=0.5)
    axA.legend(loc="upper left", fontsize=8)

    for label, (cpq, qual) in points.items():
        axB.scatter([cpq], [qual], color=color_of[label], s=90,
                    marker="*" if label == "oracle" else "o", zorder=3)
        axB.annotate(label, (cpq, qual), fontsize=8, xytext=(5, 4), textcoords="offset points")
    axB.set_xlabel("cost per query")
    axB.set_ylabel("average quality")
    axB.set_title("Cost vs quality: router approaches the oracle\nat lower cost than "
                  "always-big", fontsize=10)
    axB.grid(True, ls=":", alpha=0.5)

    fig.suptitle("LLM routing over a prefix tree: hierarchical generalization beats fixed "
                 "policies", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(__file__).parent / "tree_llm_routing.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved chart to {out}")


if __name__ == "__main__":
    main()
