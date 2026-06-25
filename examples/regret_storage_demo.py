"""Online/regret mode: the regret-vs-storage tradeoff and adaptive expansion.

Each round the learner commits to a node and follows a random path down; it competes
against the best single leaf. Committing at a shallow level is cheap in memory but pays
a per-round bias (<= spread(level)); committing at the leaves removes the bias but costs
memory and up-front exploration over many arms. The adaptive rule expands a node exactly
when it becomes bias-limited (r(v) <= spread(level)), concentrating memory near the
optimum.

Two panels:
  (A) cumulative regret vs round for adaptive and several fixed resolutions.
  (B) final regret vs peak memory: fixed-depth sweep traces the tradeoff frontier;
      adaptive sits below/left of it (better regret at less memory).

Run with:  uv run --extra plot python examples/regret_storage_demo.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from canopy.bandits import (
    TreeBandit,
    geometric_sigma,
    hierarchical_spread,
    run_adaptive,
    run_adaptive_variance,
    run_fixed_depth,
)

BRANCHING, DEPTH = 4, 4  # 256 leaves
HORIZON = 12000
N_SEEDS = 10
SIGMA = geometric_sigma(base=0.5, decay=0.55)
SPREAD = hierarchical_spread(SIGMA, DEPTH, BRANCHING, z=3.0)


def make_env(seed: int) -> TreeBandit:
    return TreeBandit.from_hierarchical_gaussian(
        BRANCHING, DEPTH, sigma=SIGMA, noise_std=0.1, rng=np.random.default_rng(seed)
    )


def main() -> None:
    depths = list(range(DEPTH + 1))  # fixed-depth 0..DEPTH
    adaptive_curves, adaptive_mem = [], []
    var_curves, var_mem = [], []
    fixed_curves = {d: [] for d in depths}
    fixed_mem = {d: 0 for d in depths}

    for seed in range(N_SEEDS):
        env = make_env(seed)
        res = run_adaptive(env, HORIZON, SPREAD, np.random.default_rng(10_000 + seed))
        adaptive_curves.append(res.cum_regret)
        adaptive_mem.append(res.memory)

        env_v = make_env(seed)
        rv = run_adaptive_variance(env_v, HORIZON, np.random.default_rng(10_000 + seed))
        var_curves.append(rv.cum_regret)
        var_mem.append(rv.memory)

        for d in depths:
            env_d = make_env(seed)
            r = run_fixed_depth(env_d, d, HORIZON, np.random.default_rng(10_000 + seed))
            fixed_curves[d].append(r.cum_regret)
            fixed_mem[d] = r.memory

    adaptive_mean = np.mean(adaptive_curves, axis=0)
    adaptive_mem_mean = float(np.mean(adaptive_mem))
    var_mean = np.mean(var_curves, axis=0)
    var_mem_mean = float(np.mean(var_mem))
    fixed_mean = {d: np.mean(fixed_curves[d], axis=0) for d in depths}

    print(f"tree: {BRANCHING**DEPTH} leaves, depth={DEPTH}, horizon={HORIZON}, "
          f"{N_SEEDS} seeds")
    print(f"adaptive (assumed spread): final regret={adaptive_mean[-1]:.0f}  mem={adaptive_mem_mean:.0f}")
    print(f"adaptive-variance (novel): final regret={var_mean[-1]:.0f}  mem={var_mem_mean:.0f}")
    for d in depths:
        print(f"fixed-depth {d}: final regret={fixed_mean[d][-1]:.0f}  mem={fixed_mem[d]}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))
    rounds = np.arange(1, HORIZON + 1)

    # Panel A: cumulative regret over time
    show = [1, 2, DEPTH]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.8, len(show)))
    for color, d in zip(cmap, show):
        label = f"fixed depth {d} (mem {fixed_mem[d]})" + (" = full leaves" if d == DEPTH else "")
        axA.plot(rounds, fixed_mean[d], color=color, lw=1.8, label=label)
    axA.plot(rounds, adaptive_mean, color="#ff7f0e", lw=2.2,
             label=f"adaptive, assumed spread (mem {adaptive_mem_mean:.0f})")
    axA.plot(rounds, var_mean, color="#d62728", lw=2.6,
             label=f"adaptive, variance-aware (mem {var_mem_mean:.0f})")
    axA.set_xlabel("round")
    axA.set_ylabel("cumulative regret")
    axA.set_title("Regret over time", fontsize=10)
    axA.grid(True, ls=":", alpha=0.5)
    axA.legend(loc="upper left", fontsize=8)

    # Panel B: final regret vs memory (tradeoff frontier)
    mems = [fixed_mem[d] for d in depths]
    finals = [fixed_mean[d][-1] for d in depths]
    axB.plot(mems, finals, "o-", color="#1f77b4", label="fixed depth (frontier)")
    for d, mm, ff in zip(depths, mems, finals):
        axB.annotate(f"d={d}", (mm, ff), fontsize=8, xytext=(4, 4),
                     textcoords="offset points")
    axB.plot([adaptive_mem_mean], [adaptive_mean[-1]], "P", color="#ff7f0e",
             markersize=13, label="adaptive (assumed spread)")
    axB.plot([var_mem_mean], [var_mean[-1]], "*", color="#d62728",
             markersize=17, label="adaptive (variance-aware)")
    axB.set_xscale("log")
    axB.set_xlabel("peak memory (nodes tracked)")
    axB.set_ylabel("final cumulative regret")
    axB.set_title("Regret vs storage", fontsize=10)
    axB.grid(True, which="both", ls=":", alpha=0.5)
    axB.legend(loc="upper right", fontsize=8)

    fig.suptitle("Adaptive expansion (expand when r(v) <= spread/heterogeneity) is "
                 "near regret-optimal at a fraction of the memory", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(__file__).parent / "tree_regret_storage.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved chart to {out}")


if __name__ == "__main__":
    main()
