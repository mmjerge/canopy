"""Prefix-cache management as online tree selection: adaptive vs LRU/LFU.

The cache is an ancestor-closed subtree of the token trie under a memory budget (storage),
and savings = tokens reused per prompt. We compare a recency-weighted, tree-aware policy
(adaptive) against LRU, LFU, and the hindsight-optimal static cache.

Panels:
  (A) savings/prompt vs memory budget on a stationary stream -- adaptive matches LFU and
      the offline optimum; LRU lags.
  (B) savings over time across a mid-stream popularity shift -- adaptive tracks the new
      distribution and beats LFU (sticky) and even the best *static* cache.

Run with:  uv run --extra plot python examples/prefix_cache_demo.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from canopy.bandits import PrefixCacheEnv, run_cache

HORIZON = 20000
N_SEEDS = 6
POLICIES = ["lru", "lfu", "adaptive", "offline"]
COLORS = {"lru": "#9467bd", "lfu": "#1f77b4", "adaptive": "#d62728", "offline": "#2ca02c"}


def main() -> None:
    budgets = [8, 16, 24, 40, 64, 96]
    print(f"prefix cache: horizon {HORIZON}, {N_SEEDS} seeds")

    # Panel A: stationary, savings vs budget
    stationary = {p: [] for p in POLICIES}
    for b in budgets:
        for p in POLICIES:
            s = [run_cache(PrefixCacheEnv(rng=np.random.default_rng(seed)).generate_stream(HORIZON),
                           b, policy=p).avg_savings for seed in range(N_SEEDS)]
            stationary[p].append(np.mean(s))
    print("\nstationary savings/prompt:")
    for p in POLICIES:
        print(f"  {p:9s} " + " ".join(f"{v:.2f}" for v in stationary[p]))

    # Panel B: non-stationary, savings over time at a fixed budget
    shift_budget = 24
    curves = {p: [] for p in POLICIES}
    for p in POLICIES:
        for seed in range(N_SEEDS):
            stream = PrefixCacheEnv(shift_at=HORIZON // 2,
                                    rng=np.random.default_rng(seed)).generate_stream(HORIZON)
            curves[p].append(run_cache(stream, shift_budget, policy=p).savings_curve)
    curve_mean = {p: np.mean(curves[p], axis=0) for p in POLICIES}
    print(f"\nnon-stationary (shift at {HORIZON // 2}, B={shift_budget}) final savings:")
    for p in POLICIES:
        print(f"  {p:9s} {curve_mean[p][-1]:.2f}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))
    for p in POLICIES:
        axA.plot(budgets, stationary[p], "o-", color=COLORS[p], lw=2,
                 label=p + (" (optimal)" if p == "offline" else ""))
    axA.set_xlabel("cache memory budget (nodes)")
    axA.set_ylabel("tokens reused per prompt")
    axA.set_title("Stationary: savings vs memory budget", fontsize=10)
    axA.grid(True, ls=":", alpha=0.5)
    axA.legend(loc="lower right", fontsize=8)

    rounds = np.arange(1, HORIZON + 1)
    for p in POLICIES:
        axB.plot(rounds, curve_mean[p], color=COLORS[p], lw=2, label=p)
    axB.axvline(HORIZON // 2, color="gray", ls="--", lw=1)
    axB.text(HORIZON // 2, 0.2, " popularity shift", color="gray", fontsize=8, rotation=90)
    axB.set_xlabel("prompts seen")
    axB.set_ylabel("tokens reused per prompt (rolling)")
    axB.set_title(f"Non-stationary (B={shift_budget}): adaptive tracks the shift", fontsize=10)
    axB.grid(True, ls=":", alpha=0.5)
    axB.legend(loc="lower left", fontsize=8)

    fig.suptitle("Prefix-cache management as online tree selection under a memory budget",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(__file__).parent / "tree_prefix_cache.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved chart to {out}")


if __name__ == "__main__":
    main()
