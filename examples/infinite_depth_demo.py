"""Infinite-depth case: finite-state compression of the regret-optimal algorithm.

Fixed arity, growing depth. The memory-bounded HOO/HCT algorithm (B-value backup + the
``T(v) >= c^2 log / spread(level)^2`` expansion rule) only ever instantiates nodes near
the optimum, so its memory stays bounded as the tree grows, while the tree itself (and
any fixed-resolution method) blows up exponentially. This is the discrete stand-in for
the continuous infinite-depth tree: under local smoothness, the explored state is finite
and governed by the near-optimality dimension, not the (infinite) tree size.

Run with:  uv run --extra plot python examples/infinite_depth_demo.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from canopy.bandits import TreeBandit, geometric_sigma, hierarchical_spread, run_hoo

BRANCHING = 3
HORIZON = 15000
N_SEEDS = 6
DEPTHS = [4, 6, 8, 10, 12]
SIGMA = geometric_sigma(base=0.5, decay=0.6)


def main() -> None:
    total_nodes, mem, mem_std, regret = [], [], [], []
    curves = {}
    for depth in DEPTHS:
        spread = hierarchical_spread(SIGMA, depth, BRANCHING, z=3.0)
        mems, regs, cs = [], [], []
        for seed in range(N_SEEDS):
            env = TreeBandit.from_hierarchical_gaussian(
                BRANCHING, depth, sigma=SIGMA, noise_std=0.1, rng=np.random.default_rng(seed)
            )
            r = run_hoo(env, HORIZON, spread, np.random.default_rng(1000 + seed),
                        memory_bounded=True)
            mems.append(r.memory)
            regs.append(r.final_regret)
            cs.append(r.cum_regret)
        total_nodes.append(sum(BRANCHING**lvl for lvl in range(depth + 1)))
        mem.append(np.mean(mems))
        mem_std.append(np.std(mems))
        regret.append(np.mean(regs))
        curves[depth] = np.mean(cs, axis=0)
        print(f"depth {depth:2d}: total_nodes={total_nodes[-1]:8d}  "
              f"HOO mem={mem[-1]:6.1f}  regret={regret[-1]:7.1f}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

    axA.plot(DEPTHS, total_nodes, "s--", color="#7f7f7f", label="total nodes in tree")
    axA.errorbar(DEPTHS, mem, yerr=mem_std, marker="o", color="#d62728", lw=2,
                 capsize=3, label="memory-bounded HOO (nodes explored)")
    axA.set_yscale("log")
    axA.set_xlabel("tree depth")
    axA.set_ylabel("nodes (log scale)")
    axA.set_title("Finite-state compression: explored memory stays\nbounded as the tree "
                  "grows exponentially", fontsize=10)
    axA.grid(True, which="both", ls=":", alpha=0.5)
    axA.legend(loc="center right", fontsize=9)

    rounds = np.arange(1, HORIZON + 1)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(DEPTHS)))
    for color, depth in zip(cmap, DEPTHS):
        axB.plot(rounds, curves[depth], color=color, lw=1.8, label=f"depth {depth}")
    axB.set_xlabel("round")
    axB.set_ylabel("cumulative regret")
    axB.set_title("Regret stays controlled as depth grows", fontsize=10)
    axB.grid(True, ls=":", alpha=0.5)
    axB.legend(loc="upper left", fontsize=8)

    fig.suptitle(f"Infinite-depth regime (arity {BRANCHING}, horizon {HORIZON}): "
                 "regret-optimal HOO compresses to a finite explored tree", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(__file__).parent / "tree_infinite_depth.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved chart to {out}")


if __name__ == "__main__":
    main()
