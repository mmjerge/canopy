"""Online/regret mode: the regret-vs-storage tradeoff and adaptive expansion.

Each round the learner commits to a node and follows a random path down, competing against
the best single leaf. Shallow commitments are cheap in memory but pay a per-round bias
(<= spread(level)); leaf-level commitments remove the bias but cost memory. The adaptive
rule expands a node when it becomes bias-limited (r(v) <= spread(level)), concentrating
memory near the optimum.

Panels: (A) cumulative regret vs round (mean over seeds, 95% CI band); (B) final regret vs
peak memory, with the fixed-depth frontier and the two adaptive rules.

Run with:  uv run --extra plot python examples/tree_bandits/regret_storage_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402

from canopy.bandits import (  # noqa: E402
    TreeBandit,
    geometric_sigma,
    hierarchical_spread,
    run_adaptive,
    run_adaptive_variance,
    run_fixed_depth,
)
from _plotstyle import PALETTE, ci_band, progress, save_figure, set_style  # noqa: E402

BRANCHING, DEPTH = 4, 4  # 256 leaves
HORIZON = 12000
N_SEEDS = 16
SIGMA = geometric_sigma(base=0.5, decay=0.55)
SPREAD = hierarchical_spread(SIGMA, DEPTH, BRANCHING, z=3.0)


def make_env(seed: int) -> TreeBandit:
    return TreeBandit.from_hierarchical_gaussian(
        BRANCHING, DEPTH, sigma=SIGMA, noise_std=0.1, rng=np.random.default_rng(seed)
    )


def run_all():
    depths = list(range(DEPTH + 1))
    adaptive, var = [], []
    fixed = {d: [] for d in depths}
    amem, vmem, fmem = [], [], {d: 0 for d in depths}
    for seed in progress(range(N_SEEDS), "regret seeds", total=N_SEEDS):
        rng = lambda: np.random.default_rng(10_000 + seed)  # noqa: E731
        r = run_adaptive(make_env(seed), HORIZON, SPREAD, rng())
        adaptive.append(r.cum_regret)
        amem.append(r.memory)
        rv = run_adaptive_variance(make_env(seed), HORIZON, rng())
        var.append(rv.cum_regret)
        vmem.append(rv.memory)
        for d in depths:
            rd = run_fixed_depth(make_env(seed), d, HORIZON, rng())
            fixed[d].append(rd.cum_regret)
            fmem[d] = rd.memory
    return (
        depths,
        np.array(adaptive),
        np.array(var),
        {d: np.array(fixed[d]) for d in depths},
        np.array(amem),
        np.array(vmem),
        fmem,
    )


def main() -> None:
    set_style()
    depths, adaptive, var, fixed, amem, vmem, fmem = run_all()
    print(f"tree: {BRANCHING**DEPTH} leaves, depth={DEPTH}, horizon={HORIZON}, {N_SEEDS} seeds")
    print(f"adaptive (assumed spread): regret={adaptive[:, -1].mean():.0f} mem={amem.mean():.0f}")
    print(f"adaptive-variance (novel): regret={var[:, -1].mean():.0f} mem={vmem.mean():.0f}")
    for d in depths:
        print(f"fixed-depth {d}: regret={fixed[d][:, -1].mean():.0f} mem={fmem[d]}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: regret over time (downsampled for a lean vector figure).
    idx = np.arange(0, HORIZON, 50)
    x = idx + 1
    ci_band(axA, x, fixed[DEPTH][:, idx].T, PALETTE["blue"], "fixed depth 4 (full)", marker="")
    ci_band(axA, x, fixed[2][:, idx].T, PALETTE["gray"], "fixed depth 2", marker="")
    ci_band(axA, x, adaptive[:, idx].T, PALETTE["orange"], "adaptive (assumed)", marker="")
    ci_band(axA, x, var[:, idx].T, PALETTE["green"], "adaptive (variance-aware)", marker="")
    axA.set_xlabel("round")
    axA.set_ylabel("cumulative regret")
    axA.set_title("Regret over time")
    axA.legend(loc="upper left")

    # Panel B: final regret vs peak memory (the tradeoff frontier).
    mems = np.array([fmem[d] for d in depths])
    finals = np.array([fixed[d][:, -1].mean() for d in depths])
    ferr = np.array([1.96 * fixed[d][:, -1].std(ddof=1) / np.sqrt(N_SEEDS) for d in depths])
    axB.errorbar(
        mems,
        finals,
        yerr=ferr,
        fmt="o-",
        color=PALETTE["blue"],
        capsize=3,
        label="fixed depth (frontier)",
    )
    axB.errorbar(
        [amem.mean()],
        [adaptive[:, -1].mean()],
        yerr=[1.96 * adaptive[:, -1].std(ddof=1) / np.sqrt(N_SEEDS)],
        fmt="P",
        color=PALETTE["orange"],
        markersize=11,
        capsize=3,
        label="adaptive (assumed)",
    )
    axB.errorbar(
        [vmem.mean()],
        [var[:, -1].mean()],
        yerr=[1.96 * var[:, -1].std(ddof=1) / np.sqrt(N_SEEDS)],
        fmt="*",
        color=PALETTE["green"],
        markersize=15,
        capsize=3,
        label="adaptive (variance-aware)",
    )
    axB.set_xscale("log")
    axB.set_xlabel("peak memory (nodes tracked)")
    axB.set_ylabel("final cumulative regret")
    axB.set_title("Regret vs storage")
    axB.legend(loc="upper right")

    fig.tight_layout()
    out = save_figure(fig, "tree_regret_storage")
    print(f"\nsaved chart to {out} (+ .png)")


if __name__ == "__main__":
    main()
