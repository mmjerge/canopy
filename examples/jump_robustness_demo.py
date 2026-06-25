"""Robustness to jump discontinuities: data-driven bias vs. assumed smoothness.

Weakening smoothness to "mostly smooth + a few sharp jumps" (the piecewise-Lipschitz /
dispersion setting). We sweep the jump width (narrow/hidden -> wide/visible) and compare,
at bounded memory:

  * assumed-smooth HOO  -- memory-bounded HOO whose bias term is the smooth spread(level)
    schedule (it expands a node only when r(v) <= spread(level)). A narrow jump violates
    smoothness, so spread under-estimates the bias and the algorithm gets stuck refining
    a cell it wrongly thinks is resolved.
  * data-driven (variance-aware) -- uses the measured within-cell variance as the bias
    proxy, so it detects the heterogeneity at a jump cell and keeps expanding.

Takeaway: the data-driven version is *robust* to how hidden the jumps are (roughly flat
regret), while assumed smoothness only does well when jumps are wide enough to show up in
coarse averages.

Run with:  uv run --extra plot python examples/jump_robustness_demo.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from canopy.bandits import (
    TreeBandit,
    geometric_sigma,
    hierarchical_spread,
    run_adaptive_variance,
    run_hoo,
    run_hybrid,
)

BRANCHING, DEPTH, HORIZON = 4, 5, 15000  # 1024 leaves
N_SEEDS = 10
N_JUMPS = 4
WIDTHS = [4, 8, 16, 32, 64]
SPREAD_SMOOTH = hierarchical_spread(geometric_sigma(0.12, 0.6), DEPTH, BRANCHING, z=3.0)


def main() -> None:
    assumed, datadriven, hybrid = [], [], []
    print(
        f"piecewise-smooth tree: {BRANCHING**DEPTH} leaves, {N_JUMPS} jumps, "
        f"horizon {HORIZON}, {N_SEEDS} seeds"
    )
    for jw in WIDTHS:
        a, v, h = [], [], []
        for seed in range(N_SEEDS):
            env = TreeBandit.from_piecewise_smooth(
                BRANCHING,
                DEPTH,
                N_JUMPS,
                jump_width=jw,
                noise_std=0.1,
                rng=np.random.default_rng(seed),
            )
            a.append(
                run_hoo(
                    env,
                    HORIZON,
                    SPREAD_SMOOTH,
                    np.random.default_rng(100 + seed),
                    memory_bounded=True,
                ).final_regret
            )
            env2 = TreeBandit.from_piecewise_smooth(
                BRANCHING,
                DEPTH,
                N_JUMPS,
                jump_width=jw,
                noise_std=0.1,
                rng=np.random.default_rng(seed),
            )
            v.append(
                run_adaptive_variance(env2, HORIZON, np.random.default_rng(100 + seed)).final_regret
            )
            env3 = TreeBandit.from_piecewise_smooth(
                BRANCHING,
                DEPTH,
                N_JUMPS,
                jump_width=jw,
                noise_std=0.1,
                rng=np.random.default_rng(seed),
            )
            h.append(
                run_hybrid(
                    env3,
                    HORIZON,
                    SPREAD_SMOOTH,
                    np.random.default_rng(100 + seed),
                    jump_factor=0.5,
                    n_min=8,
                ).final_regret
            )
        assumed.append(np.mean(a))
        datadriven.append(np.mean(v))
        hybrid.append(np.mean(h))
        print(
            f"  jump_width={jw:3d}: assumed-smooth={assumed[-1]:7.1f}  "
            f"data-driven={datadriven[-1]:7.1f}  hybrid={hybrid[-1]:7.1f}"
        )

    fig, ax = plt.subplots(figsize=(7.8, 5))
    ax.plot(WIDTHS, assumed, "s-", color="#1f77b4", lw=2, label="assumed-smooth spread(level)")
    ax.plot(
        WIDTHS, datadriven, "o-", color="#d62728", lw=2, label="data-driven (within-cell variance)"
    )
    ax.plot(
        WIDTHS,
        hybrid,
        "D-",
        color="#2ca02c",
        lw=2.4,
        label="hybrid (Lipschitz floor + jump detection)",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("jump width  (narrow / hidden  ->  wide / visible)")
    ax.set_ylabel(f"final regret ({N_SEEDS} seeds)")
    ax.set_title(
        f"Robustness to {N_JUMPS} jump discontinuities\n"
        f"hybrid dominates data-driven and avoids the assumed-smooth blow-up",
        fontsize=10,
    )
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = Path(__file__).parent / "images" / "tree_jump_robustness.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved chart to {out}")


if __name__ == "__main__":
    main()
