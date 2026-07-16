"""Ablation: estimating the smoothness constant is what buys the win.

Because the optimistic descent is inherited (HOO/HCT), the paper's actual contribution
is estimating the (local) Lipschitz constant from data rather than assuming a schedule.
This ablation isolates exactly that, on a heterogeneous-smoothness tree where no single
global constant is right (left half smooth, right half rough with the optimum hidden as
a spike; see ``heterogeneous_smoothness_leaf_means``):

  * assumed fixed L  -- run the adaptive engine with spread(l) = L * rho^l for a sweep
    of L values: too tight under-explores the rough half and misses the optimum, too
    loose over-explores the smooth half. Final regret over L is U-shaped.
  * oracle-tuned L   -- the best point on that sweep (unavailable in practice).
  * estimated L      -- ``run_local_lipschitz``: each subtree carries its own constant,
    estimated online from its measured within-cell spread and propagated to children.
    No tuning at all.

The estimated-L policy should match (or beat) the oracle-tuned fixed constant, because
it adapts a *local* constant per subtree that no single global L can match.

Run with:  uv run --extra plot python examples/tree_bandits/lipschitz_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402

from canopy.bandits import (  # noqa: E402
    TreeBandit,
    heterogeneous_smoothness_leaf_means,
    run_adaptive,
    run_local_lipschitz,
)
from _plotstyle import FIGURE_DIR, PALETTE, ci_band, save_figure, set_style  # noqa: E402

BRANCHING, DEPTH = 4, 4  # 256 leaves
HORIZON = 10_000
N_SEEDS = 10
NOISE = 0.1
RHO = 1.0 / BRANCHING
ASSUMED_L = [0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0]
TIGHT_L, LOOSE_L = 0.05, 3.0


def make_env(seed: int) -> TreeBandit:
    lm = heterogeneous_smoothness_leaf_means(BRANCHING, DEPTH, rng=np.random.default_rng(seed))
    return TreeBandit(
        BRANCHING, DEPTH, leaf_means=lm, noise_std=NOISE, rng=np.random.default_rng(100 + seed)
    )


def spread_for(lipschitz: float):
    return lambda level: lipschitz * RHO**level


def run_all():
    fixed_curves = {lip: [] for lip in ASSUMED_L}  # L -> list of cum-regret arrays
    est_curves = []
    for seed in range(N_SEEDS):
        for lip in ASSUMED_L:
            r = run_adaptive(
                make_env(seed), HORIZON, spread_for(lip), np.random.default_rng(10_000 + seed)
            )
            fixed_curves[lip].append(r.cum_regret)
        r = run_local_lipschitz(
            make_env(seed), HORIZON, np.random.default_rng(10_000 + seed), rho=RHO
        )
        est_curves.append(r.cum_regret)
    return (
        {lip: np.array(c) for lip, c in fixed_curves.items()},
        np.array(est_curves),
    )


def main() -> None:
    set_style()
    print(
        f"heterogeneous-smoothness tree: {BRANCHING**DEPTH} leaves, horizon {HORIZON}, "
        f"{N_SEEDS} seeds; assumed L sweep {ASSUMED_L}"
    )
    fixed, est = run_all()

    fixed_final = {lip: fixed[lip][:, -1] for lip in ASSUMED_L}
    est_final = est[:, -1]
    oracle_l = min(ASSUMED_L, key=lambda lip: fixed_final[lip].mean())

    print(f"{'policy':>28s} {'final regret':>13s}")
    for lip in ASSUMED_L:
        tag = "  <- oracle-tuned" if lip == oracle_l else ""
        print(f"{'assumed L = ' + str(lip):>28s} {fixed_final[lip].mean():13.0f}{tag}")
    print(f"{'estimated L (ours)':>28s} {est_final.mean():13.0f}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: U-shaped tuning curve vs the no-tuning estimator
    means = np.array([fixed_final[lip].mean() for lip in ASSUMED_L])
    errs = np.array(
        [1.96 * fixed_final[lip].std(ddof=1) / np.sqrt(N_SEEDS) for lip in ASSUMED_L]
    )
    axA.errorbar(
        ASSUMED_L, means, yerr=errs, fmt="o-", color=PALETTE["blue"], capsize=3,
        label="assumed fixed $L$",
    )
    est_mean = est_final.mean()
    est_err = 1.96 * est_final.std(ddof=1) / np.sqrt(N_SEEDS)
    axA.axhspan(est_mean - est_err, est_mean + est_err, color=PALETTE["red"], alpha=0.15)
    axA.axhline(est_mean, color=PALETTE["red"], lw=1.5, label="estimated $L$ (ours)")
    axA.plot(
        [oracle_l], [fixed_final[oracle_l].mean()], "*", color=PALETTE["green"],
        markersize=15, zorder=3, label=f"oracle-tuned $L={oracle_l}$",
    )
    axA.set_xscale("log")
    axA.set_xlabel("assumed Lipschitz constant $L$")
    axA.set_ylabel("final regret")
    axA.set_title("Estimated $L$ matches the best-tuned $L$ (no tuning)")
    axA.legend(loc="upper right", fontsize=8)

    # Panel B: cumulative regret -- mis-specified fixed L vs estimated
    x = np.arange(0, HORIZON, 50)
    ci_band(axB, x + 1, fixed[TIGHT_L][:, x].T, PALETTE["blue"],
            f"assumed tight ($L={TIGHT_L}$)", marker="")
    ci_band(axB, x + 1, fixed[LOOSE_L][:, x].T, PALETTE["orange"],
            f"assumed loose ($L={LOOSE_L}$)", marker="")
    ci_band(axB, x + 1, est[:, x].T, PALETTE["red"], "estimated $L$ (ours)", marker="")
    axB.set_xlabel("round")
    axB.set_ylabel("cumulative regret")
    axB.set_title("Mis-specified $L$ over/under-explores")
    axB.legend(loc="upper left", fontsize=8)

    fig.suptitle("Ablation: estimating the smoothness constant is what buys the win")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = save_figure(fig, "lipschitz_ablation")
    print(f"\nsaved chart to {out} (+ .png)")

    # LaTeX table (Table 1)
    rows = [
        (f"Assumed $L$ (too tight, $L = {TIGHT_L}$)", f"{fixed_final[TIGHT_L].mean():.0f}"),
        (f"Assumed $L$ (too loose, $L = {LOOSE_L}$)", f"{fixed_final[LOOSE_L].mean():.0f}"),
        (f"Oracle-tuned fixed $L = {oracle_l}$", f"{fixed_final[oracle_l].mean():.0f}"),
        ("\\textbf{Estimated $L$ (ours)}", f"\\textbf{{{est_final.mean():.0f}}}"),
    ]
    tex = (
        "% Lipschitz ablation (auto-generated by examples/tree_bandits/lipschitz_ablation.py)\n"
        "\\begin{tabular}{lr}\n\\toprule\nPolicy & Final regret \\\\\n\\midrule\n"
        + "\n".join(f"{a} & {b} \\\\" for a, b in rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIGURE_DIR / "lipschitz_ablation_table.tex").write_text(tex)
    print(f"wrote LaTeX table to {FIGURE_DIR / 'lipschitz_ablation_table.tex'}")


if __name__ == "__main__":
    main()
