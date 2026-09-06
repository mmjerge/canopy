"""Ablation: does estimating the Lipschitz constant from data matter, or is the win just HOO?

The paper is explicit that the optimistic hierarchical engine (HOO/zooming) is inherited; the
contribution is *estimating the local smoothness from data* instead of assuming a schedule. This
ablation isolates exactly that. On a heterogeneous-smoothness tree (left smooth, right rough, so
no single global Lipschitz constant is right), we compare:

  * ASSUMED-L: the inherited engine run with a fixed, hand-set global constant L -- swept over a
    range to trace the tuning curve (too tight under-explores the rough half and misses the
    optimum; too loose over-explores the smooth half). Its best point is the *oracle-tuned* L.
  * ESTIMATED-L (ours): the same engine, but each subtree estimates its own constant online
    (``run_local_lipschitz``) -- no tuning, no prior knowledge of L.

The headline: the estimated-L curve sits near the *minimum* of the assumed-L tuning curve, i.e.
the data-driven estimate matches the best hand-tuned constant without knowing it, and beats any
mis-specified one. Multi-seed with 95% CI bands.

Run with:  uv run --extra plot python examples/tree_bandits/lipschitz_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402

from canopy.bandits import TreeBandit, lipschitz_spread, run_hoo, run_local_lipschitz  # noqa: E402
from canopy.bandits.rewards import heterogeneous_smoothness_leaf_means  # noqa: E402
from _plotstyle import PALETTE, ci_band, save_figure, set_style  # noqa: E402

FIGDIR = Path(__file__).resolve().parents[1].parent / "paper" / "figures"
BRANCHING, DEPTH, HORIZON = 4, 5, 10000
N_SEEDS = 10
RHO = 1.0 / BRANCHING
L_SWEEP = [0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0]  # assumed global Lipschitz constants to try


def _env(lm, seed):
    return TreeBandit(
        BRANCHING, DEPTH, leaf_means=lm, noise_std=0.1, rng=np.random.default_rng(seed)
    )


def run() -> dict:
    # assumed-L: final regret per (L, seed); estimated-L: final regret per seed; also keep the
    # tight/loose/estimated regret-over-time curves for the second panel.
    assumed = np.zeros((len(L_SWEEP), N_SEEDS))
    estimated = np.zeros(N_SEEDS)
    curves = {"tight": [], "loose": [], "estimated": []}
    for s in range(N_SEEDS):
        lm = heterogeneous_smoothness_leaf_means(BRANCHING, DEPTH, np.random.default_rng(s))
        for i, L in enumerate(L_SWEEP):
            r = run_hoo(
                _env(lm, 100 + s),
                HORIZON,
                lipschitz_spread(L, RHO),
                np.random.default_rng(100 + s),
                memory_bounded=True,
            )
            assumed[i, s] = r.final_regret
            if i == 0:
                curves["tight"].append(r.cum_regret)
            if L == L_SWEEP[-1]:
                curves["loose"].append(r.cum_regret)
        re = run_local_lipschitz(_env(lm, 100 + s), HORIZON, np.random.default_rng(100 + s))
        estimated[s] = re.final_regret
        curves["estimated"].append(re.cum_regret)
    return {"assumed": assumed, "estimated": estimated, "curves": curves}


def main() -> None:
    set_style()
    res = run()
    assumed, estimated = res["assumed"], res["estimated"]
    a_mean = assumed.mean(axis=1)
    best_i = int(np.argmin(a_mean))
    oracle_L, oracle_regret = L_SWEEP[best_i], a_mean[best_i]
    est_mean = estimated.mean()
    print(f"heterogeneous tree ({BRANCHING**DEPTH} leaves, horizon {HORIZON}, {N_SEEDS} seeds)")
    print(
        "  assumed-L sweep final regret: "
        + "  ".join(f"L={L}:{a_mean[i]:.0f}" for i, L in enumerate(L_SWEEP))
    )
    print(f"  oracle-tuned L={oracle_L} -> {oracle_regret:.0f}")
    print(
        f"  estimated-L (ours)          -> {est_mean:.0f}  "
        f"(vs tight {a_mean[0]:.0f}, loose {a_mean[-1]:.0f})"
    )

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.4))
    # Panel A: regret vs assumed L (tuning curve) + estimated-L band + oracle marker
    ci_band(axA, L_SWEEP, assumed, PALETTE["blue"], label="assumed fixed L", marker="o")
    e_lo, e_hi = np.percentile(estimated, [2.5, 97.5])
    axA.axhspan(e_lo, e_hi, color=PALETTE["red"], alpha=0.15)
    axA.axhline(est_mean, color=PALETTE["red"], ls="-", lw=2, label="estimated L (ours)")
    axA.scatter(
        [oracle_L],
        [oracle_regret],
        color=PALETTE["green"],
        s=90,
        zorder=5,
        marker="*",
        label="oracle-tuned L",
    )
    axA.set_xscale("log")
    axA.set_xlabel("assumed Lipschitz constant $L$")
    axA.set_ylabel("final regret")
    axA.set_title("Estimated L matches the best-tuned L (no tuning)")
    axA.legend(loc="upper left")
    # Panel B: regret over time, tight vs loose vs estimated
    rounds = np.arange(1, HORIZON + 1)
    step = max(1, HORIZON // 60)  # subsample rounds so the CI band isn't overplotted
    for key, col, lab in [
        ("tight", PALETTE["blue"], f"assumed tight (L={L_SWEEP[0]})"),
        ("loose", PALETTE["orange"], f"assumed loose (L={L_SWEEP[-1]})"),
        ("estimated", PALETTE["red"], "estimated L (ours)"),
    ]:
        samples = np.array(res["curves"][key]).T[::step]  # (n_x, n_seeds)
        ci_band(axB, rounds[::step], samples, col, label=lab, marker="")
    axB.set_xlabel("round")
    axB.set_ylabel("cumulative regret")
    axB.set_title("Mis-specified L over/under-explores")
    axB.legend(loc="upper left")
    fig.suptitle("Ablation: estimating the smoothness constant is what buys the win")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = save_figure(fig, "lipschitz_ablation")
    # LaTeX table
    tex = (
        "% Lipschitz-constant ablation (auto-generated by lipschitz_ablation.py)\n"
        "\\begin{tabular}{lr}\n\\toprule\nPolicy & Final regret \\\\\n\\midrule\n"
        f"Assumed L (too tight, $L={L_SWEEP[0]}$) & {a_mean[0]:.0f} \\\\\n"
        f"Assumed L (too loose, $L={L_SWEEP[-1]}$) & {a_mean[-1]:.0f} \\\\\n"
        f"Oracle-tuned fixed $L={oracle_L}$ & {oracle_regret:.0f} \\\\\n"
        f"\\textbf{{Estimated L (ours)}} & \\textbf{{{est_mean:.0f}}} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
    )
    (FIGDIR / "lipschitz_ablation_table.tex").write_text(tex)
    print(f"\nsaved {out} (+ .png) and lipschitz_ablation_table.tex to {FIGDIR}")


if __name__ == "__main__":
    main()
