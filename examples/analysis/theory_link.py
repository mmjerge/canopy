"""Linking the regret bound to data: the sublinear rate, and where the prior lives.

Two checks connect the piecewise-Lipschitz regret bound to the experiments:

1. **Synthetic sublinear rate.** On a tree-Lipschitz value function, cumulative regret
   should grow as ``R(n) ~ n^alpha`` with ``alpha = (d+1)/(d+2) < 1`` -- i.e. the
   *average* regret ``R(n)/n`` decays toward zero as the horizon grows. We measure the
   exponent by regressing ``log R(n)`` on ``log n`` across horizons and report it with
   the near-optimality dimension the value function implies (d ~ 0 for a function with
   a single well-separated peak).

2. **Where the prior lives on real routing data.** A variance decomposition of the
   RouterBench value function (measured quality/cost of 11 real LLMs over ~36k real
   prompts): how much of the per-prompt value variance is explained by the region
   (task-category) tree vs. left within regions as per-prompt difficulty -- the
   observation noise the bandit averages over. The router needs only the regional
   optimum, so "region tree explains the regional signal, the rest is noise" is the
   empirical face of the bandit formulation. We report both the per-prompt and the
   region-mean (region-value) decompositions, at the family and category resolutions.

Run with:  uv run --extra bench --extra plot python examples/analysis/theory_link.py
(The RouterBench panel needs the ``bench`` extra + network for the first download; the
synthetic panel always runs.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm_routing"))
import matplotlib.pyplot as plt  # noqa: E402

from canopy.bandits import (  # noqa: E402
    TreeBandit,
    geometric_sigma,
    hierarchical_spread,
    run_adaptive,
)
from _plotstyle import FIGURE_DIR, PALETTE, save_figure, set_style  # noqa: E402

BRANCHING, DEPTH = 4, 4  # 256 leaves
HORIZONS = [500, 1000, 2000, 4000, 8000, 16000]
N_SEEDS = 8
SIGMA = geometric_sigma(base=0.5, decay=0.55)
SPREAD = hierarchical_spread(SIGMA, DEPTH, BRANCHING, z=3.0)
REF_LAM = 0.2  # reference cost weight for the RouterBench value function


# --- panel A: synthetic sublinear rate ------------------------------------------


def regret_exponent() -> tuple[np.ndarray, float, float]:
    """Final regret per horizon (seeds x horizons) and the fitted exponent alpha."""
    finals = np.empty((N_SEEDS, len(HORIZONS)))
    for s in range(N_SEEDS):
        for j, horizon in enumerate(HORIZONS):
            env = TreeBandit.from_hierarchical_gaussian(
                BRANCHING, DEPTH, sigma=SIGMA, noise_std=0.1, rng=np.random.default_rng(s)
            )
            r = run_adaptive(env, horizon, SPREAD, np.random.default_rng(10_000 + s))
            finals[s, j] = r.final_regret
    mean_final = finals.mean(axis=0)
    alpha, intercept = np.polyfit(np.log(HORIZONS), np.log(mean_final), 1)
    return finals, float(alpha), float(intercept)


# --- panel B: RouterBench variance decomposition ---------------------------------

# benchmark-family grouping of RouterBench eval names (coarser than category)
FAMILY_RULES = [
    ("mmlu", "knowledge"),
    ("gsm8k", "math"),
    ("mbpp", "code"),
    ("hellaswag", "commonsense"),
    ("winogrande", "commonsense"),
    ("arc", "knowledge"),
    ("mt-bench", "chat"),
    ("mt_bench", "chat"),
    ("rag", "retrieval"),
]


def family_of(region_name: str) -> str:
    name = region_name.lower()
    for key, fam in FAMILY_RULES:
        if key in name:
            return fam
    return name.split("-")[0].split("_")[0]  # fall back to the leading token


def explained_fraction(values: np.ndarray, groups: np.ndarray) -> float:
    """Fraction of Var(values) explained by the group means (between-group variance)."""
    total = float(np.var(values))
    if total == 0:
        return 0.0
    within = 0.0
    for g in np.unique(groups):
        sel = groups == g
        within += sel.mean() * float(np.var(values[sel]))
    return 1.0 - within / total


def routerbench_decomposition():
    from routerbench_routing import load_routerbench

    quality, cost, regions, models, region_names = load_routerbench()
    cost_norm = cost / cost.mean(axis=0).max()
    # the routing value function: per-prompt value of the utility-oracle choice
    f = (quality - REF_LAM * cost_norm).max(axis=1)
    families = np.array([family_of(region_names[r]) for r in regions])
    region_ids = regions

    by_category = explained_fraction(f, region_ids)
    by_family = explained_fraction(f, families)

    # region-value decomposition: how much of the *regional* signal the family tree keeps
    region_means = np.array([f[region_ids == r].mean() for r in np.unique(region_ids)])
    region_fams = np.array(
        [families[region_ids == r][0] for r in np.unique(region_ids)]
    )
    regionvalue_by_family = explained_fraction(region_means, region_fams)

    return {
        "family": by_family,
        "category": by_category,
        "within": 1.0 - by_category,
        "region_value_by_family": regionvalue_by_family,
    }


def main() -> None:
    set_style()

    print(f"[A] synthetic rate: {BRANCHING**DEPTH} leaves, horizons {HORIZONS}, {N_SEEDS} seeds")
    finals, alpha, intercept = regret_exponent()
    dhat = max(0.0, (2 * alpha - 1) / (1 - alpha)) if alpha < 1 else float("inf")
    print(f"    fitted exponent alpha = {alpha:.3f}  (sublinear iff < 1; d_hat ~ {dhat:.1f})")

    decomp = None
    try:
        decomp = routerbench_decomposition()
        print("\n[B] RouterBench variance decomposition (value fn at lam=%.2f):" % REF_LAM)
        print(f"    Var(f) explained by family tree:    {decomp['family']:.1%}")
        print(f"    Var(f) explained by category tree:  {decomp['category']:.1%}")
        print(f"    within-region (bandit noise):       {decomp['within']:.1%}")
        print(f"    region-value var. explained by family tree: {decomp['region_value_by_family']:.1%}")
    except Exception as e:  # noqa: BLE001 -- missing bench extra / no network
        print(f"\n[B] RouterBench decomposition skipped ({type(e).__name__}: {e})")
        print("    uv sync --extra bench  (and network for the first download)")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: average regret decays toward zero
    avg = finals / np.array(HORIZONS)
    mean, err = avg.mean(axis=0), 1.96 * avg.std(axis=0, ddof=1) / np.sqrt(N_SEEDS)
    axA.errorbar(
        HORIZONS, mean, yerr=err, fmt="o-", color=PALETTE["blue"], capsize=3,
        label="average regret $R(n)/n$",
    )
    axA.set_xscale("log")
    axA.set_xlabel("horizon $n$")
    axA.set_ylabel("average regret $R(n)/n$")
    axA.set_title(
        rf"Average regret $\to 0$  ($R(n) \propto n^{{{alpha:.2f}}}$, $\hat d \approx {dhat:.1f}$)"
    )
    axA.set_ylim(bottom=0)
    axA.legend(loc="upper right")

    # Panel B: cumulative variance explained by tree resolution
    if decomp is not None:
        levels = ["root", "family", "category", "prompt\n(leaf)"]
        explained = [0.0, decomp["family"], decomp["category"], 1.0]
        axB.step(range(4), explained, where="post", color=PALETTE["blue"], lw=2)
        axB.fill_between(
            range(4), explained, 1.0, step="post", color=PALETTE["orange"], alpha=0.15
        )
        axB.annotate(
            "within-region\n(bandit noise)",
            (1.55, (decomp["category"] + 1.0) / 2),
            fontsize=8,
            color=PALETTE["orange"],
        )
        axB.annotate(
            "explained by tree",
            (0.15, decomp["family"] * 0.45 + 0.02),
            fontsize=8,
            color=PALETTE["blue"],
        )
        axB.set_xticks(range(4), levels)
        axB.set_ylim(0, 1.02)
        axB.set_ylabel(r"cumulative fraction of $\mathrm{Var}(f)$ explained")
        axB.set_title(
            f"RouterBench: value is coarsely tree-structured\n"
            f"(family tree explains {decomp['region_value_by_family']:.0%} of region-value var.)"
        )
    else:
        axB.text(0.5, 0.5, "RouterBench unavailable", ha="center", va="center")
        axB.set_axis_off()

    fig.suptitle("Theory link: sublinear regret, and a coarsely tree-structured routing value")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = save_figure(fig, "theory_link")
    print(f"\nsaved chart to {out} (+ .png)")

    # LaTeX table (Table 2)
    rows = [
        ("Synthetic tree-Lipschitz value fn",
         "Regret exponent $\\alpha$",
         f"{alpha:.3f} (sublinear; $\\hat d \\approx {dhat:.1f}$)"),
    ]
    if decomp is not None:
        rows += [
            ("RouterBench: $\\mathrm{Var}(f)$ explained by family tree", "", f"{decomp['family']:.1%}".replace("%", "\\%")),
            ("RouterBench: $\\mathrm{Var}(f)$ explained by category tree", "", f"{decomp['category']:.1%}".replace("%", "\\%")),
            ("RouterBench: within-region (bandit noise)", "", f"{decomp['within']:.1%}".replace("%", "\\%")),
            ("RouterBench: region-value var.\\ explained by family tree", "", f"{decomp['region_value_by_family']:.1%}".replace("%", "\\%")),
        ]
    tex = (
        "% Theory link (auto-generated by examples/analysis/theory_link.py)\n"
        "\\begin{tabular}{llr}\n\\toprule\nQuantity & Statistic & Value \\\\\n\\midrule\n"
        + "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIGURE_DIR / "theory_link_table.tex").write_text(tex)
    print(f"wrote LaTeX table to {FIGURE_DIR / 'theory_link_table.tex'}")


if __name__ == "__main__":
    main()
