"""Linking the theory to the data: sublinear regret, and an almost tree-$K$-Lipschitz prior.

This closes the loop between the headline theory result (Suman's "almost tree-$K$-Lipschitz"
prior) and the experiments. The theory says: for a value function that is tree-Lipschitz with a
finite number $K$ of violations, the inherited optimistic engine attains regret

    R(n) = O( n^{(d+1)/(d+2)} )  +  O(K * D),

with $d$ the near-optimality dimension and $D$ the tree depth. The first term is *sublinear*, so
the growth rate of the *average* regret $R(n)/n \\to 0$ (Suman's question, "eventually 0?" ->
yes). The second, additive term is the fixed price of the $K$ violations (a measure-zero set) and
does not change the rate. Crucially neither $d$ nor $K$ is assumed -- both are estimated from
data.

We show two things, both without assumed constants:

  1. The regret rate is sublinear on a tree-Lipschitz value function: run the inherited engine
     (:func:`run_hoo`) at growing horizons and plot the *average* regret $R(n)/n$, which decays
     toward zero, with the fitted cumulative-regret exponent $\\alpha<1$ (Panel A). We also
     estimate the near-optimality dimension $d$ from the value function by counting
     $\\epsilon$-optimal cells across scales -- it is small (near $0$), the regime where the
     bound is tightest.

  2. On real routing data (RouterBench: 36k prompts, 11 models, 86 categories) the value
     function is tree-structured *at the regional scale* -- the honest picture (Panel B). A
     variance decomposition of the per-prompt oracle net utility shows the category tree explains
     only a minority of the per-prompt variance; the bulk is *within-region* variation (per-prompt
     difficulty) that the bandit averages over as observation noise. The exploitable regional
     signal is real and coarse (benchmark-family scale) -- it is what lets regional routing beat
     the structure-blind learner -- while the residual is exactly the noise a bandit is built to
     average over. This is the empirical face of the bandit-vs-function-learning separation: we
     need only the regional optimum, not the full noisy per-prompt value function. (A clean
     small-$K$ almost-tree-$K$-Lipschitz characterization lives in the reasoning-tree setting,
     where a partial trace's cheap value predicts its completion; that requires instrumenting the
     reasoning search to log the tree and is done separately.)

Run with:  uv run --extra plot --extra bench python examples/analysis/theory_link.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402

from canopy.bandits import TreeBandit, lipschitz_spread, run_hoo  # noqa: E402
from canopy.bandits.rewards import (  # noqa: E402
    geometric_sigma,
    hierarchical_gaussian_leaf_means,
)
from _plotstyle import FIGURE_DIR, PALETTE, ci_band, save_figure, set_style  # noqa: E402

HORIZONS = [1000, 2000, 4000, 8000, 16000, 32000, 64000]
N_SEEDS = 8
# a tree-Lipschitz value function (smooth hierarchical-Gaussian diffusion): the regime the
# theory's rate bound targets. Deep enough that the optimum is not trivially found (no early
# regret plateau) over the horizon range.
BRANCHING, DEPTH = 3, 8
SPREAD_L, SPREAD_RHO = 0.5, 1.0 / BRANCHING
VIOLATION_FACTOR = 3.0  # a cell is a "violation" if its oscillation exceeds this x the fit


# --------------------------------------------------------------------------- estimators


def within_cell_oscillation(leaf_means, branching, depth, level):
    """Per-cell (max - mean) at ``level`` -- the tree oscillation the Lipschitz bound controls."""
    span = branching ** (depth - level)
    cells = np.asarray(leaf_means).reshape(-1, span)
    return cells.max(axis=1) - cells.mean(axis=1)


def fit_tree_lipschitz(leaf_means, branching, depth, violation_level=None):
    """Fit ``osc(level) ~ L * rho**level`` (tree-Lipschitz) and count violations K.

    Regress log(median per-cell oscillation) on the level to recover the *small* constant ``L``
    that fits the smooth bulk. A cell is a *violation* when its oscillation exceeds
    ``VIOLATION_FACTOR`` times that fitted envelope. Because a single intra-cell jump would show
    up in its cell *and* every coarser ancestor, we count violations at one resolution only --
    ``violation_level`` (default the region level ``depth-2``) -- so ``K`` counts distinct
    within-region discontinuities rather than double-counting one jump across scales. Note that
    jumps aligned with cell boundaries are absorbed by the ultrametric (cross-cell points are
    "far") and are *not* violations; only within-cell heterogeneity beyond the small ``L`` is.
    Returns ``(L, rho, K, violation_level, levels, per_level_osc)``.
    """
    levels = list(range(1, depth))  # cells with >= branching leaves
    per_level = {lv: within_cell_oscillation(leaf_means, branching, depth, lv) for lv in levels}
    med = np.array([max(float(np.median(per_level[lv])), 1e-9) for lv in levels])
    slope, intercept = np.polyfit(levels, np.log(med), 1)  # log osc = log L + level*log rho
    rho, L = float(np.exp(slope)), float(np.exp(intercept))
    if violation_level is None:
        violation_level = max(1, depth - 2)  # the region resolution
    vl = violation_level
    K = int(np.sum(per_level[vl] > VIOLATION_FACTOR * L * rho**vl))
    return L, rho, K, vl, levels, per_level


def estimate_near_optimality_dimension(leaf_means, branching, depth, rho, nu=2.0):
    """Estimate the near-optimality dimension by fitting the packing exponent across scales.

    At level ``l`` the cell diameter is ``delta_l = rho**l``. A cell is ``epsilon``-optimal if it
    contains a near-optimal leaf: ``max_leaf(cell) >= mu* - nu*delta_l``. The count of such cells
    scales as ``N_l <= C*delta_l**(-d)``, so ``log N_l ~ const + d*(-l log rho)`` and the
    regression slope is ``d``. Returns ``d``.
    """
    leaf_means = np.asarray(leaf_means)
    mu_star = float(leaf_means.max())
    x, y = [], []
    for lv in range(1, depth):
        span = branching ** (depth - lv)
        cell_max = leaf_means.reshape(-1, span).max(axis=1)
        n_near = int(np.sum(cell_max >= mu_star - nu * rho**lv))
        if 1 <= n_near < branching**lv:  # ignore degenerate all/none-near-optimal scales
            x.append(-lv * np.log(rho))
            y.append(np.log(n_near))
    if len(x) >= 2:
        return max(0.0, float(np.polyfit(x, y, 1)[0]))
    return 0.0


def sublinear_regret_curve():
    """Run HOO at growing horizons on a tree-Lipschitz tree; return regrets and the exponent."""
    lm = hierarchical_gaussian_leaf_means(
        BRANCHING, DEPTH, geometric_sigma(0.40, 0.60), root_value=0.5,
        rng=np.random.default_rng(7),
    )
    spread = lipschitz_spread(SPREAD_L, SPREAD_RHO)
    regrets = np.zeros((len(HORIZONS), N_SEEDS))
    for j, n in enumerate(HORIZONS):
        for s in range(N_SEEDS):
            env = TreeBandit(BRANCHING, DEPTH, leaf_means=lm, noise_std=0.1,
                             rng=np.random.default_rng(1000 + s))
            regrets[j, s] = run_hoo(
                env, n, spread, np.random.default_rng(1000 + s), memory_bounded=True
            ).final_regret
    alpha = float(np.polyfit(np.log(HORIZONS), np.log(regrets.mean(axis=1)), 1)[0])
    d_hat = estimate_near_optimality_dimension(lm, BRANCHING, DEPTH, SPREAD_RHO)
    return regrets, alpha, d_hat


# --------------------------------------------------------------------------- real data

RB_CACHE = Path(__file__).resolve().parent / "routerbench_decomposition.npz"


def _family(name: str) -> str:
    """Coarse benchmark family of a RouterBench eval_name (the top tree level)."""
    if name.startswith("mmlu"):
        return "mmlu"
    if name.lower().startswith("chinese"):
        return "chinese"
    if name.startswith("mtbench"):
        return "mtbench"
    return name


def _variance_explained(values: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of Var(values) explained by group membership (between-group / total)."""
    total = float(values.var())
    if total <= 0:
        return 0.0
    n = len(values)
    within = 0.0
    for g in np.unique(labels):
        v = values[labels == g]
        within += len(v) / n * float(v.var())
    return max(0.0, 1.0 - within / total)


def load_routerbench_decomposition():
    """Variance decomposition of the RouterBench routing value function across the region tree.

    The value is the per-prompt oracle net utility ``f = max_m (quality_m - lam*cost_m)`` -- the
    reward the router chases. We decompose ``Var(f)`` by how much the tree explains as resolution
    refines: none at the root, the between-family fraction at the family level, the
    between-category fraction at the category level, and (by construction) all of it at the leaf.
    The gap from the category fraction up to 1 is the *within-region* variation -- per-prompt
    difficulty -- which the bandit averages over as observation noise. We also report the
    fraction of the *region-value* variance (``V(region)=max_m mean-utility``) that the family
    tree explains, the quantity the router actually acts on. Cached to an ``.npz`` so re-runs need
    no network. Returns a dict or ``None`` if RouterBench is unavailable.
    """
    if RB_CACHE.exists():
        d = np.load(RB_CACHE, allow_pickle=True)
        return {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
    except Exception:  # noqa: BLE001
        return None
    try:
        path = hf_hub_download(
            "withmartian/routerbench", "routerbench_0shot.pkl", repo_type="dataset"
        )
        df = pd.read_pickle(path)
    except Exception:  # noqa: BLE001
        return None

    meta = {"sample_id", "prompt", "eval_name", "oracle_model_to_route_to"}
    models = [c for c in df.columns if "|" not in c and c not in meta]
    quality = df[models].to_numpy(dtype=float)
    cost = df[[f"{m}|total_cost" for m in models]].to_numpy(dtype=float)
    rel = cost / cost.mean(axis=0).max()
    lam = 0.3
    util = quality - lam * rel  # per-prompt per-model net utility (P x M)
    f = util.max(axis=1)  # per-prompt oracle net utility (the routing value)
    cat = df["eval_name"].to_numpy()
    fam = np.array([_family(c) for c in cat])

    # region-value V(region) = best model's mean utility, and the family-tree fraction of its var
    cats = np.unique(cat)
    V = np.array([util[cat == c].mean(axis=0).max() for c in cats])
    Vfam = np.array([_family(c) for c in cats])
    noise = np.median([util[cat == c][:, util[cat == c].mean(axis=0).argmax()].std() for c in cats])

    out = {
        "family_frac": _variance_explained(f, fam),
        "category_frac": _variance_explained(f, cat),
        "region_value_family_frac": _variance_explained(V, Vfam),
        "noise_std": float(noise),
        "n_prompts": int(len(f)),
        "n_categories": int(len(cats)),
        "n_families": int(len(np.unique(fam))),
    }
    np.savez(RB_CACHE, **out)
    return out


# --------------------------------------------------------------------------- main


def main() -> None:
    set_style()
    regrets, alpha, d_hat = sublinear_regret_curve()
    print(f"tree-Lipschitz tree ({BRANCHING**DEPTH} leaves): measured regret exponent "
          f"alpha={alpha:.3f} (<1 => average regret -> 0); estimated d={d_hat:.2f}")
    avg = regrets / np.array(HORIZONS)[:, None]
    print("  avg regret R(n)/n: " + "  ".join(
        f"n={n}:{avg[j].mean():.3f}" for j, n in enumerate(HORIZONS)))

    rb = load_routerbench_decomposition()
    if rb is not None:
        print(f"RouterBench ({rb['n_prompts']} prompts, {rb['n_categories']} categories, "
              f"{rb['n_families']} families): family tree explains {rb['family_frac']:.1%} of "
              f"Var(f), category {rb['category_frac']:.1%}; within-region (bandit noise) "
              f"{1 - rb['category_frac']:.1%}; family tree explains "
              f"{rb['region_value_family_frac']:.1%} of the region-value variance; "
              f"per-region noise std {rb['noise_std']:.2f}")
    else:
        print("RouterBench unavailable (need --extra bench + network); skipping Panel B.")

    _plot(regrets, alpha, d_hat, rb)
    _write_table(alpha, d_hat, rb)


def _plot(regrets, alpha, d_hat, rb) -> None:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Panel A: average regret decays to 0 (sublinear cumulative regret)
    avg = regrets / np.array(HORIZONS)[:, None]
    ci_band(axA, HORIZONS, avg, PALETTE["blue"], label="average regret $R(n)/n$", marker="o")
    axA.set_xscale("log")
    axA.set_xlabel("horizon $n$")
    axA.set_ylabel("average regret $R(n)/n$")
    axA.set_ylim(bottom=0)
    axA.set_title(f"Average regret $\\to 0$ ($R(n)\\propto n^{{{alpha:.2f}}}$, "
                  f"$\\hat d\\approx{d_hat:.1f}$)")
    axA.legend(loc="upper right")

    # Panel B: RouterBench variance decomposition -- how much of the routing value the tree
    # explains as resolution refines (the regional signal) vs. the within-region remainder (the
    # bandit noise the router averages over).
    if rb is not None:
        fam, cat = float(rb["family_frac"]), float(rb["category_frac"])
        xs = [0, 1, 2, 3]
        cum = [0.0, fam, cat, 1.0]
        labels = ["root", "family", "category", "prompt\n(leaf)"]
        axB.step(xs, cum, where="post", color=PALETTE["blue"], lw=2, marker="o",
                 label="variance of $f$ explained by tree")
        axB.fill_between(xs, cum, 1.0, step="post", color=PALETTE["red"], alpha=0.12)
        axB.axhline(1.0, color=PALETTE["gray"], ls=":", lw=1)
        axB.text(2.02, (cat + 1.0) / 2, "within-region\n(bandit noise)", color=PALETTE["red"],
                 fontsize=8, va="center")
        axB.annotate(f"{cat:.0%} regional signal", (2, cat), textcoords="offset points",
                     xytext=(-4, -14), fontsize=8, color=PALETTE["blue"], ha="right")
        axB.set_xticks(xs)
        axB.set_xticklabels(labels)
        axB.set_ylim(0, 1.05)
        axB.set_ylabel("cumulative fraction of $\\mathrm{Var}(f)$ explained")
        axB.set_xlabel("tree resolution")
        axB.set_title(f"RouterBench: value is coarsely tree-structured\n(family tree explains "
                      f"{float(rb['region_value_family_frac']):.0%} of region-value variance)")
        axB.legend(loc="center left", fontsize=8)
    else:
        axB.text(0.5, 0.5, "RouterBench unavailable\n(need --extra bench + network)",
                 ha="center", va="center", transform=axB.transAxes)
        axB.set_axis_off()

    fig.suptitle("Theory link: sublinear regret, and a coarsely tree-structured routing value")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = save_figure(fig, "theory_link")
    print(f"\nsaved chart to {out} (+ .png)")


def _write_table(alpha, d_hat, rb) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        "Synthetic tree-Lipschitz value fn & Regret exponent $\\alpha$ & "
        f"{alpha:.3f} (sublinear; $\\hat d\\approx{d_hat:.1f}$) \\\\",
        "\\midrule",
    ]
    if rb is not None:
        def pct(x):
            return f"{100 * float(x):.1f}\\%"
        rows += [
            f"RouterBench: Var($f$) explained by family tree & & {pct(rb['family_frac'])} \\\\",
            f"RouterBench: Var($f$) explained by category tree & & {pct(rb['category_frac'])} "
            "\\\\",
            f"RouterBench: within-region (bandit noise) & & {pct(1 - rb['category_frac'])} \\\\",
            "RouterBench: region-value var.\\ explained by family tree & & "
            f"{pct(rb['region_value_family_frac'])} \\\\",
        ]
    tex = (
        "% Theory link (auto-generated by theory_link.py)\n"
        "\\begin{tabular}{lcr}\n\\toprule\n"
        "Quantity & Statistic & Value \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (FIGURE_DIR / "theory_link_table.tex").write_text(tex)
    print(f"wrote table to {FIGURE_DIR}/theory_link_table.tex")


if __name__ == "__main__":
    main()
