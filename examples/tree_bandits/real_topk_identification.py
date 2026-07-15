"""Real multi-fidelity top-k model identification on a large LLM pool (RouterEval).

This is the real-data analog of the synthetic top-k figure (examples/tree_bandits/benchmark.py):
identify the best models in a large pool at the lowest evaluation COST, exploiting a
tree over the pool exactly as the synthetic experiments do.

Setting (concrete multi-fidelity):
  * Arms are the candidate LLMs in RouterEval's "hard" pool (``linggm/RouterEval``; Huang et
    al., EMNLP 2025). The largest pool has 1000 models, which maps exactly onto a complete
    ``branching=10, depth=3`` tree (10^3 = 1000 leaves).
  * A LEAF is one model; its true value is that model's accuracy on the full evaluation
    (the expensive, unbiased fidelity -- a full benchmark run).
  * An INTERNAL node is a *group* of behaviourally-similar models; its value is the group's
    mean accuracy. Probing it is CHEAP but biased (a group average under-estimates the
    group's best member). This is the same subtree-average / value-model bias the synthetic
    tree models.
  * We organise the pool into the tree WITHOUT looking at the quantity we are trying to
    identify: models are ordered by behavioural similarity (hierarchical clustering on their
    per-prompt correctness vectors on the TRAIN split), so similar models sit in the same
    subtree. Whether subtree averages are then predictive of subtree maxima is an emergent,
    testable property of the real pool -- not baked in.

Method (the paper's two contributions, data-driven -- no assumed smoothness schedule):
  1. A data-driven local-smoothness certificate: within-subtree spread is estimated from
     cheap random-path probes (noise-deconvolved), giving the per-level ``spread`` bound that
     HierarchicalTopK needs -- estimated, not assumed.
  2. Discontinuity-guided sampling: the same probes flag the high-dispersion (violation)
     cells (multiscale edge map); HierarchicalTopK relaxes its smooth bound there and spends
     expensive leaf evaluations around those edges.

Baselines at matched COST:
  * SuccessiveEliminationTopK -- STRONG structure-blind baseline (leaf evals only).
  * UniformTopK               -- weak structure-blind baseline.

Outputs a recall@k-vs-cost table (paper/figures/real_topk_identification_table.tex) and,
with the plot extra, a figure (paper/figures/real_topk_identification.{pdf,png}).

Run:  ~/canopy/.venv/bin/python examples/tree_bandits/real_topk_identification.py \
          --pool-size 1000 --group strong_to_weak --k 10 --seeds 20
"""

from __future__ import annotations

import argparse
import io
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _plotstyle import progress  # noqa: E402

from canopy.bandits import (  # noqa: E402
    HierarchicalTopK,
    SuccessiveEliminationTopK,
    TreeBandit,
    UniformTopK,
    multiscale_edge_map,
)

FIGDIR = Path(__file__).resolve().parents[2] / "paper" / "figures"
EVALS = ["arc", "bbh", "gpqa", "gsm8k", "harness_truthfulqa_mc_0", "hellaswag", "ifeval",
         "mmlu", "humaneval", "mbpp", "winogrande", "mt_bench"]

# tree / cost model (probes cheap, leaf evals expensive) -- shared with the synthetic study
BRANCHING, DEPTH = 10, 3          # 1000 leaves == RouterEval hard pool of 1000 models
NOISE, PROBE_NOISE = 0.06, 0.06   # finite-sample evaluation noise (accuracies in [0,1])
PROBE_COST, LEAF_COST = 0.05, 1.0
BEAM = 30
CERT_SAMPLES = 12                 # random-path probes per cell for the certificate pre-pass
CERT_LEVELS = [1, 2]              # internal levels probed to estimate the certificate
Z_SPREAD = 1.0                    # multiplier on the estimated expected max-minus-mean bound


def _load_zip():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("linggm/RouterEval", "router_dataset.zip", repo_type="dataset")
    return zipfile.ZipFile(path)


def _available_evals(z) -> list[str]:
    names = {n for n in z.namelist() if n.endswith("_router_dataset.pkl")}
    return [e for e in EVALS if f"router_dataset/{e}_router_dataset.pkl" in names]


def load_eval(z, name: str, size: int, group: str):
    """Return (train_score (n_tr,M), test_score (n_te,M), model_names) for one evaluation."""
    obj = pickle.load(io.BytesIO(z.read(f"router_dataset/{name}_router_dataset.pkl")))
    node = obj["hard"][size][group]
    d = node["data"]
    return (np.asarray(d["train_score"], float),
            np.asarray(d["test_score"], float),
            list(node["model"]))


def similarity_order(train_score: np.ndarray) -> np.ndarray:
    """Order models so behaviourally-similar ones are adjacent (hierarchical clustering).

    Uses Ward linkage on the per-prompt correctness vectors (TRAIN split only) and returns
    the dendrogram leaf order. Contiguous chunks of this 1-D order therefore group similar
    models, which is what the complete tree treats as subtrees. The ordering never uses the
    per-model mean accuracy (the quantity being identified).
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import pdist

    x = train_score.T  # (M models, n_prompts)
    # de-mean per model so clustering keys on *pattern* similarity, not just overall level,
    # then Ward on euclidean distance; falls back gracefully for degenerate inputs.
    xc = x - x.mean(axis=1, keepdims=True)
    d = pdist(xc, metric="euclidean")
    if not np.all(np.isfinite(d)) or d.sum() == 0:
        return np.argsort(-x.mean(axis=1))  # degenerate: any stable order
    z = linkage(d, method="ward")
    return np.asarray(leaves_list(z))


def make_env(leaf_means: np.ndarray, seed: int) -> TreeBandit:
    return TreeBandit(
        BRANCHING, DEPTH, leaf_means=leaf_means,
        noise_std=NOISE, probe_noise_std=PROBE_NOISE,
        leaf_cost=LEAF_COST, probe_cost=PROBE_COST,
        rng=np.random.default_rng(seed),
    )


def estimate_certificate(leaf_means: np.ndarray, seed: int):
    """Data-driven certificate: (spread callable, relaxed_ranges, probe_cost_spent).

    Probes internal cells with cheap random-path plays, deconvolves the observation noise to
    estimate each cell's within-subtree spread, and turns it into (a) a per-level smoothness
    bound spread(level) = z * within_std(level) * sqrt(2 ln(#leaves under the level)) [the
    expected max-minus-mean of that many leaves], and (b) the flagged high-dispersion cells
    (multiscale edge map) as relaxed_ranges for discontinuity-guided sampling.
    """
    env = make_env(leaf_means, seed)
    em = multiscale_edge_map(
        env, np.random.default_rng(seed), levels=CERT_LEVELS, n_samples_per_cell=CERT_SAMPLES
    )
    per_level = {}
    for lvl in CERT_LEVELS:
        m = BRANCHING ** (DEPTH - lvl)  # leaves under a level-`lvl` node
        # a robust (high-quantile) within-subtree spread across cells at this level
        ws = float(np.quantile(em.within_std[lvl], 0.90))
        per_level[lvl] = Z_SPREAD * ws * np.sqrt(2.0 * np.log(max(2, m)))
    # interpolate/extend to every level (0..depth-1); level 0 uses the coarsest estimate
    coarsest = per_level[min(per_level)]

    def spread(level: int) -> float:
        if level in per_level:
            return per_level[level]
        if level < min(per_level):
            return coarsest
        return per_level[max(per_level)]  # finest estimated level for anything deeper

    n_plays = sum(BRANCHING**lvl * CERT_SAMPLES for lvl in CERT_LEVELS)
    return spread, em.finest_ranges(), n_plays * PROBE_COST


def run_methods(leaf_means: np.ndarray, k: int, budget: float, seed: int):
    """Per-seed recall@k for the three methods at a matched cost budget."""
    # ours: pay for the certificate pre-pass, then run with remaining budget; report combined
    spread, relaxed, cert_cost = estimate_certificate(leaf_means, 5000 + seed)
    e = make_env(leaf_means, seed)
    remaining = max(BRANCHING * LEAF_COST, budget - cert_cost)
    hier = (HierarchicalTopK(remaining, spread=spread, beam_width=BEAM, relaxed_ranges=relaxed)
            .run(e, k).evaluate(e, k))

    e = make_env(leaf_means, seed)
    se = SuccessiveEliminationTopK(budget).run(e, k).evaluate(e, k)

    e = make_env(leaf_means, seed)
    uni = UniformTopK(budget).run(e, k).evaluate(e, k)
    return hier, se, uni


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-size", type=int, default=1000, choices=[10, 100, 1000])
    ap.add_argument("--group", default="strong_to_weak",
                    choices=["all_strong", "all_weak", "strong_to_weak"])
    ap.add_argument("--k", type=int, default=10, help="top-k models to identify")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budgets", type=float, nargs="+",
                    default=[150, 300, 600, 1200, 2400])
    ap.add_argument("--evals", nargs="+", default=None, help="subset of evals (default: all)")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    n_leaves = BRANCHING**DEPTH
    if args.pool_size != n_leaves:
        print(f"note: tree has {n_leaves} leaves; --pool-size {args.pool_size} will be "
              f"padded/truncated to {n_leaves}.")

    try:
        z = _load_zip()
        evals = args.evals or _available_evals(z)
    except Exception as e:  # noqa: BLE001
        print(f"Could not load RouterEval ({type(e).__name__}: {e}). Needs hf + network.")
        return

    # Build one real tree per eval (ordering + leaf means), reused across seeds/budgets.
    trees: list[tuple[str, np.ndarray]] = []
    for name in progress(evals, "building trees", total=len(evals)):
        try:
            tr, te, models = load_eval(z, name, args.pool_size, args.group)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip {name}] {type(e).__name__}: {str(e)[:80]}")
            continue
        quality = te.mean(axis=0)  # true full-eval accuracy per model
        if quality.size >= n_leaves:
            # order the real pool by behavioural similarity, then keep the first n_leaves
            order = similarity_order(tr[:, :n_leaves])
            leaf_means = quality[:n_leaves][order]
        else:
            # pool smaller than the tree: order the real models, pad with weak fillers
            order = similarity_order(tr)
            pad = np.full(n_leaves - quality.size, float(quality.min()))
            leaf_means = np.concatenate([quality[order], pad])
        trees.append((name, np.ascontiguousarray(leaf_means, dtype=float)))

    if not trees:
        print("No usable evaluations.")
        return

    print(f"\nRouterEval top-k identification: pool={args.pool_size} ({args.group}), "
          f"tree {BRANCHING}^{DEPTH}={n_leaves} leaves, top-{args.k}, "
          f"{len(trees)} evals x {args.seeds} seeds, probe/leaf={PROBE_COST}")
    print(f"\n{'budget':>7s} {'Hierarchical(ours)':>19s} {'SuccessiveElim':>15s} {'Uniform':>9s}")

    labels = ["Hierarchical (ours)", "Successive elimination", "Uniform"]
    means = {lbl: [] for lbl in labels}
    sems = {lbl: [] for lbl in labels}
    rows_by_budget = []
    for budget in args.budgets:
        samples = {lbl: [] for lbl in labels}
        for _, leaf_means in trees:
            for s in range(args.seeds):
                hier, se, uni = run_methods(leaf_means, args.k, budget, s)
                samples[labels[0]].append(hier)
                samples[labels[1]].append(se)
                samples[labels[2]].append(uni)
        row = []
        for lbl in labels:
            a = np.asarray(samples[lbl])
            m = float(a.mean())
            sem = float(a.std(ddof=1) / np.sqrt(a.size))
            means[lbl].append(m)
            sems[lbl].append(sem)
            row.append((m, sem))
        rows_by_budget.append((budget, row))
        print(f"{budget:7.0f} {row[0][0]:19.3f} {row[1][0]:15.3f} {row[2][0]:9.3f}")

    _write_table(args, evals=len(trees), rows=rows_by_budget, labels=labels)
    if args.plot:
        _plot(args, means, sems, labels)


def _write_table(args, evals, rows, labels) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    header = " & ".join(["Cost budget"] + [f"{int(b)}" for b, _ in rows]) + " \\\\"
    lines = []
    for i, lbl in enumerate(labels):
        bold = i == 0
        name = f"\\textbf{{{lbl}}}" if bold else lbl
        cells = []
        for _, row in rows:
            m, sem = row[i]
            cell = f"{m:.3f}\\,$\\pm$\\,{sem:.3f}"
            cells.append(f"\\textbf{{{cell}}}" if bold else cell)
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    col = "l" + "c" * len(rows)
    tex = (
        f"% RouterEval top-k identification (auto-generated by real_topk_identification.py). "
        f"Mean top-{args.k} recall +/- SEM over {evals} evals x {args.seeds} seeds; pool "
        f"{args.pool_size} ({args.group}); tree {BRANCHING}^{DEPTH}; probe/leaf cost "
        f"{PROBE_COST}. 'ours' cost includes the certificate probing pre-pass.\n"
        f"\\begin{{tabular}}{{{col}}}\n\\toprule\n{header}\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (FIGDIR / "real_topk_identification_table.tex").write_text(tex)
    print(f"\nwrote table to {FIGDIR}/real_topk_identification_table.tex")


def _plot(args, means, sems, labels) -> None:
    try:
        import matplotlib.pyplot as plt

        from _plotstyle import PALETTE, save_figure, set_style
    except Exception:  # noqa: BLE001
        print("(install the plot extra for the chart)")
        return
    set_style()
    colors = {labels[0]: PALETTE["blue"], labels[1]: PALETTE["green"], labels[2]: PALETTE["red"]}
    markers = {labels[0]: "o", labels[1]: "^", labels[2]: "s"}
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    b = np.asarray(args.budgets, float)
    for lbl in labels:
        m = np.asarray(means[lbl])
        e = 1.96 * np.asarray(sems[lbl])
        ax.plot(b, m, marker=markers[lbl], color=colors[lbl], label=lbl)
        ax.fill_between(b, m - e, m + e, color=colors[lbl], alpha=0.18, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("evaluation cost budget (units of full model evals)")
    ax.set_ylabel(f"mean top-{args.k} recall")
    ax.set_title(f"Real LLM-pool top-{args.k} identification (RouterEval, {args.pool_size} models)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = save_figure(fig, "real_topk_identification")
    print(f"saved chart to {out} (+ .png)")


if __name__ == "__main__":
    main()
