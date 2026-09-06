"""Real-data LLM routing on RouterEval (offline; no API calls) --- quality-routing variant.

RouterEval (``linggm/RouterEval``; Huang et al., EMNLP 2025) provides 0/1 correctness of many
candidate LLMs on 12 evaluations, with candidate pools of several sizes (easy: 3,5; hard:
10,100,1000) and per-prompt embeddings. Unlike RouterBench it carries no dollar cost, so we use
it as a *quality-routing* testbed: does exploiting contextual structure let a learned router pick
the right model more often than a structure-blind learner?

Mapping to the regional prior: within each evaluation we cluster the prompt embeddings into
regions (the contextual signal), and a region-contextual UCB router learns, per region, which
pool model is most often correct. We compare, per evaluation and averaged across the 12:

  * per-prompt oracle (unreachable upper bound),
  * region-contextual router (ours),
  * structure-blind flat learner (single region --- the honest learning baseline),
  * best fixed single model (oracle-chosen on test),
  * random model.

The claim under test is the same regional one as RouterBench, on a quality axis: the contextual
router beats the flat learner and the best fixed model, approaching the oracle. Offline; needs the
``bench`` extra (datasets/hf) and scikit-learn for clustering.

Run:  ~/canopy/.venv/bin/python examples/llm_routing/routereval_routing.py \
    --pool-tier hard --pool-size 10
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

FIGDIR = Path(__file__).resolve().parents[2] / "paper" / "figures"

# the 12 RouterEval evaluations (each a separate pickle in router_dataset/)
EVALS = [
    "arc",
    "bbh",
    "gpqa",
    "gsm8k",
    "harness_truthfulqa_mc_0",
    "hellaswag",
    "ifeval",
    "mmlu",
    "humaneval",
    "mbpp",
    "winogrande",
    "mt_bench",
]
UCB_C = 0.3


def _load_zip():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("linggm/RouterEval", "router_dataset.zip", repo_type="dataset")
    return zipfile.ZipFile(path)


def _available_evals(z) -> list[str]:
    names = {n for n in z.namelist() if n.endswith("_router_dataset.pkl")}
    return [e for e in EVALS if f"router_dataset/{e}_router_dataset.pkl" in names]


def load_eval(z, name: str, tier: str, size: int, group: str):
    """Return (train_score, test_score, train_embed, test_embed, model_names) for one evaluation.

    ``score`` arrays are (n_prompts, pool_size) 0/1 correctness; ``embed`` are (n_prompts, D).
    """
    obj = pickle.load(io.BytesIO(z.read(f"router_dataset/{name}_router_dataset.pkl")))
    node = obj[tier][size][group]
    d = node["data"]
    emb = obj["embedding"]
    tr_e = np.asarray(emb["train_embed"], dtype=float)
    te_e = np.asarray(emb["test_embed"], dtype=float)
    tr_e = tr_e.reshape(tr_e.shape[0], -1)
    te_e = te_e.reshape(te_e.shape[0], -1)
    return (
        np.asarray(d["train_score"], float),
        np.asarray(d["test_score"], float),
        tr_e,
        te_e,
        list(node["model"]),
    )


def _regions(train_embed, test_embed, k, seed):
    """Cluster prompt embeddings into ``k`` regions (fit on train, assign test)."""
    from sklearn.cluster import KMeans

    k = max(1, min(k, len(train_embed)))
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(train_embed)
    return km.labels_, km.predict(test_embed), k


def eval_one(train_score, test_score, train_embed, test_embed, k, seed):
    """Region-contextual router vs baselines on one eval; returns dict of mean test accuracy."""
    rng = np.random.default_rng(seed)
    n_tr, n_models = train_score.shape
    tr_reg, te_reg, k = _regions(train_embed, test_embed, k, seed)

    # region-contextual UCB router: learn per-region best model on the train stream
    counts = np.zeros((k, n_models))
    sums = np.zeros((k, n_models))
    for t, i in enumerate(rng.permutation(n_tr), 1):
        r = tr_reg[i]
        nr = counts[r]
        mean = np.divide(sums[r], nr, out=np.zeros(n_models), where=nr > 0)
        bonus = np.where(nr > 0, UCB_C * np.sqrt(2 * np.log(t + 1) / np.maximum(nr, 1)), np.inf)
        m = int(np.argmax(mean + bonus))
        counts[r, m] += 1
        sums[r, m] += train_score[i, m]
    means = np.divide(sums, counts, out=np.full((k, n_models), -np.inf), where=counts > 0)
    global_best = int(np.nan_to_num(means, neginf=-1e9).mean(axis=0).argmax())
    region_policy = np.where(counts.sum(axis=1) > 0, means.argmax(axis=1), global_best)

    # flat learner: single region (structure-blind), best mean model on train
    flat_model = int(train_score.mean(axis=0).argmax())

    reg_choice = region_policy[te_reg]
    rows = np.arange(len(test_score))
    return {
        "oracle": test_score.max(axis=1).mean(),
        "regional": test_score[rows, reg_choice].mean(),
        "flat": test_score[:, flat_model].mean(),
        "best_fixed": test_score.mean(axis=0).max(),  # best fixed model chosen on test (generous)
        "random": test_score.mean(),  # expected accuracy of a random model
    }


def _ci(vals, seed=0, iters=2000):
    a = np.asarray(vals, float)
    if a.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    m = a[rng.integers(0, a.size, size=(iters, a.size))].mean(axis=1)
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-tier", default="hard", choices=["easy", "hard"])
    ap.add_argument("--pool-size", type=int, default=10)
    ap.add_argument(
        "--group", default="strong_to_weak", choices=["all_strong", "all_weak", "strong_to_weak"]
    )
    ap.add_argument("--regions", type=int, default=8, help="embedding clusters per evaluation")
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()

    try:
        z = _load_zip()
        evals = _available_evals(z)
    except Exception as e:  # noqa: BLE001
        print(f"Could not load RouterEval ({type(e).__name__}: {e}). Needs hf + network.")
        return
    if not evals:
        print("No RouterEval evaluations found in router_dataset.zip")
        return

    print(
        f"RouterEval quality-routing: {len(evals)} evals, pool={args.pool_tier}[{args.pool_size}]"
        f" ({args.group}), {args.regions} regions, {args.seeds} seeds"
    )
    per_policy = {k: [] for k in ("oracle", "regional", "flat", "best_fixed", "random")}
    for name in progress(evals, "routereval evals", total=len(evals)):
        try:
            tr_s, te_s, tr_e, te_e, models = load_eval(
                z, name, args.pool_tier, args.pool_size, args.group
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [skip {name}] {type(e).__name__}: {str(e)[:80]}")
            continue
        for s in range(args.seeds):
            res = eval_one(tr_s, te_s, tr_e, te_e, args.regions, s)
            for k, v in res.items():
                per_policy[k].append(v)

    order = [
        ("oracle", "per-prompt oracle (upper bound)"),
        ("regional", "region-contextual router (ours)"),
        ("best_fixed", "best fixed model (test-chosen)"),
        ("flat", "structure-blind flat learner"),
        ("random", "random model"),
    ]
    FIGDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    print("\npolicy                              accuracy [95% CI]")
    for key, label in order:
        m, lo, hi = _ci(per_policy[key], seed=hash(key) & 0xFFFF)
        bold = key == "regional"
        cell = f"\\textbf{{{m:.3f}}}" if bold else f"{m:.3f}"
        lbl = f"\\textbf{{{label}}}" if bold else label
        rows.append(f"{lbl} & {cell} [{lo:.3f}, {hi:.3f}] \\\\")
        print(f"  {label:34s} {m:.3f} [{lo:.3f},{hi:.3f}]")
    tex = (
        "% RouterEval quality-routing (auto-generated by routereval_routing.py); mean 0/1 accuracy "
        f"over {len(evals)} evals x {args.seeds} seeds, pool {args.pool_tier}[{args.pool_size}] "
        f"{args.group}, {args.regions} embedding regions.\n"
        "\\begin{tabular}{lc}\n\\toprule\nPolicy & Accuracy [95\\% CI] \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (FIGDIR / "routereval_routing_table.tex").write_text(tex)
    print(f"\nwrote table to {FIGDIR}/routereval_routing_table.tex")


if __name__ == "__main__":
    main()
