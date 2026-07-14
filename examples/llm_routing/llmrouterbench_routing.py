"""Real-data LLM routing on LLMRouterBench (offline; no API calls) --- cost/quality frontier.

LLMRouterBench (``NPULH/LLMRouterBench``; Findings of ACL 2026) records per-instance score, cost,
and tokens for many models across 27 datasets. Model coverage is sparse for most datasets, but the
four ArenaHard variants --- ``arenahard`` (general), ``arenahard_coding``,
``arenahard_creative_writing``, ``arenahard_math`` --- share a common pool of 33 models with
per-instance quality and dollar cost. That gives exactly the regional structure the tree/region
prior targets: region = task variant, arm = model, and different models win on coding vs. math vs.
creative writing.

We use it identically to RouterBench: a region-contextual UCB router learns, per variant, which
model maximizes ``quality - lam * cost``, swept over ``lam`` into a cost/quality frontier against
the fixed single models and the per-prompt oracle. Offline; needs the ``bench`` extra + network.

Run:  ~/canopy/.venv/bin/python examples/llm_routing/llmrouterbench_routing.py
"""

from __future__ import annotations

import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402

from _plotstyle import PALETTE, progress, save_figure, set_style  # noqa: E402

FIGDIR = Path(__file__).resolve().parents[2] / "paper" / "figures"
LAMBDAS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5]
N_SEEDS = 8
UCB_C = 0.3
ARENA_VARIANTS = ["arenahard", "arenahard_coding", "arenahard_creative_writing", "arenahard_math"]


def load_llmrouterbench(variants=ARENA_VARIANTS):
    """Return (quality (P,M), cost (P,M), regions (P,), models, region_names) for the ArenaHard set.

    For each variant we align every model's per-record score/cost by record index, keep the models
    common to all variants (the shared 33-model pool) and the records scored by all of them, then
    stack the variants into one matrix with region = variant.
    """
    from huggingface_hub import hf_hub_download

    tp = hf_hub_download("NPULH/LLMRouterBench", "bench-release.tar.gz", repo_type="dataset")
    variant_set = set(variants)

    # Single sequential pass over the (gzip) tar: gzip is not seekable, so repeated random-access
    # extractfile() re-decompresses the whole archive each call. Read every relevant json once.
    # data[variant][model] = {index: (score, cost)}
    data: dict = {v: {} for v in variants}
    with tarfile.open(tp, "r:gz") as t:
        for member in t:
            if not (member.isfile() and member.name.endswith(".json")):
                continue
            parts = member.name.split("/")  # bench-release/<variant>/<model>/<file>.json
            if len(parts) < 4 or parts[1] not in variant_set:
                continue
            v, m = parts[1], parts[2]
            try:
                obj = json.load(t.extractfile(member))
            except Exception:  # noqa: BLE001
                continue
            rec = {r["index"]: (float(r.get("score", 0.0)), float(r.get("cost", 0.0)))
                   for r in obj.get("records", [])}
            if rec:
                data[v][m] = rec

    common = None
    for v in variants:
        m = set(data[v])
        common = m if common is None else (common & m)
    models = sorted(common or [])

    def load_variant(d):
        per_model = {m: data[d].get(m, {}) for m in models}
        idx = sorted(set.intersection(*[set(per_model[m]) for m in models])) if models else []
        q = np.array([[per_model[m][i][0] for m in models] for i in idx], dtype=float)
        c = np.array([[per_model[m][i][1] for m in models] for i in idx], dtype=float)
        return q, c

    qs, cs, regs = [], [], []
    for ri, v in enumerate(variants):
        q, c = load_variant(v)
        if q.size:
            qs.append(q); cs.append(c); regs.append(np.full(len(q), ri))
    quality = np.vstack(qs)
    cost = np.vstack(cs)
    regions = np.concatenate(regs)
    return quality, cost, regions, models, variants


def learn_and_eval(quality, cost, cost_norm, regions, lam, rng):
    """Region-contextual UCB router: train on half, deploy greedily on the held-out half."""
    n, n_models = quality.shape
    n_regions = int(regions.max()) + 1
    idx = rng.permutation(n)
    tr, te = idx[: n // 2], idx[n // 2:]
    counts = np.zeros((n_regions, n_models))
    sums = np.zeros((n_regions, n_models))
    for t, i in enumerate(rng.permutation(tr), 1):
        r = regions[i]
        nr = counts[r]
        mean = np.divide(sums[r], nr, out=np.zeros(n_models), where=nr > 0)
        bonus = np.where(nr > 0, UCB_C * np.sqrt(2 * np.log(t + 1) / np.maximum(nr, 1)), np.inf)
        m = int(np.argmax(mean + bonus))
        counts[r, m] += 1
        sums[r, m] += quality[i, m] - lam * cost_norm[i, m]
    means = np.divide(sums, counts, out=np.full_like(sums, -np.inf), where=counts > 0)
    global_best = int(np.nan_to_num(means, neginf=-1e9).mean(axis=0).argmax())
    policy = np.where(counts.sum(axis=1) > 0, means.argmax(axis=1), global_best)
    chosen = policy[regions[te]]
    oracle = (quality[te] - lam * cost_norm[te]).argmax(axis=1)
    rows = np.arange(len(te))
    return quality[te, chosen].mean(), cost[te, chosen].mean(), quality[te[rows], oracle].mean()


def main() -> None:
    set_style()
    try:
        quality, cost, regions, models, region_names = load_llmrouterbench()
    except Exception as e:  # noqa: BLE001
        print(f"Could not load LLMRouterBench ({type(e).__name__}: {e}). Needs hf + network.")
        return
    print(f"LLMRouterBench (ArenaHard): {quality.shape[0]} prompts, {len(models)} models, "
          f"{len(region_names)} regions {region_names}")
    scale = cost.mean(axis=0).max() or 1.0
    cost_norm = cost / scale
    model_q = quality.mean(axis=0)
    model_c = cost.mean(axis=0)

    rq, rq_err, rc, oracle_q = [], [], [], []
    print("\nlam   router quality        router cost   (oracle q)")
    for lam in progress(LAMBDAS, "llmrouterbench lambda", total=len(LAMBDAS)):
        res = np.array([learn_and_eval(quality, cost, cost_norm, regions, lam,
                                       np.random.default_rng(s)) for s in range(N_SEEDS)])
        rq.append(res[:, 0].mean())
        rq_err.append(1.96 * res[:, 0].std(ddof=1) / np.sqrt(N_SEEDS))
        rc.append(res[:, 1].mean())
        oracle_q.append(res[:, 2].mean())
        print(f"lam={lam:.2f}  q={rq[-1]:.3f}  cost=${rc[-1]:.5f}  oracle={oracle_q[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.errorbar(rc, rq, yerr=rq_err, fmt="-o", color=PALETTE["blue"], capsize=3,
                label="hierarchical router (sweep $\\lambda$)")
    ax.scatter(model_c, model_q, color=PALETTE["orange"], s=30, zorder=5, label="fixed single model")
    ax.axhline(float(np.mean(oracle_q)), ls="--", lw=1, color=PALETTE["gray"],
               label="per-prompt oracle (upper bound)")
    ax.set_xlabel("mean cost per query (USD)")
    ax.set_ylabel("mean quality")
    ax.set_title(f"LLMRouterBench (ArenaHard): routing across {len(models)} real LLMs")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = save_figure(fig, "llmrouterbench_routing")

    # summary table: router frontier endpoints vs best fixed model + oracle
    FIGDIR.mkdir(parents=True, exist_ok=True)
    bi = int(model_q.argmax())
    tex = (
        "% LLMRouterBench ArenaHard routing (auto-generated by llmrouterbench_routing.py). "
        f"{quality.shape[0]} prompts, {len(models)} models, {len(region_names)} task-variant regions.\n"
        "\\begin{tabular}{lcc}\n\\toprule\nPolicy & Quality & Cost/query (\\$) \\\\\n\\midrule\n"
        f"per-prompt oracle & {float(np.mean(oracle_q)):.3f} & --- \\\\\n"
        f"\\textbf{{region router ($\\lambda{{=}}0$)}} & \\textbf{{{rq[0]:.3f}}} & {rc[0]:.5f} \\\\\n"
        f"region router (mid $\\lambda$) & {rq[len(rq)//2]:.3f} & {rc[len(rc)//2]:.5f} \\\\\n"
        f"best fixed model & {model_q[bi]:.3f} & {model_c[bi]:.5f} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
    )
    (FIGDIR / "llmrouterbench_routing_table.tex").write_text(tex)
    print(f"\nsaved chart to {out} (+ .png) and table to {FIGDIR}/llmrouterbench_routing_table.tex")


if __name__ == "__main__":
    main()
