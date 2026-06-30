"""Real-data LLM routing on RouterBench (offline; no API calls).

RouterBench (``withmartian/routerbench``) provides measured quality and dollar cost for 11
LLMs on ~36k prompts drawn from MMLU, GSM8K, MBPP, HellaSwag, Winogrande, MT-Bench, and
more. We use it as a pure-bandit routing testbed: each task category (``eval_name``) is a
region, and a hierarchical router learns, per region, which model maximizes the net utility

    u_m(prompt) = quality_m(prompt) - lam * cost_m(prompt),

competing against the per-prompt oracle and fixed-model baselines. Sweeping ``lam`` traces a
cost/quality frontier. This is the real-model instantiation of the routing application; it
needs the ``bench`` extra and downloads the dataset once (cached by Hugging Face).

Run with:  uv run --extra bench --extra plot python examples/llm_routing/routerbench_routing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402

from _plotstyle import PALETTE, save_figure, set_style  # noqa: E402

LAMBDAS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5]
N_SEEDS = 8
UCB_C = 0.3


def load_routerbench():
    """Return (quality (P,M), cost (P,M), region_ids (P,), model_names, region_names)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("withmartian/routerbench", "routerbench_0shot.pkl", repo_type="dataset")
    df = pd.read_pickle(path)
    meta = {"sample_id", "prompt", "eval_name", "oracle_model_to_route_to"}
    models = [c for c in df.columns if "|" not in c and c not in meta]
    quality = df[models].to_numpy(dtype=float)
    cost = df[[f"{m}|total_cost" for m in models]].to_numpy(dtype=float)
    regions, region_names = pd.factorize(df["eval_name"])
    return quality, cost, np.asarray(regions), models, list(region_names)


def learn_and_eval(quality, cost, cost_norm, regions, lam, rng):
    """Train our region-contextual UCB router on a train split, deploy greedily on test.

    Training is the actual bandit loop: it streams the train prompts, picks a model per
    region by a UCB index on the net utility ``quality - lam * cost``, and updates its
    estimates from the realized utility. The learned greedy policy (best estimated model per
    region) is then frozen and evaluated on the held-out test split.

    Returns ``(test_quality, test_cost, oracle_quality)``; the oracle is the per-prompt
    best-utility choice on test (an unreachable upper bound).
    """
    n, n_models = quality.shape
    n_regions = regions.max() + 1
    idx = rng.permutation(n)
    tr, te = idx[: n // 2], idx[n // 2 :]

    counts = np.zeros((n_regions, n_models))
    sums = np.zeros((n_regions, n_models))  # accumulated net utility
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
    return (
        quality[te, chosen].mean(),
        cost[te, chosen].mean(),
        quality[te[rows], oracle].mean(),
    )


def main() -> None:
    set_style()
    try:
        quality, cost, regions, models, region_names = load_routerbench()
    except Exception as e:  # noqa: BLE001
        print(f"Could not load RouterBench ({type(e).__name__}: {e}).")
        print("Install the bench extra and ensure network access: uv sync --extra bench")
        return
    print(
        f"RouterBench: {quality.shape[0]} prompts, {len(models)} models, "
        f"{len(region_names)} regions"
    )
    scale = cost.mean(axis=0).max()  # normalize so the priciest model averages ~1
    cost_norm = cost / scale
    model_q = quality.mean(axis=0)
    model_c = cost.mean(axis=0)

    rq, rq_err, rc, oracle_q = [], [], [], []
    print("\nlam   router quality         router cost    (oracle q)")
    for lam in LAMBDAS:
        res = np.array(
            [
                learn_and_eval(quality, cost, cost_norm, regions, lam, np.random.default_rng(s))
                for s in range(N_SEEDS)
            ]
        )
        q_mean = res[:, 0].mean()
        q_half = 1.96 * res[:, 0].std(ddof=1) / np.sqrt(N_SEEDS)
        rq.append(q_mean)
        rq_err.append(q_half)
        rc.append(res[:, 1].mean())
        oracle_q.append(res[:, 2].mean())
        print(f"lam={lam:.2f}  q={q_mean:.3f}  cost=${rc[-1]:.5f}  oracle={oracle_q[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.errorbar(
        rc,
        rq,
        yerr=rq_err,
        fmt="-o",
        color=PALETTE["blue"],
        capsize=3,
        label="hierarchical router (sweep $\\lambda$)",
    )
    ax.scatter(
        model_c, model_q, color=PALETTE["orange"], s=30, zorder=5, label="fixed single model"
    )
    for j in (int(model_q.argmax()), int(model_c.argmin())):
        ax.annotate(
            models[j].split("/")[-1],
            (model_c[j], model_q[j]),
            fontsize=7,
            xytext=(5, -3),
            textcoords="offset points",
        )
    ax.axhline(
        float(np.mean(oracle_q)),
        ls="--",
        lw=1,
        color=PALETTE["gray"],
        label="per-prompt oracle (upper bound)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("mean cost per query (USD)")
    ax.set_ylabel("mean quality")
    ax.set_title(f"RouterBench: routing across {len(models)} real LLMs")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = save_figure(fig, "routerbench_routing")
    print(f"\nsaved chart to {out} (+ .png)")


if __name__ == "__main__":
    main()
