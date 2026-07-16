"""Prefix-cache management on a REAL prompt stream (alpaca + tiktoken).

The synthetic demo (``prefix_cache_demo.py``) uses toy vocab-8 template prompts. This
script drives the same cache policies with a real workload: instructions from the
tatsu-lab/alpaca dataset tokenized by a real BPE tokenizer (tiktoken), so prefix
sharing comes from how real prompts actually overlap (common instruction openings like
"Write a", "Explain the", boilerplate preambles), not from a designed template
distribution.

Panels:
  (A) stationary stream: tokens reused per prompt vs. cache memory budget, adaptive vs
      LRU / LFU / offline-optimal static cache.
  (B) popularity shift: at the midpoint the prompt population switches to a disjoint
      set of instructions (so the pre-shift cache goes stale); adaptive forgets stale
      prefixes and tracks the new distribution, LFU stays sticky, the static optimum
      cannot adapt.

Run with:  uv run --extra bench --extra plot python examples/llm_routing/prefix_cache.py
(Needs the ``bench`` extra: datasets + tiktoken; downloads alpaca once.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402

from canopy.bandits import offline_optimal, run_cache  # noqa: E402
from _plotstyle import FIGURE_DIR, PALETTE, save_figure, set_style  # noqa: E402

HORIZON = 20_000
PREFIX_TOKENS = 12  # cache decisions over the first tokens of each prompt
BUDGETS = [16, 32, 64, 128, 256, 512]
SHIFT_BUDGET = 128
N_SEEDS = 6
POLICIES = ["lru", "lfu", "adaptive"]

# real instruction corpora usable as prompt pools (second corpus = robustness check)
CORPORA = {
    "alpaca": ("tatsu-lab/alpaca", "train", lambda r: r["instruction"] + (" " + r["input"] if r["input"] else "")),
    "dolly": ("databricks/databricks-dolly-15k", "train", lambda r: r["instruction"] + (" " + r["context"] if r["context"] else "")),
}
COLORS = {
    "lru": PALETTE["purple"],
    "lfu": PALETTE["blue"],
    "adaptive": PALETTE["red"],
    "offline": PALETTE["green"],
}


def load_real_prompts(corpus: str = "alpaca") -> list[tuple[int, ...]]:
    """Real instructions tokenized with tiktoken, truncated to PREFIX_TOKENS."""
    import tiktoken
    from datasets import load_dataset

    hf_name, split, to_text = CORPORA[corpus]
    enc = tiktoken.get_encoding("cl100k_base")
    ds = load_dataset(hf_name, split=split)
    prompts = []
    for row in ds:
        toks = tuple(enc.encode(to_text(row))[:PREFIX_TOKENS])
        if toks:
            prompts.append(toks)
    return prompts


def make_stream(
    prompts: list[tuple[int, ...]],
    horizon: int,
    rng: np.random.Generator,
    shift_at: int | None = None,
) -> list[tuple[int, ...]]:
    """Sample a stream from the real prompt pool (Zipf over a shuffled pool).

    With ``shift_at``, the second half draws from a disjoint half of the pool, so the
    popular prefixes before and after the shift do not overlap -- a real popularity
    shift built from real prompts.
    """
    pool = list(prompts)
    rng.shuffle(pool)
    half = len(pool) // 2
    ranks = np.arange(1, half + 1)
    weights = 1.0 / ranks**1.2
    weights /= weights.sum()
    stream = []
    for t in range(horizon):
        base = 0 if (shift_at is None or t < shift_at) else half
        stream.append(pool[base + int(rng.choice(half, p=weights))])
    return stream


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="alpaca", choices=sorted(CORPORA))
    args = ap.parse_args()
    set_style()
    try:
        prompts = load_real_prompts(args.corpus)
    except Exception as e:  # noqa: BLE001 -- missing bench extra / no network
        print(f"Could not load the real prompt corpus ({type(e).__name__}: {e}).")
        print("Install the bench extra: uv sync --extra bench  (downloads the corpus once)")
        return
    print(
        f"real prompt pool: {len(prompts)} {args.corpus} instructions, "
        f"{PREFIX_TOKENS}-token prefixes"
    )

    # Panel A: stationary savings vs budget
    stationary: dict[str, list[list[float]]] = {p: [] for p in POLICIES + ["offline"]}
    for budget in BUDGETS:
        per_policy = {p: [] for p in POLICIES + ["offline"]}
        for seed in range(N_SEEDS):
            stream = make_stream(prompts, HORIZON, np.random.default_rng(seed))
            for p in POLICIES:
                per_policy[p].append(run_cache(stream, budget, policy=p).avg_savings)
            per_policy["offline"].append(offline_optimal(stream, budget).avg_savings)
        for p, vals in per_policy.items():
            stationary[p].append(vals)
        print(
            f"  budget {budget:4d}: "
            + "  ".join(f"{p}={np.mean(v):.2f}" for p, v in per_policy.items())
        )

    # Panel B: savings over time across the shift (fixed budget)
    curves = {p: [] for p in POLICIES + ["offline"]}
    for seed in range(N_SEEDS):
        stream = make_stream(
            prompts, HORIZON, np.random.default_rng(100 + seed), shift_at=HORIZON // 2
        )
        for p in POLICIES:
            curves[p].append(run_cache(stream, SHIFT_BUDGET, policy=p).savings_curve)
        curves["offline"].append(offline_optimal(stream, SHIFT_BUDGET).savings_curve)
    # The adaptation claim lives in the transient right after the shift: given enough
    # post-shift stream even LFU eventually re-learns, so report the window where the
    # policies actually differ (and print the end-of-stream value for completeness).
    shift = HORIZON // 2
    window = slice(shift, shift + 2500)
    post = {p: float(np.mean([c[window].mean() for c in curves[p]])) for p in curves}
    final = {p: float(np.mean([c[-1] for c in curves[p]])) for p in curves}
    print(
        f"  post-shift transient (first 2500 prompts) @ budget {SHIFT_BUDGET}: "
        + "  ".join(f"{p}={v:.2f}" for p, v in post.items())
    )
    print(
        f"  end of stream (LFU has re-learned): "
        + "  ".join(f"{p}={v:.2f}" for p, v in final.items())
    )

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))
    for p in POLICIES + ["offline"]:
        arr = np.array(stationary[p])  # (budgets, seeds)
        mean = arr.mean(axis=1)
        err = 1.96 * arr.std(axis=1, ddof=1) / np.sqrt(N_SEEDS)
        axA.errorbar(
            BUDGETS, mean, yerr=err, fmt="o-", color=COLORS[p], capsize=3,
            label="offline (optimal)" if p == "offline" else ("adaptive (ours)" if p == "adaptive" else p),
        )
    axA.set_xscale("log", base=2)
    axA.set_xlabel("cache memory budget (nodes)")
    axA.set_ylabel("tokens reused per prompt")
    axA.set_title("Stationary: savings vs. memory budget")
    axA.legend(loc="upper left", fontsize=8)

    x = np.arange(HORIZON)
    for p in POLICIES + ["offline"]:
        mean_curve = np.mean(np.array(curves[p]), axis=0)
        axB.plot(
            x[::50], mean_curve[::50], color=COLORS[p], lw=1.5,
            label="offline (optimal)" if p == "offline" else ("adaptive (ours)" if p == "adaptive" else p),
        )
    axB.axvline(HORIZON // 2, color=PALETTE["gray"], ls="--", lw=1)
    axB.set_xlabel("prompts seen")
    axB.set_ylabel("tokens reused per prompt (rolling)")
    axB.set_title(f"Real popularity shift ($B={SHIFT_BUDGET}$): adaptive tracks it")
    axB.legend(loc="lower right", fontsize=8)

    fig.suptitle(f"Prefix-cache management on real prompts ({CORPORA[args.corpus][0]})")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    suffix = "" if args.corpus == "alpaca" else f"_{args.corpus}"
    out = save_figure(fig, f"prefix_cache{suffix}")
    print(f"\nsaved chart to {out} (+ .png)")

    # LaTeX table (Table 9): largest stationary budget + post-shift, per policy
    largest = {p: float(np.mean(stationary[p][-1])) for p in POLICIES + ["offline"]}
    pretty = {"lru": "LRU", "lfu": "LFU", "adaptive": "\\textbf{Adaptive (ours)}",
              "offline": "Offline-optimal (static)"}
    tex = (
        "% Prefix cache on real prompts (auto-generated by examples/llm_routing/prefix_cache.py)\n"
        "\\begin{tabular}{lrr}\n\\toprule\n"
        f"Policy & Tokens reused/prompt (stationary, $B={BUDGETS[-1]}$) & "
        f"Post-shift transient ($B={SHIFT_BUDGET}$) \\\\\n\\midrule\n"
        + "\n".join(
            f"{pretty[p]} & {largest[p]:.2f} & {post[p]:.2f} \\\\" for p in ["lru", "lfu", "adaptive", "offline"]
        )
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIGURE_DIR / f"prefix_cache{suffix}_table.tex").write_text(tex)
    print(f"wrote LaTeX table to {FIGURE_DIR}/prefix_cache{suffix}_table.tex")


if __name__ == "__main__":
    main()
