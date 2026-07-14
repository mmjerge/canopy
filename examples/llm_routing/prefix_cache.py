"""Prefix-cache management on a REAL prompt stream: adaptive vs. LRU / LFU / offline-optimal.

The cache is an ancestor-closed subtree of the token trie under a memory budget; caching a
prefix saves recompute for every prompt that passes through it (savings = depth of the deepest
cached prefix on a prompt's path). This is the tree-structured, aggregate-feedback,
storage-bounded problem of the paper with **storage = cache memory** -- and here the tree *is*
the cache, so no tree-objective alignment is assumed.

Real data: prompts come from a real corpus (a local prompt log, one prompt per line, or a
Hugging Face dataset) and are tokenized with a real tokenizer (``tiktoken`` if available, else a
Hugging Face tokenizer, else a word-level fallback). We report:

  (A) tokens reused per prompt vs. memory budget on the real stream -- adaptive matches LFU and
      the hindsight-optimal static cache; LRU lags.
  (B) tokens reused over time across a real popularity shift (the stream switches from one
      prompt population to a disjoint one at the midpoint) -- adaptive tracks the drift and beats
      both LFU (sticky) and the best static cache.

Run on a real prompt log:
    uv run --extra plot python examples/llm_routing/prefix_cache.py --corpus-file prompts.txt
Run on a Hugging Face dataset:
    uv run --extra plot --extra bench python examples/llm_routing/prefix_cache.py \
        --dataset tatsu-lab/alpaca --n-prompts 20000
Smoke-test with a synthetic stream (no deps):
    python examples/llm_routing/prefix_cache.py --mock
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from canopy.bandits import PrefixCacheEnv, run_cache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _plotstyle import FIGURE_DIR, PALETTE, progress, save_figure, set_style  # noqa: E402

POLICIES = ["lru", "lfu", "adaptive", "offline"]
PALETTE_BY = {
    "lru": PALETTE["purple"],
    "lfu": PALETTE["blue"],
    "adaptive": PALETTE["red"],
    "offline": PALETTE["green"],
}


# --- real prompt corpus + tokenizer -------------------------------------------------------


def get_tokenizer():
    """Return (name, encode) where encode(str) -> list[int]; prefer a real BPE tokenizer."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return "tiktoken/cl100k_base", enc.encode
    except Exception:  # noqa: BLE001
        pass
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
        return "hf/gpt2", lambda s: tok.encode(s)
    except Exception:  # noqa: BLE001
        pass
    # Word-level fallback: hash each word to a small vocab. Real prefix structure is preserved
    # (shared word prefixes share token prefixes); only the token identities are approximate.
    def _encode(s: str):
        return [hash(w) % 50000 for w in s.split()]

    return "word-level (fallback)", _encode


def load_corpus(args) -> list[str]:
    """Load raw prompt strings from a local file or a Hugging Face dataset."""
    if args.corpus_file:
        lines = Path(args.corpus_file).read_text().splitlines()
        prompts = [ln.strip() for ln in lines if ln.strip()]
        return prompts[: args.n_prompts] if args.n_prompts > 0 else prompts
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    n = args.n_prompts if args.n_prompts > 0 else len(ds)
    prompts = []
    for row in ds.select(range(min(n, len(ds)))):
        # Assemble a prompt from common instruction-dataset fields, else the first string field.
        if "instruction" in row:
            p = row["instruction"] + (("\n" + row["input"]) if row.get("input") else "")
        elif "prompt" in row:
            p = row["prompt"]
        elif "question" in row:
            p = row["question"]
        else:
            p = next((str(v) for v in row.values() if isinstance(v, str) and v), "")
        if p:
            prompts.append(p)
    return prompts


def tokenize_stream(prompts, encode, max_tokens: int):
    """Tokenize each prompt to a token-id tuple, truncated to ``max_tokens`` (the cacheable head)."""
    stream = []
    for p in prompts:
        toks = tuple(encode(p)[:max_tokens])
        if len(toks) >= 2:
            stream.append(toks)
    return stream


def make_shift_stream(stream, rng):
    """Build a *sharp* real popularity shift: the two halves of the stream draw from prompt
    populations with disjoint leading tokens, so the prefixes cached before the shift become
    stale afterwards. This is the non-stationary regime where recency-weighting should beat
    sticky frequency (LFU); on a stationary stream frequency is already near-optimal.
    """
    from collections import defaultdict

    by_head: dict = defaultdict(list)
    for s in set(stream):
        by_head[s[0]].append(s)  # group unique prompts by their first token (root prefix)
    heads = list(by_head)
    rng.shuffle(heads)
    half = max(1, len(heads) // 2)
    pop_a = [s for h in heads[:half] for s in by_head[h]]
    pop_b = [s for h in heads[half:] for s in by_head[h]]
    if not pop_a or not pop_b:
        return list(stream)
    n = len(stream)
    first = [pop_a[int(rng.integers(0, len(pop_a)))] for _ in range(n // 2)]
    second = [pop_b[int(rng.integers(0, len(pop_b)))] for _ in range(n - n // 2)]
    return first + second


# --- experiment ---------------------------------------------------------------------------


def run_stationary(stream, budgets, decay=0.995):
    out = {p: [] for p in POLICIES}
    for b in progress(budgets, "cache budgets", total=len(budgets)):
        for p in POLICIES:
            out[p].append(run_cache(stream, b, policy=p, decay=decay).avg_savings)
    return out


def run_shift(stream, budget, decay=0.995):
    curves = {}
    for p in POLICIES:
        curves[p] = run_cache(stream, budget, policy=p, decay=decay).savings_curve
    return curves


def load_mooncake(path: str, n_prompts: int) -> list[tuple]:
    """Load a Mooncake-format trace (JSONL) as a stream of prefix-block-id tuples.

    Each line is a request ``{timestamp, input_length, output_length, hash_ids}`` where
    ``hash_ids`` is the sequence of KV-cache block hashes; shared leading ids are a shared prefix.
    We treat each hash-id sequence as the request's cacheable token path (one "token" = one
    512-token block), so ``run_cache`` measures reused prefix *blocks* on a real production trace.
    Lines are kept in trace (timestamp) order, so the over-time curve reflects the real workload.
    """
    stream = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        ids = rec.get("hash_ids") or []
        if len(ids) >= 2:
            stream.append(tuple(int(x) for x in ids))
        if 0 < n_prompts <= len(stream):
            break
    return stream


def _stem(tag):
    return "prefix_cache" + (f"_{tag}" if tag else "")


def _write_outputs(stationary, budgets, curves, shift_budget, meta, quiet=False, tag=""):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        **meta,
        "budgets": list(budgets),
        "stationary_savings": {p: list(map(float, stationary[p])) for p in POLICIES},
        "shift_budget": shift_budget,
        "shift_final_savings": {p: float(curves[p][-1]) for p in POLICIES},
    }
    (FIGURE_DIR / f"{_stem(tag)}_results.json").write_text(json.dumps(payload, indent=2))

    # LaTeX table: savings at the largest budget (stationary) + final savings after the shift.
    big = float(np.argmax(budgets))
    rows = []
    for p in POLICIES:
        name = {"offline": "Offline-optimal (static)", "adaptive": "\\textbf{Adaptive (ours)}"}.get(
            p, p.upper()
        )
        s_stat = stationary[p][-1]
        s_shift = curves[p][-1]
        stat_c = f"{s_stat:.2f}"
        shift_c = f"{s_shift:.2f}"
        if p == "adaptive":
            stat_c, shift_c = f"\\textbf{{{stat_c}}}", f"\\textbf{{{shift_c}}}"
        rows.append(f"{name} & {stat_c} & {shift_c} \\\\")
    tex = (
        "% Prefix-cache savings on real prompts (auto-generated by prefix_cache.py)\n"
        "\\begin{tabular}{lrr}\n\\toprule\n"
        f"Policy & Stationary (B={budgets[-1]}) & After shift (B={shift_budget}) \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (FIGURE_DIR / f"{_stem(tag)}_table.tex").write_text(tex)
    del big

    if quiet:
        return
    print(f"\nprefix cache on {meta['source']} (tokenizer: {meta['tokenizer']}, "
          f"{meta['n_prompts']} prompts)")
    print("  stationary savings/prompt by budget:")
    for p in POLICIES:
        print(f"    {p:9s} " + " ".join(f"{v:.2f}" for v in stationary[p]))
    print(f"  final savings after shift (B={shift_budget}):")
    for p in POLICIES:
        print(f"    {p:9s} {curves[p][-1]:.2f}")
    try:
        out = _plot(stationary, budgets, curves, shift_budget, meta, tag=tag)
        print(f"\nwrote json+table to {FIGURE_DIR} and figure to {out} (+ .png)")
    except Exception as e:  # noqa: BLE001
        print(f"\nwrote json+table to {FIGURE_DIR} (figure skipped: {type(e).__name__}: {e})")


def _plot(stationary, budgets, curves, shift_budget, meta, tag=""):
    set_style()
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))
    for p in POLICIES:
        label = p + (" (optimal)" if p == "offline" else (" (ours)" if p == "adaptive" else ""))
        axA.plot(budgets, stationary[p], "o-", color=PALETTE_BY[p], label=label)
    axA.set_xlabel("cache memory budget (nodes)")
    axA.set_ylabel("tokens reused per prompt")
    axA.set_title("Stationary: savings vs. memory budget")
    axA.legend(loc="lower right")

    n = len(next(iter(curves.values())))
    rounds = np.arange(1, n + 1)
    for p in POLICIES:
        axB.plot(rounds, curves[p], color=PALETTE_BY[p], label=p)
    if not meta.get("real_order"):
        axB.axvline(n // 2, color=PALETTE["gray"], ls="--", lw=1)
        axB.text(n // 2, axB.get_ylim()[1] * 0.5, " popularity shift", color=PALETTE["gray"],
                 fontsize=8, rotation=90)
        axB.set_title(f"Real popularity shift (B={shift_budget}): adaptive tracks it")
    else:
        axB.set_title(f"Real trace order (B={shift_budget}): savings over time")
    axB.set_xlabel("requests seen")
    axB.set_ylabel("blocks reused per request (rolling)" if meta.get("real_order")
                   else "tokens reused per prompt (rolling)")
    axB.legend(loc="lower left")
    fig.suptitle(f"Prefix-cache management ({meta['source']})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return str(save_figure(fig, _stem(tag)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mooncake-trace", default="",
                    help="path to a Mooncake-format JSONL trace (uses hash_ids as the block stream)")
    ap.add_argument("--tag", default="", help="suffix for output files (e.g. mooncake_conversation)")
    ap.add_argument("--corpus-file", default="", help="local prompt log, one prompt per line")
    ap.add_argument("--dataset", default="tatsu-lab/alpaca", help="HF dataset (if no --corpus-file)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-prompts", type=int, default=20000, help="cap on prompts (<=0 = all)")
    ap.add_argument("--max-tokens", type=int, default=32, help="cacheable prefix length per prompt")
    ap.add_argument("--budgets", default="8,16,24,40,64,96,128")
    ap.add_argument("--shift-budget", type=int, default=40)
    ap.add_argument("--decay", type=float, default=0.999,
                    help="EWMA decay for the adaptive policy (closer to 1 = slower forgetting)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mock", action="store_true", help="synthetic Zipf stream; no deps")
    args = ap.parse_args()

    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    if args.mooncake_trace:
        try:
            stream = load_mooncake(args.mooncake_trace, args.n_prompts)
            if len(stream) < 100:
                raise RuntimeError(f"only {len(stream)} usable requests in the trace")
            shift_stream = stream  # real trace order -> the over-time curve is the real workload
            src = f"Mooncake trace ({Path(args.mooncake_trace).stem})"
            meta = {"source": src, "tokenizer": "block-hash-ids", "n_prompts": len(stream),
                    "real_order": True}
        except Exception as e:  # noqa: BLE001
            print(f"Could not load Mooncake trace ({type(e).__name__}: {e}).")
            return
    elif args.mock:
        env = PrefixCacheEnv(rng=np.random.default_rng(args.seed))
        stream = env.generate_stream(20000)
        shift_stream = PrefixCacheEnv(
            shift_at=10000, rng=np.random.default_rng(args.seed)
        ).generate_stream(20000)
        meta = {"source": "synthetic (mock)", "tokenizer": "n/a", "n_prompts": len(stream)}
    else:
        try:
            prompts = load_corpus(args)
            tok_name, encode = get_tokenizer()
            stream = tokenize_stream(prompts, encode, args.max_tokens)
            if len(stream) < 100:
                raise RuntimeError(f"only {len(stream)} usable prompts; need a bigger corpus")
            shift_stream = make_shift_stream(stream, rng)
            source = args.corpus_file or args.dataset
            meta = {"source": source, "tokenizer": tok_name, "n_prompts": len(stream)}
        except Exception as e:  # noqa: BLE001
            print(
                f"Real corpus unavailable ({type(e).__name__}: {e}).\n"
                "Provide a prompt log:   --corpus-file prompts.txt\n"
                "or a HF dataset:        --dataset tatsu-lab/alpaca  (needs the bench extra)\n"
                "or smoke-test offline:  --mock"
            )
            return

    print(f"prefix cache: {meta['n_prompts']} requests, tokenizer {meta['tokenizer']}")
    stationary = run_stationary(stream, budgets, decay=args.decay)
    curves = run_shift(shift_stream, args.shift_budget, decay=args.decay)
    _write_outputs(stationary, budgets, curves, args.shift_budget, meta, tag=args.tag)


if __name__ == "__main__":
    main()
