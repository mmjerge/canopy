"""Real vLLM serving: prefix caching ON vs OFF, and the KV-budget frontier.

Measures a live vLLM OpenAI-compatible server on a shared-prefix workload:
time-to-first-token (p50, streaming), sustained throughput (tok/s), and the engine's
own prefix-cache hit rate (from /metrics). Each run is recorded under a --label into a
results JSON; --plot renders the accumulated runs as the ON/OFF comparison (Figure 13)
and the KV-budget sweep (Figure 14).

Usage (on a GPU host):
    # 1. caching ON (vLLM enables automatic prefix caching by default on v0.9+)
    vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
    uv run python examples/systems/vllm_prefix_cache_eval.py --label on

    # 2. caching OFF
    vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --no-enable-prefix-caching
    uv run python examples/systems/vllm_prefix_cache_eval.py --label off

    # 3. KV-budget sweep (one relaunch per point)
    vllm serve ... --num-gpu-blocks-override 500     # then --label kv500
    vllm serve ... --num-gpu-blocks-override 1000    # then --label kv1000, etc.

    # 4. figures from everything recorded so far
    uv run --extra plot python examples/systems/vllm_prefix_cache_eval.py --plot

The workload: N_TEMPLATES long shared preambles (the cacheable prefix) x short unique
questions, shuffled -- the serving pattern prefix caching exists for. The same stream
drives every labeled run, so numbers are comparable. This validates the premise on
vLLM's built-in (LRU) eviction; the adaptive-vs-LRU comparison is the trace-driven
vllm_policy_eval.py, calibrated by the --calibrate measurement here.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

RESULTS = Path(__file__).parent / "vllm_prefix_cache_results.json"
N_TEMPLATES = 40
PREFIX_REPEAT = 60  # preamble length ~ PREFIX_REPEAT * 8 tokens
REQUESTS = 200
MAX_TOKENS = 64


def build_workload(seed: int = 0) -> list[str]:
    import random

    rng = random.Random(seed)
    preambles = [
        (
            f"You are expert assistant #{i}. Follow the policy strictly. "
            + " ".join(f"Rule {j}: always verify step {j} before proceeding." for j in range(PREFIX_REPEAT))
        )
        for i in range(N_TEMPLATES)
    ]
    prompts = []
    for r in range(REQUESTS):
        p = preambles[rng.randrange(N_TEMPLATES)]
        prompts.append(f"{p}\n\nQuestion {r}: what is {r} + {r % 7}? Answer briefly.")
    return prompts


def measure(base_url: str, model: str, prompts: list[str]) -> dict:
    import urllib.request

    def post_stream(prompt: str) -> tuple[float, int, float]:
        """Returns (ttft_s, completion_tokens, total_s) for one streamed request."""
        body = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": MAX_TOKENS,
                "stream": True,
                "temperature": 0.0,
            }
        ).encode()
        req = urllib.request.Request(
            f"{base_url}/v1/completions", data=body,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        ttft, n_tok = None, 0
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n_tok += 1
        return ttft or 0.0, n_tok, time.perf_counter() - t0

    def hit_rate() -> float | None:
        try:
            with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as resp:
                text = resp.read().decode()
            q = h = None
            for line in text.splitlines():
                if line.startswith("vllm:prefix_cache_queries_total"):
                    q = float(line.rsplit(" ", 1)[1])
                if line.startswith("vllm:prefix_cache_hits_total"):
                    h = float(line.rsplit(" ", 1)[1])
            return h / q if q else None
        except Exception:  # noqa: BLE001 -- older vLLM metric names differ
            return None

    ttfts, toks, total_time = [], 0, 0.0
    t_start = time.perf_counter()
    for i, prompt in enumerate(prompts):
        ttft, n_tok, dur = post_stream(prompt)
        ttfts.append(ttft)
        toks += n_tok
        total_time += dur
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(prompts)}: ttft_p50={statistics.median(ttfts):.3f}s")
    wall = time.perf_counter() - t_start
    return {
        "ttft_p50_s": statistics.median(ttfts),
        "ttft_p90_s": statistics.quantiles(ttfts, n=10)[8],
        "throughput_tok_s": toks / wall,
        "prefix_cache_hit_rate": hit_rate(),
        "n_requests": len(prompts),
    }


def calibrate(base_url: str, model: str) -> float:
    """Measure ms of prefill a cached token saves: TTFT(cold) - TTFT(warm) per token."""
    prompts = build_workload(seed=1)[:20]
    cold = measure(base_url, model, prompts)  # first pass: prefixes cold
    warm = measure(base_url, model, prompts)  # second pass: prefixes resident
    approx_prefix_tokens = PREFIX_REPEAT * 8
    ms = (cold["ttft_p50_s"] - warm["ttft_p50_s"]) * 1000 / approx_prefix_tokens
    print(f"calibration: ~{ms:.2f} ms of prefill saved per cached token")
    return ms


def plot() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import matplotlib.pyplot as plt  # noqa: E402

    from _plotstyle import PALETTE, save_figure, set_style  # noqa: E402

    runs = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    if not runs:
        print(f"no recorded runs in {RESULTS}; run with --label first")
        return
    set_style()

    if "on" in runs and "off" in runs:
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(9, 4.2))
        for ax, key, label in [(axA, "ttft_p50_s", "TTFT p50 (s)"), (axB, "throughput_tok_s", "throughput (tok/s)")]:
            vals = [runs["on"][key], runs["off"][key]]
            ax.bar(["ON", "OFF"], vals, color=[PALETTE["orange"], PALETTE["gray"]])
            ax.set_ylabel(label)
        fig.suptitle("vLLM prefix caching ON vs OFF (shared-prefix workload)")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out = save_figure(fig, "vllm_prefix_cache_onoff")
        print(f"saved {out}")

    kv = sorted(
        (int(k[2:]), v) for k, v in runs.items() if k.startswith("kv") and k[2:].isdigit()
    )
    if kv:
        budgets = [b for b, _ in kv]
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.plot(budgets, [v["ttft_p50_s"] for _, v in kv], "o-", color=PALETTE["orange"], label="TTFT p50 (s)")
        ax.set_yscale("log")
        ax.set_xlabel("KV-cache budget (blocks)")
        ax.set_ylabel("TTFT p50 (s, log)")
        ax2 = ax.twinx()
        hits = [v.get("prefix_cache_hit_rate") for _, v in kv]
        if all(h is not None for h in hits):
            ax2.plot(budgets, hits, "s--", color=PALETTE["blue"], label="hit rate")
            ax2.set_ylabel("prefix-cache hit rate")
        ax.set_title("KV-budget frontier on real vLLM")
        fig.tight_layout()
        out = save_figure(fig, "vllm_kv_sweep")
        print(f"saved {out}")

    # Table 10
    rows = []
    for name, v in runs.items():
        hit = v.get("prefix_cache_hit_rate")
        hit_s = "--" if hit is None else f"{hit:.2f}"
        rows.append(
            f"{name} & {v['ttft_p50_s']:.2f} & {v['ttft_p90_s']:.2f} & "
            f"{v['throughput_tok_s']:.0f} & {hit_s} \\\\"
        )
    from _plotstyle import FIGURE_DIR  # noqa: E402

    tex = (
        "% vLLM prefix cache measurements (auto-generated)\n"
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Run & TTFT p50 (s) & TTFT p90 (s) & Throughput (tok/s) & Hit rate \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (FIGURE_DIR / "vllm_prefix_cache_table.tex").write_text(tex)
    print(f"wrote {FIGURE_DIR / 'vllm_prefix_cache_table.tex'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--label", default=None, help="record this run under a label (on/off/kvN)")
    ap.add_argument("--calibrate", action="store_true", help="measure ms saved per cached token")
    ap.add_argument("--plot", action="store_true", help="render figures from recorded runs")
    args = ap.parse_args()

    if args.plot:
        plot()
        return
    if args.calibrate:
        calibrate(args.base_url, args.model)
        return
    if not args.label:
        print("pass --label (e.g. on / off / kv1000), --calibrate, or --plot")
        return
    prompts = build_workload()
    print(f"measuring {len(prompts)} shared-prefix requests against {args.base_url} ...")
    result = measure(args.base_url, args.model, prompts)
    print(json.dumps(result, indent=2))
    runs = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    runs[args.label] = result
    RESULTS.write_text(json.dumps(runs, indent=2))
    print(f"recorded under label {args.label!r} in {RESULTS}")


if __name__ == "__main__":
    main()
