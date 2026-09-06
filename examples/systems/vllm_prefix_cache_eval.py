"""Tier-1 systems eval: real vLLM serving with prefix caching ON vs OFF.

Turns the paper's token-reuse *proxy* (``examples/llm_routing/prefix_cache.py``) into a real
serving measurement. It launches a vLLM OpenAI-compatible server twice --- once with automatic
prefix caching enabled, once disabled --- drives the SAME request trace against each, and records
real systems metrics: time-to-first-token (TTFT), end-to-end latency, throughput, the server's
prefix-cache hit rate, and KV-cache memory usage. The workload has genuine shared-prefix structure
(a small pool of long system-prompt / few-shot templates reused across many short user queries ---
the real source of prefix reuse in serving), with an optional mid-stream popularity SHIFT.

This measures the *premise* ("the prefix tree is the cache, and reuse buys real latency/memory")
using vLLM's built-in (LRU) eviction. Testing our *adaptive* eviction policy against LRU requires
patching the engine's evictor -- see the Tier-2 note in the repo docs; this script is Tier 1.

Run on a GPU box with vLLM installed (pip install vllm):
    python examples/systems/vllm_prefix_cache_eval.py \
        --model Qwen/Qwen2.5-7B-Instruct --num-prompts 2000 --request-rate 20 --shift
Validate the harness with no GPU/vLLM (mocked timings, exercises trace+aggregation+outputs):
    python examples/systems/vllm_prefix_cache_eval.py --dry-run

Outputs (paper/figures/):
    vllm_prefix_cache_results.json   raw + aggregated metrics for both configs
    vllm_prefix_cache_table.tex      LaTeX summary table
    vllm_prefix_cache.{pdf,png}      TTFT / throughput bars (needs matplotlib)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FIGDIR = Path(__file__).resolve().parents[2] / "paper" / "figures"

# A pool of long, distinct system-prompt / few-shot "templates". Reuse of these across many
# queries is exactly what prefix caching exploits; length makes the prefill savings material.
_TEMPLATE_SEEDS = [
    "You are an expert {dom} assistant. Follow the domain policy carefully and cite steps. ",
    "System: {dom} support agent. Rules: be precise, verify inputs, never fabricate. ",
    "You are a meticulous {dom} tutor. Explain reasoning, then give the final answer. ",
    "Assistant profile: senior {dom} engineer. Prefer correctness over brevity. ",
]
_DOMAINS = ["math", "coding", "legal", "medical", "finance", "science", "history", "travel"]


def build_templates(n_templates: int, pad_tokens: int) -> list[str]:
    """Construct ``n_templates`` long, distinct shared-prefix templates (~pad_tokens words each)."""
    templates = []
    for i in range(n_templates):
        seed = _TEMPLATE_SEEDS[i % len(_TEMPLATE_SEEDS)].format(dom=_DOMAINS[i % len(_DOMAINS)])
        # pad with distinct-but-fixed filler so each template is a long, cacheable prefix
        filler = " ".join(f"guideline{i}_{k}" for k in range(pad_tokens))
        templates.append(seed + filler + "\n\nUser question: ")
    return templates


def build_trace(rng, n_prompts, n_templates, pad_tokens, shift):
    """Return a list of prompt strings: a shared template (Zipf-popular) + a unique short query.

    If ``shift`` is set, the template-popularity distribution is permuted at the midpoint, so the
    prefixes worth caching change --- the non-stationary regime.
    """
    import numpy as np

    templates = build_templates(n_templates, pad_tokens)
    ranks = np.arange(1, n_templates + 1)
    base = 1.0 / ranks**1.1
    base /= base.sum()
    shifted = base[rng.permutation(n_templates)]
    trace = []
    for t in range(n_prompts):
        p = shifted if (shift and t >= n_prompts // 2) else base
        k = int(rng.choice(n_templates, p=p))
        query = f"Q{t}: {' '.join(f'w{rng.integers(0, 9999)}' for _ in range(8))}? Answer briefly."
        trace.append(templates[k] + query)
    return trace


def load_corpus_trace(path, n_prompts):
    """Alternative workload: real prompts from a local file (one per line)."""
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    return lines[:n_prompts] if n_prompts > 0 else lines


def start_server(model, port, enable_prefix_caching, gpu_mem_util, extra_args):
    """Launch a vLLM OpenAI-compatible server; return the Popen handle (or None on failure)."""
    flag = "--enable-prefix-caching" if enable_prefix_caching else "--no-enable-prefix-caching"
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--port",
        str(port),
        flag,
        "--gpu-memory-utilization",
        str(gpu_mem_util),
    ] + list(extra_args)
    print("launching:", " ".join(cmd), flush=True)
    return subprocess.Popen(cmd)


def wait_ready(base_url, timeout_s=600):
    """Poll /health until the server is up; return True if ready."""
    import requests

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/health", timeout=5).status_code == 200:
                return True
        except Exception:  # noqa: BLE001 -- server not up yet
            pass
        time.sleep(3)
    return False


def scrape_metrics(base_url):
    """Read vLLM's Prometheus /metrics; return prefix-cache hit rate and GPU KV usage if present."""
    import re

    import requests

    out = {"prefix_cache_hit_rate": None, "gpu_cache_usage": None}
    try:
        text = requests.get(f"{base_url}/metrics", timeout=10).text
    except Exception:  # noqa: BLE001
        return out

    def _sum(pat):
        vals = [float(m) for m in re.findall(pat + r"[^ ]* ([0-9.eE+-]+)", text)]
        return sum(vals) if vals else None

    hits = _sum(r"vllm:prefix_cache_hits_total")
    queries = _sum(r"vllm:prefix_cache_queries_total")
    if hits is not None and queries and queries > 0:
        out["prefix_cache_hit_rate"] = hits / queries
    usage = _sum(r"vllm:gpu_cache_usage_perc")
    out["gpu_cache_usage"] = usage
    return out


def one_request(base_url, model, prompt, max_tokens):
    """Send one streaming completion; return (ttft_s, latency_s, ok)."""
    import requests

    t0 = time.perf_counter()
    ttft = None
    try:
        with requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": True,
            },
            stream=True,
            timeout=120,
        ) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - t0
                if line.strip() == b"data: [DONE]":
                    break
    except Exception:  # noqa: BLE001
        return None, None, False
    lat = time.perf_counter() - t0
    return (ttft if ttft is not None else lat), lat, True


def drive(base_url, model, trace, request_rate, max_tokens, concurrency):
    """Fire the trace at ~request_rate req/s (Poisson) with a thread pool; collect timings."""
    import numpy as np

    rng = np.random.default_rng(0)
    results = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for prompt in trace:
            futures.append(pool.submit(one_request, base_url, model, prompt, max_tokens))
            if request_rate > 0:
                time.sleep(float(rng.exponential(1.0 / request_rate)))
        for f in futures:
            ttft, lat, ok = f.result()
            if ok:
                results.append((ttft, lat))
    wall = time.perf_counter() - start
    return results, wall


def _pct(xs, q):
    import numpy as np

    return float(np.percentile(np.asarray(xs), q)) if xs else 0.0


def aggregate(results, wall, max_tokens):
    ttfts = [r[0] for r in results]
    lats = [r[1] for r in results]
    n = len(results)
    return {
        "n_ok": n,
        "ttft_mean": (sum(ttfts) / n if n else 0.0),
        "ttft_p50": _pct(ttfts, 50),
        "ttft_p99": _pct(ttfts, 99),
        "latency_mean": (sum(lats) / n if n else 0.0),
        "throughput_req_s": (n / wall if wall > 0 else 0.0),
        "throughput_tok_s": (n * max_tokens / wall if wall > 0 else 0.0),
        "wall_s": wall,
    }


def _write_outputs(model, configs, meta, quiet=False):
    """configs: {"on": {...agg, metrics}, "off": {...}}. Write JSON + LaTeX table + figure."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    (FIGDIR / "vllm_prefix_cache_results.json").write_text(
        json.dumps({"model": model, **meta, "configs": configs}, indent=2)
    )
    on, off = configs["on"], configs["off"]

    def _row(label, key, unit, better_low=True):
        vo, vf = on.get(key, 0.0), off.get(key, 0.0)
        gain = (vf - vo) / vf * 100 if (better_low and vf) else (vo - vf) / vf * 100 if vf else 0.0
        return f"{label} & {vo:.3f} & {vf:.3f} & {gain:+.0f}\\% \\\\"

    tput_gain_pct = (
        (on["throughput_tok_s"] - off["throughput_tok_s"])
        / max(off["throughput_tok_s"], 1e-9)
        * 100
    )
    rows = [
        _row("TTFT p50 (s)", "ttft_p50", "s"),
        _row("TTFT mean (s)", "ttft_mean", "s"),
        _row("Latency mean (s)", "latency_mean", "s"),
        f"Throughput (tok/s) & {on['throughput_tok_s']:.1f} & {off['throughput_tok_s']:.1f} & "
        f"{tput_gain_pct:+.0f}\\% \\\\",
    ]
    hr = on.get("prefix_cache_hit_rate")
    if hr is not None:
        rows.append(f"Prefix-cache hit rate & {hr:.3f} & 0.000 & --- \\\\")
    tex = (
        "% vLLM prefix caching ON vs OFF (auto-generated by vllm_prefix_cache_eval.py)\n"
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Metric & Caching ON & Caching OFF & Gain \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (FIGDIR / "vllm_prefix_cache_table.tex").write_text(tex)
    if quiet:
        return
    print(f"\nvLLM prefix caching ON vs OFF ({model}):")
    for k in ("ttft_p50", "ttft_mean", "throughput_tok_s", "throughput_req_s"):
        print(f"  {k:20s} on={on.get(k, 0):.3f}  off={off.get(k, 0):.3f}")
    if hr is not None:
        print(f"  prefix_cache_hit_rate on={hr:.3f}")
    try:
        out = _plot(model, on, off)
        print(f"\nwrote json+table to {FIGDIR} and figure to {out} (+ .png)")
    except Exception as e:  # noqa: BLE001
        print(f"\nwrote json+table to {FIGDIR} (figure skipped: {type(e).__name__}: {e})")


def _plot(model, on, off):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _plotstyle import PALETTE, save_figure, set_style

    set_style()
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9, 4.2))
    axA.bar(
        ["ON", "OFF"],
        [on["ttft_p50"], off["ttft_p50"]],
        color=[PALETTE["red"], PALETTE["gray"]],
        width=0.6,
    )
    axA.set_ylabel("TTFT p50 (s)")
    axA.set_title("Time-to-first-token")
    axB.bar(
        ["ON", "OFF"],
        [on["throughput_tok_s"], off["throughput_tok_s"]],
        color=[PALETTE["red"], PALETTE["gray"]],
        width=0.6,
    )
    axB.set_ylabel("throughput (tok/s)")
    axB.set_title("Throughput")
    fig.suptitle(f"vLLM prefix caching ON vs OFF ({model})")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return str(save_figure(fig, "vllm_prefix_cache"))


def _mock_config(trace, enable, max_tokens):
    """Dry-run: synthesize plausible timings (caching ON prefills less, so lower TTFT)."""
    import numpy as np

    rng = np.random.default_rng(0 if enable else 1)
    n = len(trace)
    base_ttft = 0.06 if enable else 0.16  # cached prefixes skip prefill -> lower TTFT
    ttfts = np.clip(rng.normal(base_ttft, base_ttft * 0.25, n), 0.005, None)
    lats = ttfts + max_tokens * (0.006 if enable else 0.007)
    wall = float(lats.sum() / max(1, 16))  # ~16-way concurrency
    agg = aggregate(list(zip(ttfts.tolist(), lats.tolist())), wall, max_tokens)
    if enable:
        agg["prefix_cache_hit_rate"] = 0.71
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--num-prompts", type=int, default=2000)
    ap.add_argument("--n-templates", type=int, default=16, help="distinct shared-prefix templates")
    ap.add_argument(
        "--pad-tokens", type=int, default=200, help="template length (longer = more prefill saved)"
    )
    ap.add_argument(
        "--request-rate", type=float, default=20.0, help="req/s (Poisson); 0 = as fast as possible"
    )
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="cap context length; keeps small KV-block budgets viable in the sweep",
    )
    ap.add_argument(
        "--shift", action="store_true", help="permute template popularity at the midpoint"
    )
    ap.add_argument(
        "--kv-block-sweep",
        default="",
        help="comma-separated --num-gpu-blocks-override values; sweeps the KV budget "
        "(prefix caching stays ON) instead of the ON/OFF comparison",
    )
    ap.add_argument(
        "--corpus-file", default="", help="use real prompts from a file instead of templates"
    )
    ap.add_argument(
        "--extra-arg", action="append", default=[], help="extra vllm serve arg (repeatable)"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="no GPU/vLLM; mock timings to test the harness"
    )
    args = ap.parse_args()

    import numpy as np

    rng = np.random.default_rng(0)
    if args.corpus_file:
        trace = load_corpus_trace(args.corpus_file, args.num_prompts)
    else:
        trace = build_trace(rng, args.num_prompts, args.n_templates, args.pad_tokens, args.shift)
    meta = {
        "n_prompts": len(trace),
        "n_templates": args.n_templates,
        "shift": args.shift,
        "request_rate": args.request_rate,
        "workload": args.corpus_file or "templates",
    }
    print(f"workload: {meta['workload']}, {len(trace)} prompts, shift={args.shift}")

    base_url = f"http://localhost:{args.port}"
    if args.kv_block_sweep.strip():
        run_budget_sweep(args, trace)
        return
    configs = {}
    for enable in (True, False):
        key = "on" if enable else "off"
        if args.dry_run:
            configs[key] = _mock_config(trace, enable, args.max_tokens)
            continue
        proc = start_server(
            args.model, args.port, enable, args.gpu_memory_utilization, args.extra_arg
        )
        try:
            if not wait_ready(base_url):
                print(f"[error] server for caching={enable} never became ready; skipping.")
                configs[key] = {}
                continue
            one_request(base_url, args.model, trace[0], args.max_tokens)  # warm up
            results, wall = drive(
                base_url, args.model, trace, args.request_rate, args.max_tokens, args.concurrency
            )
            agg = aggregate(results, wall, args.max_tokens)
            agg.update({k: v for k, v in scrape_metrics(base_url).items() if v is not None})
            configs[key] = agg
            print(
                f"  caching={enable}: {agg['n_ok']} ok, ttft_p50={agg['ttft_p50']:.3f}s, "
                f"tput={agg['throughput_tok_s']:.1f} tok/s"
            )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                proc.kill()

    if configs.get("on") and configs.get("off"):
        _write_outputs(args.model, configs, meta)
    else:
        print("[warn] incomplete configs; not writing outputs.")


# --- KV-budget sweep: savings vs cache memory budget on real GPU (prefix caching ON) --------


def run_budget_sweep(args, trace):
    """Sweep --num-gpu-blocks-override; measure TTFT / throughput / hit-rate at each KV budget.

    This is the real-hardware analog of the proxy's savings-vs-memory-budget frontier: bigger KV
    budget -> more cached prefixes -> higher hit rate -> lower TTFT. Prefix caching stays ON;
    each point differs only in the KV-cache size. Writes vllm_prefix_cache_budget.{json,tex,pdf}.
    """
    base_url = f"http://localhost:{args.port}"
    blocks = [int(x) for x in args.kv_block_sweep.split(",") if x.strip()]
    points = []
    for nb in blocks:
        if args.dry_run:
            # mock: hit-rate saturates and TTFT drops as the budget grows
            import numpy as np

            frac = nb / max(blocks)
            hr = 0.85 * (1 - np.exp(-3 * frac))
            agg = {
                "ttft_p50": 0.18 - 0.11 * hr,
                "throughput_tok_s": 1600 + 900 * hr,
                "prefix_cache_hit_rate": float(hr),
                "n_ok": len(trace),
            }
            points.append({"blocks": nb, **agg})
            continue
        proc = start_server(
            args.model,
            args.port,
            True,
            args.gpu_memory_utilization,
            list(args.extra_arg)
            + ["--max-model-len", str(args.max_model_len), "--num-gpu-blocks-override", str(nb)],
        )
        try:
            if not wait_ready(base_url):
                print(f"[error] server (blocks={nb}) not ready; skipping.")
                continue
            one_request(base_url, args.model, trace[0], args.max_tokens)
            results, wall = drive(
                base_url, args.model, trace, args.request_rate, args.max_tokens, args.concurrency
            )
            agg = aggregate(results, wall, args.max_tokens)
            agg.update({k: v for k, v in scrape_metrics(base_url).items() if v is not None})
            points.append({"blocks": nb, **agg})
            print(
                f"  blocks={nb}: ttft_p50={agg['ttft_p50']:.3f}s "
                f"tput={agg['throughput_tok_s']:.1f} hit={agg.get('prefix_cache_hit_rate')}"
            )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                proc.kill()
    _write_budget(args.model, points)


def _write_budget(model, points):
    if not points:
        return
    FIGDIR.mkdir(parents=True, exist_ok=True)
    (FIGDIR / "vllm_prefix_cache_budget_results.json").write_text(
        json.dumps({"model": model, "points": points}, indent=2)
    )
    rows = [
        f"{p['blocks']} & {p['ttft_p50']:.3f} & {p['throughput_tok_s']:.1f} & "
        f"{(p.get('prefix_cache_hit_rate') or 0):.3f} \\\\"
        for p in points
    ]
    tex = (
        "% vLLM KV-budget sweep (auto-generated by vllm_prefix_cache_eval.py)\n"
        "\\begin{tabular}{rrrr}\n\\toprule\n"
        "KV blocks & TTFT p50 (s) & Throughput (tok/s) & Hit rate \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (FIGDIR / "vllm_prefix_cache_budget_table.tex").write_text(tex)
    print(f"\nKV-budget sweep -> {FIGDIR}/vllm_prefix_cache_budget_*.{{json,tex}}")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from _plotstyle import PALETTE, save_figure, set_style

        set_style()
        import matplotlib.pyplot as plt

        b = [p["blocks"] for p in points]
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.plot(
            b, [p["ttft_p50"] for p in points], "o-", color=PALETTE["red"], label="TTFT p50 (s)"
        )
        ax.set_xlabel("KV-cache budget (blocks)")
        ax.set_ylabel("TTFT p50 (s, log scale)")
        ax.set_yscale("log")
        ax2 = ax.twinx()
        ax2.plot(
            b,
            [(p.get("prefix_cache_hit_rate") or 0) for p in points],
            "s--",
            color=PALETTE["blue"],
            label="hit rate",
        )
        ax2.set_ylabel("prefix-cache hit rate")
        ax.set_title(f"vLLM: TTFT & hit rate vs KV budget ({model})")
        fig.tight_layout()
        print("figure:", save_figure(fig, "vllm_prefix_cache_budget"))
    except Exception as e:  # noqa: BLE001
        print(f"(budget figure skipped: {type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
