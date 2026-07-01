"""Hardware-calibrated policy eval: adaptive vs LRU vs LFU cache eviction, real GPU costs.

vLLM/SGLang only ship LRU eviction and expose no pluggable policy, so a full in-engine A/B needs
a source fork (the true gold standard; see README Tier 2). This script gives the next best,
*turnkey* thing: it MEASURES the real per-token prefill time a cache hit saves on your GPU (a
calibration against a live vLLM server), then REPLAYS the paper's eviction policies (adaptive /
LRU / LFU / offline-optimal) over a real shared-prefix workload with a mid-stream popularity
shift, and reports the realized GPU time each policy saves. The eviction *decisions* are the
paper's; the *costs* are measured on the hardware. This isolates the policy question the ON/OFF
and budget-sweep experiments (`vllm_prefix_cache_eval.py`) cannot: does adaptive beat LRU under
drift, in real milliseconds?

Turnkey on a GPU box (vLLM installed):
    python examples/systems/vllm_policy_eval.py --model Qwen/Qwen2.5-7B-Instruct --kv-budget 64
Validate with no GPU (mock calibration; exercises replay + outputs):
    python examples/systems/vllm_policy_eval.py --dry-run

Outputs (paper/figures/): vllm_policy_results.json, vllm_policy_table.tex, vllm_policy.{pdf,png}.
Honest scope: this is a hardware-calibrated trace-driven comparison, not an in-engine deployment;
it grounds the policy result in real GPU prefill costs rather than a 1-token/node proxy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from canopy.bandits import run_cache

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from vllm_prefix_cache_eval import (  # noqa: E402  (reuse server + trace helpers)
    build_trace, one_request, start_server, wait_ready,
)

FIGDIR = Path(__file__).resolve().parents[2] / "paper" / "figures"
POLICIES = ["lru", "lfu", "adaptive", "offline"]


def get_encode():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base").encode
    except Exception:  # noqa: BLE001
        return lambda s: [hash(w) % 50000 for w in s.split()]


def calibrate_ms_per_token(base_url, model, lengths, max_tokens, dry_run):
    """Measure seconds of TTFT saved per cached prefix token, on the real GPU.

    For each prefix length L: a cold prompt (fresh prefix, no hit) vs a warm one (same prefix
    primed, so it hits the cache). (cold_ttft - warm_ttft)/L estimates per-token prefill saving.
    """
    if dry_run:
        return 0.0008  # ~0.8 ms/token, a plausible 7-8B prefill rate
    import numpy as np

    per_tok = []
    for i, L in enumerate(lengths):
        prefix = " ".join(f"cal{i}_{k}" for k in range(L))
        cold = f"fresh{i} " + " ".join(f"u{i}_{k}" for k in range(L)) + " ? Answer briefly."
        ttft_cold, _, ok1 = one_request(base_url, model, cold, max_tokens)
        one_request(base_url, model, prefix + " prime ? Answer briefly.", max_tokens)  # warm
        ttft_warm, _, ok2 = one_request(base_url, model, prefix + " query ? Answer briefly.", max_tokens)
        if ok1 and ok2 and ttft_cold and ttft_warm and ttft_cold > ttft_warm:
            per_tok.append((ttft_cold - ttft_warm) / max(1, L))
    val = float(np.median(per_tok)) if per_tok else 0.0
    print(f"  calibration: {len(per_tok)} usable points, ms/token={val*1000:.3f}")
    return val


def replay(stream, budget, ms_per_token):
    """Replay each eviction policy; return realized GPU seconds saved per prompt, split by regime.

    Uses the paper's cache machinery (``run_cache``): its savings curve is the cached-prefix depth
    (tokens reused) per prompt; multiplying by the measured per-token cost gives realized seconds.
    Stationary = first half of the stream; post-shift = last quarter (after the midpoint drift).
    """
    import numpy as np

    n = len(stream)
    out = {}
    for p in POLICIES:
        curve = np.asarray(run_cache(stream, budget, policy=p).savings_curve, dtype=float)
        stat = float(curve[: n // 2].mean()) if n else 0.0
        shift = float(curve[3 * n // 4:].mean()) if n else 0.0
        out[p] = {
            "tokens_saved_stationary": stat, "tokens_saved_postshift": shift,
            "ttft_saved_ms_stationary": stat * ms_per_token * 1000,
            "ttft_saved_ms_postshift": shift * ms_per_token * 1000,
        }
    return out


def _write_outputs(model, ms_per_token, budget, res, quiet=False):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    (FIGDIR / "vllm_policy_results.json").write_text(json.dumps(
        {"model": model, "ms_per_token": ms_per_token, "kv_budget": budget, "policies": res},
        indent=2))
    name = {"lru": "LRU (engine default)", "lfu": "LFU", "adaptive": "\\textbf{Adaptive (ours)}",
            "offline": "Offline-optimal"}
    rows = []
    for p in POLICIES:
        s, sh = res[p]["ttft_saved_ms_stationary"], res[p]["ttft_saved_ms_postshift"]
        cs, csh = (f"\\textbf{{{s:.1f}}}", f"\\textbf{{{sh:.1f}}}") if p == "adaptive" else (f"{s:.1f}", f"{sh:.1f}")
        rows.append(f"{name[p]} & {cs} & {csh} \\\\")
    tex = ("% vLLM-calibrated eviction policy comparison (vllm_policy_eval.py)\n"
           "\\begin{tabular}{lrr}\n\\toprule\n"
           "Policy & TTFT saved, stationary (ms) & TTFT saved, post-shift (ms) \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (FIGDIR / "vllm_policy_table.tex").write_text(tex)
    if quiet:
        return
    print(f"\nGPU-calibrated eviction policies ({model}, ms/token={ms_per_token*1000:.3f}, B={budget}):")
    print(f"  {'policy':22s} {'stationary(ms)':>15s} {'post-shift(ms)':>15s}")
    for p in POLICIES:
        print(f"  {p:22s} {res[p]['ttft_saved_ms_stationary']:15.1f} "
              f"{res[p]['ttft_saved_ms_postshift']:15.1f}")
    try:
        _plot(model, res)
    except Exception as e:  # noqa: BLE001
        print(f"(figure skipped: {type(e).__name__}: {e})")


def _plot(model, res):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _plotstyle import PALETTE, save_figure, set_style
    set_style()
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(POLICIES))
    stat = [res[p]["ttft_saved_ms_stationary"] for p in POLICIES]
    shift = [res[p]["ttft_saved_ms_postshift"] for p in POLICIES]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.bar(x - 0.2, stat, 0.4, label="stationary", color=PALETTE["gray"])
    ax.bar(x + 0.2, shift, 0.4, label="post-shift", color=PALETTE["red"])
    ax.set_xticks(x)
    ax.set_xticklabels(POLICIES)
    ax.set_ylabel("realized TTFT saved (ms)")
    ax.set_title(f"GPU-calibrated eviction: adaptive vs LRU/LFU ({model})")
    ax.legend()
    fig.tight_layout()
    print("figure:", save_figure(fig, "vllm_policy"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--num-prompts", type=int, default=4000)
    ap.add_argument("--n-templates", type=int, default=16)
    ap.add_argument("--pad-tokens", type=int, default=200)
    ap.add_argument("--kv-budget", type=int, default=64, help="cache budget in trie nodes (tokens)")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--calib-lengths", default="50,100,200,400")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--extra-arg", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import numpy as np

    rng = np.random.default_rng(0)
    text_trace = build_trace(rng, args.num_prompts, args.n_templates, args.pad_tokens, shift=True)
    encode = get_encode()
    stream = [tuple(encode(p)[:64]) for p in text_trace]
    stream = [s for s in stream if len(s) >= 2]
    print(f"policy eval: {len(stream)} prompts, {args.n_templates} templates, shift at midpoint")

    base_url = f"http://localhost:{args.port}"
    lengths = [int(x) for x in args.calib_lengths.split(",") if x.strip()]
    if args.dry_run:
        ms = calibrate_ms_per_token(base_url, args.model, lengths, args.max_tokens, True)
        _write_outputs("mock", ms, args.kv_budget, replay(stream, args.kv_budget, ms))
        return
    proc = start_server(args.model, args.port, True, args.gpu_memory_utilization, args.extra_arg)
    try:
        if not wait_ready(base_url):
            print("[error] server never became ready."); return
        one_request(base_url, args.model, text_trace[0], args.max_tokens)  # warm up
        ms = calibrate_ms_per_token(base_url, args.model, lengths, args.max_tokens, False)
        if ms <= 0:
            print("[warn] calibration failed (no measurable hit saving); reporting tokens only.")
        _write_outputs(args.model, ms, args.kv_budget, replay(stream, args.kv_budget, ms))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            proc.kill()


if __name__ == "__main__":
    main()
