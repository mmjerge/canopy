"""Combine the three vLLM systems results into one shareable figure.

Reads the JSON artifacts produced on the GPU (vllm_prefix_cache_results.json,
vllm_prefix_cache_budget_results.json, vllm_policy_results.json) and renders a single
three-panel figure summarizing the systems evidence:
  (A) prefix caching ON vs OFF (TTFT + throughput),
  (B) TTFT / hit-rate vs KV-cache budget (the storage-budget frontier, log-y),
  (C) adaptive vs LRU/LFU/offline realized TTFT saved (stationary vs post-shift).

Run locally (no GPU needed; just reads the JSONs):
    python examples/systems/combine_vllm_figures.py
Writes paper/figures/vllm_systems_combined.{pdf,png}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FIGDIR = Path(__file__).resolve().parents[2] / "paper" / "figures"


def _load(name):
    p = FIGDIR / name
    if not p.exists():
        print(f"[missing] {p} -- run the corresponding experiment first.")
        return None
    return json.loads(p.read_text())


def main() -> None:
    onoff = _load("vllm_prefix_cache_results.json")
    budget = _load("vllm_prefix_cache_budget_results.json")
    policy = _load("vllm_policy_results.json")
    if not (onoff and budget and policy):
        print("Need all three result JSONs present in paper/figures/.")
        return

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _plotstyle import PALETTE, save_figure, set_style

    set_style()
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 4.4))

    # Panel A: ON vs OFF (TTFT p50 bars; throughput annotated)
    on, off = onoff["configs"]["on"], onoff["configs"]["off"]
    axA.bar(
        ["ON", "OFF"],
        [on["ttft_p50"], off["ttft_p50"]],
        color=[PALETTE["red"], PALETTE["gray"]],
        width=0.6,
    )
    axA.set_ylabel("TTFT p50 (s)")
    axA.set_title("(a) Prefix caching ON vs OFF")
    axA.annotate(
        f"{on['throughput_tok_s']:.0f} tok/s",
        ("ON", on["ttft_p50"]),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    axA.annotate(
        f"{off['throughput_tok_s']:.0f} tok/s",
        ("OFF", off["ttft_p50"]),
        ha="center",
        va="bottom",
        fontsize=8,
    )

    # Panel B: budget frontier (TTFT log-y + hit rate twin axis)
    pts = sorted(budget["points"], key=lambda p: p["blocks"])
    b = [p["blocks"] for p in pts]
    axB.plot(b, [p["ttft_p50"] for p in pts], "o-", color=PALETTE["red"], label="TTFT p50")
    axB.set_yscale("log")
    axB.set_xlabel("KV-cache budget (blocks)")
    axB.set_ylabel("TTFT p50 (s, log)")
    axB.set_title("(b) Savings vs KV budget")
    axB2 = axB.twinx()
    axB2.plot(
        b,
        [(p.get("prefix_cache_hit_rate") or 0) for p in pts],
        "s--",
        color=PALETTE["blue"],
        label="hit rate",
    )
    axB2.set_ylabel("prefix-cache hit rate")

    # Panel C: policy comparison (stationary vs post-shift)
    order = ["lru", "lfu", "adaptive", "offline"]
    x = np.arange(len(order))
    stat = [policy["policies"][p]["ttft_saved_ms_stationary"] for p in order]
    shift = [policy["policies"][p]["ttft_saved_ms_postshift"] for p in order]
    axC.bar(x - 0.2, stat, 0.4, label="stationary", color=PALETTE["gray"])
    axC.bar(x + 0.2, shift, 0.4, label="post-shift", color=PALETTE["red"])
    axC.set_xticks(x)
    axC.set_xticklabels(["LRU", "LFU", "adaptive", "offline"])
    axC.set_ylabel("realized TTFT saved (ms)")
    axC.set_title("(c) Eviction: adaptive vs LRU/LFU")
    axC.legend(fontsize=8)

    fig.suptitle(f"Prefix caching on real vLLM ({onoff['model']}, A10G): premise, budget, policy")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    print("wrote", save_figure(fig, "vllm_systems_combined"))


if __name__ == "__main__":
    main()
