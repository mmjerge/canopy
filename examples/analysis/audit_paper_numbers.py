"""Independent audit of every headline number in the paper against raw artifacts.

Extends audit_swebench_results.py (which covers the SWE-bench families) to the remaining
Table 1 rows and headline prose claims. Each check recomputes the claimed value from the
raw per-item results JSON (never from the generated .tex tables) and compares at the
paper's displayed precision. Families with no raw JSON in the repository (MMLU routing,
top-k identification, theory link: generated tables only, raw caches on the GPU box) are
checked against their generated artifacts and flagged as artifact-level checks.

Run:  python examples/analysis/audit_paper_numbers.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FIG = Path(__file__).resolve().parents[2] / "paper" / "figures"
FAILS: list[str] = []


def check(name: str, got: float, want: float, tol: float) -> None:
    ok = abs(got - want) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: recomputed {got:.4f} vs paper {want}")
    if not ok:
        FAILS.append(name)


def j(name: str) -> dict:
    return json.loads((FIG / name).read_text())


def paired_delta_ci(bo, vg, iters=20000, seed=7):
    bo, vg = np.asarray(bo, float), np.asarray(vg, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, bo.size, size=(iters, bo.size))
    d = vg[idx].mean(axis=1) - bo[idx].mean(axis=1)
    return float((vg - bo).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def reasoning(fname: str, budget: str):
    d = j(fname)
    lv = d["levels"][budget]
    return lv["best_of_n"]["hits"], lv["value_guided"]["hits"]


def main() -> None:
    print("== Test-time search (Table 1 + RQ3 prose) ==")
    # Table 1 cells: MATH at B=59, GPQA at B=41 (budgets now stated in the table; the full
    # budget curves for every benchmark are in the appendix tables).
    bo, vg = reasoning("reasoning_search_math_results.json", "59")
    delta, lo, hi = paired_delta_ci(bo, vg)
    check("MATH llama70b vg acc (B=59)", float(np.mean(vg)), 0.617, 0.0005)
    check("MATH llama70b bo acc (B=59)", float(np.mean(bo)), 0.530, 0.0005)
    check("MATH delta", delta, 0.087, 0.0006)
    check("MATH CI lo", lo, 0.04, 0.011)
    check("MATH CI hi", hi, 0.14, 0.011)

    bo, vg = reasoning("reasoning_search_gpqa_diamond_results.json", "41")
    delta, lo, hi = paired_delta_ci(bo, vg)
    check("GPQA llama70b vg acc (B=41)", float(np.mean(vg)), 0.424, 0.0005)
    check("GPQA llama70b bo acc (B=41)", float(np.mean(bo)), 0.313, 0.0005)
    check("GPQA delta", delta, 0.111, 0.0006)
    check("GPQA CI lo", lo, 0.04, 0.011)
    check("GPQA CI hi", hi, 0.19, 0.011)

    # Within-model contrast and unreachable-regime cells use each benchmark's LARGEST budget.
    for bench, budget, want in [
        ("gsm8k", "50", -0.005),
        ("math", "95", 0.333),
        ("gpqa_diamond", "77", 0.359),
    ]:
        bo, vg = reasoning(f"reasoning_search_{bench}_sonnet45_results.json", budget)
        check(f"within-model sonnet {bench} delta", float(np.mean(vg) - np.mean(bo)), want, 0.0006)

    bo, vg = reasoning("reasoning_search_math_llama8b_results.json", "95")
    check("llama8b MATH delta (unreachable)", float(np.mean(vg) - np.mean(bo)), -0.147, 0.0006)

    bo, vg = reasoning("reasoning_search_gsm8k_results.json", "50")
    check("GSM8K llama70b delta (~0)", float(np.mean(vg) - np.mean(bo)), 0.0, 0.0006)

    for bench, lo_w, hi_w in [("humaneval", 0.01, 0.02), ("mbpp", -0.07, -0.04)]:
        d = j(f"reasoning_search_{bench}_results.json")
        deltas = []
        for lv in d["levels"].values():
            deltas.append(
                float(np.mean(lv["value_guided"]["hits"]) - np.mean(lv["best_of_n"]["hits"]))
            )
        ok = min(deltas) >= lo_w - 0.006 and max(deltas) <= hi_w + 0.006
        print(
            f"  [{'PASS' if ok else 'FAIL'}] "
            f"{bench} null range: recomputed [{min(deltas):+.3f},{max(deltas):+.3f}] vs paper "
            f"[{lo_w:+.2f},{hi_w:+.2f}]"
        )
        if not (min(deltas) >= lo_w - 0.006 and max(deltas) <= hi_w + 0.006):
            FAILS.append(f"{bench} null range")

    print("== RQ4 prior measurement ==")
    t = j("reasoning_tree_lipschitz_math_results.json")
    check("MATH pivotal edge-hit", t["edge_hit_pivotal"], 0.73, 0.005)
    check("MATH chance rate", t["random_edge_hit_rate"], 0.33, 0.005)
    s = j("swebench_tree_lipschitz_results.json")
    check("SWE tree spearman", s["spearman"], 0.86, 0.005)
    check("SWE pivotal fraction", s["n_pivotal"] / s["n_steps_total"], 0.10, 0.005)
    check("SWE pivotal edge-hit (18/18)", s["edge_hit_pivotal"], 1.0, 1e-9)
    check("SWE n candidates", s["n_candidates"], 540, 0.5)

    print("== tau-bench (5-replicate study) ==")
    reg, flat = [], []
    files = ["taubench_routing_results.json"] + [
        f"taubench_routing_rep{i}_results.json" for i in range(2, 6)
    ]
    for f in files:
        r = j(f)["results"]
        reg.append(r["routed-regional (ours)"]["success"])
        flat.append(r["routed-flat"]["success"])
    reg, flat = np.array(reg), np.array(flat)
    check("tau regional mean", reg.mean(), 0.463, 0.0006)
    check("tau regional sd", reg.std(ddof=1), 0.156, 0.0006)
    check("tau flat mean", flat.mean(), 0.323, 0.0006)
    check("tau flat sd", flat.std(ddof=1), 0.213, 0.0006)
    check("tau gap", (reg - flat).mean(), 0.14, 0.006)
    check("tau wins (4/5)", float((reg > flat).sum()), 4, 0.5)

    print("== Prefix caching / systems ==")
    # Note: an earlier draft's "5.23 vs 3.32" prompt-stream row cited a post-shift TRANSIENT
    # from a superseded analysis; the current artifact's steady-state values are below, and
    # the paper now reports the steady state (tie with LFU, both >> LRU) with the transient
    # shown in the figure. This audit finding removed that row from Table 1.
    p = j("prefix_cache_results.json")
    check(
        "prompt stream adaptive post-shift (steady)",
        p["shift_final_savings"]["adaptive"],
        1.28,
        0.006,
    )
    check("prompt stream LFU post-shift (steady)", p["shift_final_savings"]["lfu"], 1.25, 0.006)
    check("prompt stream LRU post-shift (stale)", p["shift_final_savings"]["lru"], 0.32, 0.006)
    m = j("prefix_cache_mooncake_toolagent_results.json")
    check("Mooncake adaptive B=16", m["shift_final_savings"]["adaptive"], 5.39, 0.006)
    check("Mooncake LRU B=16", m["shift_final_savings"]["lru"], 3.02, 0.006)
    check("Mooncake offline (=ours)", m["shift_final_savings"]["offline"], 5.39, 0.006)
    v = j("vllm_prefix_cache_results.json")
    check("vLLM TTFT p50 on", v["configs"]["on"]["ttft_p50"], 0.267, 0.0006)
    check("vLLM TTFT p50 off", v["configs"]["off"]["ttft_p50"], 0.961, 0.0006)
    check(
        "vLLM TTFT ratio 3.6x",
        v["configs"]["off"]["ttft_p50"] / v["configs"]["on"]["ttft_p50"],
        3.6,
        0.05,
    )
    check("vLLM throughput on", v["configs"]["on"]["throughput_tok_s"], 678, 0.5)
    check("vLLM throughput off", v["configs"]["off"]["throughput_tok_s"], 181, 0.5)
    e = j("vllm_policy_results.json")
    check(
        "eviction adaptive post-shift",
        e["policies"]["adaptive"]["ttft_saved_ms_postshift"],
        23.6,
        0.06,
    )
    check("eviction LRU post-shift", e["policies"]["lru"]["ttft_saved_ms_postshift"], 12.9, 0.06)

    print("== Trimming ==")
    lb = j("longbench_trim_results.json")
    li = lb["lambdas"].index(0.3)
    check("LongBench adaptive F1", lb["frontier"]["adaptive"][li][1], 0.362, 0.0006)
    check("LongBench adaptive tok", lb["frontier"]["adaptive"][li][0], 3544, 0.6)
    fx = [t for t in lb["trim_levels"] if t["keep_fraction"] == 0.5][0]
    check("LongBench fixed-0.5 F1", fx["f1"], 0.317, 0.0006)
    check("LongBench fixed-0.5 tok", fx["avg_input_tokens"], 3640, 0.6)
    check("LongBench full F1", lb["trim_levels"][0]["f1"], 0.440, 0.0006)
    lr = j("longbench_trim_retrieval_results.json")
    li = lr["lambdas"].index(0.3)
    check("LongBench retrieval adaptive F1", lr["frontier"]["adaptive"][li][1], 0.407, 0.0006)
    check("LongBench retrieval adaptive tok", lr["frontier"]["adaptive"][li][0], 2701, 0.6)
    fx = [t for t in lr["trim_levels"] if t["keep_fraction"] == 0.5][0]
    check("LongBench retrieval fixed-0.5 F1", fx["f1"], 0.417, 0.0006)
    bb = j("bbh_trim_results.json")
    li = bb["lambdas"].index(0.3) if 0.3 in bb["lambdas"] else len(bb["lambdas"]) // 2
    check("BBH adaptive acc", bb["frontier"]["adaptive"][li][1], 0.368, 0.0006)
    check("BBH adaptive tok", bb["frontier"]["adaptive"][li][0], 291, 0.6)
    pt = j("prompt_trim_results.json")
    li = pt["lambdas"].index(0.3) if 0.3 in pt["lambdas"] else len(pt["lambdas"]) // 2
    check("MMLU trim adaptive acc", pt["frontier"]["adaptive"][li][1], 0.668, 0.0006)
    check("MMLU trim adaptive tok", pt["frontier"]["adaptive"][li][0], 107, 0.6)

    print("== Artifact-level checks (no raw JSON in repo; caches on the GPU box) ==")
    topk = (FIG / "real_topk_identification_table.tex").read_text()
    for val in ["0.370", "0.126", "0.563", "0.428"]:
        ok = val in topk
        print(f"  [{'PASS' if ok else 'FAIL'}] top-k table contains {val}")
        if not ok:
            FAILS.append(f"topk {val}")
    mm = (FIG / "mmlu_routing_table.tex").read_text()
    for val in ["0.977", "0.254", "0.943", "0.363"]:
        ok = val in mm
        print(f"  [{'PASS' if ok else 'FAIL'}] MMLU routing table contains {val}")
        if not ok:
            FAILS.append(f"mmlu {val}")

    print(f"\nAUDIT {'PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")


if __name__ == "__main__":
    main()
