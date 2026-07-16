"""SWE-bench Lite Stage-1 gate: is the cheap localization probe informative?

Rung 3 of the code-benchmark ladder (docs/code_benchmarks.md) casts repo-level issue
resolution as the multi-fidelity tree: root -> files -> edit regions -> patches, with
retrieval as the cheap probe at the localization levels and the test suite as the
expensive leaf evaluation. Per the pre-registered protocol, before spending any model
compute we measure whether the cheap probe is informative at the top of the tree:

    Does BM25 retrieval (the standard SWE-bench cheap context probe) rank the files
    that the gold patch actually touches?

This is fully offline: princeton-nlp/SWE-bench_Lite provides the gold patches, and the
princeton-nlp/SWE-bench_Lite_bm25_13K variant provides the retrieved file set per
instance. We report recall@k of gold-patch files in the retrieval ranking, MRR, and
the fraction of instances where the probe localizes at least one gold file -- the
file-level analog of the pivotal-step hit rate on MATH (Table 8). If these are near
the random-file baseline, the theory predicts value-guided repo search cannot pay and
the race should not be run; if they are clearly informative, the edit-level gate
(which needs model calls) is next.

Run with:  uv run --extra bench --extra plot python examples/analysis/swebench_stage1.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_DIFF_FILE = re.compile(r"^diff --git a/(\S+) b/\S+", re.MULTILINE)
_RETRIEVED = re.compile(r"\[start of ([^\]]+)\]")


def gold_files(patch: str) -> list[str]:
    """Files modified by the gold patch."""
    return list(dict.fromkeys(_DIFF_FILE.findall(patch)))


def retrieved_files(text: str) -> list[str]:
    """Files included by the BM25 retriever, in ranking order."""
    return list(dict.fromkeys(_RETRIEVED.findall(text)))


def main() -> None:
    from datasets import load_dataset

    lite = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    bm25 = load_dataset("princeton-nlp/SWE-bench_Lite_bm25_13K", split="test")
    gold_by_id = {row["instance_id"]: gold_files(row["patch"]) for row in lite}

    recalls_any, rr, n_retrieved, n_gold = [], [], [], []
    recall_at = {1: [], 3: [], 5: []}
    for row in bm25:
        gold = gold_by_id.get(row["instance_id"])
        if not gold:
            continue
        ranked = retrieved_files(row["text"])
        n_retrieved.append(len(ranked))
        n_gold.append(len(gold))
        hits = [i for i, f in enumerate(ranked) if f in gold]
        recalls_any.append(bool(hits))
        rr.append(1.0 / (hits[0] + 1) if hits else 0.0)
        for k in recall_at:
            topk = set(ranked[:k])
            recall_at[k].append(len(topk & set(gold)) / len(gold))

    n = len(recalls_any)
    print(f"SWE-bench Lite Stage-1 localization gate ({n} instances)")
    print(f"  files retrieved per instance (mean): {np.mean(n_retrieved):.1f}")
    print(f"  gold-patch files per instance (mean): {np.mean(n_gold):.2f}")
    print(f"  P(>=1 gold file retrieved)  = {np.mean(recalls_any):.3f}   <- the gate")
    print(f"  MRR of first gold file      = {np.mean(rr):.3f}")
    for k, vals in recall_at.items():
        print(f"  recall@{k} (gold files)      = {np.mean(vals):.3f}")
    # a random-file probe would land a gold file in a ~13K-token context with
    # probability ~ (files that fit) / (files in repo) -- typically a few percent;
    # the measured gate value should be read against that baseline.
    print(
        "\nGate reading: clearly above the few-percent random-file baseline -> the\n"
        "localization levels of the tree have an informative cheap probe, and the\n"
        "edit-level gate (model calls; docs/code_benchmarks.md Stage 1) is worth\n"
        "running. Near the baseline -> stop; the tree prior has no signal to follow."
    )


if __name__ == "__main__":
    main()
