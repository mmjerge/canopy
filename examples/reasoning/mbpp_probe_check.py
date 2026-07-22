"""MBPP probe repair: is an execution-graded multi-test probe informative? (Stage 1+2)

The pre-registered code null used a single public assert as the cheap probe -- a one-bit,
high-variance signal that plausible-but-wrong completions pass. Per the ladder in
docs/code_benchmarks.md, before re-running any search race we (Stage 1) *measure the probe*:
sample N candidate solutions per MBPP problem and record, for each, the cheap probe score vs.
the true hidden-suite pass -- their correlation is the code analog of the cheap-vs-true
node-value correlation on MATH, the tree-Lipschitz-backbone statistic. Then (Stage 2) we
compare submission rules at identical samples:

  * first-sample     -- submit the first candidate (pass@1 reference);
  * one-assert probe -- submit by the single public assert (the null's probe);
  * multi-test probe -- submit by the continuous fraction of public asserts passed
                        (``code_eval.public_fraction``, the repaired probe).

The gate: if even the repaired probe's correlation is ~0, the theory says the search race is
not worth running on this benchmark (saturation aside); if it is informative, the sweep can
proceed with ``code_eval.public_test_value`` as the value signal in reasoning_search.py.

Run (needs AWS creds + Bedrock model access; makes real paid calls):
    uv run --extra llm --extra bench python examples/reasoning/mbpp_probe_check.py \\
        --n-problems 50 --n-samples 8 --max-spend 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from canopy.bandits.code_eval import (  # noqa: E402
    extract_code,
    grade_code,
    probe_truth_pairs,
    public_fraction,
    passes,
)

MODEL_ID = "us.meta.llama3-1-70b-instruct-v1:0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-problems", type=int, default=50)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--max-spend", type=float, default=None)
    ap.add_argument("--cache", default="examples/.cache/mbpp_probe.jsonl")
    args = ap.parse_args()

    try:
        from reasoning_search import load_mbpp  # reuse the benchmark's gold-dict loader

        from canopy.llm import BedrockClient, BudgetError, CachingLLMClient, as_generate_fn

        client = CachingLLMClient(
            BedrockClient(region=args.region, max_tokens=512),
            args.cache,
            max_calls=args.max_calls,
            max_spend_usd=args.max_spend,
        )
        items = load_mbpp(args.n_problems)
        # widen the public probe to the first TWO asserts (the sweep's loader keeps one --
        # the null's probe; the repair under test is the multi-test continuous score).
        # the hidden suite (all asserts) is untouched.
        widened = []
        for prompt, gold in items:
            tests = [
                ln for ln in gold["hidden_suffix"].splitlines() if ln.strip().startswith("assert")
            ]
            if len(tests) < 3:
                continue
            widened.append((prompt, dict(gold, public=tests[:2])))
        items = widened
        generate = as_generate_fn(client, args.model, temperature=0.7)
    except Exception as e:  # noqa: BLE001
        print(
            f"Could not initialize ({type(e).__name__}: {e}).\n"
            "Needs: uv sync --extra llm --extra bench; AWS creds with Bedrock access."
        )
        return

    print(f"MBPP: {len(items)} problems, {args.n_samples} samples each, model {args.model}")

    all_pairs: list[tuple[float, float]] = []
    first_ok, onebit_ok, multi_ok = [], [], []
    try:
        for pi, (prompt, gold) in enumerate(items):
            texts = [generate(prompt, 512, i) for i in range(args.n_samples)]
            all_pairs.extend(probe_truth_pairs(texts, gold))
            codes = [extract_code(t) for t in texts]

            first_ok.append(grade_code(codes[0], gold))
            pub = gold.get("public", [])
            onebit = max(codes, key=lambda c: float(passes(c, gold, pub[:1], 5.0)))
            onebit_ok.append(grade_code(onebit, gold))
            multi = max(codes, key=lambda c: public_fraction(c, gold))
            multi_ok.append(grade_code(multi, gold))
            if (pi + 1) % 10 == 0:
                print(
                    f"  {pi + 1}/{len(items)}: first={np.mean(first_ok):.3f} "
                    f"one-assert={np.mean(onebit_ok):.3f} multi-test={np.mean(multi_ok):.3f}"
                )
    except BudgetError as e:
        print(f"\n[budget stop] {e}  Reporting problems completed so far.")

    pairs = np.array(all_pairs)
    if len(pairs) < 10:
        print("not enough measurements")
        return
    probe, truth = pairs[:, 0], pairs[:, 1]
    rp = np.corrcoef(np.argsort(np.argsort(probe)), np.argsort(np.argsort(truth)))[0, 1]
    r = np.corrcoef(probe, truth)[0, 1]

    print("\n[Stage 1: the gate] cheap probe vs hidden truth, per candidate:")
    print(f"  Pearson r = {r:.3f}   Spearman rho = {rp:.3f}   (n = {len(pairs)})")
    print(
        f"  candidates passing all public but failing hidden: "
        f"{float(np.mean((probe == 1.0) & (truth < 1.0))):.1%}  <- the probe's blind spot"
    )
    print("\n[Stage 2] hidden-suite accuracy at identical samples:")
    print(f"  first-sample (pass@1 ref):   {np.mean(first_ok):.3f}")
    print(f"  one-assert probe (the null): {np.mean(onebit_ok):.3f}")
    print(f"  multi-test probe (repaired): {np.mean(multi_ok):.3f}")
    print(f"\n  client stats: {client.stats()}")
    print(
        "\nGate reading: if Spearman rho is near 0, do not run the search race here "
        "(the theory predicts a null); if it is clearly positive, proceed to the "
        "matched-compute sweep (reasoning_search.py --benchmark mbpp)."
    )


if __name__ == "__main__":
    main()
