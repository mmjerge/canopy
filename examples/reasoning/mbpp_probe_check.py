"""MBPP probe repair: is an execution-graded multi-test probe informative? (Stage 1+2)

The pre-registered code null used a single public assert as the cheap probe -- a
one-bit signal that plausible-but-wrong completions pass. Per the protocol in
docs/code_benchmarks.md, before re-running any search race we (Stage 1) *measure the
probe*: sample N candidate solutions per MBPP problem and record, for each, the cheap
probe score vs. the true hidden-suite pass -- their correlation is the code analog of
the cheap-vs-true node-value correlation on MATH (Table 8), the tree-Lipschitz-backbone
statistic. Then (Stage 2) we compare submission rules at identical samples:

  * first-sample     -- submit the first candidate (pass@1 reference);
  * one-assert probe -- submit the candidate passing the single public assert
                        (the null's probe);
  * multi-test probe -- submit the candidate with the highest fraction of the public
                        tests passed (the repaired, continuous probe).

The gate: if even the repaired probe's correlation is ~0, the theory says the search
race is not worth running on this benchmark (saturation aside); if it is informative,
the sweep can proceed with the same machinery as MATH.

Run (needs AWS creds + Bedrock model access; makes real paid calls):
    uv run --extra llm --extra bench python examples/reasoning/mbpp_probe_check.py \\
        --n-problems 50 --n-samples 8
"""

from __future__ import annotations

import argparse

import numpy as np

from canopy.bandits.code_llm import best_of_n_exec, extract_code, probe_truth_pairs, run_tests
from canopy.llm import as_generate_fn

MODEL_ID = "us.meta.llama3-1-70b-instruct-v1:0"

_PROMPT = (
    "Write a Python function for this task. Output only the code.\n\n"
    "Task: {text}\n\nYour function must satisfy this example:\n{example}\n\nCode:"
)


def load_mbpp(n: int):
    """MBPP problems: (prompt, public_tests, hidden_tests).

    The first assert is shown in the prompt (standard MBPP convention). The public
    probe gets the first two asserts; the hidden grade uses all three, so the probe is
    a strict subset of the truth (public-vs-hidden-suite structure).
    """
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        tests = list(row["test_list"])
        if len(tests) < 3:
            continue
        prompt = _PROMPT.format(text=row["text"], example=tests[0])
        setup = row.get("test_setup_code") or ""
        public = [setup + "\n" + t for t in tests[:2]]
        hidden = [setup + "\n" + t for t in tests]
        items.append((prompt, public, hidden))
    return items


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
        from canopy.llm import BedrockClient, BudgetError, CachingLLMClient

        client = CachingLLMClient(
            BedrockClient(region=args.region, max_tokens=512),
            args.cache,
            max_calls=args.max_calls,
            max_spend_usd=args.max_spend,
        )
        items = load_mbpp(args.n_problems)
    except Exception as e:  # noqa: BLE001
        print(
            f"Could not initialize ({type(e).__name__}: {e}).\n"
            "Needs: uv sync --extra llm --extra bench; AWS creds with Bedrock access."
        )
        return

    generate = as_generate_fn(client, args.model, temperature=0.7)
    print(f"MBPP: {len(items)} problems, {args.n_samples} samples each, model {args.model}")

    # Stage 1: probe-truth pairs pooled over problems and candidates
    all_pairs = []
    first_ok, onebit_ok, multi_ok = [], [], []
    try:
        for pi, (prompt, public, hidden) in enumerate(items):
            pairs = probe_truth_pairs(prompt, public, hidden, generate, n=args.n_samples)
            all_pairs.extend(pairs)

            # Stage 2 submission rules on the same samples (re-served from cache)
            codes = [
                extract_code(generate(prompt, 512, i)) for i in range(args.n_samples)
            ]
            first_ok.append(run_tests(codes[0], hidden) == 1.0)
            onebit = max(codes, key=lambda c: run_tests(c, public[:1]))
            onebit_ok.append(run_tests(onebit, hidden) == 1.0)
            multi = best_of_n_exec(prompt, public, hidden, generate, n=args.n_samples)
            multi_ok.append(multi.passed)
            if (pi + 1) % 10 == 0:
                print(
                    f"  {pi+1}/{len(items)}: first={np.mean(first_ok):.3f} "
                    f"one-assert={np.mean(onebit_ok):.3f} multi-test={np.mean(multi_ok):.3f}"
                )
    except BudgetError as e:
        print(f"\n[budget stop] {e}  Reporting problems completed so far.")

    pairs = np.array(all_pairs)
    if len(pairs) < 10:
        print("not enough measurements")
        return
    probe, truth = pairs[:, 0], pairs[:, 1]
    # rank (Spearman) correlation without a scipy dependency
    rp = np.corrcoef(np.argsort(np.argsort(probe)), np.argsort(np.argsort(truth)))[0, 1]
    r = np.corrcoef(probe, truth)[0, 1]

    print("\n[Stage 1: the gate] cheap probe vs hidden truth, per candidate:")
    print(f"  Pearson r = {r:.3f}   Spearman rho = {rp:.3f}   (n = {len(pairs)})")
    print(f"  candidates passing all public but failing hidden: "
          f"{float(np.mean((probe == 1.0) & (truth < 1.0))):.1%}  <- the probe's blind spot")
    print("\n[Stage 2] hidden-suite accuracy at identical samples:")
    print(f"  first-sample (pass@1 ref):   {np.mean(first_ok):.3f}")
    print(f"  one-assert probe (the null): {np.mean(onebit_ok):.3f}")
    print(f"  multi-test probe (repaired): {np.mean(multi_ok):.3f}")
    print(f"\n  client stats: {client.stats()}")
    print("\nGate reading: if Spearman rho is near 0, do not run the search race here "
          "(the theory predicts a null); if it is clearly positive, proceed to the "
          "matched-compute sweep with exec_value_fn as the value signal.")


if __name__ == "__main__":
    main()
