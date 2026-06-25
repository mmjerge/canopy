"""Real-LLM test: value-guided reasoning search vs. best-of-N on GSM8K, at equal compute.

This is the make-or-break experiment behind ``canopy.bandits.reasoning`` / ``reasoning_llm``:
does value-guided (edge-following) test-time search beat best-of-N on a *real* model? It
calls a model through Amazon Bedrock (the same client as the routing experiments), runs both
strategies on a GSM8K subset at a matched budget of generation calls, and reports accuracy.

Requires the optional extras and AWS access:
    uv sync --extra llm --extra bench
    # plus AWS credentials with Bedrock invoke permission and model access enabled
    uv run --extra llm --extra bench python examples/reasoning/gsm8k_reasoning_search.py

It makes real, paid model calls; start with a small ``--n-problems`` and small budgets.

Honest read: the synthetic result (``examples/reasoning/reasoning_search_demo.py``) shows
value-guided wins *when the value signal is informative at the decision steps*. Whether the
cheap rollout value (self-consistency, the default) is informative enough on real GSM8K traces
is exactly
what this measures -- a negative result is also informative (it would say the value signal,
not the search, is the bottleneck).
"""

from __future__ import annotations

import argparse

from canopy.bandits.reasoning_llm import best_of_n, value_guided_search

MODEL_ID = "us.meta.llama3-1-8b-instruct-v1:0"


def load_gsm8k(n: int):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for row in ds.select(range(n)):
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        items.append((row["question"], gold))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-problems", type=int, default=20)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument(
        "--bo-n",
        type=int,
        default=0,
        help="best-of-N sample count; 0 = auto-match value-guided's call budget",
    )
    ap.add_argument("--branching", type=int, default=3)
    ap.add_argument("--n-steps", type=int, default=4)
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--final-rollouts", type=int, default=5)
    args = ap.parse_args()

    try:
        from canopy.bandits.bedrock import BedrockClient

        client = BedrockClient(region=args.region, max_tokens=512)
        problems = load_gsm8k(args.n_problems)
    except Exception as e:  # noqa: BLE001 -- missing extras / creds / model access
        print(
            f"Could not initialize the real-LLM experiment: {e}\n"
            "Install extras and configure AWS:\n"
            "  uv sync --extra llm --extra bench\n"
            "  (AWS creds with Bedrock invoke permission + model access)\n"
            "The search logic itself is exercised by tests/test_reasoning_llm.py (mock LLM)."
        )
        return

    def generate(prompt: str, max_tokens: int, seed: int) -> str:
        text, _, _ = client.generate(args.model, prompt, temperature=0.7, max_tokens=max_tokens)
        return text

    # value-guided's total generation calls, used to match best-of-N's sample count
    vg_budget = args.n_steps * (args.branching * (1 + args.rollouts)) + args.final_rollouts
    bo_n = vg_budget if args.bo_n == 0 else args.bo_n

    bo_correct = vg_correct = 0
    bo_calls = vg_calls = 0
    bo_tokens = vg_tokens = 0
    for i, (q, gold) in enumerate(problems):
        bo = best_of_n(q, gold, generate, n=bo_n)
        vg = value_guided_search(
            q,
            gold,
            generate,
            branching=args.branching,
            n_steps=args.n_steps,
            rollouts=args.rollouts,
            final_rollouts=args.final_rollouts,
        )
        bo_correct += bo.correct
        vg_correct += vg.correct
        bo_calls += bo.budget.calls
        vg_calls += vg.budget.calls
        bo_tokens += bo.budget.tokens
        vg_tokens += vg.budget.tokens
        print(
            f"[{i+1}/{len(problems)}] gold={gold:>6}  best-of-N={bo.answer} ({bo.correct})  "
            f"value-guided={vg.answer} ({vg.correct})"
        )

    n = len(problems)
    print(f"\nGSM8K ({n} problems), matched budget ~{bo_n} calls, model {args.model}")
    print(
        f"  best-of-N      : acc {bo_correct/n:.2f}   avg calls {bo_calls/n:.1f}   "
        f"avg tokens {bo_tokens/n:.0f}"
    )
    print(
        f"  value-guided   : acc {vg_correct/n:.2f}   avg calls {vg_calls/n:.1f}   "
        f"avg tokens {vg_tokens/n:.0f}"
    )
    diff = (vg_correct - bo_correct) / n
    se = ((bo_correct / n) * (1 - bo_correct / n) / n) ** 0.5 + 1e-12
    print(
        f"  acc diff (vg - bo): {diff:+.3f}  (~{abs(diff) / se:.1f} SE; "
        f"n={n} is small, treat |diff| < ~{2 * se:.2f} as within noise)"
    )


if __name__ == "__main__":
    main()
