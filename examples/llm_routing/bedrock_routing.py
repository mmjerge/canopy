"""Real-LLM routing over Bedrock (requires AWS creds + the `llm` extra).

End-to-end: call several Bedrock models on a set of prompts, grade the responses to get a
real per-model quality matrix, build a PrefixTreeRouting from it, and run the hierarchical
router -- showing it learns to route cheap models where they suffice.

Setup:
  1. cd terraform && terraform apply         # provisions the Bedrock invoke role + logging
  2. Enable model access in the Bedrock console for the models you list below.
  3. uv sync --extra llm
  4. uv run --extra llm python examples/llm_routing/bedrock_routing.py

This script is NOT run in CI (it needs credentials and incurs Bedrock cost). The grader
here is a trivial stand-in; swap in a real correctness check for your task.
"""

from __future__ import annotations

import numpy as np

from canopy.bandits import PrefixTreeRouting, run_router
from canopy.llm import BedrockClient, measure_quality_matrix

# 16 prompts -> a branching=4, depth=2 prefix tree (n_leaves = 16).
PROMPTS = [
    "What is 2+2?",
    "Capital of France?",
    "Translate 'hello' to Spanish.",
    "Define entropy in one sentence.",
    "Reverse the string 'banana'.",
    "Is 17 prime?",
    "Summarize photosynthesis in one line.",
    "What year did WWII end?",
    "Compute 12 * 13.",
    "Antonym of 'fast'?",
    "Name a primary color.",
    "What is the boiling point of water in C?",
    "Spell 'necessary'.",
    "Round 3.14159 to 2 decimals.",
    "What is the plural of 'mouse'?",
    "Convert 1 km to meters.",
]
MODEL_IDS = [
    "anthropic.claude-3-5-sonnet-20240620-v1:0",  # big / costly
    "amazon.nova-micro-v1:0",  # cheap / fast
]


def trivial_grade(prompt: str, response: str) -> float:
    """Placeholder grader: non-empty, reasonably concise answers score higher.

    Replace with a real correctness check (exact match, unit test, LLM-judge, ...).
    """
    if not response.strip():
        return 0.0
    n_words = len(response.split())
    return float(np.clip(1.0 - abs(n_words - 12) / 60.0, 0.2, 1.0))


def main() -> None:
    client = BedrockClient(region="us-east-1")  # or role_arn=<terraform bedrock_app_role_arn>
    quality, costs = measure_quality_matrix(PROMPTS, MODEL_IDS, client, trivial_grade)
    print("measured per-model avg quality:", quality.mean(axis=1).round(3))
    print("measured per-model avg cost/query (USD):", costs.round(5))

    env = PrefixTreeRouting(
        branching=4,
        depth=2,
        quality=quality,
        costs=costs,
        lam=50.0,
        noise_std=0.05,
        rng=np.random.default_rng(0),
    )
    router = run_router(
        env, horizon=4000, rng=np.random.default_rng(1), strategy="hierarchical", resolution=1
    )
    big = run_router(
        PrefixTreeRouting(4, 2, quality, costs, lam=50.0, rng=np.random.default_rng(0)),
        4000,
        np.random.default_rng(1),
        strategy="all_largest",
    )
    print(f"router:      regret={router.final_regret:.2f}  cost/query={router.total_cost/4000:.5f}")
    print(f"always-big:  regret={big.final_regret:.2f}  cost/query={big.total_cost/4000:.5f}")


if __name__ == "__main__":
    main()
