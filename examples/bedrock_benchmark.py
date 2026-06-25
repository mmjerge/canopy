"""Real Bedrock benchmark for prefix-tree routing.

Runs a small, objectively-gradeable benchmark (easy facts/arithmetic + harder multi-step
questions) across several Bedrock models, measures real per-model accuracy and token cost,
then routes over the resulting prefix tree and compares to fixed policies.

Easy questions are leaves 0-7, hard are 8-15, so the routing tree's regions correlate with
difficulty -- a good router should send easy prompts to the cheap model and hard ones to
the strong model.

Run:  uv run --extra llm --extra plot python examples/bedrock_benchmark.py
(Needs AWS creds with Bedrock access; caches results to bedrock_benchmark.npz.)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from oco.bandits import PrefixTreeRouting, run_router
from oco.bandits.bedrock import BedrockClient, measure_quality_matrix

# (prompt, answer) -- first 8 easy, last 8 hard.
BENCH: list[tuple[str, str]] = [
    ("What is 7 + 5? Answer with just the number.", "12"),
    ("What is the capital of Japan? One word.", "tokyo"),
    ("What is 9 times 6? Just the number.", "54"),
    ("How many days are in a week? Number only.", "7"),
    ("What color do you get mixing blue and yellow? One word.", "green"),
    ("What is the chemical symbol for water? Just the formula.", "h2o"),
    ("What is 100 divided by 4? Number only.", "25"),
    ("What planet is known as the Red Planet? One word.", "mars"),
    ("A train travels 60 miles in 1.5 hours. Average speed in mph? Number only.", "40"),
    ("Next number in the sequence 2, 6, 12, 20, 30, ? Number only.", "42"),
    ("A shirt costs $40 after a 20% discount. Original price in dollars? Number only.", "50"),
    ("How many prime numbers are between 10 and 20? Number only.", "4"),
    ("If today is Monday, what day is it in 100 days? One word.", "wednesday"),
    ("Sum of the interior angles of a pentagon in degrees? Number only.", "540"),
    ("At 3:15, the angle between a clock's hour and minute hands in degrees? Number only.", "7.5"),
    ("What is 15% of 240? Number only.", "36"),
]
MODELS = [
    "amazon.nova-pro-v1:0",        # strong, costly
    "amazon.nova-lite-v1:0",       # mid
    "mistral.mistral-small-2402-v1:0",
    "amazon.nova-micro-v1:0",      # cheap
]


def grade(prompt: str, response: str) -> float:
    answer = ANSWERS[PROMPTS.index(prompt)]
    r = response.lower().replace(",", "")
    if re.fullmatch(r"-?\d+(\.\d+)?", answer):
        nums = re.findall(r"-?\d+\.?\d*", r)
        return 1.0 if any(abs(float(x) - float(answer)) < 1e-6 for x in nums) else 0.0
    return 1.0 if answer.lower() in r else 0.0


PROMPTS = [q for q, _ in BENCH]
ANSWERS = [a for _, a in BENCH]


def main() -> None:
    cache = Path(__file__).parent / "bedrock_benchmark.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        quality, costs, models = data["quality"], data["costs"], list(data["models"])
        print(f"loaded cached results from {cache.name}")
    else:
        client = BedrockClient(region="us-east-1", max_tokens=256)
        quality, costs = measure_quality_matrix(PROMPTS, MODELS, client, grade)
        models = MODELS
        np.savez(cache, quality=quality, costs=costs, models=np.array(models))
        print(f"saved results to {cache.name}")

    print("\nper-model accuracy (overall / easy / hard) and cost:")
    for i, m in enumerate(models):
        acc, easy, hard = quality[i].mean(), quality[i, :8].mean(), quality[i, 8:].mean()
        print(f"  {m:34s} {acc:.2f} / {easy:.2f} / {hard:.2f}   cost/q={costs[i]:.6f}")

    # rescale costs to relative units so the cost term is comparable to accuracy gaps
    rel_costs = costs / costs.max()
    results = {}
    for strat, kw in [("hierarchical", {"resolution": 1}), ("best_single", {}),
                      ("all_largest", {}), ("oracle", {})]:
        env = PrefixTreeRouting(4, 2, quality, rel_costs, lam=0.3, noise_std=0.05,
                                rng=np.random.default_rng(0))
        r = run_router(env, 4000, np.random.default_rng(1), strategy=strat, **kw)
        results[r.label] = r
        print(f"  {r.label:18s} regret={r.final_regret:7.1f}  "
              f"avg_quality={r.avg_quality:.3f}  rel_cost/q={r.total_cost/4000:.3f}")


if __name__ == "__main__":
    main()
