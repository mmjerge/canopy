"""Real MMLU-by-subject routing benchmark over diverse Bedrock models.

Each MMLU subject is a prefix region of the routing tree, so the router can learn which
model to use per subject. Models span providers and tiers (Llama 70B/8B, Nova Pro/Lite/
Micro, Mistral-Small) so they have genuinely complementary strengths -- the condition the
earlier toy benchmark lacked.

Run:  uv run --extra bench --extra plot python examples/llm_routing/mmlu_routing.py
Needs AWS creds with Bedrock access. Caches per-model results to mmlu_routing.npz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from canopy.bandits import PrefixTreeRouting, run_router
from canopy.llm import DEFAULT_PRICING, BedrockClient

SUBJECTS = [
    "elementary_mathematics",
    "abstract_algebra",
    "high_school_biology",
    "college_computer_science",
    "world_religions",
    "moral_scenarios",
    "global_facts",
    "high_school_government_and_politics",
]
Q_PER = 4  # questions per subject -> 8 subjects * 4 = 32 leaves (branching=2, depth=5)
MODELS = [
    "us.meta.llama3-1-70b-instruct-v1:0",
    "us.meta.llama3-1-8b-instruct-v1:0",
    "amazon.nova-pro-v1:0",
    "amazon.nova-lite-v1:0",
    "amazon.nova-micro-v1:0",
    "mistral.mistral-small-2402-v1:0",
]
LETTERS = "ABCD"


def load_questions() -> tuple[list[str], list[str]]:
    from datasets import load_dataset

    prompts, answers = [], []
    for subject in SUBJECTS:
        ds = load_dataset("cais/mmlu", subject, split="test")
        for row in ds.select(range(Q_PER)):
            opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(row["choices"]))
            prompts.append(
                f"Question: {row['question']}\n{opts}\n"
                "Answer with only the letter A, B, C, or D."
            )
            answers.append(LETTERS[int(row["answer"])])
    return prompts, answers


def parse_letter(text: str) -> str:
    for ch in text.upper():
        if ch in LETTERS:
            return ch
    return "?"


def main() -> None:
    cache = Path(__file__).parent / "mmlu_routing.npz"
    prompts, answers = load_questions()
    n = len(prompts)
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        quality, costs = data["quality"], data["costs"]
        print(f"loaded cached results from {cache.name}")
    else:
        client = BedrockClient(region="us-east-1", max_tokens=8)
        quality = np.zeros((len(MODELS), n))
        costs = np.zeros(len(MODELS))
        total_fails = 0
        for mi, model in enumerate(MODELS):
            in_p, out_p = DEFAULT_PRICING.get(model, (0.001, 0.001))
            fails = 0
            for pi, prompt in enumerate(prompts):
                try:
                    text, it, ot = client.generate(model, prompt)
                    quality[mi, pi] = 1.0 if parse_letter(text) == answers[pi] else 0.0
                    costs[mi] += in_p * it / 1000 + out_p * ot / 1000
                except Exception as e:  # noqa: BLE001
                    fails += 1
                    if fails <= 1:
                        print(f"   {model} call failed: {type(e).__name__}: {str(e)[:120]}")
            costs[mi] /= n
            total_fails += fails
            print(
                f"  {model:36s} acc={quality[mi].mean():.2f}  cost/q={costs[mi]:.6f}  fails={fails}"
            )
        if total_fails > 0.2 * n * len(MODELS):
            print(
                f"\nABORTING: {total_fails} calls failed (likely expired creds or model "
                "access). Not caching. Refresh creds and re-run."
            )
            return
        np.savez(cache, quality=quality, costs=costs, models=np.array(MODELS))
        print(f"saved results to {cache.name}")

    # per-subject accuracy: do models have complementary strengths?
    print("\nper-subject accuracy (rows=models):")
    print("  " + "  ".join(f"{s[:10]:>10s}" for s in SUBJECTS))
    for mi, model in enumerate(MODELS):
        per_sub = [quality[mi, s * Q_PER : (s + 1) * Q_PER].mean() for s in range(len(SUBJECTS))]
        print(f"  {model.split('.')[-1][:14]:14s} " + " ".join(f"{v:10.2f}" for v in per_sub))

    rel_costs = costs / costs.max()
    print("\nrouting (branching=2, depth=5, region=subject at resolution 3):")
    results = {}
    for strat, kw in [
        ("hierarchical", {"resolution": 3}),
        ("best_single", {}),
        ("all_largest", {}),
        ("oracle", {}),
    ]:
        env = PrefixTreeRouting(
            2, 5, quality, rel_costs, lam=0.3, noise_std=0.05, rng=np.random.default_rng(0)
        )
        r = run_router(env, 6000, np.random.default_rng(1), strategy=strat, **kw)
        results[r.label] = r
        print(
            f"  {r.label:18s} regret={r.final_regret:7.1f}  "
            f"avg_quality={r.avg_quality:.3f}  rel_cost/q={r.total_cost/6000:.3f}"
        )

    _plot(results, quality)


def _plot(results: dict, quality: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "hierarchical(r=3)": "#d62728",
        "best_single": "#1f77b4",
        "all_largest": "#ff7f0e",
        "oracle": "#2ca02c",
    }
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))
    for label, r in results.items():
        axA.plot(
            np.arange(1, len(r.cum_regret) + 1),
            r.cum_regret,
            color=colors.get(label),
            lw=2,
            label=label,
        )
    axA.set_xlabel("prompts seen")
    axA.set_ylabel("cumulative routing regret")
    axA.set_title("Real MMLU routing: hierarchical beats fixed policies", fontsize=10)
    axA.grid(True, ls=":", alpha=0.5)
    axA.legend(loc="upper left", fontsize=8)
    for label, r in results.items():
        axB.scatter(
            [r.total_cost / len(r.cum_regret)],
            [r.avg_quality],
            color=colors.get(label),
            s=110,
            marker="*" if label == "oracle" else "o",
            zorder=3,
        )
        axB.annotate(
            label,
            (r.total_cost / len(r.cum_regret), r.avg_quality),
            fontsize=8,
            xytext=(5, 4),
            textcoords="offset points",
        )
    axB.set_xlabel("relative cost per query")
    axB.set_ylabel("average quality (accuracy)")
    axB.set_title("Cost vs quality across diverse Bedrock models", fontsize=10)
    axB.grid(True, ls=":", alpha=0.5)
    fig.suptitle("Routing over real MMLU subjects (8 subjects, 6 Bedrock models)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(__file__).parent.parent / "images" / "mmlu_routing.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved chart to {out}")


if __name__ == "__main__":
    main()
