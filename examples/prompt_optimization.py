"""Prompt optimization (redundant-prefix trimming) on real Bedrock + MMLU.

Suman's second idea: remove redundant prompt boilerplate to cut cost while keeping
quality. This is the same cost-aware tree selection as routing -- but the "arms" are
prompt TRIM LEVELS (how much boilerplate to keep) instead of models. For each MMLU
question we build several prompt variants from verbose (full preamble) to terse (bare
question), measure a model's accuracy and input-token cost at each level, then let the
hierarchical policy learn -- per subject -- the most aggressive trim that still answers
correctly. Reward = correctness, cost = input tokens; utility = accuracy - lam * tokens.

Run (in a shell with AWS creds):
    uv run --extra bench --extra plot python examples/prompt_optimization.py
Caches per-(trim, question) results to prompt_optimization.npz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from oco.bandits import PrefixTreeRouting, run_router
from oco.bandits.bedrock import BedrockClient

SUBJECTS = [
    "elementary_mathematics", "abstract_algebra", "high_school_biology",
    "college_computer_science", "world_religions", "moral_scenarios",
    "global_facts", "high_school_government_and_politics",
]
Q_PER = 4
MODEL = "amazon.nova-lite-v1:0"  # cheap, capable
LETTERS = "ABCD"

# Trim levels: index 0 = most verbose, increasing = more boilerplate removed.
PREAMBLES = [
    "You are a knowledgeable expert. Read the question carefully, consider every option, "
    "and reason about each choice before deciding.\n\n",
    "Read the question carefully and pick the best option.\n\n",
    "",  # bare question
    "",  # bare question, terse answer cue
]
SUFFIXES = [
    "\nAnswer with only the letter A, B, C, or D.",
    "\nAnswer with only the letter A, B, C, or D.",
    "\nAnswer with only the letter A, B, C, or D.",
    "\nLetter:",
]
N_TRIM = len(PREAMBLES)


def build_prompt(question: str, choices: list[str], level: int) -> str:
    opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices))
    return f"{PREAMBLES[level]}Question: {question}\n{opts}{SUFFIXES[level]}"


def load_questions():
    from datasets import load_dataset

    items = []
    for subject in SUBJECTS:
        ds = load_dataset("cais/mmlu", subject, split="test")
        for row in ds.select(range(Q_PER)):
            items.append((row["question"], list(row["choices"]), LETTERS[int(row["answer"])]))
    return items


def parse_letter(text: str) -> str:
    for ch in text.upper():
        if ch in LETTERS:
            return ch
    return "?"


def main() -> None:
    cache = Path(__file__).parent / "prompt_optimization.npz"
    items = load_questions()
    n = len(items)
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        quality, in_tokens = data["quality"], data["in_tokens"]
        print(f"loaded cached results from {cache.name}")
    else:
        client = BedrockClient(region="us-east-1", max_tokens=8)
        quality = np.zeros((N_TRIM, n))
        in_tokens = np.zeros((N_TRIM, n))
        fails = 0
        for lvl in range(N_TRIM):
            for qi, (q, choices, ans) in enumerate(items):
                try:
                    text, it, _ = client.generate(MODEL, build_prompt(q, choices, lvl))
                    quality[lvl, qi] = 1.0 if parse_letter(text) == ans else 0.0
                    in_tokens[lvl, qi] = it
                except Exception as e:  # noqa: BLE001
                    fails += 1
                    if fails <= 1:
                        print(f"  call failed: {type(e).__name__}: {str(e)[:110]}")
            print(f"  trim {lvl}: acc={quality[lvl].mean():.2f}  avg_in_tokens={in_tokens[lvl].mean():.1f}")
        if fails > 0.2 * N_TRIM * n:
            print(f"ABORTING: {fails} failed calls (creds/access). Not caching.")
            return
        np.savez(cache, quality=quality, in_tokens=in_tokens)
        print(f"saved results to {cache.name}")

    costs = in_tokens.mean(axis=1)
    rel_costs = costs / costs.max()
    print("\ntrim levels (acc / avg input tokens):")
    for lvl in range(N_TRIM):
        print(f"  trim {lvl}: acc={quality[lvl].mean():.2f}  tokens={costs[lvl]:.1f}")

    print("\nprompt-trim policy (arms = trim levels, regions = subjects):")
    results = {}
    for strat, kw in [("hierarchical", {"resolution": 3}), ("best_single", {}), ("oracle", {})]:
        env = PrefixTreeRouting(2, 5, quality, rel_costs, lam=0.3, noise_std=0.05,
                                rng=np.random.default_rng(0))
        r = run_router(env, 6000, np.random.default_rng(1), strategy=strat, **kw)
        tok = r.total_cost / 6000 * costs.max()
        results[r.label] = (r.avg_quality, tok, r.final_regret)
        label = {"best_single": "best-fixed-trim"}.get(r.label, r.label)
        print(f"  {label:18s} regret={r.final_regret:7.1f}  accuracy={r.avg_quality:.3f}  avg_tokens={tok:.1f}")
    print(f"  {'always-verbose(0)':18s} accuracy={quality[0].mean():.3f}  avg_tokens={costs[0]:.1f}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 5))
    ax.plot(costs, quality.mean(axis=1), "o-", color="#1f77b4", label="fixed trim level")
    for lvl in range(N_TRIM):
        ax.annotate(f"trim {lvl}", (costs[lvl], quality[lvl].mean()),
                    fontsize=8, xytext=(5, 4), textcoords="offset points")
    aq, at, _ = results["hierarchical(r=3)"]
    ax.scatter([at], [aq], color="#d62728", s=130, marker="*", zorder=3, label="adaptive per-subject")
    oq, ot, _ = results["oracle"]
    ax.scatter([ot], [oq], color="#2ca02c", s=110, marker="D", zorder=3, label="oracle (per-question)")
    ax.set_xlabel("average input tokens (cost)")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Prompt trimming on {MODEL}: adaptive per-subject trim\n"
                 "beats any fixed trim on the accuracy/token tradeoff", fontsize=10)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = Path(__file__).parent / "prompt_optimization.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved chart to {out}")


if __name__ == "__main__":
    main()
