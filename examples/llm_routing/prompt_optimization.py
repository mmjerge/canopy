"""Prompt trimming (redundant-prefix removal) on real Bedrock + MMLU.

The "arms" are prompt TRIM LEVELS (how much boilerplate/preamble to keep), the "regions" are
MMLU subjects, and the tree bandit learns -- per subject -- the most aggressive trim that still
answers correctly. Reward = correctness; cost = input tokens; utility = accuracy - lam*tokens.
This is the same cost-aware tree selection as routing, with trim levels in place of models.

Real data: every (trim, question) pair is answered by a real Bedrock model; per-call results
are cached to a JSONL response cache and aggregated to an .npz so re-runs are free. The final
policy comparison, the LaTeX table, and the figure are written to ``paper/figures/`` (like the
routing experiments).

Run (AWS creds with Bedrock access):
    uv run --extra bench --extra plot python examples/llm_routing/prompt_optimization.py \
        --n-subjects 8 --q-per 8 --max-spend 5
Smoke-test the aggregation/plot/table with no deps or creds:
    python examples/llm_routing/prompt_optimization.py --mock
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from canopy.bandits import PrefixTreeRouting, run_router

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _plotstyle import FIGURE_DIR, PALETTE, save_figure, set_style  # noqa: E402

ALL_SUBJECTS = [
    "elementary_mathematics",
    "abstract_algebra",
    "high_school_biology",
    "college_computer_science",
    "world_religions",
    "moral_scenarios",
    "global_facts",
    "high_school_government_and_politics",
    "anatomy",
    "astronomy",
    "college_chemistry",
    "computer_security",
    "formal_logic",
    "high_school_mathematics",
    "machine_learning",
    "professional_medicine",
]
MODEL = "amazon.nova-lite-v1:0"  # cheap, capable
LETTERS = "ABCD"

# Trim levels: index 0 = most verbose (long redundant preamble), increasing = more removed.
# Every level keeps a PARSEABLE answer instruction, so accuracy differences reflect the trim,
# not a broken output format. Token savings come from the shrinking preamble.
PREAMBLES = [
    "You are a highly knowledgeable expert assistant. Carefully read the question below. "
    "Consider each of the answer options in turn, weigh the evidence for and against each, "
    "and think carefully before committing to the single best answer.\n\n",
    "You are a knowledgeable expert. Read the question carefully and consider every option "
    "before deciding.\n\n",
    "Read the question carefully and pick the best option.\n\n",
    "Answer the following multiple-choice question.\n\n",
    "",
    "",
]
SUFFIXES = [
    "\nAnswer with only the letter A, B, C, or D.",
    "\nAnswer with only the letter A, B, C, or D.",
    "\nAnswer with only the letter A, B, C, or D.",
    "\nAnswer with only the letter A, B, C, or D.",
    "\nAnswer with only the letter A, B, C, or D.",
    "\nAnswer (A/B/C/D):",  # terse but still parseable
]
N_TRIM = len(PREAMBLES)


def build_prompt(question: str, choices: list[str], level: int) -> str:
    opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices))
    return f"{PREAMBLES[level]}Question: {question}\n{opts}{SUFFIXES[level]}"


def load_questions(subjects: list[str], q_per: int):
    from datasets import load_dataset

    items = []
    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split="test")
        for row in ds.select(range(min(q_per, len(ds)))):
            items.append((row["question"], list(row["choices"]), LETTERS[int(row["answer"])]))
    return items


def parse_letter(text: str) -> str:
    for ch in text.upper():
        if ch in LETTERS:
            return ch
    return "?"


def measure_real(items, region, cache, max_calls, max_spend, npz_cache, checkpoint_every=25):
    """Answer every (trim, question) with real Bedrock; resumable via the .npz aggregate.

    Returns (quality[N_TRIM, n], in_tokens[N_TRIM, n]). Partial progress is written to the .npz
    every ``checkpoint_every`` calls so an interrupted or budget-stopped run resumes for free.
    """
    from canopy.llm import BedrockClient, BudgetError, CachingLLMClient

    n = len(items)
    if npz_cache.exists():
        data = np.load(npz_cache, allow_pickle=True)
        if data["quality"].shape == (N_TRIM, n):
            quality, in_tokens, done = data["quality"], data["in_tokens"], data["done"]
            if bool(done.all()):
                print(f"loaded complete cached results from {npz_cache.name}")
                return quality, in_tokens
            print(f"resuming from {npz_cache.name}: {int(done.sum())}/{done.size} cells done")
        else:
            quality = np.zeros((N_TRIM, n))
            in_tokens = np.zeros((N_TRIM, n))
            done = np.zeros((N_TRIM, n), dtype=bool)
    else:
        quality = np.zeros((N_TRIM, n))
        in_tokens = np.zeros((N_TRIM, n))
        done = np.zeros((N_TRIM, n), dtype=bool)

    client = CachingLLMClient(
        BedrockClient(region=region, max_tokens=8), cache, max_calls=max_calls,
        max_spend_usd=max_spend,
    )
    fails = 0
    since_ckpt = 0

    def _save():
        np.savez(npz_cache, quality=quality, in_tokens=in_tokens, done=done)

    try:
        for lvl in range(N_TRIM):
            for qi, (q, choices, ans) in enumerate(items):
                if done[lvl, qi]:
                    continue
                try:
                    text, it, _ = client.generate(MODEL, build_prompt(q, choices, lvl))
                    quality[lvl, qi] = 1.0 if parse_letter(text) == ans else 0.0
                    in_tokens[lvl, qi] = it
                    done[lvl, qi] = True
                except BudgetError:
                    raise
                except Exception as e:  # noqa: BLE001
                    fails += 1
                    if fails <= 1:
                        print(f"  call failed: {type(e).__name__}: {str(e)[:110]}")
                since_ckpt += 1
                if since_ckpt >= checkpoint_every:
                    _save()
                    since_ckpt = 0
            print(f"  trim {lvl}: acc={quality[lvl].mean():.2f} in_tok={in_tokens[lvl].mean():.1f}")
    except BudgetError as e:
        print(f"\n[budget stop] {e}  Saving progress; re-run with the same args to resume.")
        _save()
        raise
    _save()
    print(f"  budget: {client.stats()}")
    if fails > 0.2 * N_TRIM * n:
        raise RuntimeError(f"{fails} failed calls (creds/access?)")
    return quality, in_tokens


def measure_mock(items, seed=0):
    """Fabricate plausible (quality, in_tokens) so the aggregation/plot/table run with no deps."""
    rng = np.random.default_rng(seed)
    n = len(items)
    quality = np.zeros((N_TRIM, n))
    in_tokens = np.zeros((N_TRIM, n))
    # verbose prompts cost more tokens; accuracy stays ~flat until the terse final cue
    base_tokens = np.array([70.0, 55.0, 40.0, 30.0, 22.0, 20.0])
    base_acc = np.array([0.80, 0.81, 0.81, 0.80, 0.79, 0.72])
    for lvl in range(N_TRIM):
        quality[lvl] = (rng.random(n) < base_acc[lvl]).astype(float)
        in_tokens[lvl] = base_tokens[lvl] + rng.normal(0, 3, n)
    return quality, in_tokens


LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]


def _pad_to_pow2(mat, depth):
    n = mat.shape[1]
    width = 2**depth
    if width == n:
        return mat
    idx = np.arange(width - n) % n
    return np.concatenate([mat, mat[:, idx]], axis=1)


def _sweep_lambda(quality, in_tokens, n_subjects, lambdas, horizon=6000):
    """Sweep the cost weight lam; return per-policy (tokens, accuracy) frontiers.

    Pure post-hoc computation over the cached per-(trim, question) measurements -- no new model
    calls. Regions = subjects at resolution log2(n_subjects); phantom leaves are tiled in if the
    question count is not a power of two.
    """
    n = quality.shape[1]
    depth = max(1, int(np.ceil(np.log2(n))))
    q = _pad_to_pow2(quality, depth)
    tk = _pad_to_pow2(in_tokens, depth)
    resolution = max(1, min(depth, int(round(np.log2(max(1, n_subjects))))))
    costs = in_tokens.mean(axis=1)  # per-trim token cost (unpadded)
    rel_costs = tk.mean(axis=1) / tk.mean(axis=1).max()
    frontier = {"adaptive": [], "best_single": [], "oracle": []}
    for lam in lambdas:
        env = PrefixTreeRouting(2, depth, q, rel_costs, lam=lam, noise_std=0.05,
                                rng=np.random.default_rng(0))
        for strat, kw, key in [
            ("hierarchical", {"resolution": resolution}, "adaptive"),
            ("best_single", {}, "best_single"),
            ("oracle", {}, "oracle"),
        ]:
            r = run_router(env, horizon, np.random.default_rng(1), strategy=strat, **kw)
            frontier[key].append((r.total_cost / horizon * costs.max(), r.avg_quality))
    return costs, frontier


def _write_outputs(items, quality, in_tokens, model, n_subjects, quiet=False):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    acc = quality.mean(axis=1)
    costs, frontier = _sweep_lambda(quality, in_tokens, n_subjects, LAMBDAS)
    best_fixed = int(np.argmax(acc))
    li = LAMBDAS.index(0.3) if 0.3 in LAMBDAS else len(LAMBDAS) // 2
    adapt_tok, adapt_acc = frontier["adaptive"][li]
    orc_tok, orc_acc = frontier["oracle"][li]

    payload = {
        "model": model, "n_questions": len(items), "n_subjects": n_subjects, "lambdas": LAMBDAS,
        "trim_levels": [
            {"level": t, "accuracy": float(acc[t]), "avg_input_tokens": float(costs[t])}
            for t in range(N_TRIM)
        ],
        "frontier": {k: [[float(a), float(b)] for a, b in v] for k, v in frontier.items()},
    }
    (FIGURE_DIR / "prompt_trim_results.json").write_text(json.dumps(payload, indent=2))

    rows = [
        f"Always-verbose (trim 0) & {acc[0]:.3f} & {costs[0]:.1f} \\\\",
        f"Best fixed trim & {acc[best_fixed]:.3f} & {costs[best_fixed]:.1f} \\\\",
        f"\\textbf{{Adaptive per-subject (ours)}} & \\textbf{{{adapt_acc:.3f}}} & "
        f"\\textbf{{{adapt_tok:.1f}}} \\\\",
        f"Per-question oracle & {orc_acc:.3f} & {orc_tok:.1f} \\\\",
    ]
    tex = (
        "% Prompt trimming on real MMLU (auto-generated by prompt_optimization.py)\n"
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "Policy & Accuracy & Avg.\\ input tokens \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (FIGURE_DIR / "prompt_trim_table.tex").write_text(tex)

    if quiet:
        return
    print(f"\nprompt trimming on {model} ({len(items)} questions, {n_subjects} subjects):")
    for t in range(N_TRIM):
        print(f"  trim {t}: acc={acc[t]:.3f}  tokens={costs[t]:.1f}")
    print(f"  adaptive(lam=0.3): acc={adapt_acc:.3f} tok={adapt_tok:.1f}  |  "
          f"best-fixed: acc={acc[best_fixed]:.3f} tok={costs[best_fixed]:.1f}  |  "
          f"oracle: acc={orc_acc:.3f} tok={orc_tok:.1f}")
    try:
        out = _plot(acc, costs, frontier, model)
        print(f"\nwrote json+table to {FIGURE_DIR} and figure to {out} (+ .png)")
    except Exception as e:  # noqa: BLE001
        print(f"\nwrote json+table to {FIGURE_DIR} (figure skipped: {type(e).__name__}: {e})")


def _plot(acc, costs, frontier, model):
    set_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    order = np.argsort(costs)
    ax.plot(np.asarray(costs)[order], np.asarray(acc)[order], "o-", color=PALETTE["blue"],
            label="fixed trim level")
    for t in range(N_TRIM):
        ax.annotate(f"trim {t}", (costs[t], acc[t]), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    for key, col, mk, lab in [
        ("adaptive", PALETTE["red"], "*", "adaptive per-subject (ours)"),
        ("oracle", PALETTE["green"], "D", "per-question oracle"),
    ]:
        pts = sorted(frontier[key])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, color=col, label=lab)
    ax.set_xlabel("average input tokens (cost)")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Prompt trimming on {model}: adaptive frontier ($\\lambda$ sweep)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return str(save_figure(fig, "prompt_trim"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--n-subjects", type=int, default=len(ALL_SUBJECTS))
    ap.add_argument("--q-per", type=int, default=16, help="questions per subject")
    ap.add_argument("--max-calls", type=int, default=None, help="hard cap on API calls")
    ap.add_argument("--max-spend", type=float, default=None, help="hard cap on est. USD spend")
    ap.add_argument("--cache", default="examples/.cache/prompt_optimization.jsonl")
    ap.add_argument("--mock", action="store_true", help="fabricate data; no deps/creds")
    args = ap.parse_args()

    subjects = ALL_SUBJECTS[: args.n_subjects]
    npz_cache = Path(__file__).parent / "prompt_optimization.npz"

    if args.mock:
        items = [(f"q{i}", ["a", "b", "c", "d"], "A") for i in range(len(subjects) * args.q_per)]
        quality, in_tokens = measure_mock(items)
        model_label = "mock"
    else:
        try:
            items = load_questions(subjects, args.q_per)
            quality, in_tokens = measure_real(
                items, args.region, args.cache, args.max_calls, args.max_spend, npz_cache
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"Real run unavailable ({type(e).__name__}: {e}).\n"
                "Install extras and configure AWS:\n"
                "  uv sync --extra bench --extra plot\n"
                "Smoke-test the aggregation/plot/table with no deps: --mock"
            )
            return
        model_label = MODEL

    _write_outputs(items, quality, in_tokens, model_label, len(subjects))


if __name__ == "__main__":
    main()
