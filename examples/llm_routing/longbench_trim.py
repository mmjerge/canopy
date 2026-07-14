"""Context trimming on real Bedrock + LongBench (long-context prompt compression).

The prompt-trim task, on a benchmark built for it. The "arms" are context-keep fractions (how
much of a long retrieved context to keep), the "regions" are LongBench task categories, and the
tree bandit learns --- per task --- the most aggressive trim that still answers correctly. Reward
= QA-F1 against gold; cost = input tokens; utility = F1 - lam * tokens. Long contexts give real
trimming headroom, and different tasks tolerate different trims (answers early vs. spread through
the context), so there is genuine regional heterogeneity for the router to exploit.

Real data: every (trim, item) is answered by a real Bedrock model; per-call results cache to a
JSONL response cache and aggregate to an .npz so re-runs/resumes are free. The frontier, LaTeX
table, and figure are written to ``paper/figures/`` (like the other routing/trim experiments).

Run (AWS creds with Bedrock access):
    ~/canopy/.venv/bin/python examples/llm_routing/longbench_trim.py --n-per 30 --max-spend 8
Smoke-test the aggregation/plot/table with no deps or creds:
    python examples/llm_routing/longbench_trim.py --mock
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _plotstyle import FIGURE_DIR, PALETTE, progress, save_figure, set_style  # noqa: E402
from prompt_optimization import LAMBDAS, _sweep_lambda  # noqa: E402  reuse the frontier sweep

# English QA subsets of LongBench (scored by span-overlap F1 against gold answers).
LONGBENCH_TASKS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique"]
MODEL = "us.amazon.nova-lite-v1:0"  # cheap, long context window
# Context-keep fractions: level 0 keeps the full (capped) context; higher levels trim harder.
TRIM_FRACTIONS = [1.0, 0.66, 0.5, 0.33, 0.15]
N_TRIM = len(TRIM_FRACTIONS)
CONTEXT_WORD_CAP = 6000  # cap the full context to bound cost; fractions are taken of this


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred: str, golds: list[str]) -> float:
    """Max SQuAD-style token-overlap F1 of ``pred`` against any gold answer."""
    p = _normalize(pred).split()
    best = 0.0
    for g in golds:
        gt = _normalize(g).split()
        if not p or not gt:
            best = max(best, float(p == gt))
            continue
        common = Counter(p) & Counter(gt)
        ns = sum(common.values())
        if ns == 0:
            continue
        prec, rec = ns / len(p), ns / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def build_prompt(context: str, question: str, frac: float) -> str:
    words = context.split()[:CONTEXT_WORD_CAP]
    kept = words[: max(1, int(len(words) * frac))]
    trimmed = " ".join(kept)
    return (f"Read the context and answer the question as concisely as possible, using only a "
            f"short phrase.\n\nContext:\n{trimmed}\n\nQuestion: {question}\nAnswer:")


def load_longbench(tasks: list[str], n_per: int):
    """Return items [(context, question, gold_answers, task_idx)] and the task names (regions)."""
    from datasets import load_dataset

    items = []
    for ti, task in enumerate(tasks):
        try:
            ds = load_dataset("THUDM/LongBench", task, split="test")
        except Exception:  # noqa: BLE001 -- some configs live under a "_e" (LongBench-E) name
            ds = load_dataset("THUDM/LongBench", task + "_e", split="test")
        for row in ds.select(range(min(n_per, len(ds)))):
            golds = row["answers"] if isinstance(row["answers"], list) else [row["answers"]]
            items.append((row["context"], row["input"], [str(g) for g in golds], ti))
    return items, tasks


def measure_real(items, region, cache, max_calls, max_spend, npz_cache, checkpoint_every=25):
    """Answer every (trim, item) with a real Bedrock model; resumable via the .npz aggregate."""
    from canopy.llm import BedrockClient, BudgetError, CachingLLMClient

    n = len(items)
    if npz_cache.exists() and np.load(npz_cache, allow_pickle=True)["quality"].shape == (N_TRIM, n):
        d = np.load(npz_cache, allow_pickle=True)
        quality, in_tokens, done = d["quality"].copy(), d["in_tokens"].copy(), d["done"].copy()
        if bool(done.all()):
            print(f"loaded complete cached results from {npz_cache.name}")
            return quality, in_tokens
        print(f"resuming from {npz_cache.name}: {int(done.sum())}/{done.size} cells done")
    else:
        quality = np.zeros((N_TRIM, n)); in_tokens = np.zeros((N_TRIM, n))
        done = np.zeros((N_TRIM, n), dtype=bool)

    client = CachingLLMClient(BedrockClient(region=region, max_tokens=64), cache,
                              max_calls=max_calls, max_spend_usd=max_spend)
    fails = since = 0

    def _save():
        np.savez(npz_cache, quality=quality, in_tokens=in_tokens, done=done)

    try:
        for lvl in range(N_TRIM):
            for qi, (ctx, q, golds, _ti) in enumerate(
                progress(items, f"trim {lvl}/{N_TRIM - 1}", total=len(items))
            ):
                if done[lvl, qi]:
                    continue
                try:
                    text, it, _ = client.generate(MODEL, build_prompt(ctx, q, TRIM_FRACTIONS[lvl]))
                    quality[lvl, qi] = qa_f1(text, golds)
                    in_tokens[lvl, qi] = it
                    done[lvl, qi] = True
                except BudgetError:
                    raise
                except Exception as e:  # noqa: BLE001
                    fails += 1
                    if fails <= 1:
                        print(f"  call failed: {type(e).__name__}: {str(e)[:110]}")
                since += 1
                if since >= checkpoint_every:
                    _save(); since = 0
            print(f"  trim {lvl} (keep {TRIM_FRACTIONS[lvl]:.2f}): "
                  f"F1={quality[lvl].mean():.3f} in_tok={in_tokens[lvl].mean():.0f}")
    except BudgetError as e:
        print(f"\n[budget stop] {e}  Saving progress; re-run to resume.")
        _save(); raise
    _save()
    print(f"  budget: {client.stats()}")
    if fails > 0.2 * N_TRIM * n:
        raise RuntimeError(f"{fails} failed calls (creds/access/context-length?)")
    return quality, in_tokens


def measure_mock(items, seed=0):
    rng = np.random.default_rng(seed)
    n = len(items)
    quality = np.zeros((N_TRIM, n)); in_tokens = np.zeros((N_TRIM, n))
    base_tok = np.array([3200.0, 2150.0, 1650.0, 1100.0, 520.0])
    base_f1 = np.array([0.42, 0.42, 0.41, 0.37, 0.28])  # F1 degrades as context is trimmed away
    for lvl in range(N_TRIM):
        quality[lvl] = np.clip(base_f1[lvl] + rng.normal(0, 0.05, n), 0, 1)
        in_tokens[lvl] = base_tok[lvl] + rng.normal(0, 80, n)
    return quality, in_tokens


def _write_outputs(items, quality, in_tokens, model, n_tasks, quiet=False):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    f1 = quality.mean(axis=1)
    costs, frontier = _sweep_lambda(quality, in_tokens, n_tasks, LAMBDAS)
    best_fixed = int(np.argmax(f1))
    li = LAMBDAS.index(0.3) if 0.3 in LAMBDAS else len(LAMBDAS) // 2
    adapt_tok, adapt_f1 = frontier["adaptive"][li]
    orc_tok, orc_f1 = frontier["oracle"][li]

    payload = {
        "model": model, "n_items": len(items), "n_tasks": n_tasks, "lambdas": LAMBDAS,
        "trim_levels": [
            {"keep_fraction": TRIM_FRACTIONS[t], "f1": float(f1[t]),
             "avg_input_tokens": float(costs[t])} for t in range(N_TRIM)
        ],
        "frontier": {k: [[float(a), float(b)] for a, b in v] for k, v in frontier.items()},
    }
    (FIGURE_DIR / "longbench_trim_results.json").write_text(json.dumps(payload, indent=2))

    rows = [
        f"Full context (keep 1.00) & {f1[0]:.3f} & {costs[0]:.0f} \\\\",
        f"Best fixed trim & {f1[best_fixed]:.3f} & {costs[best_fixed]:.0f} \\\\",
        f"\\textbf{{Adaptive per-task (ours)}} & \\textbf{{{adapt_f1:.3f}}} & "
        f"\\textbf{{{adapt_tok:.0f}}} \\\\",
        f"Per-item oracle & {orc_f1:.3f} & {orc_tok:.0f} \\\\",
    ]
    tex = (
        "% Context trimming on real LongBench (auto-generated by longbench_trim.py). "
        f"{len(items)} items across {n_tasks} tasks, QA-F1.\n"
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "Policy & QA-F1 & Avg.\\ input tokens \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (FIGURE_DIR / "longbench_trim_table.tex").write_text(tex)

    if quiet:
        return
    print(f"\ncontext trimming on {model} ({len(items)} items, {n_tasks} LongBench tasks):")
    for t in range(N_TRIM):
        print(f"  keep {TRIM_FRACTIONS[t]:.2f}: F1={f1[t]:.3f}  tokens={costs[t]:.0f}")
    print(f"  adaptive(lam=0.3): F1={adapt_f1:.3f} tok={adapt_tok:.0f}  |  "
          f"best-fixed: F1={f1[best_fixed]:.3f} tok={costs[best_fixed]:.0f}  |  "
          f"oracle: F1={orc_f1:.3f} tok={orc_tok:.0f}")
    try:
        out = _plot(f1, costs, frontier, model)
        print(f"\nwrote json+table to {FIGURE_DIR} and figure to {out} (+ .png)")
    except Exception as e:  # noqa: BLE001
        print(f"\nwrote json+table to {FIGURE_DIR} (figure skipped: {type(e).__name__}: {e})")


def _plot(f1, costs, frontier, model):
    set_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    order = np.argsort(costs)
    ax.plot(np.asarray(costs)[order], np.asarray(f1)[order], "o-", color=PALETTE["blue"],
            label="fixed keep-fraction")
    for t in range(N_TRIM):
        ax.annotate(f"keep {TRIM_FRACTIONS[t]:.2f}", (costs[t], f1[t]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    for key, col, mk, lab in [
        ("adaptive", PALETTE["red"], "*", "adaptive per-task (ours)"),
        ("oracle", PALETTE["green"], "D", "per-item oracle"),
    ]:
        pts = sorted(frontier[key])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, color=col, label=lab)
    ax.set_xlabel("average input tokens (cost)")
    ax.set_ylabel("QA-F1")
    ax.set_title(f"LongBench context trimming on {model.split('.')[-1]}: adaptive frontier")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return str(save_figure(fig, "longbench_trim"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--tasks", default=",".join(LONGBENCH_TASKS))
    ap.add_argument("--n-per", type=int, default=30, help="items per task")
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--max-spend", type=float, default=None)
    ap.add_argument("--cache", default="examples/.cache/longbench_trim.jsonl")
    ap.add_argument("--mock", action="store_true", help="fabricate data; no deps/creds")
    args = ap.parse_args()

    tasks = [t for t in args.tasks.split(",") if t.strip()]
    npz_cache = Path(__file__).parent / "longbench_trim.npz"

    if args.mock:
        items = [("ctx " * 500, "q?", ["a"], i % len(tasks)) for i in range(len(tasks) * args.n_per)]
        quality, in_tokens = measure_mock(items)
        model_label = "mock"
    else:
        try:
            items, tasks = load_longbench(tasks, args.n_per)
            quality, in_tokens = measure_real(items, args.region, args.cache, args.max_calls,
                                              args.max_spend, npz_cache)
        except Exception as e:  # noqa: BLE001
            print(f"Real run unavailable ({type(e).__name__}: {e}).\n"
                  "Needs the bench extra + AWS Bedrock. Smoke-test with: --mock")
            return
        model_label = MODEL

    _write_outputs(items, quality, in_tokens, model_label, len(tasks))


if __name__ == "__main__":
    main()
