"""Few-shot / instruction trimming on real Bedrock + BBH (Big-Bench Hard).

Third prompt-trim benchmark, on a different axis from MMLU (preamble trimming) and LongBench
(context trimming): here the "arms" are FEW-SHOT trim levels (how many in-context exemplars +
how much instruction to keep), the "regions" are BBH's 27 hard-reasoning tasks, and the tree
bandit learns --- per task --- the cheapest prompt that still answers correctly. Reward =
exact-match accuracy; cost = input tokens; utility = accuracy - lam * tokens. Different reasoning
tasks need different amounts of demonstration (some are trivial 0-shot, some need worked
exemplars), so there is genuine regional heterogeneity for the router to exploit.

Real data: every (trim, item) is answered by a real Bedrock model; per-call results cache to a
JSONL cache and aggregate to an .npz so re-runs/resumes are free. Frontier, table, and figure go
to ``paper/figures/`` like the other trim experiments.

Run:  ~/canopy/.venv/bin/python examples/llm_routing/bbh_trim.py --n-per 15
Smoke-test:  python examples/llm_routing/bbh_trim.py --mock
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/ (for _plotstyle)
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))  # examples/llm_routing/ (siblings)
from _plotstyle import FIGURE_DIR, PALETTE, progress, save_figure, set_style  # noqa: E402
from prompt_optimization import LAMBDAS, _sweep_lambda  # noqa: E402  reuse the frontier sweep

MODEL = "us.amazon.nova-lite-v1:0"
N_DEMOS_MAX = 3
# Trim levels: (instruction text, number of few-shot demos). Level 0 = verbose + 3-shot; higher
# levels strip instruction and demos. Every level keeps a parseable "A:" answer cue.
TRIM_SPECS = [
    ("Solve the following reasoning task. Read the problem, reason about it, then give the "
     "final answer after 'A:'.\n\n", 3),
    ("Solve the following reasoning task.\n\n", 2),
    ("Solve the following reasoning task.\n\n", 1),
    ("Solve the following reasoning task.\n\n", 0),
    ("", 0),
]
N_TRIM = len(TRIM_SPECS)


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = s.split("\n")[0]
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def exact_match(pred: str, target: str) -> float:
    p = pred
    if "a:" in p.lower():
        p = p.lower().split("a:")[-1]
    return 1.0 if _norm(p) == _norm(target) else 0.0


def load_bbh(tasks, n_per: int, n_demos: int = N_DEMOS_MAX):
    """Return (items[(question,target,task_idx)], demos_by_task{task_idx:[(q,a)]}, task_names).

    The first ``n_demos`` examples of each task are held out as the few-shot pool; the next
    ``n_per`` are the test items.
    """
    from datasets import get_dataset_config_names, load_dataset

    if not tasks:
        tasks = get_dataset_config_names("lukaemon/bbh")
    items, demos_by_task = [], {}
    for ti, task in enumerate(tasks):
        ds = load_dataset("lukaemon/bbh", task, split="test")
        demos_by_task[ti] = [(ds[i]["input"], ds[i]["target"]) for i in range(min(n_demos, len(ds)))]
        for i in range(n_demos, min(n_demos + n_per, len(ds))):
            items.append((ds[i]["input"], ds[i]["target"], ti))
    return items, demos_by_task, tasks


def build_prompt(question: str, task_idx: int, level: int, demos_by_task) -> str:
    instr, k = TRIM_SPECS[level]
    demo_str = ""
    for q, a in demos_by_task.get(task_idx, [])[:k]:
        demo_str += f"Q: {q}\nA: {a}\n\n"
    return f"{instr}{demo_str}Q: {question}\nA:"


def measure_real(items, demos_by_task, region, cache, max_calls, max_spend, npz_cache, ckpt=25):
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

    client = CachingLLMClient(BedrockClient(region=region, max_tokens=24), cache,
                              max_calls=max_calls, max_spend_usd=max_spend)
    fails = since = 0

    def _save():
        np.savez(npz_cache, quality=quality, in_tokens=in_tokens, done=done)

    try:
        for lvl in range(N_TRIM):
            for qi, (q, tgt, ti) in enumerate(
                progress(items, f"trim {lvl}/{N_TRIM - 1}", total=len(items))
            ):
                if done[lvl, qi]:
                    continue
                try:
                    text, it, _ = client.generate(MODEL, build_prompt(q, ti, lvl, demos_by_task))
                    quality[lvl, qi] = exact_match(text, tgt)
                    in_tokens[lvl, qi] = it
                    done[lvl, qi] = True
                except BudgetError:
                    raise
                except Exception as e:  # noqa: BLE001
                    fails += 1
                    if fails <= 1:
                        print(f"  call failed: {type(e).__name__}: {str(e)[:110]}")
                since += 1
                if since >= ckpt:
                    _save(); since = 0
            spec = TRIM_SPECS[lvl]
            print(f"  trim {lvl} ({spec[1]}-shot): acc={quality[lvl].mean():.3f} "
                  f"in_tok={in_tokens[lvl].mean():.0f}")
    except BudgetError as e:
        print(f"\n[budget stop] {e}  Saving progress; re-run to resume.")
        _save(); raise
    _save()
    print(f"  budget: {client.stats()}")
    if fails > 0.2 * N_TRIM * n:
        raise RuntimeError(f"{fails} failed calls (creds/access?)")
    return quality, in_tokens


def measure_mock(items, seed=0):
    rng = np.random.default_rng(seed)
    n = len(items)
    quality = np.zeros((N_TRIM, n)); in_tokens = np.zeros((N_TRIM, n))
    base_tok = np.array([420.0, 300.0, 210.0, 90.0, 70.0])
    base_acc = np.array([0.62, 0.60, 0.57, 0.50, 0.44])
    for lvl in range(N_TRIM):
        quality[lvl] = (rng.random(n) < base_acc[lvl]).astype(float)
        in_tokens[lvl] = base_tok[lvl] + rng.normal(0, 12, n)
    return quality, in_tokens


def _write_outputs(items, quality, in_tokens, model, n_tasks, quiet=False):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    acc = quality.mean(axis=1)
    costs, frontier = _sweep_lambda(quality, in_tokens, n_tasks, LAMBDAS)
    best_fixed = int(np.argmax(acc))
    li = LAMBDAS.index(0.3) if 0.3 in LAMBDAS else len(LAMBDAS) // 2
    adapt_tok, adapt_acc = frontier["adaptive"][li]
    orc_tok, orc_acc = frontier["oracle"][li]

    payload = {
        "model": model, "n_items": len(items), "n_tasks": n_tasks, "lambdas": LAMBDAS,
        "trim_levels": [
            {"n_demos": TRIM_SPECS[t][1], "accuracy": float(acc[t]),
             "avg_input_tokens": float(costs[t])} for t in range(N_TRIM)
        ],
        "frontier": {k: [[float(a), float(b)] for a, b in v] for k, v in frontier.items()},
    }
    (FIGURE_DIR / "bbh_trim_results.json").write_text(json.dumps(payload, indent=2))

    rows = [
        f"Full 3-shot (verbose) & {acc[0]:.3f} & {costs[0]:.0f} \\\\",
        f"Best fixed trim & {acc[best_fixed]:.3f} & {costs[best_fixed]:.0f} \\\\",
        f"\\textbf{{Adaptive per-task (ours)}} & \\textbf{{{adapt_acc:.3f}}} & "
        f"\\textbf{{{adapt_tok:.0f}}} \\\\",
        f"Per-item oracle & {orc_acc:.3f} & {orc_tok:.0f} \\\\",
    ]
    tex = (
        "% Few-shot trimming on real BBH (auto-generated by bbh_trim.py). "
        f"{len(items)} items across {n_tasks} tasks, exact-match.\n"
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "Policy & Accuracy & Avg.\\ input tokens \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (FIGURE_DIR / "bbh_trim_table.tex").write_text(tex)

    if quiet:
        return
    print(f"\nfew-shot trimming on {model} ({len(items)} items, {n_tasks} BBH tasks):")
    for t in range(N_TRIM):
        print(f"  {TRIM_SPECS[t][1]}-shot: acc={acc[t]:.3f}  tokens={costs[t]:.0f}")
    print(f"  adaptive(lam=0.3): acc={adapt_acc:.3f} tok={adapt_tok:.0f}  |  "
          f"best-fixed: acc={acc[best_fixed]:.3f} tok={costs[best_fixed]:.0f}  |  "
          f"oracle: acc={orc_acc:.3f} tok={orc_tok:.0f}")
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
            label="fixed few-shot level")
    for t in range(N_TRIM):
        ax.annotate(f"{TRIM_SPECS[t][1]}-shot", (costs[t], acc[t]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    for key, col, mk, lab in [
        ("adaptive", PALETTE["red"], "*", "adaptive per-task (ours)"),
        ("oracle", PALETTE["green"], "D", "per-item oracle"),
    ]:
        pts = sorted(frontier[key])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, color=col, label=lab)
    ax.set_xlabel("average input tokens (cost)")
    ax.set_ylabel("exact-match accuracy")
    ax.set_title(f"BBH few-shot trimming on {model.split('.')[-1]}: adaptive frontier")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return str(save_figure(fig, "bbh_trim"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--tasks", default="", help="comma-separated BBH tasks (empty = all 27)")
    ap.add_argument("--n-per", type=int, default=15, help="test items per task")
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--max-spend", type=float, default=None)
    ap.add_argument("--cache", default="examples/.cache/bbh_trim.jsonl")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    tasks = [t for t in args.tasks.split(",") if t.strip()]
    npz_cache = Path(__file__).parent / "bbh_trim.npz"

    if args.mock:
        tasks = tasks or [f"task{i}" for i in range(12)]
        items = [("q?", "a", i % len(tasks)) for i in range(len(tasks) * args.n_per)]
        quality, in_tokens = measure_mock(items)
        model_label, n_tasks = "mock", len(tasks)
    else:
        try:
            items, demos_by_task, tasks = load_bbh(tasks, args.n_per)
            quality, in_tokens = measure_real(items, demos_by_task, args.region, args.cache,
                                              args.max_calls, args.max_spend, npz_cache)
        except Exception as e:  # noqa: BLE001
            print(f"Real run unavailable ({type(e).__name__}: {e}). Needs bench extra + Bedrock. "
                  "Smoke-test: --mock")
            return
        model_label, n_tasks = MODEL, len(tasks)

    _write_outputs(items, quality, in_tokens, model_label, n_tasks)


if __name__ == "__main__":
    main()
