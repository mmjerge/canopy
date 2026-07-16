"""Value-guided reasoning search vs. best-of-N at matched compute, on real benchmarks.

The flagship real-model test of the multi-fidelity machinery, over three benchmarks
selected to span the reachability axis:

  * ``--benchmark math``          -- MATH-500 subset: long, reachable-but-unreliable
    derivations (the headline regime; boxed-answer grading).
  * ``--benchmark gpqa_diamond``  -- graduate-level science, multiple choice
    (letter grading; same budget curve as MATH).
  * ``--benchmark gsm8k``         -- near-saturated short-chain control (the regime
    where the theory predicts the gap vanishes).

For each budget point we run value-guided search (branching candidates scored by
self-consistency rollouts) and set best-of-N's sample count to value-guided's realized
generation-call budget, so the x-axis is matched calls. Per-problem correctness is
recorded and the paired per-problem gap is reported with bootstrap 95% CIs.

Run (needs AWS creds + Bedrock model access; makes real paid calls):
    uv sync --extra llm --extra bench --extra plot
    uv run --extra llm --extra bench --extra plot \\
        python examples/reasoning/reasoning_search.py --benchmark math --n-problems 300

Responses are cached on disk (JSONL), so reruns and budget sweeps reuse paid calls;
--max-calls / --max-spend give hard caps. Start small (--n-problems 20).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from canopy.bandits.reasoning_llm import (
    _normalize,
    best_of_n,
    extract_answer,
    extract_boxed,
    extract_letter,
    value_guided_search,
)
from canopy.llm import as_generate_fn

MODEL_ID = "us.meta.llama3-1-70b-instruct-v1:0"
N_STEPS_SWEEP = [2, 4, 6, 8, 10]  # -> budgets 23, 41, 59, 77, 95 calls at b=3, r=2, f=5
LETTERS = "ABCD"

_BOXED_SOLVE = (
    "Solve the problem step by step. End with your final answer in \\boxed{{...}}.\n\n"
    "Problem: {q}\n\nSolution:"
)
_BOXED_CONTINUE = (
    "Solve the problem step by step. End with your final answer in \\boxed{{...}}.\n\n"
    "Problem: {q}\n\nSolution so far:\n{prefix}\nNext step:"
)
_BOXED_ROLLOUT = (
    "Solve the problem step by step. End with your final answer in \\boxed{{...}}.\n\n"
    "Problem: {q}\n\nSolution so far:\n{prefix}\nFinish the solution:"
)
_MC_SOLVE = (
    "Answer the multiple-choice question. Reason step by step, then end with "
    "'Answer: <letter>'.\n\n{q}\n\nReasoning:"
)
_MC_CONTINUE = (
    "Answer the multiple-choice question. Reason step by step, then end with "
    "'Answer: <letter>'.\n\n{q}\n\nReasoning so far:\n{prefix}\nNext step:"
)
_MC_ROLLOUT = (
    "Answer the multiple-choice question. Reason step by step, then end with "
    "'Answer: <letter>'.\n\n{q}\n\nReasoning so far:\n{prefix}\nFinish the reasoning:"
)


def load_benchmark(name: str, n: int, seed: int = 0):
    """Return ``(items, extract_fn, templates)`` for the named benchmark.

    ``items`` is a list of ``(question, gold)``; ``templates`` is
    ``(solve, continue, rollout)`` (None -> module defaults in reasoning_llm).
    """
    from datasets import load_dataset

    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = [
            (row["question"], row["answer"].split("####")[-1].strip().replace(",", ""))
            for row in ds.select(range(n))
        ]
        return items, extract_answer, (None, None, None)

    if name == "math":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = []
        for row in ds.select(range(min(n, len(ds)))):
            gold = _normalize(str(row["answer"]).strip().replace(" ", ""))
            items.append((row["problem"], gold))
        return items, extract_boxed, (_BOXED_SOLVE, _BOXED_CONTINUE, _BOXED_ROLLOUT)

    if name == "gpqa_diamond":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        rng = np.random.default_rng(seed)
        items = []
        for row in ds.select(range(min(n, len(ds)))):
            options = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            order = rng.permutation(4)
            gold = LETTERS[int(np.where(order == 0)[0][0])]
            opts = "\n".join(
                f"{LETTERS[i]}) {options[int(o)].strip()}" for i, o in enumerate(order)
            )
            items.append((f"Question: {row['Question'].strip()}\n{opts}", gold))
        return items, extract_letter, (_MC_SOLVE, _MC_CONTINUE, _MC_ROLLOUT)

    raise ValueError(f"unknown benchmark {name!r} (gsm8k | math | gpqa_diamond)")


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of (a - b) with a bootstrap 95% CI over problems (paired)."""
    rng = np.random.default_rng(seed)
    diffs = a.astype(float) - b.astype(float)
    boots = np.array(
        [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(n_boot)]
    )
    return float(diffs.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def marginal_ci(x: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boots = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="math", choices=["gsm8k", "math", "gpqa_diamond"])
    ap.add_argument("--n-problems", type=int, default=300)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--branching", type=int, default=3)
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--final-rollouts", type=int, default=5)
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--max-spend", type=float, default=None)
    ap.add_argument("--cache", default=None, help="response cache path (default: per-benchmark)")
    ap.add_argument("--out", default=None, help="results JSON path (default: per-benchmark)")
    args = ap.parse_args()

    cache_path = args.cache or f"examples/.cache/reasoning_{args.benchmark}.jsonl"
    out_path = Path(args.out or f"examples/.cache/reasoning_{args.benchmark}_results.json")

    try:
        from canopy.llm import BedrockClient, BudgetError, CachingLLMClient

        base = BedrockClient(region=args.region, max_tokens=768)
        client = CachingLLMClient(
            base, cache_path, max_calls=args.max_calls, max_spend_usd=args.max_spend
        )
        items, extract, (solve_t, cont_t, roll_t) = load_benchmark(
            args.benchmark, args.n_problems
        )
    except Exception as e:  # noqa: BLE001 -- missing extras / creds / gated dataset
        print(
            f"Could not initialize ({type(e).__name__}: {e}).\n"
            "Needs: uv sync --extra llm --extra bench; AWS creds with Bedrock access;\n"
            "for gpqa_diamond, HF authentication (the dataset is gated)."
        )
        return

    generate = as_generate_fn(client, args.model, temperature=0.7)
    print(f"{args.benchmark}: {len(items)} problems, model {args.model}")

    results: dict[int, dict] = {}
    try:
        for n_steps in N_STEPS_SWEEP:
            vg_ok = np.zeros(len(items), dtype=bool)
            bo_ok = np.zeros(len(items), dtype=bool)
            realized_calls = []
            for i, (q, gold) in enumerate(items):
                vg = value_guided_search(
                    q,
                    gold,
                    generate,
                    branching=args.branching,
                    n_steps=n_steps,
                    rollouts=args.rollouts,
                    final_rollouts=args.final_rollouts,
                    extract=extract,
                    continue_template=cont_t,
                    rollout_template=roll_t,
                )
                bo = best_of_n(
                    q, gold, generate, n=vg.budget.calls, extract=extract, solve_template=solve_t
                )
                vg_ok[i], bo_ok[i] = vg.correct, bo.correct
                realized_calls.append(vg.budget.calls)
                if (i + 1) % 25 == 0:
                    print(
                        f"  [steps={n_steps}] {i+1}/{len(items)}: "
                        f"vg={vg_ok[: i + 1].mean():.3f} bo={bo_ok[: i + 1].mean():.3f}"
                    )
            budget = int(np.mean(realized_calls))
            d, lo, hi = paired_bootstrap(vg_ok, bo_ok)
            bo_lo, bo_hi = marginal_ci(bo_ok.astype(float))
            vg_lo, vg_hi = marginal_ci(vg_ok.astype(float), seed=1)
            results[budget] = {
                "n_steps": n_steps,
                "bo": float(bo_ok.mean()),
                "bo_ci": [bo_lo, bo_hi],
                "vg": float(vg_ok.mean()),
                "vg_ci": [vg_lo, vg_hi],
                "delta": d,
                "delta_ci": [lo, hi],
                "vg_correct": vg_ok.tolist(),
                "bo_correct": bo_ok.tolist(),
            }
            print(
                f"budget={budget:3d} calls: bo={bo_ok.mean():.3f} [{bo_lo:.2f},{bo_hi:.2f}]  "
                f"vg={vg_ok.mean():.3f} [{vg_lo:.2f},{vg_hi:.2f}]  "
                f"D={d:+.3f} [{lo:+.2f},{hi:+.2f}]"
            )
    except BudgetError as e:
        print(f"\n[budget stop] {e}  Reporting budgets completed so far.")

    if not results:
        print("no completed budget points")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote results to {out_path}")
    print(f"  client stats: {client.stats()}")
    _plot_and_table(results, args.benchmark)


def _plot_and_table(results: dict[int, dict], benchmark: str) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import matplotlib.pyplot as plt  # noqa: E402

    from _plotstyle import FIGURE_DIR, PALETTE, save_figure, set_style  # noqa: E402

    set_style()
    budgets = sorted(results)
    bo = [results[b]["bo"] for b in budgets]
    vg = [results[b]["vg"] for b in budgets]
    bo_ci = np.array([results[b]["bo_ci"] for b in budgets])
    vg_ci = np.array([results[b]["vg_ci"] for b in budgets])

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(budgets, bo, "s-", color=PALETTE["orange"], label="best-of-N")
    ax.fill_between(budgets, bo_ci[:, 0], bo_ci[:, 1], color=PALETTE["orange"], alpha=0.15)
    ax.plot(budgets, vg, "o-", color=PALETTE["red"], label="value-guided (ours)")
    ax.fill_between(budgets, vg_ci[:, 0], vg_ci[:, 1], color=PALETTE["red"], alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("matched compute budget (generation calls)")
    ax.set_ylabel(f"{benchmark.upper()} accuracy")
    ax.set_title(f"Value-guided vs. best-of-N at matched compute ({benchmark})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = save_figure(fig, f"reasoning_search_{benchmark}")
    print(f"saved chart to {out} (+ .png)")

    rows = []
    for b in budgets:
        r = results[b]
        rows.append(
            f"{b} & {r['bo']:.3f} [{r['bo_ci'][0]:.2f}, {r['bo_ci'][1]:.2f}] & "
            f"{r['vg']:.3f} [{r['vg_ci'][0]:.2f}, {r['vg_ci'][1]:.2f}] & "
            f"${r['delta']:+.3f}$ [${r['delta_ci'][0]:+.2f}$, ${r['delta_ci'][1]:+.2f}$] \\\\"
        )
    tex = (
        f"% {benchmark} matched-compute sweep (auto-generated by reasoning_search.py)\n"
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "Budget (calls) & best-of-N acc [95\\% CI] & value-guided acc [95\\% CI] & "
        "$\\Delta$ (paired) [95\\% CI] \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIGURE_DIR / f"reasoning_search_{benchmark}_table.tex").write_text(tex)
    print(f"wrote LaTeX table to {FIGURE_DIR / f'reasoning_search_{benchmark}_table.tex'}")


if __name__ == "__main__":
    main()
