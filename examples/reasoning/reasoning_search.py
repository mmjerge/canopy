"""Flagship real-LLM test: value-guided (multi-fidelity) reasoning search vs. best-of-N.

The only real-model experiment that exercises the paper's *multi-fidelity + tree-depth*
machinery (the routing experiments are flat/depth-one). The scientific question:

    At a *matched* budget of generation calls, does value-guided (edge-following) test-time
    search find more correct answers than best-of-N (self-consistency)?

The advantage is exponential-vs-polynomial in the number of decision steps K, so the headline
benchmark is **MATH** (long multi-step derivations, reachable-but-unreliable for capable
models) rather than the near-saturated, short-chain GSM8K, which we keep as a baseline. Pick
with ``--benchmark {math,gsm8k}``.

The cheap, biased probe is a short rollout / self-consistency score on a *partial* chain; the
expensive, unbiased leaf is a full generation graded against the gold answer (numeric for
GSM8K, boxed-answer symbolic grading for MATH). We sweep a compute budget (search depth), and
at each level match best-of-N's sample count to value-guided's realized call budget so the
x-axis is honest compute. Per-problem correctness is recorded for bootstrap confidence bands.

Outputs (written to ``paper/figures/``, per benchmark):
    reasoning_search_<bench>_results.json   machine-readable curve + metadata
    reasoning_search_<bench>_table.tex       LaTeX accuracy/compute table
    reasoning_search_<bench>.{pdf,png}        accuracy-vs-compute figure (needs the plot extra)

Requires the optional extras and AWS access for the real run:
    uv sync --extra llm --extra bench
    uv run --extra llm --extra bench python examples/reasoning/reasoning_search.py --resume
It makes real, paid model calls; start with a small ``--n-problems`` and ``--max-spend``.

Smoke-test the whole harness (sweep, checkpoint, resume, table, figure) with no deps/creds:
    python examples/reasoning/reasoning_search.py --mock

Honest read: a negative result is also informative -- it would say the cheap rollout value
(self-consistency, the default) is not informative enough at the pivotal steps, i.e. the
bottleneck is the value signal, not the search.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from canopy.bandits.reasoning_llm import (
    best_of_n,
    extract_answer,
    extract_boxed_answer,
    grade_math,
    value_guided_search,
)

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    tqdm = None

# paper/figures (canopy root is two levels up from examples/reasoning/).
FIGDIR = Path(__file__).resolve().parents[2] / "paper" / "figures"

MODEL_ID = "us.meta.llama3-1-8b-instruct-v1:0"


def _grade_numeric(answer, gold):
    from canopy.bandits.reasoning_llm import _normalize

    return answer is not None and answer == _normalize(gold)


# (loader, extract_fn, grade_fn) per benchmark.
def load_gsm8k(n: int):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        items.append((row["question"], gold))
    return items


def load_math(n: int):
    """MATH-500: long multi-step problems with boxed final answers."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        items.append((row["problem"], str(row["answer"])))
    return items


BENCHMARKS = {
    "math": (load_math, extract_boxed_answer, grade_math),
    "gsm8k": (load_gsm8k, extract_answer, _grade_numeric),
}


def vg_call_budget(branching: int, n_steps: int, rollouts: int, final_rollouts: int) -> int:
    """Exact number of generation calls value-guided search spends (used to match best-of-N)."""
    return n_steps * (branching * (1 + rollouts)) + final_rollouts


# --- deterministic mock generator so the harness runs with no deps/creds ------------------


def make_mock_generate(seed: int = 0):
    """A stand-in ``generate(prompt, max_tokens, call_seed) -> text`` for smoke tests."""
    import hashlib
    import random
    import re

    def generate(prompt: str, max_tokens: int, call_seed: int) -> str:
        rng = random.Random(hash((prompt, call_seed, seed)) & 0xFFFFFFFF)
        m = re.search(r"what is (\d+) plus (\d+)", prompt)
        guided = "Solution so far:" in prompt
        p_correct = 0.65 if guided else 0.35
        if m and rng.random() < p_correct:
            val = (int(m.group(1)) + int(m.group(2))) % 100
        else:
            val = int(hashlib.md5(f"{prompt}{call_seed}".encode()).hexdigest(), 16) % 100
        return f"step reasoning ... #### {val}"

    return generate


def load_mock_problems(n: int):
    return [(f"Mock problem number {i}: what is {i} plus {i}?", str((2 * i) % 100)) for i in range(n)]


def run_level(problems, generate, cfg, extract_fn, grade_fn, budget_error, workers=8):
    """Run both strategies over all problems at one budget level; return per-problem 0/1 lists.

    Problems are independent, so they run on a thread pool (``workers``); the model calls are
    I/O-bound, giving a near-linear speedup. If a spend/call cap trips mid-level, the whole level
    is discarded (re-raised) so a partial, biased level is never saved -- the caller resumes it.
    """
    branching, n_steps, rollouts, final_rollouts = cfg
    bo_n = vg_call_budget(branching, n_steps, rollouts, final_rollouts)

    def _one(q, gold):
        bo = best_of_n(q, gold, generate, n=bo_n, extract_fn=extract_fn, grade_fn=grade_fn)
        vg = value_guided_search(
            q, gold, generate, branching=branching, n_steps=n_steps, rollouts=rollouts,
            final_rollouts=final_rollouts, extract_fn=extract_fn, grade_fn=grade_fn,
        )
        return int(bo.correct), int(vg.correct), bo.budget.calls, vg.budget.calls

    bo_hits, vg_hits = [], []
    bo_calls = vg_calls = 0
    hit_budget = None
    bar = tqdm(total=len(problems), unit="prob", desc=f"budget {bo_n}") if tqdm else None
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_one, q, gold) for q, gold in problems]
        for fut in as_completed(futs):
            try:
                bc, vc, bca, vca = fut.result()
            except budget_error as e:  # noqa: PERF203
                hit_budget = e
                continue
            bo_hits.append(bc); vg_hits.append(vc); bo_calls += bca; vg_calls += vca
            if bar is not None:
                bar.update(1)
    if bar is not None:
        bar.close()
    if hit_budget is not None:
        raise hit_budget  # discard this (partial) level; caller keeps completed levels
    n = max(1, len(bo_hits))
    return {
        "matched_budget": bo_n,
        "n_problems": len(bo_hits),
        "best_of_n": {"hits": bo_hits, "avg_calls": bo_calls / n},
        "value_guided": {"hits": vg_hits, "avg_calls": vg_calls / n},
    }


def _bootstrap_ci(hits, iters=2000, seed=0):
    import numpy as np

    a = np.asarray(hits, dtype=float)
    if a.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(iters, a.size))].mean(axis=1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _paired_delta_ci(bo_hits, vg_hits, iters=4000, seed=0):
    """Paired bootstrap of the gap ``value_guided - best_of_N`` over the *same* problems.

    Best-of-N and value-guided are graded on identical problems, so the honest significance test
    is the paired difference (per-problem), not a comparison of the two marginal CIs, which
    overlap even when the paired gap is clearly positive. Returns ``(delta, lo, hi)``.
    """
    import numpy as np

    bo, vg = np.asarray(bo_hits, float), np.asarray(vg_hits, float)
    if bo.size == 0 or bo.size != vg.size:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, bo.size, size=(iters, bo.size))
    d = vg[idx].mean(axis=1) - bo[idx].mean(axis=1)
    return float((vg - bo).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _stem(bench, tag=""):
    """Output basename; a non-empty ``tag`` (e.g. a model label) keeps comparison runs separate."""
    return f"reasoning_search_{bench}" + (f"_{tag}" if tag else "")


def _write_outputs(results, model, bench, n_problems, quiet=False, tag=""):
    if not results:
        return
    FIGDIR.mkdir(parents=True, exist_ok=True)
    stem = _stem(bench, tag)
    (FIGDIR / f"{stem}_results.json").write_text(
        json.dumps({"benchmark": bench, "model": model, "n_problems": n_problems,
                    "levels": results}, indent=2)
    )

    levels = sorted(results.values(), key=lambda r: r["matched_budget"])
    rows = []
    for r in levels:
        b = r["matched_budget"]
        bo_m, bo_lo, bo_hi = _bootstrap_ci(r["best_of_n"]["hits"], seed=b)
        vg_m, vg_lo, vg_hi = _bootstrap_ci(r["value_guided"]["hits"], seed=b + 1)
        d_m, d_lo, d_hi = _paired_delta_ci(r["best_of_n"]["hits"], r["value_guided"]["hits"], seed=b)
        rows.append(
            f"{b} & {bo_m:.3f} [{bo_lo:.2f}, {bo_hi:.2f}] & "
            f"{vg_m:.3f} [{vg_lo:.2f}, {vg_hi:.2f}] & "
            f"${d_m:+.3f}$ [{d_lo:+.2f}, {d_hi:+.2f}] \\\\"
        )
    tex = (
        f"% {bench.upper()} value-guided vs best-of-N (auto-generated by reasoning_search.py). "
        "Delta is the paired (same-problem) gap with a 95% bootstrap CI.\n"
        "\\begin{tabular}{rccc}\n\\toprule\n"
        "Budget (calls) & best-of-N acc [95\\% CI] & value-guided acc [95\\% CI] "
        "& $\\Delta$ (paired) [95\\% CI] \\\\\n\\midrule\n" + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (FIGDIR / f"{stem}_table.tex").write_text(tex)

    if quiet:
        return
    print(f"\n{bench.upper()} reasoning search ({n_problems} problems), model {model}")
    print(f"  {'budget':>8s} {'best-of-N':>18s} {'value-guided':>18s} {'delta (paired) [95% CI]':>26s}")
    for r in levels:
        b = r["matched_budget"]
        bo_m, bo_lo, bo_hi = _bootstrap_ci(r["best_of_n"]["hits"], seed=b)
        vg_m, vg_lo, vg_hi = _bootstrap_ci(r["value_guided"]["hits"], seed=b + 1)
        d_m, d_lo, d_hi = _paired_delta_ci(r["best_of_n"]["hits"], r["value_guided"]["hits"], seed=b)
        print(f"  {b:8d} {bo_m:.3f} [{bo_lo:.2f},{bo_hi:.2f}] "
              f"{vg_m:.3f} [{vg_lo:.2f},{vg_hi:.2f}] {d_m:+.3f} [{d_lo:+.2f},{d_hi:+.2f}]")
    try:
        out = _plot_figure(results, model, bench, n_problems, tag=tag)
        print(f"\nwrote json+table to {FIGDIR} and figure to {out} (+ .png)")
    except Exception as e:  # noqa: BLE001
        print(f"\nwrote json+table to {FIGDIR} (figure skipped: {type(e).__name__}: {e})")


def _plot_figure(results, model, bench, n_problems, tag=""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _plotstyle import PALETTE, ci_band, save_figure, set_style

    set_style()
    import matplotlib.pyplot as plt
    import numpy as np

    levels = sorted(results.values(), key=lambda r: r["matched_budget"])
    budgets = [r["matched_budget"] for r in levels]

    def _samples(key):
        cols = []
        for r in levels:
            m, lo, hi = _bootstrap_ci(r[key]["hits"], seed=r["matched_budget"])
            half = max(hi - m, m - lo, 1e-6)
            cols.append(np.array([m - half / 1.96, m + half / 1.96]))
        return np.array(cols)

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ci_band(ax, budgets, _samples("best_of_n"), PALETTE["orange"], label="best-of-N", marker="s")
    ci_band(ax, budgets, _samples("value_guided"), PALETTE["red"],
            label="value-guided (ours)", marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("matched compute budget (generation calls)")
    ax.set_ylabel(f"{bench.upper()} accuracy")
    title_model = model.split(".")[-1][:24]
    ax.set_title(f"Value-guided vs. best-of-N at matched compute "
                 f"({n_problems} {bench.upper()}, {title_model})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return str(save_figure(fig, _stem(bench, tag)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=list(BENCHMARKS), default="math",
                    help="math (headline, long chains) or gsm8k (baseline)")
    ap.add_argument("--n-problems", type=int, default=100)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--branching", type=int, default=3)
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--final-rollouts", type=int, default=5)
    ap.add_argument("--depth-sweep", default="2,4,6,8",
                    help="comma-separated n_steps values; each is one matched-budget level")
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--max-spend", type=float, default=None)
    ap.add_argument("--cache", default="")
    ap.add_argument("--resume", action="store_true",
                    help="skip budget levels already in the benchmark's results JSON")
    ap.add_argument("--workers", type=int, default=8,
                    help="problems to run concurrently per level (I/O-bound; ~linear speedup)")
    ap.add_argument("--tag", default="",
                    help="suffix for output files, e.g. a model label, so runs on different "
                         "models don't overwrite each other (empty = the headline file names)")
    ap.add_argument("--mock", action="store_true", help="deterministic mock model, no deps/creds")
    args = ap.parse_args()

    depths = [int(x) for x in args.depth_sweep.split(",") if x.strip()]
    loader, extract_fn, grade_fn = BENCHMARKS[args.benchmark]
    cache = args.cache or f"examples/.cache/reasoning_{args.benchmark}.jsonl"

    if args.mock:
        generate = make_mock_generate()
        problems = load_mock_problems(args.n_problems)
        client = None

        class _NoBudgetError(Exception):
            pass

        budget_error = _NoBudgetError
        extract_fn, grade_fn = extract_answer, _grade_numeric  # mock emits #### numbers
    else:
        try:
            from canopy.llm import BedrockClient, BudgetError, CachingLLMClient, as_generate_fn

            base = BedrockClient(region=args.region, max_tokens=512)
            client = CachingLLMClient(
                base, cache, max_calls=args.max_calls, max_spend_usd=args.max_spend
            )
            generate = as_generate_fn(client, args.model, temperature=0.7)
            problems = loader(args.n_problems)
            budget_error = BudgetError
        except Exception as e:  # noqa: BLE001
            print(
                f"Could not initialize the real-LLM experiment: {e}\n"
                "Install extras and configure AWS:\n"
                "  uv sync --extra llm --extra bench\n"
                "  (AWS creds with Bedrock invoke permission + model access)\n"
                "Smoke-test the harness with no deps: --mock"
            )
            return

    model_label = "mock" if args.mock else args.model
    print(f"{args.benchmark.upper()} reasoning search: {len(problems)} problems, "
          f"model {model_label}, depth sweep {depths}")

    stem = _stem(args.benchmark, args.tag)
    results: dict[str, dict] = {}
    if args.resume:
        rp = FIGDIR / f"{stem}_results.json"
        if rp.exists():
            results = dict(json.loads(rp.read_text()).get("levels", {}))
            if results:
                print(f"resuming: {len(results)} budget levels already done: {sorted(results)}")

    start = time.monotonic()
    for i, n_steps in enumerate(depths):
        cfg = (args.branching, n_steps, args.rollouts, args.final_rollouts)
        key = str(vg_call_budget(*cfg))
        if key in results:
            continue
        try:
            level = run_level(problems, generate, cfg, extract_fn, grade_fn, budget_error,
                              workers=args.workers)
        except budget_error as e:
            print(f"\n[budget stop] {e}  (completed {i}/{len(depths)} levels)")
            break
        results[key] = level
        bo_acc = sum(level["best_of_n"]["hits"]) / max(1, level["n_problems"])
        vg_acc = sum(level["value_guided"]["hits"]) / max(1, level["n_problems"])
        print(f"  [budget {key}] best-of-N acc={bo_acc:.3f}  value-guided acc={vg_acc:.3f}  "
              f"({(time.monotonic() - start) / 60:.1f}m elapsed)")
        _write_outputs(results, model_label, args.benchmark, len(problems), quiet=True,
                       tag=args.tag)

    _write_outputs(results, model_label, args.benchmark, len(problems), tag=args.tag)
    if client is not None:
        print(f"  budget: {client.stats()}")


if __name__ == "__main__":
    main()
