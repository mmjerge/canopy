"""Is the reasoning value function almost tree-$K$-Lipschitz? (the flagship characterization).

This is where Suman's "almost tree-$K$-Lipschitz" prior lives naturally. A node is a partial
reasoning trace; its *true value* is the probability a completion from it is correct (estimated
by grading a few rollouts against gold); the *cheap value* is the biased self-consistency probe
the search actually uses. The prior says the value function is tree-Lipschitz -- nearby nodes
(same prefix) have similar value, so the cheap probe is informative -- except at a finite number
$K$ of *pivotal* steps where one continuation is decisively better (a near-discontinuity: a
wrong turn is unrecoverable). Those pivotal steps are the violations, and they are what
value-guided search must get right.

We instrument :func:`value_guided_search` (via ``trace_log``) to record, per candidate node, the
cheap probe value and the true (graded) value, then measure:

  1. **Informativeness (tree-Lipschitz backbone).** The correlation between the cheap probe and
     the true node value across all nodes. High correlation is the empirical premise that makes
     the cheap probe a usable value edge (Panel A).
  2. **Violations $K$.** Per problem, the number of steps whose sibling true-value *spread*
     (max $-$ min over the ``branching`` candidates) exceeds a threshold -- a pivotal step where
     the choice matters sharply. Few such steps per problem is the "$K$ small / measure-zero"
     part of the prior (Panel B).

Run a real characterization on the GPU box (Bedrock creds), e.g.:
    uv run --extra llm --extra bench python examples/analysis/reasoning_tree_lipschitz.py \
        --benchmark math --n-problems 100 --branching 3 --n-steps 6 --rollouts 4 --resume
Smoke-test the harness with no deps/creds:
    python examples/analysis/reasoning_tree_lipschitz.py --mock
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))  # examples/
sys.path.insert(0, str(HERE.parents[1] / "reasoning"))

from canopy.bandits.reasoning_llm import (  # noqa: E402
    extract_answer,
    extract_boxed_answer,
    grade_math,
    value_guided_search,
)

FIGDIR = HERE.parents[2] / "paper" / "figures"


def _grade_numeric(answer, gold):
    from canopy.bandits.reasoning_llm import _normalize

    return answer is not None and answer == _normalize(gold)


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    def rank(a):
        order = np.argsort(a, kind="stable")
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(a))
        return r
    return _pearson(rank(np.asarray(x, float)), rank(np.asarray(y, float)))


def characterize(problems, generate, cfg, extract_fn, grade_fn, tau, budget_error):
    """Run value-guided search with tree logging over problems; return aggregated records."""
    branching, n_steps, rollouts = cfg
    all_cheap, all_true = [], []
    per_problem_K, per_problem_steps = [], []
    path_value_by_step = [[] for _ in range(n_steps)]  # chosen-candidate true value per step
    for q, gold in problems:
        trace: list[dict] = []
        try:
            value_guided_search(
                q, gold, generate, branching=branching, n_steps=n_steps, rollouts=rollouts,
                final_rollouts=rollouts, extract_fn=extract_fn, grade_fn=grade_fn,
                trace_log=trace,
            )
        except budget_error:
            raise
        by_step: dict[int, list[dict]] = {}
        for rec in trace:
            all_cheap.append(rec["cheap_value"]); all_true.append(rec["true_value"])
            by_step.setdefault(rec["step"], []).append(rec)
        K = 0
        for step, recs in by_step.items():
            tv = [r["true_value"] for r in recs]
            if max(tv) - min(tv) > tau:
                K += 1
            for r in recs:
                if r["chosen"]:
                    path_value_by_step[step].append(r["true_value"])
        per_problem_K.append(K)
        per_problem_steps.append(len(by_step))
    return {
        "cheap": all_cheap, "true": all_true,
        "K": per_problem_K, "n_steps_seen": per_problem_steps,
        "path_value_by_step": [float(np.mean(v)) if v else float("nan")
                               for v in path_value_by_step],
        "tau": tau, "branching": branching, "n_steps": n_steps, "rollouts": rollouts,
    }


def _write_outputs(agg, model, bench, n_problems):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    r_p = _pearson(agg["cheap"], agg["true"])
    r_s = _spearman(agg["cheap"], agg["true"])
    K = np.array(agg["K"], float)
    steps = np.array(agg["n_steps_seen"], float)
    viol_frac = float(K.sum() / max(1.0, steps.sum()))
    stem = f"reasoning_tree_lipschitz_{bench}"
    payload = {
        "benchmark": bench, "model": model, "n_problems": n_problems,
        "pearson_cheap_true": r_p, "spearman_cheap_true": r_s,
        "mean_K": float(K.mean()) if K.size else 0.0,
        "median_K": float(np.median(K)) if K.size else 0.0,
        "max_steps": int(steps.max()) if steps.size else 0,
        "violation_step_fraction": viol_frac,
        "path_value_by_step": agg["path_value_by_step"],
        "tau": agg["tau"], "branching": agg["branching"], "n_steps": agg["n_steps"],
        "rollouts": agg["rollouts"], "n_nodes": len(agg["cheap"]),
    }
    (FIGDIR / f"{stem}_results.json").write_text(json.dumps(payload, indent=2))

    tex = (
        f"% Reasoning-tree Lipschitz characterization ({bench.upper()}, "
        "auto-generated by reasoning_tree_lipschitz.py)\n"
        "\\begin{tabular}{lr}\n\\toprule\nQuantity & Value \\\\\n\\midrule\n"
        f"Cheap-vs-true node value (Pearson $r$) & {r_p:.3f} \\\\\n"
        f"Cheap-vs-true node value (Spearman $\\rho$) & {r_s:.3f} \\\\\n"
        f"Mean violations per problem $\\bar K$ & {payload['mean_K']:.2f} \\\\\n"
        f"Median violations per problem & {payload['median_K']:.1f} \\\\\n"
        f"Pivotal (violation) step fraction & {100 * viol_frac:.1f}\\% \\\\\n"
        f"Steps per problem (max) & {payload['max_steps']} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
    )
    (FIGDIR / f"{stem}_table.tex").write_text(tex)

    print(f"\n{bench.upper()} reasoning-tree characterization ({n_problems} problems, "
          f"{payload['n_nodes']} nodes), model {model}")
    print(f"  cheap-vs-true value: Pearson r={r_p:.3f}, Spearman rho={r_s:.3f}")
    print(f"  violations K per problem: mean={payload['mean_K']:.2f}, "
          f"median={payload['median_K']:.1f}; pivotal-step fraction={viol_frac:.1%}")
    print(f"  chosen-path true value by step: "
          + ", ".join(f"{v:.2f}" for v in agg['path_value_by_step']))
    try:
        out = _plot(agg, payload, model, bench, n_problems)
        print(f"  wrote json+table to {FIGDIR} and figure to {out} (+ .png)")
    except Exception as e:  # noqa: BLE001
        print(f"  wrote json+table to {FIGDIR} (figure skipped: {type(e).__name__}: {e})")


def _plot(agg, payload, model, bench, n_problems):
    from _plotstyle import PALETTE, save_figure, set_style

    set_style()
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))
    # Panel A: cheap probe vs true node value (informativeness)
    cheap, true = np.asarray(agg["cheap"]), np.asarray(agg["true"])
    jit = 0.01 * np.random.default_rng(0).standard_normal(cheap.shape)
    axA.scatter(cheap + jit, true + jit, s=10, alpha=0.25, color=PALETTE["blue"])
    if cheap.size >= 2:
        b, a = np.polyfit(cheap, true, 1)
        xs = np.array([cheap.min(), cheap.max()])
        axA.plot(xs, a + b * xs, "-", color=PALETTE["red"], lw=2,
                 label=f"fit (Pearson $r$={payload['pearson_cheap_true']:.2f})")
    axA.set_xlabel("cheap probe value (self-consistency)")
    axA.set_ylabel("true node value (fraction of rollouts correct)")
    axA.set_title("Cheap probe predicts true value (tree-Lipschitz backbone)")
    axA.legend(loc="upper left")

    # Panel B: distribution of violations K per problem
    K = np.array(agg["K"], int)
    if K.size:
        bins = np.arange(0, K.max() + 2) - 0.5
        axB.hist(K, bins=bins, color=PALETTE["green"], alpha=0.8, rwidth=0.9)
        axB.axvline(K.mean(), color=PALETTE["red"], ls="--", lw=2,
                    label=f"mean $\\bar K$={K.mean():.2f}")
    axB.set_xlabel(f"pivotal steps per problem $K$ (sibling value spread $>\\tau={agg['tau']}$)")
    axB.set_ylabel("number of problems")
    axB.set_title(f"Few violations per problem (of $\\leq{payload['max_steps']}$ steps)")
    axB.legend(loc="upper right")

    fig.suptitle(f"Reasoning value function is almost tree-$K$-Lipschitz "
                 f"({n_problems} {bench.upper()}, model {model})")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return str(save_figure(fig, f"reasoning_tree_lipschitz_{bench}"))


BENCHMARKS = {"math": extract_boxed_answer, "gsm8k": extract_answer}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=list(BENCHMARKS), default="math")
    ap.add_argument("--n-problems", type=int, default=100)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--model", default="us.meta.llama3-1-8b-instruct-v1:0")
    ap.add_argument("--branching", type=int, default=3)
    ap.add_argument("--n-steps", type=int, default=6)
    ap.add_argument("--rollouts", type=int, default=4, help="rollouts per node (true-value est.)")
    ap.add_argument("--tau", type=float, default=0.5, help="sibling value spread => pivotal step")
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--max-spend", type=float, default=None)
    ap.add_argument("--cache", default="")
    ap.add_argument("--mock", action="store_true", help="deterministic mock model, no deps/creds")
    args = ap.parse_args()

    extract_fn = BENCHMARKS[args.benchmark]
    grade_fn = grade_math if args.benchmark == "math" else _grade_numeric
    cfg = (args.branching, args.n_steps, args.rollouts)

    if args.mock:
        from reasoning_search import load_mock_problems, make_mock_generate

        generate = make_mock_generate()
        problems = load_mock_problems(args.n_problems)
        extract_fn, grade_fn = extract_answer, _grade_numeric

        class _NoBudget(Exception):
            pass

        budget_error, model_label = _NoBudget, "mock"
    else:
        try:
            from reasoning_search import BENCHMARKS as RS_BENCH

            from canopy.llm import BedrockClient, BudgetError, CachingLLMClient, as_generate_fn

            cache = args.cache or f"examples/.cache/reasoning_{args.benchmark}.jsonl"
            base = BedrockClient(region=args.region, max_tokens=512)
            client = CachingLLMClient(
                base, cache, max_calls=args.max_calls, max_spend_usd=args.max_spend
            )
            generate = as_generate_fn(client, args.model, temperature=0.7)
            problems = RS_BENCH[args.benchmark][0](args.n_problems)
            budget_error, model_label = BudgetError, args.model
        except Exception as e:  # noqa: BLE001
            print(f"Could not initialize the real-LLM characterization: {e}\n"
                  "  uv sync --extra llm --extra bench  (+ AWS Bedrock access)\n"
                  "  smoke-test with: --mock")
            return

    print(f"{args.benchmark.upper()} reasoning-tree characterization: {len(problems)} problems, "
          f"model {model_label}, branching {args.branching}, {args.n_steps} steps, "
          f"{args.rollouts} rollouts/node")
    try:
        agg = characterize(problems, generate, cfg, extract_fn, grade_fn, args.tau, budget_error)
    except budget_error as e:
        print(f"[budget stop] {e}")
        return
    _write_outputs(agg, model_label, args.benchmark, len(problems))


if __name__ == "__main__":
    main()
