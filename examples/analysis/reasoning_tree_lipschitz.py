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

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    tqdm = None

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


TAU_SWEEP = [0.25, 0.5, 0.75]  # sibling true-value spreads counted as "pivotal" (violation)


def _one_problem(q, gold, generate, cfg, extract_fn, grade_fn, eps):
    """Run value-guided search on one problem and return its per-step contributions.

    Independent of every other problem, so many of these run concurrently. Different problems
    have different prompts, so their cache keys don't collide; the search *within* a problem is
    sequential (each step depends on the chosen prefix).
    """
    branching, n_steps, rollouts = cfg
    trace: list[dict] = []
    value_guided_search(
        q, gold, generate, branching=branching, n_steps=n_steps, rollouts=rollouts,
        final_rollouts=rollouts, extract_fn=extract_fn, grade_fn=grade_fn, trace_log=trace,
    )
    by_step: dict[int, list[dict]] = {}
    cheap, true = [], []
    for rec in trace:
        cheap.append(rec["cheap_value"]); true.append(rec["true_value"])
        by_step.setdefault(rec["step"], []).append(rec)
    spreads, ehits, egaps, pv = [], [], [], {}
    for step, recs in by_step.items():
        tv = np.array([r["true_value"] for r in recs])
        cv = np.array([r["cheap_value"] for r in recs])
        spreads.append(float(tv.max() - tv.min()))
        ehits.append(float(tv[int(cv.argmax())] >= tv.max() - eps))
        egaps.append(float(tv.max() - tv[int(cv.argmax())]))
        pv[step] = next(r["true_value"] for r in recs if r["chosen"])
    return {"cheap": cheap, "true": true, "spreads": spreads, "ehits": ehits,
            "egaps": egaps, "pv": pv, "nsteps": len(by_step)}


def characterize(problems, generate, cfg, extract_fn, grade_fn, tau, budget_error, workers=8):
    """Run value-guided search with tree logging over problems (in parallel); aggregate records.

    Beyond the cheap-vs-true correlation, we record the metric that directly tests the value
    edge: at each step, whether the highest-*cheap*-value sibling is also a highest-*true*-value
    sibling (``edge_hit``), and the true value lost by following the cheap edge instead of the
    best sibling (``edge_gap``). Problems run on a thread pool (``workers``): the model calls are
    I/O-bound, so this is a near-linear speedup, and the per-call progress bar is updated from a
    thread-safe wrapper.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    branching, n_steps, rollouts = cfg
    all_cheap, all_true, sibling_spreads, edge_hits, edge_gaps = [], [], [], [], []
    per_problem_steps = []
    path_value_by_step = [[] for _ in range(n_steps)]
    eps = 1.0 / max(1, rollouts) / 2.0
    calls_per_problem = n_steps * branching * (1 + rollouts) + rollouts
    bar = (tqdm(total=len(problems) * calls_per_problem, unit="call", desc="reasoning-tree")
           if tqdm else None)

    def gen(prompt, max_tokens, seed):
        out = generate(prompt, max_tokens, seed)
        if bar is not None:
            bar.update(1)  # tqdm.update is thread-safe
        return out

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(_one_problem, q, gold, gen, cfg, extract_fn, grade_fn, eps): (q, gold)
                   for q, gold in problems}
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except budget_error:
                break  # spend/call cap hit; aggregate whatever finished
            all_cheap += r["cheap"]; all_true += r["true"]
            sibling_spreads += r["spreads"]; edge_hits += r["ehits"]; edge_gaps += r["egaps"]
            for step, v in r["pv"].items():
                path_value_by_step[step].append(v)
            per_problem_steps.append(r["nsteps"])
            done += 1
            if bar is not None:
                hr = float(np.mean(edge_hits)) if edge_hits else 0.0
                sp = float(np.mean(sibling_spreads)) if sibling_spreads else 0.0
                bar.set_postfix_str(f"{done}/{len(problems)} prob, edge-hit={hr:.2f} "
                                    f"spread={sp:.2f}")
    if bar is not None:
        bar.close()
    spreads = np.array(sibling_spreads)
    ehits = np.array(edge_hits)
    egaps = np.array(edge_gaps)
    # The decisive number: edge-following *on the pivotal steps* (sibling spread > tau), where the
    # choice actually matters. The unconditional edge-hit is inflated by the many smooth steps
    # whose siblings are ~equal (any pick trivially "hits"), so we report both.
    pivotal = spreads > tau
    n_piv = int(pivotal.sum())
    return {
        "cheap": all_cheap, "true": all_true,
        "sibling_spreads": sibling_spreads,
        "edge_hit_rate": float(ehits.mean()) if ehits.size else 0.0,
        "edge_gap": float(egaps.mean()) if egaps.size else 0.0,
        "edge_hit_pivotal": float(ehits[pivotal].mean()) if n_piv else float("nan"),
        "edge_gap_pivotal": float(egaps[pivotal].mean()) if n_piv else float("nan"),
        "n_pivotal": n_piv, "n_steps_total": int(spreads.size),
        "K_by_tau": {t: float(np.mean(spreads > t)) if spreads.size else 0.0 for t in TAU_SWEEP},
        "n_steps_seen": per_problem_steps,
        "path_value_by_step": [float(np.mean(v)) if v else float("nan")
                               for v in path_value_by_step],
        "tau": tau, "branching": branching, "n_steps": n_steps, "rollouts": rollouts,
    }


def _write_outputs(agg, model, bench, n_problems):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    r_p = _pearson(agg["cheap"], agg["true"])
    r_s = _spearman(agg["cheap"], agg["true"])
    steps = np.array(agg["n_steps_seen"], float)
    base = 1.0 / max(2, agg["branching"])  # edge-hit rate a random pick would achieve
    stem = f"reasoning_tree_lipschitz_{bench}"
    payload = {
        "benchmark": bench, "model": model, "n_problems": n_problems,
        "pearson_cheap_true": r_p, "spearman_cheap_true": r_s,
        "edge_hit_rate": agg["edge_hit_rate"], "random_edge_hit_rate": base,
        "edge_gap": agg["edge_gap"],
        "edge_hit_pivotal": agg["edge_hit_pivotal"], "edge_gap_pivotal": agg["edge_gap_pivotal"],
        "n_pivotal": agg["n_pivotal"], "n_steps_total": agg["n_steps_total"], "tau": agg["tau"],
        "K_by_tau": agg["K_by_tau"],
        "max_steps": int(steps.max()) if steps.size else 0,
        "path_value_by_step": agg["path_value_by_step"],
        "branching": agg["branching"], "n_steps": agg["n_steps"],
        "rollouts": agg["rollouts"], "n_nodes": len(agg["cheap"]),
    }
    (FIGDIR / f"{stem}_results.json").write_text(json.dumps(payload, indent=2))

    ktau = "; ".join(f"$\\tau{{=}}{t}$: {100 * v:.0f}\\%" for t, v in agg["K_by_tau"].items())
    tex = (
        f"% Reasoning-tree Lipschitz characterization ({bench.upper()}, "
        "auto-generated by reasoning_tree_lipschitz.py)\n"
        "\\begin{tabular}{lr}\n\\toprule\nQuantity & Value \\\\\n\\midrule\n"
        f"Cheap-vs-true node value (Spearman $\\rho$) & {r_s:.3f} \\\\\n"
        f"Cheap-vs-true node value (Pearson $r$) & {r_p:.3f} \\\\\n"
        f"Edge-following hit rate, all steps (chance {base:.2f}) & {agg['edge_hit_rate']:.3f} "
        "\\\\\n"
        f"Edge-following hit rate, pivotal steps only & {agg['edge_hit_pivotal']:.3f} "
        f"($n$={agg['n_pivotal']}) \\\\\n"
        f"Mean true value lost per step (edge gap) & {agg['edge_gap']:.3f} \\\\\n"
        f"Pivotal-step fraction & {ktau} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
    )
    (FIGDIR / f"{stem}_table.tex").write_text(tex)

    print(f"\n{bench.upper()} reasoning-tree characterization ({n_problems} problems, "
          f"{payload['n_nodes']} nodes), model {model}")
    print(f"  cheap-vs-true value: Spearman rho={r_s:.3f} (headline), Pearson r={r_p:.3f}")
    print(f"  edge-following hit rate={agg['edge_hit_rate']:.3f} (chance {base:.2f}); "
          f"mean edge gap={agg['edge_gap']:.3f}")
    print(f"  edge-following hit rate on PIVOTAL steps (spread>{agg['tau']})="
          f"{agg['edge_hit_pivotal']:.3f}  (n={agg['n_pivotal']}/{agg['n_steps_total']}; "
          f"gap={agg['edge_gap_pivotal']:.3f})  <- the decisive number")
    print("  pivotal-step fraction: "
          + ", ".join(f"tau={t}:{100 * v:.0f}%" for t, v in agg["K_by_tau"].items()))
    print("  chosen-path true value by step: "
          + ", ".join(f"{v:.2f}" for v in agg["path_value_by_step"]))
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
    # Panel A: cheap probe vs true node value (informativeness); headline is Spearman + edge-hit
    cheap, true = np.asarray(agg["cheap"]), np.asarray(agg["true"])
    jit = 0.01 * np.random.default_rng(0).standard_normal(cheap.shape)
    axA.scatter(cheap + jit, true + jit, s=10, alpha=0.25, color=PALETTE["blue"])
    if cheap.size >= 2:
        b, a = np.polyfit(cheap, true, 1)
        xs = np.array([cheap.min(), cheap.max()])
        axA.plot(xs, a + b * xs, "-", color=PALETTE["red"], lw=2, label="linear fit")
    axA.set_xlabel("cheap probe value (self-consistency)")
    axA.set_ylabel("true node value (fraction of rollouts correct)")
    axA.set_title(f"Cheap probe tracks true value (Spearman $\\rho$="
                  f"{payload['spearman_cheap_true']:.2f}, edge-hit "
                  f"{payload['edge_hit_rate']:.2f} vs {payload['random_edge_hit_rate']:.2f})")
    axA.legend(loc="upper left")

    # Panel B: distribution of sibling true-value spreads (the "violation size" spectrum)
    spreads = np.asarray(agg["sibling_spreads"])
    if spreads.size:
        axB.hist(spreads, bins=np.linspace(0, 1, 11), color=PALETTE["green"], alpha=0.8,
                 rwidth=0.9)
        for t, v in agg["K_by_tau"].items():
            axB.axvline(t, color=PALETTE["gray"], ls=":", lw=1)
            axB.text(t, axB.get_ylim()[1] * 0.9, f" {100 * v:.0f}%>$\\tau$", fontsize=7,
                     color=PALETTE["gray"])
    axB.set_xlabel("sibling true-value spread per step (max $-$ min)")
    axB.set_ylabel("number of steps")
    axB.set_title("Most steps are smooth; few are pivotal (violations)")

    fig.suptitle(f"Reasoning value function: tree-Lipschitz backbone + few violations "
                 f"({n_problems} {bench.upper()}, {model})")
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
    ap.add_argument("--rollouts", type=int, default=8,
                    help="rollouts per node: more = less-noisy true-value estimate (>=8 advised)")
    ap.add_argument("--tau", type=float, default=0.5,
                    help="reference pivotal-step threshold (a sweep is always reported too)")
    ap.add_argument("--workers", type=int, default=8,
                    help="problems to run concurrently (I/O-bound; ~linear speedup)")
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
          f"{args.rollouts} rollouts/node, {args.workers} workers")
    try:
        agg = characterize(problems, generate, cfg, extract_fn, grade_fn, args.tau,
                           budget_error, workers=args.workers)
    except budget_error as e:
        print(f"[budget stop] {e}")
        return
    _write_outputs(agg, model_label, args.benchmark, len(problems))


if __name__ == "__main__":
    main()
