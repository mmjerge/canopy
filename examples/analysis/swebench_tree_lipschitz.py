"""Is the repository-level code value function almost tree-$K$-Lipschitz? (real-code pairing).

This is the code counterpart of ``reasoning_tree_lipschitz.py`` and the real-data instantiation
of the synthetic ``violation_family`` / ``adversarial_spike`` study (examples/tree_bandits/
violation_regret_demo.py). The synthetic task posits a tree whose value is smooth except at a few
pivotal decisions (violations); here we *measure* whether repository-level patch search has that
same structure.

A node is a candidate patch for a SWE-bench issue. Its
  * cheap value (biased, cheap): the fraction of the issue's FAIL_TO_PASS tests it makes pass
    (``swebench_eval.cheap_value``) -- ignores regressions, so it is a cheap biased proxy;
  * true value (expensive, unbiased): the official ``resolved`` grade (all FAIL_TO_PASS pass AND
    all PASS_TO_PASS preserved; ``swebench_eval.is_resolved``).

We run the same value-guided patch search as the flagship SWE-bench experiment, logging per
candidate the (cheap, true) pair and, per refinement step, the sibling true-value spread, then
measure:
  1. **Informative edge (tree-Lipschitz backbone).** Correlation of cheap probe vs true resolution
     across candidates, and how often the highest-cheap candidate is also a highest-true one.
  2. **Violations $K$.** The fraction of steps whose sibling true-value spread is decisive (some
     sibling resolves, others do not) -- the pivotal, near-discontinuity steps.

The LLM generations reuse the flagship run's response cache (so re-running is free of API cost);
only the official Docker harness grading is recomputed per candidate. Run on the experiment box:
    ~/canopy/.venv/bin/python examples/analysis/swebench_tree_lipschitz.py \
        --dataset princeton-nlp/SWE-bench_Verified --n-instances 40 \
        --model us.anthropic.claude-sonnet-4-5-20250929-v1:0 --branching 3 --depth 2 --resume
Smoke-test the whole pipeline (no Docker / creds / deps):
    python examples/analysis/swebench_tree_lipschitz.py --mock --n-instances 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))            # examples/
sys.path.insert(0, str(HERE.parents[1] / "reasoning"))

from canopy.bandits.swebench_eval import (  # noqa: E402
    SWEBENCH_PROMPTS,
    build_patch,
    cheap_value,
    failing_f2p_feedback,
    fetch_oracle_files,
    format_files,
    is_resolved,
    patched_paths,
)

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    tqdm = None

FIGDIR = HERE.parents[2] / "paper" / "figures"
MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
TAU_SWEEP = [0.25, 0.5, 0.75]  # sibling true-value spreads counted as pivotal (violation)


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a):
    a = np.asarray(a, float)
    order = a.argsort(kind="stable")
    ranks = np.empty(len(a), float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _spearman(x, y):
    return _pearson(_rankdata(x), _rankdata(y))


def _bootstrap_ci(vals, iters=2000, seed=0):
    a = np.asarray(vals, float)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(iters, a.size))].mean(axis=1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def measure_instance(inst, generate, grader, branching, depth, max_tokens):
    """Run value-guided patch search on one instance, logging the tree structure.

    Returns per-candidate (cheap, true) records and per-step sibling true-value spreads. The search
    itself is identical to the flagship value_guided_swe (select on cheap probe, refine on failing
    -test feedback); we only additionally record the graded structure.
    """
    files = inst.get("_files", {})
    files_str = format_files(files)
    solve = SWEBENCH_PROMPTS[0].format(repo=inst["repo"], q=inst["problem_statement"],
                                       files=files_str)
    cheap_all, true_all, spreads, edge_hits = [], [], [], []

    # round 0: B fresh candidates
    patches = [build_patch(generate(solve, max_tokens, c), files) for c in range(branching)]
    outs = grader(inst, patches, tag="lip0")
    cv = [cheap_value(o) for o in outs]
    tv = [float(is_resolved(o)) for o in outs]
    cheap_all += cv
    true_all += tv
    spreads.append(max(tv) - min(tv))
    edge_hits.append(float(tv[int(np.argmax(cv))] >= max(tv)))
    bi = int(np.argmax(cv))
    best_patch, best_out = patches[bi], outs[bi]

    # rounds 1..D: refine current best on failing-test feedback; siblings are the new candidates
    for step in range(1, depth + 1):
        feedback = failing_f2p_feedback(best_out)
        refine = SWEBENCH_PROMPTS[1].format(
            repo=inst["repo"], q=inst["problem_statement"], files=files_str,
            prefix=best_patch or "(empty patch)", feedback=feedback,
        )
        cand = [build_patch(generate(refine, max_tokens, 1000 * step + c), files)
                for c in range(branching)]
        couts = grader(inst, cand, tag=f"lip{step}")
        ccv = [cheap_value(o) for o in couts]
        ctv = [float(is_resolved(o)) for o in couts]
        cheap_all += ccv
        true_all += ctv
        spreads.append(max(ctv) - min(ctv))
        edge_hits.append(float(ctv[int(np.argmax(ccv))] >= max(ctv)))
        pool = [best_patch] + cand
        pool_out = [best_out] + couts
        pcv = [cheap_value(o) for o in pool_out]
        bi = int(np.argmax(pcv))
        best_patch, best_out = pool[bi], pool_out[bi]

    return {"cheap": cheap_all, "true": true_all, "spreads": spreads, "edge_hits": edge_hits}


def _load_instances(n, dataset, difficulty, repo):
    from swebench_search import load_swebench

    return load_swebench(n, dataset, difficulty=difficulty, repo=repo)


def characterize(instances, generate, grader, branching, depth, max_tokens, per_instance,
                 checkpoint):
    """Measure the tree structure over instances (checkpointed per instance for resume)."""
    bar = tqdm(total=len(instances), unit="inst", desc="swe-tree") if tqdm else None
    for inst in instances:
        iid = inst["instance_id"]
        if iid in per_instance:
            if bar is not None:
                bar.update(1)
            continue
        if inst.get("base_commit") and "_files" not in inst:
            inst["_files"] = fetch_oracle_files(
                inst["repo"], inst["base_commit"], patched_paths(inst.get("patch", "")))
        try:
            rec = measure_instance(inst, generate, grader, branching, depth, max_tokens)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip {iid}] {type(e).__name__}: {str(e)[:120]}")
            if bar is not None:
                bar.update(1)
            continue
        per_instance[iid] = rec
        checkpoint(per_instance)
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()


def _aggregate(per_instance, tau):
    cheap, true, spreads, edge_hits = [], [], [], []
    for rec in per_instance.values():
        cheap += rec["cheap"]
        true += rec["true"]
        spreads += rec["spreads"]
        edge_hits += rec["edge_hits"]
    spreads = np.asarray(spreads, float)
    edge_hits = np.asarray(edge_hits, float)
    pivotal = spreads > tau
    n_piv = int(pivotal.sum())
    piv_m, piv_lo, piv_hi = _bootstrap_ci(edge_hits[pivotal]) if n_piv else (float("nan"),) * 3
    return {
        "cheap": cheap, "true": true, "spreads": spreads.tolist(),
        "spearman": _spearman(cheap, true), "pearson": _pearson(cheap, true),
        "edge_hit_rate": float(edge_hits.mean()) if edge_hits.size else 0.0,
        "edge_hit_pivotal": piv_m, "edge_hit_pivotal_lo": piv_lo, "edge_hit_pivotal_hi": piv_hi,
        "n_pivotal": n_piv, "n_steps_total": int(spreads.size),
        "K_by_tau": {t: float(np.mean(spreads > t)) if spreads.size else 0.0 for t in TAU_SWEEP},
        "tau": tau, "n_candidates": len(cheap),
    }


def _write_outputs(agg, model, dataset, n_instances, branching, depth):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    base = 1.0 / max(2, branching)
    agg_summary = {k: v for k, v in agg.items() if k not in ("cheap", "true")}
    payload = {"benchmark": "swebench", "dataset": dataset, "model": model,
               "n_instances": n_instances, "branching": branching, "depth": depth,
               "random_edge_hit_rate": base, **agg_summary}
    (FIGDIR / "swebench_tree_lipschitz_results.json").write_text(json.dumps(payload, indent=2))
    ktau = "; ".join(f"$\\tau{{=}}{t}$: {100 * v:.0f}\\%" for t, v in agg["K_by_tau"].items())
    tex = (
        "% SWE-bench tree-Lipschitz characterization (auto-generated by "
        "swebench_tree_lipschitz.py). Cheap probe = FAIL_TO_PASS pass fraction; true = official "
        "resolved.\n"
        "\\begin{tabular}{lr}\n\\toprule\nQuantity & Value \\\\\n\\midrule\n"
        f"Cheap-vs-true patch value (Spearman $\\rho$) & {agg['spearman']:.3f} \\\\\n"
        f"Cheap-vs-true patch value (Pearson $r$) & {agg['pearson']:.3f} \\\\\n"
        f"Edge-following hit rate, all steps (chance {base:.2f}) & {agg['edge_hit_rate']:.3f} "
        "\\\\\n"
        f"Edge-following hit rate, pivotal steps only & {agg['edge_hit_pivotal']:.3f} "
        f"[{agg['edge_hit_pivotal_lo']:.2f}, {agg['edge_hit_pivotal_hi']:.2f}] "
        f"($n$={agg['n_pivotal']}) \\\\\n"
        f"Pivotal-step fraction & {ktau} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
    )
    (FIGDIR / "swebench_tree_lipschitz_table.tex").write_text(tex)
    print(f"\nSWE-bench tree-Lipschitz ({dataset}, {n_instances} instances, "
          f"{agg['n_candidates']} candidate patches), model {model}")
    print(f"  cheap-vs-true resolved: Spearman rho={agg['spearman']:.3f}, "
          f"Pearson r={agg['pearson']:.3f}")
    print(f"  edge-hit rate={agg['edge_hit_rate']:.3f} (chance {base:.2f}); "
          f"pivotal edge-hit={agg['edge_hit_pivotal']:.3f} (n={agg['n_pivotal']})")
    print("  pivotal-step fraction: "
          + ", ".join(f"tau={t}:{100 * v:.0f}%" for t, v in agg["K_by_tau"].items()))
    try:
        _plot(agg, model, dataset, n_instances)
    except Exception as e:  # noqa: BLE001
        print(f"  (figure skipped: {type(e).__name__}: {e})")


def _plot(agg, model, dataset, n_instances):
    from _plotstyle import PALETTE, save_figure, set_style

    set_style()
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))
    cheap, true = np.asarray(agg["cheap"]), np.asarray(agg["true"])
    jit = 0.02 * np.random.default_rng(0).standard_normal(cheap.shape)
    axA.scatter(cheap + jit, true + jit, s=12, alpha=0.3, color=PALETTE["blue"])
    if cheap.size >= 2 and cheap.std() > 0:
        b, a = np.polyfit(cheap, true, 1)
        xs = np.array([cheap.min(), cheap.max()])
        axA.plot(xs, a + b * xs, "-", color=PALETTE["red"], lw=2, label="linear fit")
        axA.legend(loc="center right")
    axA.set_xlabel("cheap probe (FAIL\\_TO\\_PASS pass fraction)")
    axA.set_ylabel("true value (official resolved)")
    axA.set_title(f"Cheap test probe tracks resolution (Spearman $\\rho$={agg['spearman']:.2f})")

    spreads = np.asarray(agg["spreads"])
    if spreads.size:
        axB.hist(spreads, bins=np.linspace(0, 1, 11), color=PALETTE["green"], alpha=0.8, rwidth=0.9)
        for t, v in agg["K_by_tau"].items():
            axB.axvline(t, color=PALETTE["gray"], ls=":", lw=1)
    axB.set_xlabel("sibling true-value spread per refinement step (max $-$ min)")
    axB.set_ylabel("number of steps")
    axB.set_title("Most steps smooth; few are pivotal (violations)")
    fig.suptitle(f"Repository-level code value function: informative edge + few violations "
                 f"({n_instances} SWE-bench issues)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, "swebench_tree_lipschitz")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    ap.add_argument("--difficulty", default="")
    ap.add_argument("--repo", default="")
    ap.add_argument("--n-instances", type=int, default=40)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--branching", type=int, default=3)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--harness-workers", type=int, default=4)
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--run-id", default="canopy_swe_lip")
    ap.add_argument("--work-dir", default="examples/.cache/swebench")
    ap.add_argument("--cache", default="examples/.cache/reasoning_swebench.jsonl")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    if args.mock:
        from canopy.bandits.swebench_eval import MockGrader
        from swebench_search import load_mock_instances, make_mock_generate

        generate = make_mock_generate()
        instances = load_mock_instances(args.n_instances)
        mock = MockGrader()

        def grader(instance, patches, tag="c"):
            return mock.grade(instance, patches)

        model_label = "mock"
    else:
        try:
            from canopy.bandits.swebench_eval import grade_candidates
            from canopy.llm import BedrockClient, CachingLLMClient, as_generate_fn

            base = BedrockClient(region=args.region, max_tokens=args.max_tokens)
            client = CachingLLMClient(base, args.cache)
            generate = as_generate_fn(client, args.model, temperature=0.7)
            instances = _load_instances(args.n_instances, args.dataset, args.difficulty, args.repo)

            def grader(instance, patches, tag="c"):
                return grade_candidates(
                    instance, patches, dataset_name=args.dataset, run_id=args.run_id,
                    workers=args.harness_workers, namespace=args.namespace,
                    work_dir=args.work_dir, tag=f"{instance['instance_id']}_{tag}",
                )

            model_label = args.model
        except Exception as e:  # noqa: BLE001
            print(f"Could not initialize SWE-bench characterization: {type(e).__name__}: {e}\n"
                  "  ~/canopy/.venv/bin/pip install swebench datasets ; --mock to smoke-test")
            return

    results_path = FIGDIR / "swebench_tree_lipschitz_results.json"
    per_instance: dict = {}
    if args.resume and results_path.exists():
        prior = json.loads(results_path.read_text())
        per_instance = prior.get("per_instance", {}) or {}
        if per_instance:
            print(f"resume: {len(per_instance)} instances already measured")

    def checkpoint(pi):
        # keep raw per-instance records so a crash resumes; overwritten with full aggregate at end
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps({"per_instance": pi}, indent=2))

    print(f"SWE-bench tree-Lipschitz: {len(instances)} instances, model {model_label}, "
          f"branching {args.branching}, depth {args.depth}")
    characterize(instances, generate, grader, args.branching, args.depth, args.max_tokens,
                 per_instance, checkpoint)
    if per_instance:
        agg = _aggregate(per_instance, args.tau)
        # preserve raw records alongside the aggregate for resume/reproducibility
        payload = json.loads(results_path.read_text()) if results_path.exists() else {}
        _write_outputs(agg, model_label, args.dataset, len(per_instance), args.branching,
                       args.depth)
        merged = json.loads(results_path.read_text())
        merged["per_instance"] = per_instance
        results_path.write_text(json.dumps(merged, indent=2))
        _ = payload


if __name__ == "__main__":
    main()
