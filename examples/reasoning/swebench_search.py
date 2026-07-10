"""Repository-level value-guided search vs. best-of-N on SWE-bench (real LLM, real repos).

The reasoning-search flagship (``reasoning_search.py``) tops out at single-file code
(HumanEval/MBPP), where the cheap probe is a lone public assert and problems are near-saturated ---
so the value-guided gain there is (honestly) a null. SWE-bench moves the same comparison to
*repository-level* issue resolution, which is (a) what current reviewers expect and (b) the regime
the theory says should favour value-guided most: the repo ships a real test suite, so the cheap
probe (the issue's FAIL_TO_PASS tests) is genuinely informative, and the localize->edit->refine
chain has many decision steps K.

Two strategies at a matched budget of B*(D+1) generation calls:

* best-of-N: sample N = B*(D+1) independent candidate patches from the issue, select the one that
  passes the most FAIL_TO_PASS tests (cheap probe), and grade the selection with the official
  "resolved" criterion (leaf).
* value-guided: sample B patches, keep the best by cheap probe, then run D refinement rounds that
  condition on the failing-test feedback (descending the value edge); grade the final best.

Both are selected on the SAME public signal (FAIL_TO_PASS pass fraction); only the final choice is
scored by the full hidden criterion (FAIL_TO_PASS all pass AND PASS_TO_PASS preserved), so the
comparison is fair. Per-instance 0/1 resolution is recorded for paired bootstrap CIs --- the honest
test, since a benchmark task with a flawed test (a known SWE-bench issue) rejects correct patches
from *both* arms and largely cancels in the paired difference.

Setup on the experiment box (needs Docker + ~120GB disk; pulls prebuilt swebench images):
    ~/canopy/.venv/bin/pip install swebench
    ~/canopy/.venv/bin/python examples/reasoning/swebench_search.py \
        --dataset princeton-nlp/SWE-bench_Verified --n-instances 20 \
        --model us.anthropic.claude-sonnet-4-5-20250929-v1:0 --branching 3 --depth 1 --resume

Smoke-test the whole pipeline (search, feedback loop, matched budget, resume, table, figure) with
no Docker/creds/deps:
    python examples/reasoning/swebench_search.py --mock --n-instances 12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from canopy.bandits.swebench_eval import (
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

FIGDIR = Path(__file__).resolve().parents[2] / "paper" / "figures"
MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


def load_swebench(n: int, dataset_name: str, difficulty: str = "", repo: str = "") -> list[dict]:
    """Load ``n`` SWE-bench instances (issue + test metadata), optionally filtered and repo-balanced.

    Taking the first ``n`` rows biases to one repo (they are sorted by instance id, so SWE-bench
    Verified starts with all-astropy, one of the hardest repos). Instead we optionally filter by
    ``difficulty`` (e.g. "<15 min fix") and/or ``repo``, then round-robin across repos so a small
    pilot is difficulty-controlled and repo-diverse rather than an unsolvable corner.
    """
    from collections import OrderedDict

    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="test")
    rows = list(ds)
    has_diff = "difficulty" in (ds.features or {})
    if difficulty and has_diff:
        rows = [r for r in rows if r.get("difficulty") == difficulty]
    if repo:
        rows = [r for r in rows if r.get("repo") == repo]
    by_repo: "OrderedDict[str, list]" = OrderedDict()
    for r in rows:
        by_repo.setdefault(r["repo"], []).append(r)
    ordered = []
    while any(by_repo.values()) and len(ordered) < len(rows):
        for lst in by_repo.values():
            if lst:
                ordered.append(lst.pop(0))
    items = []
    for row in ordered[:n]:
        items.append({
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "problem_statement": row["problem_statement"],
            "base_commit": row["base_commit"],
            "patch": row["patch"],  # gold patch: used ONLY to localize oracle files (not shown)
        })
    return items


def matched_budget(branching: int, depth: int) -> int:
    """Generation calls value-guided spends = best-of-N's sample count = B*(D+1)."""
    return branching * (depth + 1)


def _best_index(outcomes: list[dict]) -> int:
    """Index of the highest cheap-value candidate (ties -> first)."""
    best_i, best_v = 0, -1.0
    for i, o in enumerate(outcomes):
        v = cheap_value(o)
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def best_of_n_swe(instance, generate, grader, n, max_tokens, run_id):
    """Sample ``n`` patches from the issue, select by cheap probe, grade the selection (leaf)."""
    files = instance.get("_files", {})
    prompt = SWEBENCH_PROMPTS[0].format(repo=instance["repo"], q=instance["problem_statement"],
                                        files=format_files(files))
    patches, calls = [], 0
    for i in range(n):
        text = generate(prompt, max_tokens, i)
        calls += 1
        patches.append(build_patch(text, files))
    outcomes = grader(instance, patches, tag="bo")
    sel = _best_index(outcomes)
    return {"resolved": int(is_resolved(outcomes[sel])), "calls": calls}


def value_guided_swe(instance, generate, grader, branching, depth, max_tokens, run_id):
    """Sample B patches, keep best by cheap probe, then D feedback-conditioned refine rounds."""
    files = instance.get("_files", {})
    files_str = format_files(files)
    solve = SWEBENCH_PROMPTS[0].format(repo=instance["repo"], q=instance["problem_statement"],
                                       files=files_str)
    calls = 0
    # round 0: B fresh candidate patches
    patches = []
    for c in range(branching):
        text = generate(solve, max_tokens, c)
        calls += 1
        patches.append(build_patch(text, files))
    outcomes = grader(instance, patches, tag="vg0")
    bi = _best_index(outcomes)
    best_patch, best_outcome = patches[bi], outcomes[bi]

    # rounds 1..D: refine the current best, conditioned on its failing-test feedback
    for step in range(1, depth + 1):
        feedback = failing_f2p_feedback(best_outcome)
        refine = SWEBENCH_PROMPTS[1].format(
            repo=instance["repo"], q=instance["problem_statement"], files=files_str,
            prefix=best_patch or "(empty patch)", feedback=feedback,
        )
        cand_patches = []
        for c in range(branching):
            text = generate(refine, max_tokens, 1000 * step + c)
            calls += 1
            cand_patches.append(build_patch(text, files))
        cand_outcomes = grader(instance, cand_patches, tag=f"vg{step}")
        # keep the best of {current best} U {new candidates} by cheap value (monotone descent)
        pool = [best_patch] + cand_patches
        pool_out = [best_outcome] + cand_outcomes
        bi = _best_index(pool_out)
        best_patch, best_outcome = pool[bi], pool_out[bi]

    return {"resolved": int(is_resolved(best_outcome)), "calls": calls}


def run_level(instances, generate, grader, branching, depth, max_tokens, run_id):
    """Run both strategies over all instances at the (single) matched budget; return 0/1 lists."""
    n = matched_budget(branching, depth)
    bo_hits, vg_hits = [], []
    bar = tqdm(total=len(instances), unit="inst", desc=f"budget {n}") if tqdm else None
    for inst in instances:
        # oracle context: fetch the current contents of the files the gold patch touches (real
        # instances only; mock instances carry no base_commit and fall back to raw-diff parsing)
        if inst.get("base_commit") and "_files" not in inst:
            inst["_files"] = fetch_oracle_files(
                inst["repo"], inst["base_commit"], patched_paths(inst.get("patch", "")))
        try:
            bo = best_of_n_swe(inst, generate, grader, n, max_tokens, run_id)
            vg = value_guided_swe(inst, generate, grader, branching, depth, max_tokens, run_id)
        except Exception as e:  # noqa: BLE001 -- skip an instance whose harness/grader errored
            print(f"  [skip {inst['instance_id']}] {type(e).__name__}: {str(e)[:120]}")
            if bar is not None:
                bar.update(1)
            continue
        bo_hits.append(bo["resolved"])
        vg_hits.append(vg["resolved"])
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()
    return {"matched_budget": n, "n_instances": len(bo_hits),
            "best_of_n": {"hits": bo_hits}, "value_guided": {"hits": vg_hits}}


def _bootstrap_ci(hits, iters=2000, seed=0):
    import numpy as np

    a = np.asarray(hits, dtype=float)
    if a.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(iters, a.size))].mean(axis=1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _paired_delta_ci(bo_hits, vg_hits, iters=4000, seed=0):
    import numpy as np

    bo, vg = np.asarray(bo_hits, float), np.asarray(vg_hits, float)
    if bo.size == 0 or bo.size != vg.size:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, bo.size, size=(iters, bo.size))
    d = vg[idx].mean(axis=1) - bo[idx].mean(axis=1)
    return float((vg - bo).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _write_outputs(result, model, dataset_name, n_instances, tag="swebench"):
    """Write results JSON + LaTeX table (+ figure) for the single matched-budget SWE-bench level."""
    if not result or result["n_instances"] == 0:
        return
    FIGDIR.mkdir(parents=True, exist_ok=True)
    stem = f"reasoning_search_{tag}"
    (FIGDIR / f"{stem}_results.json").write_text(
        json.dumps({"benchmark": tag, "dataset": dataset_name, "model": model,
                    "n_instances": n_instances, "level": result}, indent=2)
    )
    b = result["matched_budget"]
    bo_m, bo_lo, bo_hi = _bootstrap_ci(result["best_of_n"]["hits"], seed=b)
    vg_m, vg_lo, vg_hi = _bootstrap_ci(result["value_guided"]["hits"], seed=b + 1)
    d_m, d_lo, d_hi = _paired_delta_ci(result["best_of_n"]["hits"],
                                       result["value_guided"]["hits"], seed=b)
    tex = (
        "% SWE-bench value-guided vs best-of-N (auto-generated by swebench_search.py). Resolved "
        "rate; Delta is the paired (same-instance) gap with a 95% bootstrap CI over instances.\n"
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "Budget (calls) & best-of-N resolved [95\\% CI] & value-guided resolved [95\\% CI] "
        "& $\\Delta$ (paired) [95\\% CI] \\\\\n\\midrule\n"
        f"{b} & {bo_m:.3f} [{bo_lo:.2f}, {bo_hi:.2f}] & {vg_m:.3f} [{vg_lo:.2f}, {vg_hi:.2f}] & "
        f"${d_m:+.3f}$ [{d_lo:+.2f}, {d_hi:+.2f}] \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
    )
    (FIGDIR / f"{stem}_table.tex").write_text(tex)
    print(f"\nSWE-bench ({dataset_name}, {result['n_instances']} instances), model {model}")
    print(f"  budget {b}: best-of-N resolved={bo_m:.3f} [{bo_lo:.2f},{bo_hi:.2f}]  "
          f"value-guided resolved={vg_m:.3f} [{vg_lo:.2f},{vg_hi:.2f}]  "
          f"delta={d_m:+.3f} [{d_lo:+.2f},{d_hi:+.2f}]")
    print(f"  wrote json+table to {FIGDIR}")


def make_mock_generate(seed: int = 0):
    """Deterministic patch generator: refine prompts (with feedback) emit a 'FIX' patch more often,
    giving value-guided a real gradient over best-of-N in the mock grader."""
    import hashlib
    import random

    def generate(prompt: str, max_tokens: int, call_seed: int) -> str:
        rng = random.Random(hash((prompt, call_seed, seed)) & 0xFFFFFFFF)
        refining = "Test feedback" in prompt
        p_fix = 0.55 if refining else 0.30
        body = "FIX\n" if rng.random() < p_fix else ""
        h = hashlib.md5(f"{prompt}{call_seed}".encode()).hexdigest()[:8]
        return (f"```diff\ndiff --git a/mod.py b/mod.py\n--- a/mod.py\n+++ b/mod.py\n"
                f"@@ -1,1 +1,2 @@\n {body}+# patch {h}\n```")

    return generate


def load_mock_instances(n: int):
    return [{"instance_id": f"mock__repo-{i}", "repo": "mock/repo",
             "problem_statement": f"Mock issue #{i}: the widget miscomputes the total."} for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    ap.add_argument("--difficulty", default="",
                    help="filter by SWE-bench difficulty tier, e.g. '<15 min fix' (Verified only)")
    ap.add_argument("--repo", default="", help="restrict to one repo, e.g. django/django")
    ap.add_argument("--n-instances", type=int, default=20)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--branching", type=int, default=3)
    ap.add_argument("--depth", type=int, default=1, help="refinement rounds; budget = B*(depth+1)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--harness-workers", type=int, default=4)
    ap.add_argument("--namespace", default=None,
                    help="swebench image namespace; '' forces local builds (arm64). Default: DockerHub.")
    ap.add_argument("--run-id", default="canopy_swe")
    ap.add_argument("--work-dir", default="examples/.cache/swebench")
    ap.add_argument("--cache", default="examples/.cache/reasoning_swebench.jsonl")
    ap.add_argument("--tag", default="swebench")
    ap.add_argument("--resume", action="store_true",
                    help="skip if the results JSON for this tag already has the level")
    ap.add_argument("--mock", action="store_true", help="deterministic mock model+grader, no Docker")
    args = ap.parse_args()

    if args.mock:
        from canopy.bandits.swebench_eval import MockGrader

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
            instances = load_swebench(args.n_instances, args.dataset,
                                       difficulty=args.difficulty, repo=args.repo)

            def grader(instance, patches, tag="c"):
                return grade_candidates(
                    instance, patches, dataset_name=args.dataset, run_id=args.run_id,
                    workers=args.harness_workers, namespace=args.namespace,
                    work_dir=args.work_dir, tag=f"{instance['instance_id']}_{tag}",
                )

            model_label = args.model
        except Exception as e:  # noqa: BLE001
            print(f"Could not initialize SWE-bench experiment: {type(e).__name__}: {e}\n"
                  "Install extras + swebench and configure Docker/AWS on the box:\n"
                  "  ~/canopy/.venv/bin/pip install swebench datasets\n"
                  "Smoke-test with no deps: --mock")
            return

    stem = f"reasoning_search_{args.tag}"
    if args.resume and (FIGDIR / f"{stem}_results.json").exists():
        print(f"resume: {stem}_results.json already exists; nothing to do (delete it to rerun)")
        return

    print(f"SWE-bench search: {len(instances)} instances, dataset {args.dataset}, "
          f"model {model_label}, budget B*(D+1)={matched_budget(args.branching, args.depth)}")
    start = time.monotonic()
    result = run_level(instances, generate, grader, args.branching, args.depth,
                       args.max_tokens, args.run_id)
    _write_outputs(result, model_label, args.dataset, len(instances), tag=args.tag)
    print(f"  ({(time.monotonic() - start) / 60:.1f}m elapsed)")


if __name__ == "__main__":
    main()
