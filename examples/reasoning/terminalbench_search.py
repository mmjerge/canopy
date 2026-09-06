"""Value-guided search vs. best-of-N on Terminal-Bench 2.x (Harbor-backed, fresh container).

The terminal-task counterpart to ``swebench_search.py``: the same value-guided multi-fidelity
method (cheap probe guides expansion/refinement of full candidate solutions; matched generation
budget vs. best-of-N), applied to Terminal-Bench 2.x tasks via the Harbor harness. Each candidate
is a full bash solution graded in a FRESH container by the task's verifier -- no live-terminal
state cloning (which is what sank the ALFWorld agentic attempt).

  * best-of-N: sample N=B*(D+1) candidate scripts, select the one passing the most tests (cheap
    probe), grade the selection by the official all-tests-pass criterion (leaf).
  * value-guided: sample B scripts, keep the best by cheap probe, then D test-feedback refinement
    rounds; grade the final best.

Both select on the same public signal (fraction of tests passing); only the final choice is graded
for full resolution. Per-task 0/1 resolution is recorded for paired bootstrap CIs.

Runs in the Python 3.12 harbor venv on the box (Harbor requires >=3.12), against a local
terminal-bench 2.x tasks directory (git clone of github.com/harbor-framework/terminal-bench-2-1):
    ~/harbor-venv/bin/python examples/reasoning/terminalbench_search.py \
        --tasks-dir examples/.cache/terminalbench/tb21/tasks --n-tasks 20 \
        --model bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
        --branching 3 --depth 1 --resume

Smoke-test the whole pipeline (search, feedback, matched budget, resume, table) with no Harbor:
    python examples/reasoning/terminalbench_search.py --mock --n-tasks 12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from canopy.bandits.terminalbench_eval import (
    TERMINALBENCH_PROMPTS,
    cheap_value,
    extract_script,
    is_resolved,
    load_terminalbench_tasks,
    test_feedback,
)

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    tqdm = None

FIGDIR = Path(__file__).resolve().parents[2] / "paper" / "figures"


def matched_budget(branching: int, depth: int) -> int:
    return branching * (depth + 1)


def _best_index(outcomes: list[dict]) -> int:
    best_i, best_v = 0, -1.0
    for i, o in enumerate(outcomes):
        v = cheap_value(o)
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def best_of_n_tb(task, generate, grader, n, max_tokens):
    prompt = TERMINALBENCH_PROMPTS[0].format(q=task["instruction"])
    scripts, calls = [], 0
    for i in range(n):
        text = generate(prompt, max_tokens, i)
        calls += 1
        scripts.append(extract_script(text))
    outcomes = grader(task, scripts, tag="bo")
    sel = _best_index(outcomes)
    return {"resolved": int(is_resolved(outcomes[sel])), "calls": calls}


def value_guided_tb(task, generate, grader, branching, depth, max_tokens):
    solve = TERMINALBENCH_PROMPTS[0].format(q=task["instruction"])
    calls = 0
    scripts = []
    for c in range(branching):
        text = generate(solve, max_tokens, c)
        calls += 1
        scripts.append(extract_script(text))
    outcomes = grader(task, scripts, tag="vg0")
    bi = _best_index(outcomes)
    best_script, best_outcome = scripts[bi], outcomes[bi]

    for step in range(1, depth + 1):
        refine = TERMINALBENCH_PROMPTS[1].format(
            q=task["instruction"],
            prefix=best_script or "(empty)",
            feedback=test_feedback(best_outcome),
        )
        cand = []
        for c in range(branching):
            text = generate(refine, max_tokens, 1000 * step + c)
            calls += 1
            cand.append(extract_script(text))
        cand_out = grader(task, cand, tag=f"vg{step}")
        pool, pool_out = [best_script] + cand, [best_outcome] + cand_out
        bi = _best_index(pool_out)
        best_script, best_outcome = pool[bi], pool_out[bi]

    return {"resolved": int(is_resolved(best_outcome)), "calls": calls}


def run_level(tasks, generate, grader, branching, depth, max_tokens, per_task, checkpoint):
    n = matched_budget(branching, depth)
    bar = tqdm(total=len(tasks), unit="task", desc=f"budget {n}") if tqdm else None
    for task in tasks:
        tid = task["task_id"]
        if tid in per_task:
            if bar is not None:
                bar.update(1)
            continue
        try:
            bo = best_of_n_tb(task, generate, grader, n, max_tokens)
            vg = value_guided_tb(task, generate, grader, branching, depth, max_tokens)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip {tid}] {type(e).__name__}: {str(e)[:120]}")
            if bar is not None:
                bar.update(1)
            continue
        per_task[tid] = {"bo": bo["resolved"], "vg": vg["resolved"]}
        checkpoint(per_task)
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()
    return n


def _hits(per_task):
    return [v["bo"] for v in per_task.values()], [v["vg"] for v in per_task.values()]


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


def _stem(tag):
    return f"reasoning_search_{tag}"


def write_checkpoint(per_task, budget, model, dataset, tag):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    (FIGDIR / f"{_stem(tag)}_results.json").write_text(
        json.dumps(
            {
                "benchmark": tag,
                "dataset": dataset,
                "model": model,
                "n_tasks": len(per_task),
                "level": {"matched_budget": budget, "per_task": per_task},
            },
            indent=2,
        )
    )


def _write_outputs(per_task, budget, model, dataset, tag="terminalbench"):
    if not per_task:
        return
    write_checkpoint(per_task, budget, model, dataset, tag)
    b = budget
    bo_hits, vg_hits = _hits(per_task)
    bo_m, bo_lo, bo_hi = _bootstrap_ci(bo_hits, seed=b)
    vg_m, vg_lo, vg_hi = _bootstrap_ci(vg_hits, seed=b + 1)
    d_m, d_lo, d_hi = _paired_delta_ci(bo_hits, vg_hits, seed=b)
    tex = (
        "% Terminal-Bench value-guided vs best-of-N (auto-generated by terminalbench_search.py). "
        "Resolved rate; Delta is the paired (same-task) gap with a 95% bootstrap CI over tasks.\n"
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "Budget (calls) & best-of-N resolved [95\\% CI] & value-guided resolved [95\\% CI] "
        "& $\\Delta$ (paired) [95\\% CI] \\\\\n\\midrule\n"
        f"{b} & {bo_m:.3f} [{bo_lo:.2f}, {bo_hi:.2f}] & {vg_m:.3f} [{vg_lo:.2f}, {vg_hi:.2f}] & "
        f"${d_m:+.3f}$ [{d_lo:+.2f}, {d_hi:+.2f}] \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
    )
    (FIGDIR / f"{_stem(tag)}_table.tex").write_text(tex)
    print(f"\nTerminal-Bench ({dataset}, {len(per_task)} tasks), model {model}")
    print(
        f"  budget {b}: best-of-N resolved={bo_m:.3f} [{bo_lo:.2f},{bo_hi:.2f}]  "
        f"value-guided resolved={vg_m:.3f} [{vg_lo:.2f},{vg_hi:.2f}]  "
        f"delta={d_m:+.3f} [{d_lo:+.2f},{d_hi:+.2f}]"
    )


def make_mock_generate(seed: int = 0):
    import random

    def generate(prompt: str, max_tokens: int, call_seed: int) -> str:
        rng = random.Random(hash((prompt, call_seed, seed)) & 0xFFFFFFFF)
        refining = "Test feedback" in prompt
        body = "FIX\n" if rng.random() < (0.55 if refining else 0.30) else ""
        return f"```bash\n#!/bin/bash\n{body}echo done {call_seed}\n```"

    return generate


def load_mock_tasks(n: int):
    return [
        {
            "task_id": f"mock-task-{i}",
            "instruction": f"Mock task #{i}: create a file and populate it correctly.",
        }
        for i in range(n)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tasks-dir",
        default="examples/.cache/terminalbench/tb21/tasks",
        help="local tasks directory (a terminal-bench 2.x git clone's tasks/)",
    )
    ap.add_argument("--n-tasks", type=int, default=20)
    ap.add_argument("--model", default="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--branching", type=int, default=3)
    ap.add_argument("--depth", type=int, default=1, help="refinement rounds; budget = B*(depth+1)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--jobs-dir", default="examples/.cache/terminalbench_jobs")
    ap.add_argument("--harbor-bin", default="harbor")
    ap.add_argument("--cache", default="examples/.cache/reasoning_terminalbench.jsonl")
    ap.add_argument("--tag", default="terminalbench")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--mock", action="store_true", help="deterministic mock model+grader, no Harbor"
    )
    args = ap.parse_args()

    if args.mock:
        from canopy.bandits.terminalbench_eval import MockGrader

        generate = make_mock_generate()
        tasks = load_mock_tasks(args.n_tasks)
        mock = MockGrader()

        def grader(task, scripts, tag="c"):
            return mock.grade(task, scripts)

        model_label = "mock"
    else:
        try:
            from canopy.bandits.terminalbench_eval import grade_candidates
            from canopy.llm import BedrockClient, CachingLLMClient, as_generate_fn

            base = BedrockClient(region=args.region, max_tokens=args.max_tokens)
            client = CachingLLMClient(base, args.cache)
            # model label may be prefixed 'bedrock/'; BedrockClient wants the bare id
            gen_model = args.model.split("bedrock/")[-1]
            generate = as_generate_fn(client, gen_model, temperature=0.7)
            tasks = load_terminalbench_tasks(args.n_tasks, args.tasks_dir)
            if not tasks:
                print(
                    f"No tasks found under {args.tasks_dir} -- clone a terminal-bench 2.x "
                    "repo there, e.g.\n  git clone --depth 1 "
                    "https://github.com/harbor-framework/terminal-bench-2-1 "
                    "examples/.cache/terminalbench/tb21"
                )
                return

            def grader(task, scripts, tag="c"):
                return grade_candidates(
                    task,
                    scripts,
                    jobs_dir=args.jobs_dir,
                    harbor_bin=args.harbor_bin,
                    timeout=args.timeout,
                    tag=tag,
                )

            model_label = args.model
        except Exception as e:  # noqa: BLE001
            print(
                f"Could not initialize Terminal-Bench experiment: {type(e).__name__}: {e}\n"
                "Run in ~/harbor-venv (Python 3.12) with harbor + terminal-bench + canopy "
                "installed and Docker running. Smoke-test with no deps: --mock"
            )
            return

    budget = matched_budget(args.branching, args.depth)
    stem = _stem(args.tag)
    per_task: dict = {}
    if args.resume and (FIGDIR / f"{stem}_results.json").exists():
        prior = json.loads((FIGDIR / f"{stem}_results.json").read_text())
        per_task = prior.get("level", {}).get("per_task", {}) or {}
        if per_task:
            print(f"resume: {len(per_task)} tasks already graded; continuing")

    dataset_label = "mock" if args.mock else str(args.tasks_dir)

    def checkpoint(pt):
        write_checkpoint(pt, budget, model_label, dataset_label, args.tag)

    print(
        f"Terminal-Bench search: {len(tasks)} tasks from {dataset_label}, "
        f"model {model_label}, budget B*(D+1)={budget}"
    )
    start = time.monotonic()
    run_level(
        tasks, generate, grader, args.branching, args.depth, args.max_tokens, per_task, checkpoint
    )
    _write_outputs(per_task, budget, model_label, dataset_label, tag=args.tag)
    print(f"  ({(time.monotonic() - start) / 60:.1f}m elapsed)")


if __name__ == "__main__":
    main()
