"""Terminal-Bench grading for value-guided-vs-best-of-N search (Harbor-backed, fresh container).

Mirror of ``swebench_eval.py`` for repository-of-terminal-tasks. Each candidate is a full solution
script; we grade it by running it in a FRESH Harbor container (via the ``CanopyScriptAgent``) and
reading the verifier's per-test outcomes. This realizes our value-guided method on Terminal-Bench
without cloning live terminal state (the ALFWorld failure mode):

  * leaf (expensive, unbiased): the task is "resolved" iff every verifier test passes.
  * cheap probe (biased): the fraction of the task's tests that pass -- a cheaper, partial signal
    that guides which candidate to expand/refine (best-of-N and value-guided both select on it).

Honest scope: this fits the *scriptable* subset of Terminal-Bench tasks (produce commands graded by
tests); genuinely interactive tasks are out of scope and excluded, noted in the paper.

SECURITY / RESOURCES: runs model-generated scripts inside Harbor's Docker sandbox; box only. A
deterministic ``mock`` grader lets the search/aggregation/table pipeline run with no Harbor/Docker.

INTEGRATION NOTES (validate on the box against harbor 0.18): the ``harbor run`` flags in
``run_harbor`` and the results-JSON layout in ``parse_results`` are written to the documented CLI
and should be confirmed with one ``-a oracle`` run before the real pilot.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
from pathlib import Path

_FENCE = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

_TB_SOLVE = (
    "You are solving a task in a Linux terminal (a fresh Docker container). Complete the task by "
    "writing a single self-contained bash script that runs non-interactively from the container's "
    "default working directory.\n\n## Task\n{q}\n\n"
    "Respond with ONLY the script inside a ```bash code block. Do not ask questions or explain; "
    "the script must perform every step needed to satisfy the task's tests."
)
_TB_REFINE = (
    "You are solving a task in a Linux terminal. Your previous script did not pass all of the "
    "task's tests.\n\n## Task\n{q}\n\n## Your current script\n{prefix}\n\n## Test feedback\n"
    "{feedback}\n\nProduce an improved single bash script that fixes the failing tests. Respond "
    "with ONLY the script inside a ```bash code block."
)
_TB_ROLLOUT = _TB_SOLVE

TERMINALBENCH_PROMPTS = (_TB_SOLVE, _TB_REFINE, _TB_ROLLOUT)

_SOLUTION_ENV = "CANOPY_SOLUTION_B64"
AGENT_IMPORT = "canopy.bandits.harbor_agent:CanopyScriptAgent"


def extract_script(text: str) -> str:
    """Pull a bash solution script from a model completion (last ```bash block, else raw text)."""
    if not text:
        return ""
    blocks = _FENCE.findall(text)
    if blocks:
        return blocks[-1].strip() + "\n"
    return text.strip() + "\n" if text.strip() else ""


def run_harbor(task_path: str, script: str, *, jobs_dir: str, harbor_bin: str = "harbor",
               timeout: int = 1800, extra_args: list | None = None) -> None:
    """Run one candidate script in a fresh Harbor container for the task at ``task_path``.

    Validated against harbor 0.18 on the box: a single local task runs via ``harbor run -p
    <task_dir>`` (no registry needed); results land under ``jobs_dir/<timestamp>/<trial>/`` with
    ``verifier/ctrf.json`` (per-test outcomes) and ``verifier/reward.txt`` (0/1 resolution). The
    candidate script is passed base64-encoded in the environment; the CanopyScriptAgent decodes
    and executes it inside the container, then the task's verifier grades the state.
    """
    env = dict(os.environ)
    env[_SOLUTION_ENV] = base64.b64encode(script.encode()).decode()
    cmd = [
        harbor_bin, "run",
        "-p", str(task_path),
        "-a", AGENT_IMPORT,
        "--ae", f"{_SOLUTION_ENV}={env[_SOLUTION_ENV]}",  # pass through to the agent phase
        "-k", "1",
        "-o", str(jobs_dir),
        "-y", "--quiet",
    ]
    if extra_args:
        cmd += list(extra_args)
    subprocess.run(cmd, env=env, timeout=timeout, check=False)


def _find_verifier_dir(jobs_dir: Path) -> Path | None:
    """Newest ``verifier/`` directory under a Harbor jobs dir (one trial per invocation).

    Verified layout (harbor 0.18): ``jobs_dir/<timestamp>/<task>__<id>/verifier/`` containing
    ``ctrf.json`` (per-test outcomes) and ``reward.txt`` ("1" iff the task is resolved).
    """
    cands = [p for p in jobs_dir.rglob("verifier") if p.is_dir()]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def _count_tests(obj) -> tuple[int, int, bool | None]:
    """Best-effort extraction of (n_pass, n_total, resolved) from a Harbor result object.

    Harbor/Terminal-Bench verifiers report per-test outcomes; formats vary across versions, so we
    look for common shapes: a ``tests`` list of {name,status/passed}, or pass/fail counts, or a
    top-level ``resolved``/``passed``. Returns resolved=None if nothing recognizable is found.
    """
    if isinstance(obj, dict):
        # explicit per-test list
        for key in ("tests", "test_results", "results"):
            v = obj.get(key)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                total = len(v)
                npass = sum(1 for t in v if _test_passed(t))
                return npass, total, (npass == total and total > 0)
        # explicit counts
        if "num_passed" in obj and "num_total" in obj:
            npass, total = int(obj["num_passed"]), int(obj["num_total"])
            return npass, total, (npass == total and total > 0)
        for key in ("resolved", "passed", "is_resolved", "success"):
            if key in obj and isinstance(obj[key], bool):
                return (1 if obj[key] else 0), 1, obj[key]
        # recurse into nested dicts (e.g. {"verifier": {...}})
        for v in obj.values():
            r = _count_tests(v)
            if r[2] is not None:
                return r
    return 0, 0, None


def _test_passed(t: dict) -> bool:
    st = t.get("status") or t.get("outcome") or t.get("result")
    if isinstance(st, str):
        return st.lower() in ("passed", "pass", "ok", "success")
    for key in ("passed", "success", "ok"):
        if isinstance(t.get(key), bool):
            return t[key]
    return False


def parse_results(jobs_dir: str | Path, task_id: str) -> dict:
    """Read Harbor verifier results into {resolved, tests_pass, tests_total}.

    Primary source: the trial's ``verifier/ctrf.json`` summary (pytest CTRF: tests/passed) plus
    ``verifier/reward.txt`` (the official 0/1 resolution). Falls back to the generic result-JSON
    walker if the layout differs.
    """
    out = {"resolved": False, "tests_pass": 0, "tests_total": 0}
    vd = _find_verifier_dir(Path(jobs_dir))
    if vd is not None:
        try:
            ctrf = json.loads((vd / "ctrf.json").read_text())
            summary = ctrf.get("results", {}).get("summary", {})
            out["tests_total"] = int(summary.get("tests", 0))
            out["tests_pass"] = int(summary.get("passed", 0))
        except Exception:  # noqa: BLE001 -- verifier may have crashed before writing ctrf
            pass
        try:
            out["resolved"] = (vd / "reward.txt").read_text().strip() == "1"
        except Exception:  # noqa: BLE001
            out["resolved"] = out["tests_total"] > 0 and out["tests_pass"] == out["tests_total"]
        return out
    # fallback: any result JSON with a recognizable shape
    cands = list(Path(jobs_dir).rglob("result*.json"))
    for p in sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            obj = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        npass, total, resolved = _count_tests(obj)
        if resolved is not None:
            out.update(tests_pass=npass, tests_total=total, resolved=bool(resolved))
            break
    return out


def cheap_value(outcome: dict) -> float:
    """Cheap probe in [0,1]: fraction of the task's tests that pass (biased, partial credit)."""
    if outcome.get("tests_total", 0) <= 0:
        return 1.0 if outcome.get("resolved") else 0.0
    return outcome["tests_pass"] / outcome["tests_total"]


def is_resolved(outcome: dict) -> bool:
    """Leaf grade: the task is resolved iff every verifier test passes."""
    return bool(outcome.get("resolved", False))


def test_feedback(outcome: dict) -> str:
    """Short feedback for the refine prompt: how many tests still fail."""
    total = outcome.get("tests_total", 0)
    nfail = max(0, total - outcome.get("tests_pass", 0))
    if total == 0:
        return "The task's tests did not run (the script may have errored or produced no output)."
    if nfail == 0:
        return "All tests pass."
    return f"{nfail} of {total} tests still fail after your script. Reconsider the approach."


def grade_candidates(task: dict, scripts: list[str], *, jobs_dir: str | Path,
                     harbor_bin: str = "harbor", timeout: int = 1800,
                     tag: str = "c") -> list[dict]:
    """Grade several candidate scripts for ONE task, each in its own fresh Harbor container.

    Unlike SWE-bench (where one harness run graded many predictions), Harbor grades one agent run
    per container, so we invoke ``harbor run -p <task_path>`` once per non-empty script. Returns
    one outcome dict per input script, in order. Empty scripts grade as non-resolving without
    invoking Harbor.
    """
    task_id = task["task_id"]
    task_path = task["task_path"]
    jd = Path(jobs_dir)
    outcomes: list[dict] = [{"resolved": False, "tests_pass": 0, "tests_total": 0} for _ in scripts]
    for k, script in enumerate(scripts):
        if not script:
            continue
        cell_dir = jd / f"{task_id}_{tag}_cand{k}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        run_harbor(task_path, script, jobs_dir=str(cell_dir), harbor_bin=harbor_bin,
                   timeout=timeout)
        outcomes[k] = parse_results(cell_dir, task_id)
    return outcomes


class MockGrader:
    """Deterministic no-Harbor grader for smoke tests (mirrors swebench_eval.MockGrader)."""

    def __init__(self, seed: int = 0, n_tests: int = 4):
        self.seed = seed
        self.n_tests = n_tests

    def grade(self, task: dict, scripts: list[str]) -> list[dict]:
        import hashlib

        out = []
        for script in scripts:
            if not script:
                out.append({"resolved": False, "tests_pass": 0, "tests_total": self.n_tests})
                continue
            h = int(hashlib.md5(f"{self.seed}:{script}".encode()).hexdigest(), 16)
            quality = (h % 1000) / 1000.0
            if "FIX" in script:  # mock generator's "solved" signal
                quality = min(1.0, quality + 0.5)
            npass = round(quality * self.n_tests)
            out.append({"resolved": npass == self.n_tests, "tests_pass": npass,
                        "tests_total": self.n_tests})
        return out


def load_terminalbench_tasks(n: int, tasks_dir: str | Path) -> list[dict]:
    """Load up to ``n`` Terminal-Bench tasks from a local tasks directory.

    Validated layout (terminal-bench-2.x git repo, e.g. github.com/harbor-framework/
    terminal-bench-2-1): ``tasks/<task-name>/`` each containing ``instruction.md``, ``task.toml``,
    ``environment/`` and ``tests/``. Returns ``{task_id, task_path, instruction}`` dicts, sorted
    by task name for a deterministic subset.
    """
    root = Path(tasks_dir)
    tasks: list[dict] = []
    for d in sorted(root.iterdir()):
        if len(tasks) >= n:
            break
        instr = d / "instruction.md"
        if not d.is_dir() or not instr.exists():
            continue
        tasks.append({"task_id": d.name, "task_path": str(d),
                      "instruction": instr.read_text(errors="replace")})
    return tasks
