"""SWE-bench grading for the value-guided-vs-best-of-N harness (repo-level code generation).

This extends the reasoning-search comparison from single-file code (HumanEval/MBPP, see
``code_eval.py``) to *repository-level* issue resolution, which is what recent reviewers expect
and --- more importantly --- is the regime where the paper's multi-fidelity claim should be
*strongest*: a real repo ships a test suite, so the cheap probe (running the issue's
FAIL_TO_PASS tests on a candidate patch) is genuinely informative, unlike the single public
assert that made the HumanEval/MBPP probe weak; and the localize -> edit -> refine chain is long
(many decision steps K), which is exactly where value-guided search is predicted to beat
best-of-N.

Multi-fidelity mapping (identical in spirit to ``code_eval.py``):

  * leaf (expensive, unbiased): the OFFICIAL SWE-bench "resolved" criterion --- the candidate
    patch makes every FAIL_TO_PASS test pass AND keeps every PASS_TO_PASS test passing.
  * cheap probe (biased): the fraction of the (typically few) FAIL_TO_PASS tests that pass. This
    ignores regressions (the potentially large PASS_TO_PASS suite), so it is a cheaper, biased
    estimate of a patch's promise --- the signal the value edge / best-of-N selection may use.

Grading is delegated to the official ``swebench.harness.run_evaluation`` harness (containerized,
reproducible, and the number reviewers trust). We give each candidate patch a distinct
``model_name_or_path`` so many candidates for one instance can be graded in a single harness run,
then read each per-instance ``report.json`` for the FAIL_TO_PASS / PASS_TO_PASS breakdown.

SECURITY / RESOURCES: the harness runs model-generated patches inside per-repo Docker containers.
Run it only on a disposable machine (the experiment box), which needs Docker, ~120GB free disk,
and network access to pull the prebuilt ``swebench`` images. A deterministic ``mock`` grader is
provided so the search/aggregation/plotting pipeline is testable with no Docker or credentials.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

# A model may wrap the diff in a ```diff / ```patch fence, in <patch>...</patch>, or emit it raw.
_DIFF_FENCE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_PATCH_TAG = re.compile(r"<patch>\s*(.*?)\s*</patch>", re.DOTALL | re.IGNORECASE)
_DIFF_START = re.compile(r"^(?:diff --git |--- )", re.MULTILINE)


def extract_patch(text: str) -> str:
    """Pull a unified-diff patch out of a model completion.

    Prefers an explicit ``<patch>`` block, then a fenced ```diff block, then the raw text from the
    first ``diff --git`` / ``---`` line onward. Returns a patch string (possibly empty). A trailing
    newline is guaranteed when non-empty, since ``git apply`` requires it.
    """
    if not text:
        return ""
    m = _PATCH_TAG.search(text)
    body = m.group(1) if m else None
    if body is None:
        blocks = _DIFF_FENCE.findall(text)
        body = blocks[-1] if blocks else None
    if body is None:
        m2 = _DIFF_START.search(text)
        body = text[m2.start():] if m2 else ""
    body = body.strip()
    return body + "\n" if body else ""


# --- prompts (repo-level) -------------------------------------------------------------------
# Kept here (not in reasoning_llm) because they are SWE-bench-shaped: issue + repo context in,
# unified diff out. The three-template convention (solve, continue/refine, rollout) matches
# reasoning_llm.MATH_PROMPTS / CODE_PROMPTS so the same search code drives them.

_EDIT_INSTRUCTIONS = (
    "Propose the fix as one or more SEARCH/REPLACE edit blocks. For each edit, copy the EXACT "
    "current lines to change between `<<<<<<< SEARCH` and `=======`, and the new lines between "
    "`=======` and `>>>>>>> REPLACE`. The SEARCH text must match the file byte-for-byte "
    "(indentation included), and include a few surrounding lines so it is unique. Use this format, "
    "one block per edit:\n"
    "### path/to/file.py\n<<<<<<< SEARCH\n<exact current lines, copied verbatim>\n=======\n"
    "<replacement lines>\n>>>>>>> REPLACE\n\n"
    "A standard unified diff (```diff block) is also acceptable if you include enough context "
    "lines. Output only the edits, no explanation."
)
_SWE_SOLVE = (
    "You are fixing a bug in the `{repo}` repository.\n\n## Issue\n{q}\n\n"
    "## Relevant files (current contents)\n{files}\n\n" + _EDIT_INSTRUCTIONS
)
_SWE_REFINE = (
    "You are fixing a bug in the `{repo}` repository. Your previous attempt did not fully resolve "
    "the issue.\n\n## Issue\n{q}\n\n## Relevant files (current contents)\n{files}\n\n"
    "## Your current patch\n{prefix}\n\n## Test feedback\n{feedback}\n\n"
    "Produce improved edits that make the failing tests pass without breaking others. "
    + _EDIT_INSTRUCTIONS
)
_SWE_ROLLOUT = _SWE_SOLVE  # a rollout is a fresh full attempt (patches are not "continued")

SWEBENCH_PROMPTS = (_SWE_SOLVE, _SWE_REFINE, _SWE_ROLLOUT)


# --- oracle file context + search/replace -> unified diff -----------------------------------

_PLUS_FILE = re.compile(r"^\+\+\+ b/(\S+)", re.MULTILINE)
_GIT_FILE = re.compile(r"^diff --git a/\S+ b/(\S+)", re.MULTILINE)
_EDIT_BLOCK = re.compile(
    r"###\s*(?P<path>[^\n]+?)\s*\n+<<<<<<<[ ]*SEARCH\s*\n(?P<search>.*?)\n?=======\s*\n"
    r"(?P<replace>.*?)\n?>>>>>>>[ ]*REPLACE",
    re.DOTALL,
)


def patched_paths(gold_patch: str) -> list[str]:
    """The file paths a (gold) patch modifies -- used for oracle file localization."""
    paths = _PLUS_FILE.findall(gold_patch or "")
    if not paths:
        paths = _GIT_FILE.findall(gold_patch or "")
    # de-dup, preserve order, drop /dev/null (pure deletions)
    seen, out = set(), []
    for p in paths:
        if p != "dev/null" and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def fetch_oracle_files(repo: str, base_commit: str, paths: list[str], timeout: float = 20.0) -> dict:
    """Fetch exact base-commit contents of ``paths`` from GitHub raw (all SWE-bench repos are public).

    Returns ``{path: content}`` for files that fetched successfully (HTTP 200). This is the oracle
    setting: we localize to the files the gold patch edits and give the model their current text,
    isolating the search question from retrieval quality.
    """
    import urllib.request

    files: dict[str, str] = {}
    for path in paths:
        url = f"https://raw.githubusercontent.com/{repo}/{base_commit}/{path}"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                if resp.status == 200:
                    files[path] = resp.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 -- a missing/renamed file just isn't offered as context
            continue
    return files


def format_files(files: dict, max_chars_per_file: int = 24000) -> str:
    """Render ``{path: content}`` as labelled code blocks for the prompt (large files truncated)."""
    parts = []
    for path, content in files.items():
        body = content if len(content) <= max_chars_per_file else (
            content[:max_chars_per_file] + "\n# ... (file truncated) ...\n"
        )
        parts.append(f"### {path}\n```python\n{body}\n```")
    return "\n\n".join(parts) if parts else "(no files retrieved)"


def parse_edits(text: str) -> list[tuple[str, str, str]]:
    """Parse SEARCH/REPLACE blocks into ``[(path, search, replace), ...]`` in document order."""
    out = []
    for m in _EDIT_BLOCK.finditer(text or ""):
        out.append((m.group("path").strip(), m.group("search"), m.group("replace")))
    return out


def _unified_diff(path: str, before: str, after: str) -> str:
    import difflib

    diff = difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}",
    )
    return "".join(diff)


def diff_to_edits(text: str) -> list[tuple[str, str, str]]:
    """Convert a model-authored unified diff into content-anchored ``(path, old, new)`` edits.

    Models (esp. instruct models) tend to emit a ```diff block regardless of instructions, and
    their hunk line numbers are usually wrong -- which is why ``git apply`` fails. We ignore the
    (unreliable) ``@@`` line numbers and instead reconstruct, per hunk, the old text (context +
    removed lines) and new text (context + added lines); re-anchoring that old text against the
    real file (below) produces a correct diff. Robust to the model's natural output.
    """
    m = _PATCH_TAG.search(text or "")
    body = m.group(1) if m else None
    if body is None:
        blocks = _DIFF_FENCE.findall(text or "")
        body = blocks[-1] if blocks else None
    if body is None:
        ms = _DIFF_START.search(text or "")
        body = text[ms.start():] if ms else ""

    edits: list[tuple[str, str, str]] = []
    path, old, new, in_hunk = None, [], [], False

    def flush():
        nonlocal old, new
        if path and (old or new):
            edits.append((path, "\n".join(old), "\n".join(new)))
        old, new = [], []

    for line in body.splitlines():
        if line.startswith("+++ "):
            flush()
            p = line[4:].strip()
            path = p[2:] if p.startswith(("a/", "b/")) else p
            in_hunk = False
        elif line.startswith("--- ") or line.startswith("diff --git"):
            flush()
            in_hunk = False
        elif line.startswith("@@"):
            flush()
            in_hunk = True
        elif in_hunk and line[:1] in (" ", "-", "+"):
            c, content = line[0], line[1:]
            if c in (" ", "-"):
                old.append(content)
            if c in (" ", "+"):
                new.append(content)
    flush()
    return [(p, o, n) for (p, o, n) in edits if o != n]


def _resolve_path(path: str, files: dict) -> str | None:
    """Map an edit's file path to a key in ``files`` (exact, then a/ b/-stripped, then suffix)."""
    if path in files:
        return path
    stripped = path[2:] if path.startswith(("a/", "b/")) else path
    if stripped in files:
        return stripped
    cand = [k for k in files if k.endswith(stripped) or stripped.endswith(k)]
    return cand[0] if len(cand) == 1 else None


def _apply_edits(files: dict, edits: list[tuple[str, str, str]]) -> tuple[dict, list[str]]:
    """Apply content-anchored edits to the real file text (first exact match each); returns
    (updated_files, changed_paths). Edits whose ``old`` text is not found exactly are skipped."""
    after = dict(files)
    changed: list[str] = []
    for path, old, new in edits:
        key = _resolve_path(path, after)
        if key is None or not old or old not in after[key]:
            continue
        after[key] = after[key].replace(old, new, 1)
        if key not in changed:
            changed.append(key)
    return after, changed


def build_patch(text: str, files: dict | None) -> str:
    """Turn a model completion into a mechanically-correct unified-diff patch.

    Accepts either SEARCH/REPLACE blocks (preferred) or a model-authored ```diff (re-anchored by
    content), applies the edits to the exact oracle file text, and regenerates a clean unified diff
    via ``difflib`` -- sidestepping the wrong-line-number problem that makes raw model diffs fail to
    apply. Falls back to :func:`extract_patch` when no oracle files are available (the mock path).
    """
    if not files:
        return extract_patch(text)
    edits = parse_edits(text) or diff_to_edits(text)
    if not edits:
        return extract_patch(text)
    after, changed = _apply_edits(files, edits)
    pieces = [_unified_diff(p, files[p], after[p]) for p in changed if after[p] != files[p]]
    patch = "".join(pieces)
    return patch if (not patch or patch.endswith("\n")) else patch + "\n"


# --- official-harness grading ---------------------------------------------------------------


def write_predictions(path: str | Path, predictions: list[dict]) -> None:
    """Write a SWE-bench predictions JSONL (one object per line).

    Each prediction needs ``instance_id``, ``model_name_or_path`` (we use it to keep multiple
    candidates for the same instance in separate report dirs), and ``model_patch``.
    """
    with open(path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")


def run_harness(
    dataset_name: str,
    predictions_path: str | Path,
    run_id: str,
    instance_ids: list[str],
    workers: int = 4,
    namespace: str | None = None,
    timeout: int = 1800,
    cwd: str | Path | None = None,
) -> None:
    """Invoke ``python -m swebench.harness.run_evaluation`` on a predictions file.

    ``namespace=''`` forces local image builds (needed on arm64 / when DockerHub images are
    missing); ``None`` uses the default ``swebench`` DockerHub namespace (x86_64, much faster).
    Writes per-instance reports under ``<cwd>/logs/run_evaluation/<run_id>/<model>/<instance>/``.
    Raises ``subprocess.CalledProcessError`` on a non-zero exit that is not a per-instance failure.
    """
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset_name,
        "--predictions_path", str(predictions_path),
        "--run_id", run_id,
        "--max_workers", str(workers),
    ]
    if instance_ids:
        cmd += ["--instance_ids", *instance_ids]
    if namespace is not None:
        cmd += ["--namespace", namespace]
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, timeout=timeout, check=False)


def _report_path(log_root: Path, run_id: str, model_name: str, instance_id: str) -> Path:
    return log_root / "run_evaluation" / run_id / model_name / instance_id / "report.json"


def parse_report(log_root: str | Path, run_id: str, model_name: str, instance_id: str) -> dict:
    """Read one per-instance ``report.json`` into a normalized outcome dict.

    Returns ``{resolved, f2p_pass, f2p_total, p2p_pass, p2p_total, applied}``. A missing report
    (patch failed to apply, harness error, or empty patch) grades as an unresolved, zero-pass
    non-application rather than raising.
    """
    p = _report_path(Path(log_root), run_id, model_name, instance_id)
    out = {"resolved": False, "f2p_pass": 0, "f2p_total": 0, "p2p_pass": 0,
           "p2p_total": 0, "applied": False}
    if not p.exists():
        return out
    try:
        rep = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return out
    # report.json is keyed by instance_id at the top level.
    inst = rep.get(instance_id, rep)
    out["resolved"] = bool(inst.get("resolved", False))
    out["applied"] = bool(inst.get("patch_successfully_applied", inst.get("applied", False)))
    status = inst.get("tests_status", {}) or {}
    f2p = status.get("FAIL_TO_PASS", {}) or {}
    p2p = status.get("PASS_TO_PASS", {}) or {}
    fs, ff = f2p.get("success", []) or [], f2p.get("failure", []) or []
    ps, pf = p2p.get("success", []) or [], p2p.get("failure", []) or []
    out.update(f2p_pass=len(fs), f2p_total=len(fs) + len(ff),
               p2p_pass=len(ps), p2p_total=len(ps) + len(pf))
    return out


def cheap_value(outcome: dict) -> float:
    """Cheap probe in [0,1]: fraction of FAIL_TO_PASS tests the patch makes pass.

    Ignores the PASS_TO_PASS regression suite, so it is biased-but-cheap --- the SWE-bench analogue
    of the public-test probe in ``code_eval.public_test_value``. A patch that fails to apply scores
    0. When an instance has no FAIL_TO_PASS listed (should not happen), falls back to ``resolved``.
    """
    if outcome.get("f2p_total", 0) <= 0:
        return 1.0 if outcome.get("resolved") else 0.0
    return outcome["f2p_pass"] / outcome["f2p_total"]


def is_resolved(outcome: dict) -> bool:
    """Leaf grade: the official SWE-bench resolved criterion (all F2P pass, all P2P preserved)."""
    return bool(outcome.get("resolved", False))


def failing_f2p_feedback(outcome: dict, max_lines: int = 20) -> str:
    """A short, human-readable feedback string for the refine prompt (which F2P tests still fail)."""
    n_fail = max(0, outcome.get("f2p_total", 0) - outcome.get("f2p_pass", 0))
    if not outcome.get("applied", False):
        return "The patch did not apply cleanly to the repository (check file paths and context)."
    if n_fail == 0 and outcome.get("p2p_total", 0) and outcome.get("p2p_pass", 0) < outcome["p2p_total"]:
        return ("All target (FAIL_TO_PASS) tests pass, but the patch breaks existing "
                "(PASS_TO_PASS) tests. Make the fix narrower so it does not regress other behavior.")
    if n_fail == 0:
        return "All target tests pass."
    return (f"{n_fail} of {outcome['f2p_total']} target tests still fail after your patch. "
            "Revisit the root cause and adjust the fix.")


# --- batch grading orchestration + mock -----------------------------------------------------


def _cand_model_name(tag: str, k: int) -> str:
    """Per-candidate ``model_name_or_path`` so K candidates for one instance get separate reports."""
    return f"{tag}__cand{k}"


def grade_candidates(
    instance: dict,
    patches: list[str],
    *,
    dataset_name: str,
    run_id: str,
    workers: int = 4,
    log_root: str | Path = "logs",
    namespace: str | None = None,
    work_dir: str | Path = ".",
    tag: str = "c",
) -> list[dict]:
    """Grade several candidate patches for ONE instance via a single official-harness run.

    Returns one normalized outcome dict per input patch (see :func:`parse_report`), in order.
    Empty patches are graded as non-applying without being sent to the harness. All non-empty
    candidates are written to one predictions file (distinct ``model_name_or_path`` each) and
    graded in a single ``run_evaluation`` invocation, then their reports are parsed back.
    """
    iid = instance["instance_id"]
    # Absolute so the path is unambiguous regardless of the harness subprocess cwd (which we set
    # to ``work`` so its ``logs/`` land there); a relative path would be resolved against cwd twice.
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    preds, index = [], {}
    for k, patch in enumerate(patches):
        if not patch:
            continue
        mname = _cand_model_name(tag, k)
        index[k] = mname
        preds.append({"instance_id": iid, "model_name_or_path": mname, "model_patch": patch})

    outcomes: list[dict] = [
        {"resolved": False, "f2p_pass": 0, "f2p_total": 0, "p2p_pass": 0, "p2p_total": 0,
         "applied": False}
        for _ in patches
    ]
    if not preds:
        return outcomes

    preds_path = work / f"preds_{run_id}_{tag}.jsonl"
    write_predictions(preds_path, preds)
    run_harness(dataset_name, preds_path, run_id, [iid], workers=workers, namespace=namespace,
                cwd=work)
    for k, mname in index.items():
        outcomes[k] = parse_report(work / "logs", run_id, mname, iid)
    return outcomes


class MockGrader:
    """Deterministic, no-Docker grader for smoke tests of the search/aggregation/plot pipeline.

    It fabricates a plausible F2P/P2P outcome from a hash of the patch text so that (a) different
    candidates get different scores (search has something to select on) and (b) "better-looking"
    patches (those a mock generator marks with more ``+`` lines / a ``FIX`` token) tend to resolve,
    giving value-guided refinement a gradient to climb. No real grading occurs.
    """

    def __init__(self, seed: int = 0, f2p_total: int = 3, p2p_total: int = 5):
        self.seed = seed
        self.f2p_total = f2p_total
        self.p2p_total = p2p_total

    def grade(self, instance: dict, patches: list[str]) -> list[dict]:
        import hashlib

        out = []
        for patch in patches:
            if not patch:
                out.append({"resolved": False, "f2p_pass": 0, "f2p_total": self.f2p_total,
                            "p2p_pass": 0, "p2p_total": self.p2p_total, "applied": False})
                continue
            h = int(hashlib.md5(f"{self.seed}:{patch}".encode()).hexdigest(), 16)
            quality = (h % 1000) / 1000.0
            if "FIX" in patch:  # mock generator's signal that it "solved" the issue
                quality = min(1.0, quality + 0.5)
            f2p_pass = round(quality * self.f2p_total)
            p2p_pass = self.p2p_total if quality > 0.3 else round(quality * self.p2p_total)
            resolved = (f2p_pass == self.f2p_total) and (p2p_pass == self.p2p_total)
            out.append({"resolved": resolved, "f2p_pass": f2p_pass, "f2p_total": self.f2p_total,
                        "p2p_pass": p2p_pass, "p2p_total": self.p2p_total, "applied": True})
        return out
