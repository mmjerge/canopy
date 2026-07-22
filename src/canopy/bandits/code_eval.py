"""Code-generation grading for the reasoning-search harness (HumanEval / MBPP).

This lets the value-guided-vs-best-of-N comparison run on a *code* domain, not just math. The
multi-fidelity structure maps cleanly:

  * leaf (expensive, unbiased): run the candidate program against the FULL (hidden) assert suite.
  * cheap probe (biased): run it against a single PUBLIC assert -- far cheaper, and the signal the
    value edge / best-of-N selection is allowed to use (hidden tests are only for final scoring).

Both HumanEval and MBPP reduce to "run the program, then a list of ``assert`` statements". We
normalise them to a common ``gold`` dict: ``{code_prefix, entry_point, public, hidden}`` where
``public``/``hidden`` are lists of assert strings and ``code_prefix`` is the function
header/docstring to prepend if a completion emitted only the body.

SECURITY: grading executes model-generated code. It runs in a separate Python subprocess with a
wall-clock timeout and no stdin, which is the standard HumanEval-style harness. Run it only on a
disposable machine (e.g. the experiment box), never on a host with anything sensitive.
"""

from __future__ import annotations

import re
import subprocess
import sys

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Pull the Python solution out of a model completion.

    Prefers the last fenced ```python block (instruct models wrap code in fences); falls back to
    the raw text (already-bare code). Returns a code string (possibly empty).
    """
    if text is None:
        return ""
    blocks = _FENCE.findall(text)
    if blocks:
        return blocks[-1].strip()
    return text.strip()


def _run_script(script: str, timeout: float) -> bool:
    """Execute ``script`` in a fresh subprocess; True iff it exits 0 within ``timeout``."""
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-"],  # -I: isolated (ignore env/user site) for a bit of safety
            input=script,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return proc.returncode == 0
    except Exception:  # noqa: BLE001 -- timeout, OSError, etc. all count as "failed"
        return False


def _build_script(code: str, gold: dict, asserts: list) -> str:
    """Assemble a runnable script: (optional header) + completion + candidate alias + asserts."""
    header = ""
    ep = gold.get("entry_point") or ""
    if ep and f"def {ep}" not in code:
        header = gold.get("code_prefix", "")  # completion was body-only: restore the signature
    alias = f"\ncandidate = {ep}\n" if ep else "\n"  # HumanEval asserts call ``candidate``
    return header + "\n" + code + alias + "\n".join(asserts) + "\n"


def passes(code: str, gold: dict, asserts: list, timeout: float) -> bool:
    """True iff ``code`` passes every assert in ``asserts`` (empty list / empty code -> False)."""
    if not code or not asserts:
        return False
    return _run_script(_build_script(code, gold, asserts), timeout)


def grade_code(code: str, gold: dict, timeout: float = 10.0) -> bool:
    """Leaf grade: does the completion pass the full hidden test suite?

    ``gold['hidden_suffix']`` is the benchmark's native test tail appended after the completion
    (HumanEval: the ``check`` function + ``check(candidate)``; MBPP: its joined ``assert`` list).
    """
    if not code:
        return False
    header = ""
    ep = gold.get("entry_point") or ""
    if ep and f"def {ep}" not in code:
        header = gold.get("code_prefix", "")  # body-only completion: restore signature
    alias = f"\ncandidate = {ep}\n" if ep else "\n"
    script = header + "\n" + code + alias + gold.get("hidden_suffix", "") + "\n"
    return _run_script(script, timeout)


def public_test_value(rollout_texts, gold: dict, timeout: float = 5.0) -> float:
    """Cheap probe: fraction of rollouts whose extracted code passes the PUBLIC assert(s).

    This is the code analogue of self-consistency -- a cheap, biased estimate of a partial
    solution's promise, using only the public test(s), never the hidden suite.
    """
    pub = gold.get("public", [])
    if not rollout_texts or not pub:
        return 0.0
    ok = sum(1 for t in rollout_texts if passes(extract_code(t), gold, pub, timeout))
    return ok / len(rollout_texts)


def select_by_public_tests(answers, gold: dict, timeout: float = 5.0):
    """best-of-N / final selection for code: pick the candidate passing the most public asserts.

    Uses only public information (fair vs. value-guided, which also only sees public tests);
    ties and total failures fall back to the first candidate. ``answers`` are extracted code
    strings. Returns the chosen code string (or ``None`` if empty).
    """
    answers = [a for a in answers if a]
    if not answers:
        return None
    pub = gold.get("public", [])
    if not pub:
        return answers[0]
    # grade each public assert separately so we can rank partial-passers, not just all-or-nothing
    for a in answers:  # fast path: return the first candidate passing all public asserts
        if passes(a, gold, pub, timeout):
            return a
    best, best_score = answers[0], -1
    for a in answers:
        score = sum(1 for asrt in pub if passes(a, gold, [asrt], timeout))
        if score > best_score:
            best, best_score = a, score
    return best


def public_fraction(code: str, gold: dict, timeout: float = 5.0) -> float:
    """Fraction of the PUBLIC asserts that ``code`` passes individually (continuous in [0,1]).

    Unlike :func:`passes` (all-or-nothing on the list), each assert runs separately, so a
    plausible-but-partially-wrong candidate scores between 0 and 1. This is the repaired,
    continuous cheap probe of the code ladder (docs/code_benchmarks.md, rung 2).
    """
    pub = gold.get("public", [])
    if not code or not pub:
        return 0.0
    return sum(passes(code, gold, [a], timeout) for a in pub) / len(pub)


def probe_truth_pairs(candidate_texts, gold: dict, timeout: float = 5.0):
    """Stage-1 gate measurement: per candidate, (cheap public score, true hidden pass).

    Sample candidates elsewhere, then record for each the continuous public-test score (the
    cheap value the search would follow) and the hidden-suite pass (the truth). The correlation
    across candidates -- pooled over problems -- is the code analog of the cheap-vs-true
    node-value correlation measured on MATH: if it is near zero, the theory predicts
    value-guided search cannot beat best-of-N here, and the race is not worth running.
    """
    pairs = []
    for text in candidate_texts:
        code = extract_code(text)
        pairs.append((public_fraction(code, gold, timeout), float(grade_code(code, gold))))
    return pairs
