"""Independent audit of the SWE-bench headline + sweep numbers.

Recomputes, directly from the per-instance resolved flags in the raw results JSONs and
without reusing combine_swebench_sweep.py's code paths:
  * best-of-N and value-guided resolved rates
  * paired per-instance Delta and a fresh 95% bootstrap CI (independent RNG stream)
  * discordant-pair counts (vg-only wins vs bo-only wins) and an exact two-sided
    binomial (McNemar) p-value as a bootstrap-free cross-check

Compares against the numbers printed in the paper and flags any mismatch.

Run:  python examples/analysis/audit_swebench_results.py [--figdir paper/figures]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# Paper claims to verify against: (file stem, bo, vg, delta, lo, hi) -- flagship, then sweep.
FLAGSHIP = ("reasoning_search_swebench_results", 0.195, 0.318, 0.123, 0.08, 0.17)
SWEEP = {  # (model tag, depth) -> (delta, lo, hi) as printed in tab:swesweep
    ("sonnet45", 1): (0.100, 0.02, 0.19),
    ("sonnet45", 2): (0.125, 0.04, 0.21),
    ("sonnet45", 3): (0.062, -0.01, 0.14),
    ("nova-pro", 1): (0.100, 0.03, 0.18),
    ("nova-pro", 2): (0.062, -0.01, 0.15),
    ("nova-pro", 3): (0.025, -0.04, 0.09),
    ("llama70b", 1): (0.050, -0.01, 0.12),
    ("llama70b", 2): (0.087, 0.02, 0.16),
    ("llama70b", 3): (0.025, -0.04, 0.09),
    ("mistral-large", 1): (0.013, -0.04, 0.06),
    ("mistral-large", 2): (0.013, -0.04, 0.06),
    ("mistral-large", 3): (-0.013, -0.06, 0.04),
    ("llama8b", 1): (-0.013, -0.08, 0.05),
    ("llama8b", 2): (0.050, 0.00, 0.11),
    ("llama8b", 3): (0.037, -0.02, 0.10),
}


def paired_bootstrap(bo, vg, iters=20000, seed=12345):
    """Fresh paired bootstrap, deliberately different iters/seed than the paper pipeline."""
    bo, vg = np.asarray(bo, float), np.asarray(vg, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, bo.size, size=(iters, bo.size))
    d = vg[idx].mean(axis=1) - bo[idx].mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def mcnemar_p(bo, vg):
    """Exact two-sided binomial test on discordant pairs."""
    b = sum(1 for x, y in zip(bo, vg) if x == 1 and y == 0)  # bo-only wins
    c = sum(1 for x, y in zip(bo, vg) if x == 0 and y == 1)  # vg-only wins
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n * 2
    return b, c, min(1.0, p)


def audit_file(path: Path, name: str, expect=None):
    d = json.loads(path.read_text())
    pi = d["level"]["per_instance"]
    bo = [v["bo"] for v in pi.values()]
    vg = [v["vg"] for v in pi.values()]
    n = len(bo)
    bo_m, vg_m = float(np.mean(bo)), float(np.mean(vg))
    delta = vg_m - bo_m
    lo, hi = paired_bootstrap(bo, vg)
    b, c, p = mcnemar_p(bo, vg)
    ratio = vg_m / bo_m if bo_m > 0 else float("inf")
    print(
        f"{name}: n={n}  bo={bo_m:.3f}  vg={vg_m:.3f}  delta={delta:+.3f} "
        f"CI[{lo:+.3f},{hi:+.3f}]  ratio={ratio:.2f}x  "
        f"discordant bo-only={b} vg-only={c}  McNemar p={p:.4f}"
    )
    ok = True
    if expect is not None:
        e_bo, e_vg, e_d, e_lo, e_hi = expect
        checks = [
            ("bo", bo_m, e_bo, 0.0005),
            ("vg", vg_m, e_vg, 0.0005),
            ("delta", delta, e_d, 0.0005),
            ("lo", lo, e_lo, 0.015),
            ("hi", hi, e_hi, 0.015),
        ]
        for label, got, want, tol in checks:
            if abs(got - want) > tol:
                print(f"  MISMATCH {label}: recomputed {got:+.4f} vs paper {want:+.4f}")
                ok = False
    return ok, delta, lo, hi, bo_m, vg_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default=None)
    args = ap.parse_args()
    figdir = (
        Path(args.figdir)
        if args.figdir
        else (Path(__file__).resolve().parents[2] / "paper" / "figures")
    )

    all_ok = True

    stem, *expect = FLAGSHIP
    p = figdir / f"{stem}.json"
    if p.exists():
        print("== Flagship (261 issues, sonnet, B=9) ==")
        ok, *_ = audit_file(p, "flagship", tuple(expect))
        all_ok &= ok
    else:
        print(f"flagship results not found at {p}")
        all_ok = False

    print("\n== Sweep cells (5 models x 3 budgets, 80 issues/cell) ==")
    found = 0
    for (tag, depth), (e_d, e_lo, e_hi) in SWEEP.items():
        p = figdir / f"reasoning_search_swebench_sweep_{tag}_d{depth}_results.json"
        if not p.exists():
            continue
        found += 1
        ok, delta, lo, hi, _, _ = audit_file(p, f"{tag} d{depth}")
        # delta tolerance is half an ulp of the printed third decimal (cells landing exactly
        # on x.xx5, e.g. 5/80 = 0.0625, legitimately round either way across print paths)
        if abs(delta - e_d) > 0.00055 or abs(lo - e_lo) > 0.015 or abs(hi - e_hi) > 0.015:
            print(f"  MISMATCH vs paper: paper delta={e_d:+.3f} [{e_lo:+.2f},{e_hi:+.2f}]")
            ok = False
        all_ok &= ok
    if found == 0:
        print("no sweep cell JSONs in this figdir (they may live on the GPU box)")

    print(
        f"\nAUDIT {'PASSED' if all_ok else 'FAILED'} "
        f"(flagship{' + ' + str(found) + ' sweep cells' if found else ' only'})"
    )


if __name__ == "__main__":
    main()
