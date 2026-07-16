"""The scope of the value-guided advantage: probe informativeness x saturation.

This is the synthetic bridge between the theory and the real-benchmark pattern. The
theory (docs/reasoning_search.md, paper 3.4) predicts that value-guided search beats
best-of-N only when (a) problems are *reachable but unreliable* -- headroom exists, the
effective number of pivotal decisions K is positive -- and (b) the cheap probe is
*informative* at the pivotal steps. Real benchmarks differ on exactly these two knobs:

  * GSM8K / HumanEval / MBPP (strong model): near-saturated -> no headroom, and for the
    code pair the one-public-assert probe is nearly uninformative -> both knobs fail,
    the measured gap is ~0 (the pre-registered null).
  * MATH / GPQA (mid model): unsaturated, self-consistency probe moderately informative
    -> a consistent moderate gain.
  * Repo-level coding (SWE-bench / RepoBench style): unsaturated for a capable agent,
    and execution-graded probes (patch applies, imports, subset tests) are strongly
    informative with a real cost gradient -> the regime where the theory predicts the
    largest advantage. (The single-function null does NOT contradict it; it sits in the
    dead corner of this map.)

The demo sweeps both knobs on the synthetic reasoning tree and plots the paired
(value-guided - best-of-N) success gap: positive only in the informative-unsaturated
region, ~0 along the saturated edge, and <= 0 along the uninformative edge.

Run with:  uv run --extra plot python examples/reasoning/probe_scope_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402

from canopy.bandits import scoped_success_rate  # noqa: E402
from _plotstyle import PALETTE, save_figure, set_style  # noqa: E402

DEPTH = N_DECISIONS = 8
BUDGET = 1024
SIGMA = 0.3
SEEDS = 200
SATURATIONS = [0.0, 0.25, 0.5, 0.75, 0.95]
INFORMATIVENESS = [0.0, 0.25, 0.5, 0.75, 1.0]

# where the real benchmarks sit on the two knobs (annotation only; the placements are
# qualitative: saturation ~ strong-model pass rate, informativeness ~ measured
# probe-vs-truth quality -- e.g. MATH's pivotal-step hit rate 0.73 vs 0.33 chance)
BENCHMARKS = {
    "GSM8K": (0.95, 0.75),
    "HumanEval (1 assert)": (0.80, 0.25),
    "MATH": (0.50, 0.75),
    "SWE-bench (exec probes)": (0.25, 1.0),
}


def gap_grid() -> np.ndarray:
    """Paired success gap (value-guided - best-of-N) over the (s, q) grid."""
    gaps = np.empty((len(SATURATIONS), len(INFORMATIVENESS)))
    for i, s in enumerate(SATURATIONS):
        bo = scoped_success_rate(
            "best_of_n", DEPTH, N_DECISIONS, BUDGET, SIGMA, saturation=s, seeds=SEEDS
        )
        for j, q in enumerate(INFORMATIVENESS):
            vg = scoped_success_rate(
                "value_guided",
                DEPTH,
                N_DECISIONS,
                BUDGET,
                SIGMA,
                saturation=s,
                probe_informativeness=q,
                seeds=SEEDS,
            )
            gaps[i, j] = vg - bo
    return gaps


def main() -> None:
    set_style()
    print(
        f"scope sweep: depth={DEPTH}, K={N_DECISIONS}, budget={BUDGET}, "
        f"{SEEDS} seeds per cell"
    )
    gaps = gap_grid()
    print(f"{'s \\ q':>8s} " + " ".join(f"{q:6.2f}" for q in INFORMATIVENESS))
    for i, s in enumerate(SATURATIONS):
        print(f"{s:8.2f} " + " ".join(f"{g:+6.2f}" for g in gaps[i]))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Panel A: the phase diagram
    vmax = float(np.abs(gaps).max())
    im = axA.imshow(
        gaps, origin="upper", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax
    )
    axA.set_xticks(range(len(INFORMATIVENESS)), [f"{q:.2f}" for q in INFORMATIVENESS])
    axA.set_yticks(range(len(SATURATIONS)), [f"{s:.2f}" for s in SATURATIONS])
    axA.set_xlabel("probe informativeness $q$")
    axA.set_ylabel("saturation $s$")
    axA.set_title("Value-guided $-$ best-of-N success gap")
    fig.colorbar(im, ax=axA, shrink=0.9)
    for i in range(len(SATURATIONS)):
        for j in range(len(INFORMATIVENESS)):
            axA.text(
                j,
                i,
                f"{gaps[i, j]:+.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    # place the real benchmarks on the map (grid coordinates via interpolation)
    for name, (s, q) in BENCHMARKS.items():
        x = float(np.interp(q, INFORMATIVENESS, np.arange(len(INFORMATIVENESS))))
        y = float(np.interp(s, SATURATIONS, np.arange(len(SATURATIONS))))
        axA.plot(x, y, "*", color=PALETTE["green"], markersize=12, zorder=3)
        near_right_edge = x > len(INFORMATIVENESS) - 1.5
        axA.annotate(
            name,
            (x, y),
            fontsize=7,
            xytext=(-8, -10) if near_right_edge else (4, -8),
            ha="right" if near_right_edge else "left",
            textcoords="offset points",
            color=PALETTE["green"],
        )

    # Panel B: the two failure edges as slices, with benchmark annotations
    axB.plot(
        INFORMATIVENESS,
        gaps[0],
        "o-",
        color=PALETTE["blue"],
        label="unsaturated ($s=0$): gain needs an informative probe",
    )
    axB.plot(
        INFORMATIVENESS,
        gaps[-1],
        "s-",
        color=PALETTE["orange"],
        label=f"saturated ($s={SATURATIONS[-1]}$): nothing left to buy",
    )
    axB.axhline(0.0, color=PALETTE["gray"], lw=1, ls="--")
    axB.set_xlabel("probe informativeness $q$")
    axB.set_ylabel("success gap")
    axB.set_title("The two ways the advantage dies")
    axB.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    out = save_figure(fig, "probe_scope")
    print(f"\nsaved chart to {out} (+ .png)")
    print("\nreading: the null on 1-assert HumanEval/MBPP and the win on MATH are the")
    print("same theory -- the benchmarks sit at different (s, q); repo-level coding")
    print("with execution probes sits in the winning corner.")


if __name__ == "__main__":
    main()
