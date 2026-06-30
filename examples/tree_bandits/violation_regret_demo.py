"""Graceful degradation in the number of Lipschitz violations (the value of structure).

Multi-fidelity regime where the tree provably pays: cheap biased internal probes
(probe_cost << leaf_cost) and a tight identification budget. The number of violations is the
number of ``adversarial_spike`` spikes (the canonical "subtree averages are uninformative"
axis). Three methods at a fixed tight budget:

  * assume-smooth -- HierarchicalTopK trusting one smoothness bound (misled by spikes);
  * blind         -- SuccessiveEliminationTopK (no structure; can't afford enough leaves);
  * hybrid        -- detect the spike cells from data, then relax the smooth bound there.

The hybrid degrades gracefully with the violation count and dominates both baselines. Lines
show mean top-1 accuracy over seeds with a shaded 95% CI band.

Run with:  uv run --extra plot python examples/tree_bandits/violation_regret_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402

from canopy.bandits import (  # noqa: E402
    HierarchicalTopK,
    SuccessiveEliminationTopK,
    TreeBandit,
    detect_violations,
)
from canopy.bandits.rewards import adversarial_spike_leaf_means  # noqa: E402
from _plotstyle import PALETTE, ci_band, save_figure, set_style  # noqa: E402

BRANCHING, DEPTH, LEVEL = 4, 5, 3
BUDGET = 400.0
PROBE_COST, LEAF_COST, NOISE = 0.05, 1.0, 0.05
SPREAD, BEAM, FLOOR = 0.3, 8, 0.08
N_SEEDS = 30
K_VALUES = [1, 2, 4, 8, 16, 32, 64]


def env_for(lm: np.ndarray, seed: int) -> TreeBandit:
    return TreeBandit(
        BRANCHING,
        DEPTH,
        leaf_means=lm,
        noise_std=NOISE,
        leaf_cost=LEAF_COST,
        probe_cost=PROBE_COST,
        rng=np.random.default_rng(seed),
    )


def run_all():
    nk, ns = len(K_VALUES), N_SEEDS
    detected = np.zeros((nk, ns))
    hier = np.zeros((nk, ns))
    blind = np.zeros((nk, ns))
    hybrid = np.zeros((nk, ns))
    cell = BRANCHING ** (DEPTH - LEVEL)
    for i, k in enumerate(K_VALUES):
        for s in range(N_SEEDS):
            lm = adversarial_spike_leaf_means(BRANCHING, DEPTH, k, rng=np.random.default_rng(s))
            e = env_for(lm, 40 + s)
            hier[i, s] = (
                HierarchicalTopK(BUDGET, 0.1, spread=SPREAD, beam_width=BEAM)
                .run(e, 1)
                .evaluate(e, 1)
            )
            e = env_for(lm, 40 + s)
            blind[i, s] = SuccessiveEliminationTopK(BUDGET, 0.1).run(e, 1).evaluate(e, 1)
            rep = detect_violations(
                env_for(lm, 80 + s),
                LEVEL,
                lambda _l: FLOOR,
                np.random.default_rng(80 + s),
                n_samples_per_cell=40,
            )
            detected[i, s] = rep.count
            ranges = [(c * cell, (c + 1) * cell) for c in rep.detected]
            e = env_for(lm, 40 + s)
            hybrid[i, s] = (
                HierarchicalTopK(BUDGET, 0.1, spread=SPREAD, beam_width=BEAM, relaxed_ranges=ranges)
                .run(e, 1)
                .evaluate(e, 1)
            )
    return detected, hier, blind, hybrid


def main() -> None:
    set_style()
    print(
        f"multi-fidelity: branching {BRANCHING}, depth {DEPTH}, budget {BUDGET}, "
        f"probe/leaf={PROBE_COST}, {N_SEEDS} seeds"
    )
    detected, hier, blind, hybrid = run_all()
    print(f"{'K':>4s} {'detected':>9s} {'assume':>7s} {'blind':>7s} {'hybrid':>7s}")
    for i, k in enumerate(K_VALUES):
        print(
            f"{k:4d} {detected[i].mean():9.1f} {hier[i].mean():7.2f} "
            f"{blind[i].mean():7.2f} {hybrid[i].mean():7.2f}"
        )

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))
    ci_band(axA, K_VALUES, detected, PALETTE["blue"], "detected", "o")
    axA.plot(K_VALUES, K_VALUES, "--", color=PALETTE["gray"], lw=1, label="true $K$")
    axA.set_xscale("log", base=2)
    axA.set_yscale("log", base=2)
    axA.set_xlabel("true violations $K$")
    axA.set_ylabel("detected violations")
    axA.set_title("Violation count inferred from data")
    axA.legend(loc="upper left")

    ci_band(axB, K_VALUES, hybrid, PALETTE["green"], "hybrid (detect+relax)", "o")
    ci_band(axB, K_VALUES, hier, PALETTE["orange"], "assume-smooth", "s")
    ci_band(axB, K_VALUES, blind, PALETTE["red"], "blind (no structure)", "^")
    axB.set_xscale("log", base=2)
    axB.set_xlabel("number of violations $K$")
    axB.set_ylabel("top-1 accuracy @ fixed budget")
    axB.set_title("Hybrid degrades gracefully and dominates")
    axB.set_ylim(0, 1.02)
    axB.legend(loc="upper right")

    fig.tight_layout()
    out = save_figure(fig, "tree_violation_regret")
    print(f"\nsaved chart to {out} (+ .png)")


if __name__ == "__main__":
    main()
