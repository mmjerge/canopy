"""Spectral edge isolation -> targeted sampling: the sample-efficiency payoff.

This is the crux Suman pointed at: the spectral / Laplacian structure should be used to
*isolate the sharp edges* (the Lipschitz violations) and then *sample around them*, instead
of merely describing where the energy lives. The within-cell variance estimated from cheap
random-path probes is the data-driven proxy for the local Laplacian/Dirichlet energy -- it
spikes exactly at the sharp edges -- so concentrating expensive leaf evaluations on the
high-energy (edge) cells is "sampling around the sharp edge."

Concretely, on a tree where the optimum is a spike hidden inside a violation (low subtree
average, so average-based methods prune it), we compare top-1 identification accuracy at a
fixed cost budget for:
  * blind            -- SuccessiveEliminationTopK (no structure, no targeting);
  * assume-smooth    -- HierarchicalTopK trusting one smoothness bound (prunes the hidden
                        optimum -> misses);
  * edge-targeted    -- detect the violation cells from cheap probes (``detect_violations``),
                        then HierarchicalTopK that relaxes the bound there and spends its
                        expensive evaluations on those edge cells.

Result (sample-efficiency curve): edge-targeted reaches high accuracy at a small fraction of
the budget the blind method needs -- it samples around the edges instead of everywhere.

Run with:  uv run python examples/tree_bandits/targeted_sampling_demo.py
           uv run --extra plot python examples/tree_bandits/targeted_sampling_demo.py   # + PNG
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from canopy.bandits import (
    HierarchicalTopK,
    SuccessiveEliminationTopK,
    TreeBandit,
    detect_violations,
)
from canopy.bandits.rewards import adversarial_spike_leaf_means

BRANCHING, DEPTH, LEVEL = 4, 5, 3
K_VIOLATIONS = 8
NOISE, PROBE_COST, LEAF_COST = 0.05, 0.05, 1.0
SPREAD, BEAM, FLOOR = 0.3, 8, 0.08
N_SEEDS = 40
BUDGETS = [150, 250, 400, 600, 900, 1400, 2000]
TARGET_ACC = 0.7


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


def accuracy_at(budget: float) -> tuple[float, float, float]:
    blind, smooth, edge = [], [], []
    cell = BRANCHING ** (DEPTH - LEVEL)
    for seed in range(N_SEEDS):
        lm = adversarial_spike_leaf_means(
            BRANCHING, DEPTH, K_VIOLATIONS, rng=np.random.default_rng(seed)
        )
        e = env_for(lm, 40 + seed)
        blind.append(SuccessiveEliminationTopK(budget, 0.1).run(e, 1).evaluate(e, 1))
        e = env_for(lm, 40 + seed)
        smooth.append(
            HierarchicalTopK(budget, 0.1, spread=SPREAD, beam_width=BEAM).run(e, 1).evaluate(e, 1)
        )
        # edge isolation from cheap probes -> relax + target those cells
        det = detect_violations(
            env_for(lm, 80 + seed),
            LEVEL,
            lambda _l: FLOOR,
            np.random.default_rng(80 + seed),
            n_samples_per_cell=30,
        ).detected
        ranges = [(c * cell, (c + 1) * cell) for c in det]
        e = env_for(lm, 40 + seed)
        edge.append(
            HierarchicalTopK(budget, 0.1, spread=SPREAD, beam_width=BEAM, relaxed_ranges=ranges)
            .run(e, 1)
            .evaluate(e, 1)
        )
    return float(np.mean(blind)), float(np.mean(smooth)), float(np.mean(edge))


def budget_to_reach(curve: list[float], target: float) -> float | None:
    for b, acc in zip(BUDGETS, curve):
        if acc >= target:
            return b
    return None


def main() -> None:
    print(
        f"tree: branching {BRANCHING}, depth {DEPTH}, {K_VIOLATIONS} hidden violations, "
        f"probe/leaf = {PROBE_COST}, {N_SEEDS} seeds"
    )
    print(f"{'budget':>7s} {'blind':>7s} {'assume-smooth':>14s} {'edge-targeted':>14s}")
    blind_c, smooth_c, edge_c = [], [], []
    for b in BUDGETS:
        bl, sm, ed = accuracy_at(b)
        blind_c.append(bl)
        smooth_c.append(sm)
        edge_c.append(ed)
        print(f"{b:7d} {bl:7.2f} {sm:14.2f} {ed:14.2f}")

    b_edge = budget_to_reach(edge_c, TARGET_ACC)
    b_blind = budget_to_reach(blind_c, TARGET_ACC)
    if b_edge and b_blind:
        print(
            f"\nbudget to reach {TARGET_ACC:.0%} accuracy: edge-targeted {b_edge}, "
            f"blind {b_blind}  ->  {b_blind / b_edge:.1f}x more sample-efficient"
        )
    print(
        "edge-targeted isolates the sharp edges from cheap probes and samples around them;\n"
        "blind needs far more budget, assume-smooth prunes the hidden optimum and stalls."
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(install the 'plot' extra for the chart: uv run --extra plot ...)")
        return

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(BUDGETS, edge_c, "-o", color="#2ca02c", lw=2, label="edge-targeted (detect+relax)")
    ax.plot(BUDGETS, blind_c, "-^", color="#d62728", lw=2, label="blind (no structure)")
    ax.plot(BUDGETS, smooth_c, "-s", color="#ff7f0e", lw=2, label="assume-smooth")
    ax.axhline(TARGET_ACC, color="#999", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("cost budget")
    ax.set_ylabel("top-1 accuracy (hidden optimum)")
    ax.set_title("Sample efficiency of sampling around the sharp edge", fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = Path(__file__).parent.parent / "images" / "tree_targeted_sampling.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved chart to {out}")


if __name__ == "__main__":
    main()
