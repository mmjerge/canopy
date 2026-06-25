"""Multiscale (scale-adaptive) tree edge map.

The single-resolution edge detector (``detect_violations`` at a fixed level) forces a bad
tradeoff: a coarse level detects an edge but localizes it only to a large cell, while a fine
level localizes tightly but misses wide / diluted features (and probes every cell at that
level). ``multiscale_edge_map`` estimates the within-cell spread at several levels with a
*per-level, data-driven* floor (a multiple of that level's 75th-percentile spread, never
below the estimation-noise level -- no assumed Lipschitz constant), so it catches violations
at whatever scale they live and localizes each at the finest level that still detects it.

On a tree with mixed-width violations (wide blocks + single-leaf spikes, plus the optimum),
the multiscale map catches them all and gives a clean per-leaf edge score that spikes at the
violations -- the signal used to "sample around the sharp edge".

Run with:  uv run python examples/multiscale_edge_demo.py
           uv run --extra plot python examples/multiscale_edge_demo.py   # + PNG
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from oco.bandits import TreeBandit, detect_violations, multiscale_edge_map
from oco.bandits.rewards import geometric_sigma, hierarchical_gaussian_leaf_means

BRANCHING, DEPTH = 4, 5
N = BRANCHING**DEPTH
NOISE = 0.08
N_SEEDS = 20
LEVELS = [2, 3, 4]


def family(seed: int):
    rng = np.random.default_rng(seed)
    m = np.clip(hierarchical_gaussian_leaf_means(
        BRANCHING, DEPTH, geometric_sigma(0.04, 0.6), root_value=0.4, rng=rng), 0, 1)
    regions = []
    for _ in range(2):  # wide blocks
        s = int(rng.integers(0, N - 64))
        m[s:s + 64] = 0.8
        regions.append((s, s + 64))
    for _ in range(2):  # narrow spikes
        s = int(rng.integers(0, N))
        m[s] = 0.95
        regions.append((s, s + 1))
    return m, regions


def covers(ranges, regions) -> float:
    return float(np.mean([any(s < b and a < e for s, e in ranges) for a, b in regions]))


def main() -> None:
    ms_recall, ms_loc = [], []
    single = {lvl: [] for lvl in LEVELS}
    for seed in range(N_SEEDS):
        lm, regions = family(seed)
        env = TreeBandit(BRANCHING, DEPTH, leaf_means=lm, noise_std=NOISE,
                         rng=np.random.default_rng(20 + seed))
        em = multiscale_edge_map(env, np.random.default_rng(20 + seed), levels=LEVELS,
                                 n_samples_per_cell=40)
        ranges = em.finest_ranges()
        ms_recall.append(covers(ranges, regions))
        ms_loc.append(np.mean([e - s for s, e in ranges]) if ranges else N)
        for lvl in LEVELS:
            det = detect_violations(env, lvl, lambda _l: 0.06, np.random.default_rng(20 + seed),
                                    n_samples_per_cell=40).detected
            cs = BRANCHING ** (DEPTH - lvl)
            single[lvl].append(covers([(c * cs, (c + 1) * cs) for c in det], regions))

    print(f"tree branching {BRANCHING}, depth {DEPTH}, mixed-width violations, {N_SEEDS} seeds")
    print(f"{'method':18s} {'recall':>7s}")
    for lvl in LEVELS:
        cs = BRANCHING ** (DEPTH - lvl)
        print(f"{'single level ' + str(lvl):18s} {np.mean(single[lvl]):7.2f}   (cell size {cs})")
    print(f"{'multiscale':18s} {np.mean(ms_recall):7.2f}   (avg flagged cell {np.mean(ms_loc):.0f})")
    print("\nmultiscale catches violations at every scale and localizes each at its own scale;\n"
          "any single level trades recall (coarse misses narrow / fine misses wide) for "
          "localization.")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(install the 'plot' extra for the chart: uv run --extra plot ...)")
        return

    lm, regions = family(0)
    env = TreeBandit(BRANCHING, DEPTH, leaf_means=lm, noise_std=NOISE, rng=np.random.default_rng(0))
    em = multiscale_edge_map(env, np.random.default_rng(0), levels=LEVELS, n_samples_per_cell=60)

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axA.plot(lm, color="#1f77b4", lw=0.8)
    for a, b in regions:
        axA.axvspan(a, b, color="#d62728", alpha=0.25)
    axA.set_ylabel("leaf value")
    axA.set_title("Function with mixed-width violations (shaded)", fontsize=10)
    axB.plot(em.leaf_score, color="#2ca02c", lw=0.8)
    for a, b in regions:
        axB.axvspan(a, b, color="#d62728", alpha=0.25)
    axB.axhline(1.0, color="#999", ls="--", lw=1)
    axB.set_ylabel("multiscale edge score")
    axB.set_xlabel("leaf index")
    axB.set_title("Multiscale edge score spikes at every violation (all scales)", fontsize=10)
    fig.tight_layout()
    out = Path(__file__).parent / "tree_multiscale_edge.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved chart to {out}")


if __name__ == "__main__":
    main()
