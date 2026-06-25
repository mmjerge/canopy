# Spectral edge isolation → targeted sampling (the sample-efficiency payoff)

The directions-to-capture results (`docs/spectral_trees.md`) are descriptive: they show
*where* the energy lives (region-structured signals are sparse in a few horocyclic
directions; sharp cliffs have full angular spread). The actual goal (Suman) is the **active**
use: isolate the sharp edges with the spectral / Laplacian structure and **sample around
them**, so the bandit resolves a hidden optimum with far fewer samples than uniform.

## The edge map is local Laplacian energy, estimated from cheap probes

The within-cell variance of random-path probes, with the known observation noise
deconvolved, is the data-driven proxy for the **local Dirichlet / Laplacian energy**
`Σ_edges (f_u − f_v)²` of a cell. It is small on smooth (Lipschitz) cells and **spikes
exactly at the sharp edges** (the violations). `detect_violations` computes it and flags the
edge cells — with no assumed Lipschitz constant. Concentrating expensive leaf evaluations on
those cells is literally "sampling around the sharp edge."

## Experiment: identify a hidden optimum at a fixed budget

A tree whose optimum is a spike hidden inside a violation (low subtree average, so
average-based methods prune it). Top-1 accuracy vs cost budget (branching 4, depth 5,
8 hidden violations, `probe/leaf = 0.05`, 40 seeds):

| budget | blind | assume-smooth | **edge-targeted** |
| --- | --- | --- | --- |
| 150 | 0.12 | 0.28 | **0.40** |
| 250 | 0.12 | 0.30 | **0.55** |
| 400 | 0.20 | 0.33 | **0.72** |
| 600 | 0.30 | 0.33 | **0.72** |
| 900 | 0.20 | 0.35 | **0.68** |
| 1400 | 0.85 | 0.35 | 0.78 |
| 2000 | 0.90 | 0.35 | 0.80 |

* **edge-targeted** (detect the edge cells from cheap probes, relax the smooth bound there,
  spend expensive evaluations on them) reaches 70% accuracy at budget **400**;
* **blind** (`SuccessiveEliminationTopK`, no structure) needs budget **1400** to match —
  **≈3.5× more sample-efficient**;
* **assume-smooth** (`HierarchicalTopK` trusting one bound) prunes the hidden optimum and
  stalls around 0.35 regardless of budget.

So isolating the edges and sampling around them is the win; blind eventually catches up only
once the budget is large enough to sample everything, and trusting smoothness never finds the
hidden optimum.

```bash
uv run python examples/targeted_sampling_demo.py
uv run --extra plot python examples/targeted_sampling_demo.py   # sample-efficiency curve -> PNG
```

## Honest notes / next

* The edge map here is **single-resolution** within-cell variance. A genuinely multiscale
  spectral localizer (per-level Haar/horocyclic detail energy) would pinpoint edges at the
  right scale automatically and sub-cell; that is the natural refinement.
* The crossover (blind overtakes once the budget covers the whole tree) is expected and
  honest: targeting matters in the **tight-budget / large-action** regime, which is the
  relevant one for test-time-compute routing.
* Theory to formalize next: a fixed-budget / simple-regret bound of the form
  `accuracy ≥ 1 − f(K · (probe+confirm cost) / budget)`, i.e. the cost of isolating and
  confirming `K` edges, versus the blind `Σ 1/Δ²` leaf complexity.


## Multiscale (scale-adaptive) edge map

The single-resolution detector above forces a tradeoff: a coarse level detects an edge but
localizes it only to a large cell, while a fine level localizes tightly but misses wide /
diluted features and probes every `b**level` cell. `multiscale_edge_map` removes the need to
pick a level: it estimates the within-cell spread at several levels with a **per-level,
data-driven floor** — a multiple of that level's 75th-percentile spread, never below the
estimation-noise level `σ/√(2·probes)` — so a median-collapse-to-zero on mostly-smooth levels
can't blow up the normalization. The per-leaf `leaf_score` is the maximum over levels of the
normalized anomaly `within_std / floor`, large wherever the function has a sharp edge at *any*
scale; `finest_ranges()` localizes each edge at the finest level that still detects it.

On a tree with **mixed-width** violations (two wide blocks + two single-leaf spikes, depth 5,
20 seeds), recall of the violation regions:

| method | recall | localization |
| --- | --- | --- |
| single level 2 | 0.56 | cell size 64 |
| single level 3 | 0.89 | cell size 16 |
| single level 4 | 0.82 | cell size 4 |
| **multiscale** | **1.00** | finest per edge (avg ≈ 17) |

The multiscale map catches violations at every scale (a single level always misses the
wrong-scale ones) and localizes each at its own scale, with no assumed Lipschitz constant.

```bash
uv run python examples/multiscale_edge_demo.py
uv run --extra plot python examples/multiscale_edge_demo.py   # function + edge-score panels -> PNG
```

**Honest caveat (efficiency).** The robust version probes every cell at every chosen level,
so it costs more than a single fine level. A coarse-to-fine **cascade** (probe coarse, refine
only anomalous cells) is much cheaper but *fragile*: a premature prune at a coarse level drops
an edge permanently (recall fell to ~0.5 in tests). Making it both cheap and robust needs
confidence-based refinement (only prune when statistically sure), i.e. HOO/zooming-style
expansion — the natural next step.
