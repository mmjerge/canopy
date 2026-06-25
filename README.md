# Canopy

**Exploiting smooth tree priors for bandits.** A multi-fidelity bandit over a complete tree
where an internal node's value is the *average* reward of its leaf-subtree. Probing an internal
node is a **cheap but biased** signal (its average underestimates its best leaf); evaluating a
leaf is **expensive but unbiased**. The question this repo answers: *when does trusting the tree
structure (the smoothness prior) let you identify good leaves, or minimize regret, at lower cost
than a structure-blind baseline?*

Target venue: ICLR / ICML (short). The companion long paper develops the analysis toolbox
(`harmonic-trees`).

## Install (uv)

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check src/ tests/ examples/
```

Optional extras: `--extra plot` (matplotlib charts), `--extra llm` (Bedrock routing),
`--extra bench` (datasets).

## Core idea

* `TreeBandit` (`oco.bandits.tree`) — the multi-fidelity environment. `leaf_cost` vs
  `probe_cost`; budgets are measured in **cost**. Scenarios: `from_hierarchical_gaussian`,
  `from_adversarial_spikes`, `from_piecewise_smooth`.
* Two regimes: **pure exploration** (identify the top-k leaves at least cost) and
  **regret minimization** (commit each round, compete with the best leaf).

## What's in the repo

| Module | Role |
| --- | --- |
| `oco.bandits.tree` | `TreeBandit` multi-fidelity environment |
| `oco.bandits.topk` | `HierarchicalTopK`, `UniformTopK` (top-k identification) |
| `oco.bandits.baselines` | `SuccessiveEliminationTopK` (strong structure-blind baseline) |
| `oco.bandits.online` | regret mode: `run_hoo`, `run_adaptive(_variance)`, `run_fixed_depth`, `run_hybrid`, `run_local_lipschitz`; `detect_violations` |
| `oco.bandits.maxmean` | noise-deconvolved high-probability bound on `max − mean` (certifies the bias term) |
| `oco.bandits.rewards` | reward families (hierarchical-Gaussian, adversarial spikes, piecewise-smooth, violation) |
| `oco.bandits.routing`, `oco.bandits.prefix_cache` | applied bandit instantiations (LLM routing, prefix-cache) |
| `oco.bandits.exp3`, `oco.experts`, `oco.algorithms` | classical OCO/MAB baselines (Hazan) |

## Key results (honest, reproducible)

**Multi-fidelity top-k — the prior pays off only with cheap probes.** 1024 leaves, top-5, 30
seeds, budget 1500:

| probe/leaf cost | Hierarchical | SuccElim (strong) |
| --- | --- | --- |
| 1.0 (no advantage) | 0.29 | 0.58 |
| 0.5 | 0.80 | 0.58 |
| 0.05 | 0.84 | 0.58 |

When an internal probe costs as much as a leaf, the descent is wasted overhead and the tree
loses; it overtakes the strong baseline once probes are ≥2× cheaper.

**Regret/memory — adaptive matches full resolution at a fraction of the memory.** 256 leaves,
12k rounds: adaptive reaches near full-leaf regret at ~1/3 the memory (variance-aware: ~6×
less), sitting below the fixed-depth tradeoff frontier (`run_hoo` is the regret-optimal rule).

**Graceful degradation in the number of violations.** Detect-and-relax hybrid strictly
dominates assume-smooth and structure-blind, declining to the blind floor only as violations
fill the tree (`docs/violation_regret.md`).

**Fixed-budget bound.** `H_edge = c₀(d) + Σ_{k≤K} c_ℓ·h_k` gives
`P(error) ≤ Õ(N)·exp(−κB/H_edge)` — recovers the smooth rate at `K=0`, grows additively in the
violation count `K`, beats the structure-blind `H_blind`, saturates back to it as violations
proliferate (`docs/fixed_budget_bound.md`).

```bash
uv run --extra plot python examples/benchmark.py            # fidelity + budget sweep
uv run --extra plot python examples/regret_storage_demo.py  # regret vs memory
uv run --extra plot python examples/violation_regret_demo.py
```

## Docs

`fixed_budget_bound.md`, `ucb_optimal.md`, `infinite_tree.md`, `lipschitz_regret.md`,
`maxmean_bound.md`, `regret_storage_note.md`, `violation_regret.md`, `llm_routing.md`.
