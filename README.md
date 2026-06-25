# Canopy

[![CI](https://github.com/mmjerge/canopy/actions/workflows/ci.yml/badge.svg)](https://github.com/mmjerge/canopy/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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

* `TreeBandit` (`canopy.bandits.tree`) — the multi-fidelity environment. `leaf_cost` vs
  `probe_cost`; budgets are measured in **cost**. Scenarios: `from_hierarchical_gaussian`,
  `from_adversarial_spikes`, `from_piecewise_smooth`.
* Two regimes: **pure exploration** (identify the top-k leaves at least cost) and
  **regret minimization** (commit each round, compete with the best leaf).

## Applications: three real LLM use cases

The same machinery instantiates three LLM applications, each mapping a real problem onto the
tree (leaf, arm/region, and budget). These are the empirical contributions of the paper.

### 1. LLM routing
*Mapping:* leaf = prompt, arm = model, region = subject / token-prefix. A hierarchical router
learns which model to use for which prefix region, generalizing across prompts that share a
prefix.
*Result:* real MMLU + 6 Bedrock models — **0.84 quality vs 0.69** best-fixed-model, matching
always-largest at ~**½ the cost** (≈12× lower regret than the best fixed policy).
*Code:* `canopy.bandits.routing`; `examples/llm_routing/mmlu_routing.py`, `examples/llm_routing/llm_routing_demo.py`,
`examples/llm_routing/bedrock_routing.py`. *Docs:* `docs/llm_routing.md`.

### 2. Prefix caching
*Mapping:* cache = ancestor-closed subtree of the token trie, storage = memory budget; caching
a prefix saves recompute for every prompt through it.
*Result:* matches LFU and the hindsight optimum on a stationary stream, and **beats both LFU
and the best static cache under a popularity shift** (it tracks the drift).
*Code:* `canopy.bandits.prefix_cache`; `examples/llm_routing/prefix_cache_demo.py`.
*Note:* the cleanest of the three — the prefix tree **is** the actual cache data structure, so
there is no tree-alignment assumption.

### 3. Prompt optimization (trimming)
*Mapping:* arm = trim level, region = subject. Adaptive per-subject prompt trimming.
*Result:* real MMLU + nova-lite — adaptive trim beats the best fixed trim on accuracy
(**0.77 vs 0.75**) and tokens (**97 vs 101**).
*Code:* `examples/llm_routing/prompt_optimization.py`.
*Honest caveat:* this run used an 8-token output cap that penalized verbose prompts; it is the
weakest of the three and needs a re-run at a larger output budget to be airtight.

```bash
uv run --extra plot python examples/llm_routing/llm_routing_demo.py      # routing (synthetic + chart)
uv run --extra plot python examples/llm_routing/prefix_cache_demo.py     # caching under drift
uv run --extra plot --extra llm python examples/llm_routing/mmlu_routing.py        # real Bedrock routing
uv run --extra plot --extra llm python examples/llm_routing/prompt_optimization.py # real Bedrock trimming
```

## What's in the repo

| Module | Role |
| --- | --- |
| `canopy.bandits.tree` | `TreeBandit` multi-fidelity environment |
| `canopy.bandits.topk` | `HierarchicalTopK`, `UniformTopK` (top-k identification) |
| `canopy.bandits.baselines` | `SuccessiveEliminationTopK` (strong structure-blind baseline) |
| `canopy.bandits.online` | regret mode: `run_hoo`, `run_adaptive(_variance)`, `run_fixed_depth`, `run_hybrid`, `run_local_lipschitz`; `detect_violations` |
| `canopy.bandits.maxmean` | noise-deconvolved high-probability bound on `max − mean` (certifies the bias term) |
| `canopy.bandits.rewards` | reward families (hierarchical-Gaussian, adversarial spikes, piecewise-smooth, violation) |
| `canopy.bandits.routing`, `canopy.bandits.prefix_cache` | applied bandit instantiations (LLM routing, prefix-cache) |
| `canopy.bandits.exp3`, `canopy.experts`, `canopy.algorithms` | classical OCO/MAB baselines (Hazan) |

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
uv run --extra plot python examples/tree_bandits/benchmark.py            # fidelity + budget sweep
uv run --extra plot python examples/tree_bandits/regret_storage_demo.py  # regret vs memory
uv run --extra plot python examples/tree_bandits/violation_regret_demo.py
```

## Docs

`applications.md` (the three use cases), `fixed_budget_bound.md`, `ucb_optimal.md`,
`infinite_tree.md`, `lipschitz_regret.md`, `maxmean_bound.md`, `regret_storage_note.md`,
`violation_regret.md`, `llm_routing.md`.
