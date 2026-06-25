# Online hierarchical bandit: regret vs. storage, and a variance-aware expansion rule

Follow-up to the question: *when do we refine a parent's estimate vs. expand it into
children, and is there a regret-optimal way to do it?* Chart:
`examples/images/tree_regret_storage.png`.

## Setup

Online hierarchical bandit on a complete b-ary tree (here b=4, depth=4, 256 leaves).
Each round the learner commits to a node and follows a random path down; the realized
reward is a random leaf under that node + noise. Expected reward = subtree mean f(v), so
competing against the best single leaf, the per-round regret of committing at level ℓ is
≤ spread(ℓ) — the bias of mean-backup at that resolution. Horizon 12k, 10 seeds, smooth
hierarchical-Gaussian rewards.

## The regret–storage tradeoff (Panel B)

Fixed-resolution play traces a clean frontier — deeper = less bias but exponential
memory:

| strategy | peak memory | final regret |
| --- | --- | --- |
| fixed depth 0 (root only) | 1 | 12790 |
| fixed depth 2 | 16 | 3457 |
| fixed depth 3 | 64 | 1518 |
| fixed depth 4 (full leaves) | 256 | 955 |

## The regret-optimal expansion rule

Per-node crossover: **refine a node while it is statistically limited, expand it the
moment it becomes bias-limited**, i.e. when its confidence radius r(v) ≤ spread(level).
While r(v) > spread(ℓ) the dominant uncertainty is statistical and cheap to reduce; once
r(v) ≤ spread(ℓ) you are bias-limited and the only way forward is to cut the bias by
expanding. This is the HOO/HCT-style trigger; it concentrates memory near the optimum
(memory scales with the near-optimality dimension, not the whole tree).

| strategy | peak memory | final regret |
| --- | --- | --- |
| adaptive (assumed spread schedule) | 82 | 1024 |

→ near full-resolution regret at ~1/3 the memory.

## Novel piece: variance-aware expansion (no assumed spread)

Two observations specific to this setup:

1. Playing a node via random-path has reward variance σ_within(v)² + noise² — coarse
   nodes are both more *biased* and more *noisy*. So the confidence radius should be
   empirical-Bernstein (variance-adaptive), not fixed-noise Hoeffding.
2. The *measured* within-subtree variance is a **data-driven surrogate for spread(ℓ)**.
   Estimate σ_within(v) from data (subtract known noise) and use
   σ_within(v)·√(2 log|L(v)|) — the expected max-minus-mean of |L(v)| leaves — as the
   bias proxy. This removes the assumed, level-only spread schedule and makes expansion
   location/data-dependent: drill into heterogeneous regions, leave homogeneous ones
   coarse.

| strategy | peak memory | final regret |
| --- | --- | --- |
| **adaptive, variance-aware (novel)** | **44** | **969** |

→ matches full-resolution regret (955 @ memory 256) at **~6× less memory**, and beats the
assumed-spread version on both axes — without needing the spread schedule as input.

## Open / next

- The variance proxy uses σ_within (a soft signal) where the sound bound wants max−mean.
  **Addressed:** `docs/maxmean_bound.md` derives a high-probability, noise-deconvolved
  empirical-MGF bound on max−mean. Finding: it's a good *certifier* but a conservative
  *driver*, so drive with the heuristic and certify with the MGF bound.
- Still on synthetic smooth rewards; the honest next step is a real instance (test-time
  compute) and a published multi-fidelity / tree-BAI baseline.
- DAG generalization inherits the same value-as-linear-functional structure.
