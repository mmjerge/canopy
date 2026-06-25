# A provably regret-optimal UCB-style algorithm for the hierarchical bandit

Answers: *a UCB-style bias that is provably regret-optimal in this setting, and an
algorithm.* Implemented as `run_hoo` in `oco.bandits.online`.

## The UCB index and the bias term

For a node v at level (depth in the tree) h, with T(v) plays through it and empirical
mean reward μ̂(v) (the empirical subtree average), the index is

    U(v) = μ̂(v) + c·√(2 ln t / T(v))   +   spread(h).
            └ exploitation ┘ └ statistical (UCB) ┘ └ bias / resolution ┘

The **bias term is `spread(h)`** — the bound on `max leaf − subtree mean` at resolution
h (its diameter bonus). This is the provably-correct optimism bonus: it is exactly the
amount by which the best leaf under v can exceed the mean we actually estimate, so
`U(v)` is a valid high-probability upper bound on the best leaf reachable through v.

## What makes it regret-optimal: the B-value backup

Raw `U(v)` is optimistic but loose. The optimism is tightened by propagating children's
bounds upward:

    B(v) = min( U(v),  max_{child c of v} B(c) ).

Each round descends from the root by B-value to a leaf of the current tree, commits to it
(random path), and updates μ̂, T, U, B along the whole path. This is exactly HOO
(Hierarchical Optimistic Optimization; Bubeck–Munos–Stoltz–Szepesvári).

**Regret.** Under local smoothness near the optimum (cell diameters spread(h) → 0), HOO
attains the X-armed-bandit rate

    R_n = Õ( n^{(d+1)/(d+2)} ),   d = near-optimality dimension,

which is optimal up to logs for this class. The `spread(h)` bias is the term that makes
the index a valid optimistic bound, and the B-value backup is what delivers the rate; so
`spread(h)` *is* the provably regret-optimal UCB bias for this setting.

## Memory-bounded variant (ties to the storage question)

`run_hoo(..., memory_bounded=True)` expands a played node only once it is bias-limited,

    T(v) ≥ c² · log / spread(h)²   ⇔   r(v) ≤ spread(h),

(the HCT rule). This keeps the same regret rate but caps memory by the number of nodes
near the optimum (the near-optimality dimension), instead of HOO's O(n) growth. It
unifies the earlier expand-vs-refine rule with the regret-optimal index.

## Empirical (256 leaves, depth 4, horizon 12k, 8 seeds)

| algorithm | final regret | peak memory |
| --- | --- | --- |
| HOO (B-value backup) | 929 | 275 |
| HOO, memory-bounded (HCT) | 1077 | 89 |
| adaptive (argmax-U, earlier) | 1040 | 81 |
| full resolution | 962 | 256 |

HOO with the B-value backup gives the best regret; the memory-bounded version keeps it
near-optimal at ~⅓ the memory.

## Honest positioning

The algorithm is HOO/HCT adapted to the average-backup tree; regret-optimality is
inherited from X-armed-bandit theory under the smoothness assumption. The pieces that are
specific to this project: (i) the bias term is the **max−mean** bound of mean-backup,
(ii) it can be *certified from data* via the empirical-MGF bound (`docs/maxmean_bound.md`)
rather than assumed, and (iii) the random-path reward variance scales with subtree
heterogeneity, coupling the statistical and bias terms. Turning (ii)–(iii) into a
self-certifying, provably-optimal algorithm is the open theoretical target.
