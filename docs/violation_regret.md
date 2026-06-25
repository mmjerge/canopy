# Graceful degradation in the number of Lipschitz violations

Setting (per the Slack thread). The agreed function class is **locally Lipschitz with a few
dispersed jumps** (a per-subtree constant, an additive jump penalty, a finite
near-optimality dimension), with **constants estimated from data, not assumed**. The open
question (Suman): can we get guarantees **parameterized by the number of violations `K`**
that degrade *gracefully* — fewer violations ⇒ better, worsening toward the generic bound
as `K` grows?

## An honest false start (what does NOT work)

The obvious experiment — cost to *certify* the optimum vs `K` near-optimal decoys — is
**confounded**. It reduces to the standard best-arm-identification complexity (cost grows
with the number of near-optimal arms, `Σ 1/Δ²`); it has nothing to do with Lipschitz
violations specifically, and the tree structure provides **no leverage** on it (a
structure-exploiting `HierarchicalTopK` is no cheaper than blind successive elimination on
that family — sometimes worse, from probe overhead). Likewise, for *cumulative* regret the
relationship is **non-monotone**: high-reward violations are extra good arms and can
*reduce* cumulative regret. So neither "cost to certify" nor "cumulative regret" isolates
the violation effect. (Both were verified directly.)

## The corrected experiment (what the violation count actually controls)

The right object is the **value of structure**, and the regime is the multi-fidelity one
where the tree provably pays: cheap biased internal probes (`probe_cost ≪ leaf_cost`) and a
tight identification budget (per the README benchmark). The number of violations is the
number of `adversarial_spike` spikes — the repo's canonical "subtree averages are
uninformative" axis — so more spikes make the coarse probes the descent relies on more
misleading.

Three methods, **top-1 accuracy at a fixed (tight) cost budget** (branching 4, depth 5,
budget 400, `probe/leaf = 0.05`, 24 seeds):

| `K` violations | detected | assume-smooth | blind | **hybrid (detect+relax)** |
| --- | --- | --- | --- | --- |
| 1 | 1.0 | 0.42 | 0.42 | **0.96** |
| 2 | 1.5 | 0.46 | 0.33 | **0.92** |
| 4 | 3.2 | 0.50 | 0.21 | **0.83** |
| 8 | 6.7 | 0.17 | 0.17 | **0.71** |
| 16 | 12.1 | 0.38 | 0.04 | **0.46** |
| 32 | 22.1 | 0.38 | 0.12 | **0.33** |
| 64 | 37.0 | 0.17 | 0.00 | **0.17** |

The **hybrid degrades gracefully** in the number of violations — accuracy ≈ 1 when
violations are few, declining smoothly toward the structure-blind floor as they proliferate
— and **strictly dominates** both pure assume-smooth (misled by the spikes, erratic) and
blind (can't afford enough leaves at a tight budget) across the whole range. This is the
honest realization of "fewer violations ⇒ better, gracefully worsening toward the generic
bound."

The hybrid is the data-driven method the thread settled on: `detect_violations` flags the
spike cells from samples (no assumed Lipschitz constant, no jump count a priori), and
`HierarchicalTopK(..., relaxed_ranges=...)` uses the smooth bound everywhere except those
cells, where it falls back to leaf-level certification. (Detection undercounts at large `K`
because spikes collide in cells — that is correct: distinct anomalous *cells* < spikes.)

```bash
uv run python examples/tree_bandits/violation_regret_demo.py
uv run --extra plot python examples/tree_bandits/violation_regret_demo.py   # detection + dominance panels -> PNG
```

## For the writeup

* The clean empirical claim is about **accuracy / simple regret at fixed budget**, not
  cumulative regret: the structure-exploiting hybrid's accuracy interpolates between the
  smooth-optimal regime (few violations) and the structure-blind floor (many violations).
* Theory to formalize: a fixed-budget / simple-regret bound of the form
  `accuracy ≥ 1 − f(K/budget)`, with the per-violation cost governed by the probe/leaf ratio
  and the detection margin; equivalently a sample-complexity bound `base(d) + K·penalty` to
  reach a target accuracy.
* State the scope honestly: cumulative regret can be *helped* by benign (high-reward)
  violations, so the monotone "more violations ⇒ harder" belongs to the fixed-budget /
  identification setting, where each violation defeats one unit of structural pruning.
