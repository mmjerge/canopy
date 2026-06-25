# Tree-Lipschitz functions, a regret bound with jumps, and the hybrid algorithm

This note (i) gives a natural definition of Lipschitzness for tree functions, (ii) states
a regret bound for the piecewise-Lipschitz (smooth + K jumps) case, and (iii) describes
the hybrid algorithm that achieves robustness empirically.

Standing assumption (per Suman): **bounded range** — leaf rewards lie in [0, B] (we use
B = 1). This makes the empirical-Bernstein / MGF concentration steps clean and bounds all
jump heights by B.

## A natural Lipschitzness for tree functions

Order is the wrong primitive for a tree; the natural metric is the **ultrametric** given
by the lowest common ancestor (LCA):

    d(x, y) = rho ** level( LCA(x, y) ),    rho in (0, 1).

Two leaves are close iff they share a deep ancestor. **f is L-Lipschitz on the tree** iff

    |f(x) − f(y)| ≤ L · d(x, y)   for all leaves x, y,

equivalently: the oscillation of f within any level-ℓ subtree is at most `L·rho**ℓ`. This
immediately gives the bias schedule we had been *assuming*:

    spread(ℓ) = max_{leaf in cell} f − (cell mean) ≤ L · rho**ℓ      (`lipschitz_spread`).

So one constant L (and discount rho, naturally 1/branching) *derives* the whole schedule.
For **prefix/suffix trees the ultrametric is exactly the longest-common-prefix distance**,
which is why this is the right notion for the LLM-routing use case below.

## Piecewise-Lipschitz with K jumps

Weaken to: f is L-Lipschitz **except** at a set D of K discontinuities (cliffs of height
≤ B), satisfying a *dispersion* condition — no small region contains too many of them
(Balcan–Dick–Vitercik). Then for a level-ℓ cell C:

* if C contains no discontinuity: `max − mean ≤ L·rho**ℓ` (smooth bound);
* if C straddles a jump: `max − mean ≤ L·rho**ℓ + (heights of the ≤ O(1) jumps in C) ≤ L·rho**ℓ + B`.

Only O(K) cells per level straddle a jump, i.e. O(K·depth) "bad" cells total (down to a
given resolution).

## Regret bound (statement + sketch)

**Claim.** Run the optimistic algorithm (UCB index `μ̂ + √(2 ln t / T) + bias`, B-value
backup) with `bias = spread(ℓ) = L·rho**ℓ` on smooth cells and `bias = B` on the O(K·depth)
jump cells. Then after n rounds,

    R_n  ≤  C_1 · n^{(d+1)/(d+2)}  +  C_2 · K · depth · B,

where d is the near-optimality (zooming) dimension of the Lipschitz part and C_1 depends
on (L, rho, d). The first term is the standard X-armed-bandit Lipschitz regret; the second
is an additive, horizon-independent price for the K jumps.

**Sketch.** Partition explored cells into (a) smooth and (b) jump cells. For (a), the
analysis is HOO/zooming verbatim with diameter `L·rho**ℓ`, giving the `n^{(d+1)/(d+2)}`
term — because on smooth cells `spread(ℓ)` is a valid optimism bonus and the B-value backup
prunes sub-optimal regions at the zooming rate. For (b), each of the O(K·depth) jump cells
is visited at most Õ(1/Δ²) times before its UCB separates it (Δ the relevant gap), and its
per-visit regret is ≤ B; dispersion ensures these cells do not pile up at a single scale,
so their total contribution is O(K·depth·B). Summing gives the bound. (This is a sketch
leaning on the HOO regret theorem and the dispersion machinery, not a complete proof.)

Two consequences match the experiments:
* With **no jumps** (K = 0) it reduces to the clean Lipschitz rate.
* The jump cost is **additive and horizon-independent**, so for fixed K it is dominated by
  the first term as n grows — i.e. a few jumps don't change the rate, only the constant.

## The hybrid algorithm (`run_hybrid`) and what it buys

The catch: on a *jump* cell the assumed bias `L·rho**ℓ` is invalid (too small), so a method
that trusts it under-explores and stalls. The fix is to **detect** jumps from data: in a
Lipschitz cell the within-cell variance obeys `σ_within ≲ L·rho**ℓ`, so

    measured σ_within > (factor) · L·rho**ℓ   ⟹   the cell provably straddles a jump,

and there we inflate the bias to the data-driven `σ_within·√(2 log m)`. This is `run_hybrid`:
tight Lipschitz bias where the data looks smooth, inflated bias only where a discontinuity
is detected.

Empirically (1024 leaves, 4 jumps, horizon 15k, 10 seeds; `examples/jump_robustness_demo.py`):

| jump width | assumed-smooth | data-driven | hybrid |
| --- | --- | --- | --- |
| 4 (hidden) | 3279 | 745 | **679** |
| 8 | 1673 | 972 | **806** |
| 16 | 1459 | 1108 | **990** |
| 32 | 895 | 1144 | 1035 |
| 64 (visible) | **272** | 1209 | 1088 |

The hybrid **dominates the data-driven method at every width** and beats assumed-smoothness
on hidden/medium jumps, never suffering its blow-up (3279). It only loses to assumed-smooth
on fully-visible jumps — exactly the regime where you'd know smoothness holds. Net: the
hybrid is the robust choice across the whole spectrum, for a modest constant premium.

## Use case: routing to LLMs (Suman)

Instantiate the tree as a **prefix (or suffix) tree over tokens**; a leaf is a prompt/
input, an internal node is the class of inputs sharing that prefix, and `f(leaf)` is the
quality (or −error) of a given LLM on that input. The LCA ultrametric **is** the
longest-common-prefix distance, so "tree-Lipschitz" reads as *inputs sharing a long prefix
get similar quality* — a plausible prior — and **jumps are exactly the sharp
prompt-sensitivity cliffs** where one token flips behavior. Multi-fidelity fits too: a cheap
value-model score on a prefix is the cheap biased probe; a full generation + grader is the
expensive leaf. Routing across LLMs = a per-model tree function f_m (replant the error
function per model) and pick model × input region.

Honest caveat: real prompt→quality is probably *not* very Lipschitz and likely has many
cliffs, so the dispersion assumption is the thing to validate on real data — which is
precisely why the jump-robust / data-driven hybrid (rather than pure assumed smoothness) is
the right tool here.
