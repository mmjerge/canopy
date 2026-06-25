# A fixed-budget / simple-regret bound parameterized by the number of violations

This note formalizes the empirical result of `docs/targeted_sampling.md`: in the
multi-fidelity regime, isolating the sharp edges and sampling around them identifies the
optimum with a budget that grows gracefully with the number of Lipschitz violations `K`. We
state the guarantee in the **fixed-budget best-arm-identification** framework (a complexity
`H` and an exponential error in `B/H`), tie it to the near-optimality dimension of the smooth
part, and verify the predicted two-regime scaling against the experiments.

This is a *sketch* — it composes three known results (zooming/HOO, fixed-budget BAI, and an
empirical-Bernstein detection step), not a self-contained proof.

## Setting and assumptions

* Complete `b`-ary tree, depth `D`, `N = b**D` leaves; leaf means `μ(x) ∈ [0, 1]`.
* **Multi-fidelity feedback.** A random-path *probe* of an internal node returns
  `μ(L) + η` for a uniformly random leaf `L` under it (a cheap, biased aggregate), at cost
  `c_p`; a *leaf evaluation* returns `μ(x) + η` (expensive, unbiased) at cost `c_ℓ ≥ c_p`.
  Noise `η` is sub-Gaussian with scale `σ`.
* **Locally Lipschitz with `K` violations.** There is a set `V` of `K` cells such that *off*
  `V` the function is tree-Lipschitz — within-cell `max − mean ≤ spread(ℓ) = L·ρ**ℓ` (so the
  smooth bound is a valid optimism/pruning bonus) — while *on* the `K` cells of `V` that
  bound fails (sharp edges). The smooth part has near-optimality (zooming) dimension `d`.
* **Goal.** Identify the top-1 leaf `x*` (which may be *hidden inside* a violation), at a
  fixed cost budget `B`. Let `Δ` be the top-1 gap and `Δ_x = μ* − μ(x)` the per-leaf gaps.

## The edge-targeted algorithm (recap)

1. **Detect (cheap probes).** Estimate each cell's within-cell spread by deconvolving the
   noise (`max(0, Var̂ − σ²)`) and flag the cells whose spread exceeds a data-driven floor
   (`detect_violations` / `multiscale_edge_map`). Call the flagged set `V̂`.
2. **Search.** Run the optimistic multi-fidelity descent (HOO/zooming) using the smooth bound
   `spread(ℓ)` on cells `∉ V̂` (cheap probes prune whole subtrees), and **relax** to
   leaf-level certification on cells `∈ V̂` (`HierarchicalTopK(relaxed_ranges=V̂)`).

## Complexity and the bound

Define the **edge-targeted complexity**

    H_edge  =  c_0(d, Δ; c_p)  +  Σ_{k=1}^{K} c_ℓ · h_k ,
    h_k = Σ_{x ∈ C_k} σ² / max(Δ_x, Δ)²   (≤ m_k σ² / Δ²),

where `c_0` is the cost of the structural search over the smooth part (governed by the
near-optimality dimension `d`, paid in *cheap* probes `c_p`), and `h_k` is the fixed-budget
hardness of leaf-certifying the `k`-th violation cell (`m_k` leaves), paid in *expensive*
evaluations `c_ℓ`.

**Theorem (fixed-budget, informal).** There is a constant `κ > 0` such that the edge-targeted
algorithm at budget `B` misidentifies `x*` with probability

    P(error)  ≤  δ_det  +  Õ(N) · exp( − κ · B / H_edge ),

where `δ_det` is the detection-failure probability. Equivalently, error `≤ δ` is achieved
once `B ≳ H_edge · log(Õ(N)/δ) + B_det`, with the detection budget
`B_det = O(c_p · (σ²/γ²) · (#cells) · log(#cells/δ))` for a detection margin `γ`.

### Proof sketch

* **Detection.** Within-cell spread is estimated by an empirical-Bernstein bound; with
  `O((σ²/γ²) log(#cells/δ))` probes per cell every true violation is flagged and every smooth
  cell is not, w.p. `≥ 1 − δ_det`. This phase uses only cheap probes.
* **Smooth part.** Off `V̂` the bound `spread(ℓ)` is valid, so the descent is the standard
  zooming/HOO algorithm; its fixed-budget guarantee gives error `exp(−κ B / c_0)` for locating
  the optimum *when it lies in the smooth part*, with `c_0` set by the near-optimality
  dimension `d` (and cheap probes).
* **Hidden optimum.** If `x*` sits in a violation cell, that cell is in `V̂` (detection),
  is relaxed, and is leaf-certified by standard fixed-budget BAI (successive rejects /
  Audibert–Bubeck) at hardness `h_k`.
* **Compose.** `x*` lives in exactly one place; the budget splits across the structural
  search and the `≤ K` violation certifications, whose complexities **add**, giving
  `H_edge`. A union bound over the failure events yields the stated probability.

## Corollaries (and how they match the experiments)

* **Recovers the smooth rate (`K = 0`).** `H_edge = c_0`: cheap structural search governed by
  `d`, no leaf certification.
* **Graceful in `K` (sparse regime).** `H_edge` grows **additively**: `c_0 + Σ_{k≤K} c_ℓ h_k`.
  Each violation adds a bounded, horizon-independent certification cost — "fewer violations ⇒
  better."
* **Beats blind when violations are sparse and probes cheap.** The structure-blind method has
  `H_blind = c_ℓ · Σ_{all x} σ²/Δ_x²` (every leaf at full cost); `H_edge ≪ H_blind` because
  the bulk is searched with cheap probes (`c_0`) and only `K` cells incur leaf cost.
* **Saturation / crossover.** As `K → #cells`, `V̂` covers the tree, `H_edge → H_blind`, and
  the advantage vanishes — consistent with the budget-sweep crossover (blind catches up once
  `B` can cover all leaves).

## Empirical validation (the two regimes)

Budget needed for the edge-targeted method to reach 50% top-1 accuracy vs `K`
(`adversarial_spike`, branching 4, depth 5, level-3 cells = 64 cells, 24 seeds):

| `K` | 2 | 4 | 8 | 16 | 24 |
| --- | --- | --- | --- | --- | --- |
| budget to 50% | ≤150 | ≤150 | 250 | 2500 | not reached |

The cost is **near-flat / additive while violations are sparse** (`K ≪ 64`) and **rises
sharply as `K` approaches the cell count** — exactly the `H_edge = c_0 + Σ h_k` regime
saturating to `H_blind`. The matching accuracy-vs-budget curve at fixed `K` (≈3.5× efficiency
over blind) is in `docs/targeted_sampling.md`.

## Honest caveats / open

* The detection floor must be calibrated (a median floor collapses on mostly-smooth levels;
  the 75th-percentile-with-noise-guard floor is what works — see `multiscale_edge_map`); the
  empirical-Bernstein step assumes bounded/sub-Gaussian rewards.
* `c_0` inherits the zooming-dimension dependence and the validity of the smooth bound off
  `V`; making the detector both cheap and robust (vs. probing all scales) needs
  confidence-based refinement, which is also what would turn this sketch into a clean,
  self-contained fixed-budget theorem.
* A matching **lower bound** (that `H_edge` is the right complexity, not just an upper bound)
  is open.
