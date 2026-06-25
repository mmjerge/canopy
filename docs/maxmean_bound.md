# A data-driven, noise-deconvolved bound on (max leaf − subtree mean)

Goal: replace the *assumed* smoothness schedule `spread(ℓ)` (and the *invalid* light-tail
heuristic `σ_within·√(2 log m)`) with a **high-probability, data-driven** upper bound on
the bias term

    B(v) = max_{ℓ ∈ L(v)} μ(ℓ) − f(v),     f(v) = (1/m) Σ_{ℓ∈L(v)} μ(ℓ),   m = |L(v)|,

computed only from random-path plays of node v. Each play returns `X = μ(L) + η` for a
uniformly random leaf `L ∈ L(v)` and zero-mean noise `η` with known sub-Gaussian scale σ.
We never observe individual leaf means.

## The bound

**Step 1 — log-sum-exp (softmax) upper bound on the max.** For any λ > 0,

    max_ℓ μ(ℓ) ≤ (1/λ) log Σ_ℓ exp(λ μ(ℓ)) = (1/λ) [ log m + log E_L exp(λ μ(L)) ].

So `B(v) ≤ (log m)/λ + (1/λ) log M_μ(λ) − f(v)`, where `M_μ(λ) = E_L exp(λ μ(L))` is the
(uncentered) MGF of the leaf-mean distribution.

**Step 2 — deconvolve the known noise.** Since `X = μ(L) + η` with η ⫫ L and MGF
`M_η(λ) ≤ exp(λ²σ²/2)`,

    E exp(λX) = M_μ(λ) · M_η(λ)   ⟹   M_μ(λ) = E exp(λX) · exp(−λ²σ²/2).

**Step 3 — estimate `E exp(λX)` with a one-sided confidence.** With n i.i.d. plays and
leaf means in `[0,1]` (so `Y_i = exp(λ X_i)` is bounded), an empirical-Bernstein upper
confidence gives, w.p. ≥ 1 − δ simultaneously over a finite grid Λ,

    E exp(λX) ≤ Ĝ(λ) + √(2 V̂(λ) ln(2|Λ|/δ) / n) + (7 R_λ ln(2|Λ|/δ)) / (3(n−1)),

where `Ĝ(λ) = (1/n)Σ exp(λX_i)`, `V̂(λ)` is its sample variance, and
`R_λ = exp(λ·hi) − exp(λ·lo)` is the range of `exp(λX)`.

**Resulting bound.** With probability ≥ 1 − δ,

    B(v) ≤ min_{λ ∈ Λ} { (1/λ)[ log m + log Ĝ_upper(λ) − λ²σ²/2 ] − x̄ }.

This is `mgf_bound` / `mgf_bound_from_moments` in `canopy.bandits.maxmean`. It needs only
O(|Λ|) running moments per node (`Σ exp(λX_i)`, `Σ exp(2λX_i)`), so it is cheap online.

## Why it is the right object (special cases)

* **Sub-Gaussian leaf means** (variance proxy σ_w²): `M_μ(λ) ≤ exp(λ f(v) + λ²σ_w²/2)`, and
  minimizing `(log m)/λ + λσ_w²/2` gives `σ_w·√(2 log m)` at `λ* = √(2 log m)/σ_w`. So the
  bound **recovers the light-tail rate** — but as a guarantee, not a heuristic.
* **Worst case (a single spike):** the log-sum-exp tends to the true max, so the bound
  degrades gracefully to ≈ `max − mean` (and is comparable to the distribution-free
  Samuelson bound `σ_w√(m−1)`), instead of failing like the sub-Gaussian heuristic.

It is therefore a **data-adaptive interpolation** between the loose worst-case bound and
the light-tail rate, valid throughout.

## Validation (256 leaves, σ=0.1, n=2000; see test_maxmean.py)

| instance | true B | Samuelson (hard) | sub-Gaussian (heuristic) | **MGF (ours)** |
| --- | --- | --- | --- | --- |
| clustered / light tail | 0.153 | 0.865 | 0.180 ✓ | 0.752 ✓ |
| single spike | 0.747 | 1.026 | 0.214 ✗ | 1.050 ✓ |
| gaussian spread | 0.505 | 3.232 | 0.674 ✓ | 0.747 ✓ |
| few spikes | 0.686 | 1.432 | 0.299 ✗ | 1.039 ✓ |

The sub-Gaussian heuristic is *invalid* on spikes (it under-estimates). The MGF bound is
valid throughout and dominates Samuelson whenever there is genuine spread.

## Driver vs. certifier (an honest finding)

Plugging the rigorous bound in as the optimistic bonus (`run_adaptive_mgf`) *hurts*
regret and memory (it over-explores, because a valid bound is necessarily conservative at
small n). The clean design is therefore: **drive exploration with the tight heuristic**
(`run_adaptive_variance`, σ_w·√(2 log m)) and **use the MGF bound to certify** — e.g. for
the high-probability soundness of pruning / stopping, where validity, not tightness, is
what matters. The two roles want different bounds.

## Novelty positioning (honest)

The ingredients are classical — log-sum-exp/MGF max bounds (Boucheron–Lugosi–Massart),
empirical Bernstein (Maurer–Pontil), Samuelson's inequality, and MGF/characteristic-
function deconvolution. The contribution is the **synthesis and the application**: a
*noise-deconvolved, empirical-MGF, high-probability* bound on `max − mean` from mixture
samples, used to make a hierarchical bandit *self-certifying* — its bias/resolution term
estimated and guaranteed from data rather than assumed via `spread(ℓ)`.

## Open

- Rigor for unbounded (sub-Gaussian) noise needs a sub-exponential concentration on
  `exp(λX)` rather than the bounded-reward Hoeffding/Bernstein step.
- Continuous optimization over λ (vs. a grid) and a matching lower bound would sharpen it.
- The DAG generalization keeps the same structure: a node value is a linear functional of
  the leaf vector, so the same MGF deconvolution applies to weighted subset averages.
