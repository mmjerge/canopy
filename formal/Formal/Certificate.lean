/-
Copyright (c) 2026 Michael Jerge. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Michael Jerge
-/
import Mathlib

/-!
# Mechanized soundness core of the aggregation-bias certificate (Theorem 1)

Machine-checked deterministic skeleton of the paper's data-driven certificate:

* `max_le_logSumExp` — Step 1: for finitely many leaf means and any `lam > 0`,
  `max_i mu i ≤ (1/lam) * (log m + log ((1/m) * Σ_i exp (lam * mu i)))`.
* `aggregation_bias_bound` — the bound restated for `B(v) = max mu - f(v)`.
* `subgaussian_rate` — the value of the sub-Gaussian relaxation at the optimizer
  `lam* = sqrt (2 log m) / s` is `s * sqrt (2 log m)` (the light-tail rate).
* `subgaussian_rate_le` — that value is the true minimum: for every `lam > 0`,
  `s * sqrt (2 log m) ≤ log m / lam + lam * s^2 / 2` (AM–GM direction).

Not mechanized (paper-cited): the empirical-Bernstein confidence step
(Maurer–Pontil) and the Gaussian-MGF deconvolution, which are probabilistic;
this mirrors the manuscript's disclosure of what is derived vs. invoked.
-/

open Finset Real

namespace Canopy

/-- **Step 1 of Theorem 1 (log-sum-exp bound).** For a finite nonempty index set `s`,
leaf means `mu : ι → ℝ`, and `lam > 0`:
`max_i mu i ≤ (1/lam) * (log #s + log ((#s)⁻¹ * Σ_{i∈s} exp (lam * mu i)))`. -/
theorem max_le_logSumExp {ι : Type*} (s : Finset ι) (hs : s.Nonempty)
    (mu : ι → ℝ) (lam : ℝ) (hlam : 0 < lam) :
    s.sup' hs mu ≤ (1 / lam) *
      (Real.log s.card + Real.log ((s.card : ℝ)⁻¹ * ∑ i ∈ s, Real.exp (lam * mu i))) := by
  have hcard : (0 : ℝ) < s.card := by
    exact_mod_cast Finset.card_pos.mpr hs
  have hsum : (0 : ℝ) < ∑ i ∈ s, Real.exp (lam * mu i) :=
    Finset.sum_pos (fun i _ => Real.exp_pos _) hs
  -- log m + log (m⁻¹ * S) = log S
  have hlog : Real.log s.card + Real.log ((s.card : ℝ)⁻¹ * ∑ i ∈ s, Real.exp (lam * mu i))
      = Real.log (∑ i ∈ s, Real.exp (lam * mu i)) := by
    rw [Real.log_mul (inv_ne_zero (ne_of_gt hcard)) (ne_of_gt hsum), Real.log_inv]
    ring
  rw [hlog]
  obtain ⟨j, hj, hjmax⟩ := Finset.exists_mem_eq_sup' hs mu
  rw [hjmax]
  have hterm : Real.exp (lam * mu j) ≤ ∑ i ∈ s, Real.exp (lam * mu i) :=
    Finset.single_le_sum (f := fun i => Real.exp (lam * mu i))
      (fun i _ => (Real.exp_pos _).le) hj
  have hle : lam * mu j ≤ Real.log (∑ i ∈ s, Real.exp (lam * mu i)) := by
    rw [← Real.log_exp (lam * mu j)]
    exact Real.log_le_log (Real.exp_pos _) hterm
  calc mu j = (1 / lam) * (lam * mu j) := by field_simp
    _ ≤ (1 / lam) * Real.log (∑ i ∈ s, Real.exp (lam * mu i)) := by
        apply mul_le_mul_of_nonneg_left hle
        positivity

/-- **The certificate's target form.** The aggregation bias `B(v) = max mu - f(v)`
obeys the certified bound with the empirical mixture moment. -/
theorem aggregation_bias_bound {ι : Type*} (s : Finset ι) (hs : s.Nonempty)
    (mu : ι → ℝ) (lam : ℝ) (hlam : 0 < lam) (fv : ℝ) :
    s.sup' hs mu - fv ≤ (1 / lam) *
      (Real.log s.card + Real.log ((s.card : ℝ)⁻¹ * ∑ i ∈ s, Real.exp (lam * mu i))) - fv := by
  have := max_le_logSumExp s hs mu lam hlam
  linarith

/-- **Sub-Gaussian rate, value at the optimizer.** At `lam* = sqrt (2 log m) / s` the
relaxation `log m / lam + lam * s^2 / 2` equals `s * sqrt (2 * log m)`. -/
theorem subgaussian_rate (m sw : ℝ) (hm : 1 < m) (hsw : 0 < sw) :
    Real.log m / (Real.sqrt (2 * Real.log m) / sw) +
      (Real.sqrt (2 * Real.log m) / sw) * sw ^ 2 / 2 = sw * Real.sqrt (2 * Real.log m) := by
  have hlogm : 0 < Real.log m := Real.log_pos hm
  set t := Real.sqrt (2 * Real.log m) with htdef
  have ht : t ^ 2 = 2 * Real.log m := Real.sq_sqrt (by linarith)
  have htpos : 0 < t := Real.sqrt_pos.mpr (by linarith)
  have hlog_eq : Real.log m = t ^ 2 / 2 := by linarith
  rw [hlog_eq]
  field_simp
  ring

/-- **Sub-Gaussian rate is the minimum (AM–GM direction).** For every `lam > 0`,
`s * sqrt (2 * log m) ≤ log m / lam + lam * s^2 / 2`. -/
theorem subgaussian_rate_le (m sw lam : ℝ) (hm : 1 < m) (hlam : 0 < lam) :
    sw * Real.sqrt (2 * Real.log m) ≤ Real.log m / lam + lam * sw ^ 2 / 2 := by
  have hlogm : 0 < Real.log m := Real.log_pos hm
  set t := Real.sqrt (2 * Real.log m) with htdef
  have ht : t ^ 2 = 2 * Real.log m := Real.sq_sqrt (by linarith)
  have hlog_eq : Real.log m = t ^ 2 / 2 := by linarith
  rw [← sub_nonneg, hlog_eq]
  have expand : t ^ 2 / 2 / lam + lam * sw ^ 2 / 2 - sw * t
      = (t - lam * sw) ^ 2 / (2 * lam) := by
    field_simp
    ring
  rw [expand]
  positivity

end Canopy
