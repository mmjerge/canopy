/-
Copyright (c) 2026 Michael Jerge. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Michael Jerge
-/
import Mathlib

/-!
# Composition steps of the fixed-budget identification theorem (Theorem 2)

Machine-checked deterministic composition of Theorem 2's proof, with the component
guarantees (detection, per-phase fixed-budget errors from successive rejects) supplied
as hypotheses --- the same convention as GIF's use of hypothesis-supplied guarantees:

* `budget_split` — Step 4: equalizing the phase exponents under `B_s + B_c = B` gives
  `B_s / c0 = B / (c0 + H)` and `B_c / H = B / (c0 + H)`.
* `error_synthesis` — the union-bound synthesis: component errors
  `δ_det + A·exp(-κ·B_s/c0) + K·exp(-κ·B_c/H)` collapse, under the equalized split, to
  `δ_det + (A + K)·exp(-κ·B/(c0 + H))`.
-/

open Real

namespace Canopy

/-- **Theorem 2, Step 4 (budget split).** If `B_s / c0 = B_c / H` and `B_s + B_c = B`
with `c0, H > 0`, then both ratios equal `B / (c0 + H)`. -/
theorem budget_split (B Bs Bc c0 H : ℝ) (hc0 : 0 < c0) (hH : 0 < H)
    (heq : Bs / c0 = Bc / H) (hsum : Bs + Bc = B) :
    Bs / c0 = B / (c0 + H) ∧ Bc / H = B / (c0 + H) := by
  have hc0H : 0 < c0 + H := by linarith
  have hBs : Bs * H = Bc * c0 := by
    field_simp at heq
    linarith
  constructor
  · rw [div_eq_div_iff (ne_of_gt hc0) (ne_of_gt hc0H)]
    nlinarith
  · rw [div_eq_div_iff (ne_of_gt hH) (ne_of_gt hc0H)]
    nlinarith

/-- **Theorem 2, synthesis.** Under the equalized budget split, the union bound over the
three failure events gives the stated form: if the detection error is at most `d`, the
smooth-phase error at most `A * exp (-k * (Bs / c0))`, and the certification error at most
`K * exp (-k * (Bc / H))`, then the total error is at most
`d + (A + K) * exp (-k * (B / (c0 + H)))`. -/
theorem error_synthesis (B Bs Bc c0 H d A K k e1 e2 e3 : ℝ)
    (hc0 : 0 < c0) (hH : 0 < H)
    (heq : Bs / c0 = Bc / H) (hsum : Bs + Bc = B)
    (h1 : e1 ≤ d) (h2 : e2 ≤ A * Real.exp (-k * (Bs / c0)))
    (h3 : e3 ≤ K * Real.exp (-k * (Bc / H))) :
    e1 + e2 + e3 ≤ d + (A + K) * Real.exp (-k * (B / (c0 + H))) := by
  obtain ⟨hs, hc⟩ := budget_split B Bs Bc c0 H hc0 hH heq hsum
  rw [hs] at h2
  rw [hc] at h3
  nlinarith [Real.exp_pos (-k * (B / (c0 + H)))]

end Canopy
