/-
Copyright (c) 2026 Michael Jerge. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Michael Jerge
-/
import Mathlib

/-!
# Composition steps of the piecewise-Lipschitz regret theorem (Theorem 3)

Machine-checked bookkeeping of Theorem 3's proof:

* `jump_cells_per_level_le` — the tree-dispersion count: at any level, the cells
  containing at least one of `K` boundary points number at most `K` (so over `D`
  levels the jump cells number at most `K * D`).
* `jump_regret_sum_le` — Step 2's summation: if each of at most `n_cells ≤ K * D`
  jump cells contributes regret at most `C * log n / Δ_min`, the total jump regret
  is at most `C * K * D * log n / Δ_min`.
-/

open Finset

namespace Canopy

/-- **Tree-dispersion count.** At a fixed level, cells are the image of the `K` boundary
points under the point-to-cell map, so at most `K` cells contain a boundary point. -/
theorem jump_cells_per_level_le {P C : Type*} [DecidableEq C] (boundary : Finset P)
    (cellOf : P → C) :
    (boundary.image cellOf).card ≤ boundary.card :=
  Finset.card_image_le

/-- **Theorem 3, Step 2 (jump-regret summation).** If there are at most `K * D` jump cells
and each contributes regret at most `c`, the total jump regret is at most `K * D * c`. -/
theorem jump_regret_sum_le {V : Type*} (cells : Finset V) (regret : V → ℝ)
    (K D : ℕ) (c : ℝ) (hc : 0 ≤ c)
    (hcard : cells.card ≤ K * D) (hbound : ∀ v ∈ cells, regret v ≤ c) :
    ∑ v ∈ cells, regret v ≤ (K * D : ℝ) * c := by
  calc ∑ v ∈ cells, regret v ≤ ∑ _v ∈ cells, c := Finset.sum_le_sum hbound
    _ = cells.card * c := by rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ (K * D : ℝ) * c := by
        apply mul_le_mul_of_nonneg_right _ hc
        exact_mod_cast hcard

end Canopy
