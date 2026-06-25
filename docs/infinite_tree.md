# The infinite-depth tree: how weak can the smoothness assumption be?

Setting (per Suman): **fixed arity b, infinite depth**. Leaves become infinite paths;
the value of an internal node is still the average reward over its (now infinite) subtree.
The question: the *finite-state compression* — keeping only a finite explored tree — needs
some smoothness; how weak can it be?

First, an honest note: this is **not** in the Hazan OCO book (which covers continuous
*convex* optimization). The relevant theory is X-armed / continuous-armed bandits:
Bubeck–Munos–Stoltz–Szepesvári (HOO), Munos' monograph *From Bandits to Monte-Carlo Tree
Search*, and the zooming algorithm (Kleinberg–Slivkins–Upfal).

## Formalization

Index nodes by finite strings over an alphabet of size b; an infinite path x ∈ {1..b}^∞
is a "leaf" with reward mean f(x). A depth-h node v owns the cell C(v) of all paths
through it, and its value is f(v) = E[f(x) | x ∈ C(v)] (the subtree average). Define the
optimum f* = sup_x f(x).

## The smoothness assumption (and how weak it can be)

The bias of committing at node v (level h) is, as before,

    B(v) = sup_{x ∈ C(v)} f(x) − f(v) ≤ spread(h),

and we need spread(h) → 0 as h → ∞ so that deep commitments are nearly bias-free. The
strength of the assumption is set by *where* and *how fast* this must hold:

1. **Global, two-sided, known rate** (strongest, what our toy uses): spread(h) ≤ ν ρ^h
   for all nodes. Easy but unrealistically strong.
2. **Local near the optimum only** (HOO): the bound need only hold for cells containing a
   near-optimal path, i.e. for v with f(v) ≥ f* − spread(h). This is one-sided and local
   — much weaker — and is the standard assumption. The cost of search is then governed by
   the **near-optimality dimension** d: the number of ε-optimal cells at resolution ε
   grows like ε^{-d}.
3. **Unknown rate (ν, ρ)** (weaker still): SOO / StoSOO (Munos; Valko–Carpentier–Munos)
   and POO (Grill–Valko–Munos) achieve near-optimal simple/cumulative regret *without
   knowing* (ν, ρ), by running/ aggregating over a ladder of candidate smoothness levels.
4. **Weakest meaningful**: only require that along the optimal path the per-level bias is
   summable enough that d < ∞ — equivalently, that the optimum is *detectable at finite
   resolution*: each step down the optimal path reduces the bias, so a finite-resolution
   probe eventually distinguishes the optimal subtree.

**Lower-bound intuition (why some assumption is unavoidable):** with no smoothness, a
single great path can hide arbitrarily deep in a subtree whose averages look ordinary, so
no finite exploration can find it — you would have to descend forever. Some link between
the average (what you observe) and the supremum (what you want) is necessary; spread(h)→0
is the minimal such link, and the near-optimality dimension quantifies how much it buys.

## Finite-state compression (the payoff)

Under assumption (2), HOO/HCT only ever expand nodes near the optimum: after n rounds the
explored tree has Õ(n^{d/(d+2)}) nodes — **finite and independent of the (infinite) tree**.
So the infinite tree is compressed to a finite explored state whose size is set by d, not
by depth.

Empirically (`examples/infinite_depth_demo.py`, arity 3, horizon 15k, memory-bounded HOO):

| depth | total nodes in tree | explored memory | final regret |
| --- | --- | --- | --- |
| 4 | 121 | 55 | 663 |
| 6 | 1,093 | 80 | 1495 |
| 8 | 9,841 | 75 | 1454 |
| 10 | 88,573 | 69 | 1562 |
| 12 | 797,161 | 70 | 1702 |

Total nodes grow ~6500x while explored memory stays ~70 and regret stays controlled —
the discrete witness of finite-state compression in the infinite-depth limit.

## For Tuesday

- Weakest assumption to target: local, one-sided smoothness near the optimum with
  *unknown* rate (POO/StoSOO style), so finite-state compression holds with d < ∞ and no
  knowledge of (ν, ρ).
- Open: pair this with the data-driven max−mean certificate (`docs/maxmean_bound.md`) so
  the per-level bias is *estimated*, giving an adaptive infinite-depth algorithm that
  needs neither the spread schedule nor its rate.
