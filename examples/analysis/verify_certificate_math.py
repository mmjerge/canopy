"""Mechanical verification of the paper's certificate algebra (sympy + randomized checks).

Scope (honest): this does NOT formally verify the theorems --- the concentration steps
(empirical Bernstein) and the composed bandit analyses (zooming, fixed-budget BAI) are cited
results outside sympy's reach, and Theorems 2-3 are stated as proof sketches. What it DOES
verify mechanically is every self-contained algebraic step in the manuscript:

  T1.1  Log-sum-exp bound (Thm 1, Step 1):
          max_x mu(x) <= (1/lam) [log m + log E_L e^{lam mu(L)}]      (symbolic identity +
        the residual >= 0 argument, plus randomized numeric checks)
  T1.2  Noise deconvolution (Thm 1, Step 2): for eta ~ N(0, sigma^2) independent of L,
          E e^{lam X} = E e^{lam mu(L)} * e^{lam^2 sigma^2 / 2}       (symbolic Gaussian
        integral + factorization under independence, numeric check)
  T1.3  Assembly (Thm 1): given G_up >= E e^{lam X} and f(v) >= Xbar - eps_mean,
          B(v) <= (1/lam)[log m + log G_up - lam^2 sigma^2/2] - (Xbar - eps_mean)
        (inequality chain checked on randomized exact instances -- no estimation, so any
        violation would be an algebra error, not noise)
  T1.4  Sub-Gaussian recovery (Smoothness sec., "Properties"): min over lam of
          log(m)/lam + lam sigma_w^2 / 2  equals  sigma_w sqrt(2 log m)   (symbolic minimize)
  T2.4  Budget-split algebra (Thm 2, Step 4): B_smooth/c0 = B_cert/H_cert and
          B_smooth + B_cert = B  imply both ratios equal B/(c0 + H_cert)  (symbolic solve)
  T3.2  Jump-cell count (Thm 3 / dispersion): at most K cells straddle >= 1 of K boundary
        points per level, so |V_jump| <= K * D                          (exhaustive check on
        random trees)

Run:  .venv/bin/python examples/analysis/verify_certificate_math.py
Exits nonzero if any check fails.
"""

from __future__ import annotations

import sys

import numpy as np
import sympy as sp

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ------------------------------------------------------------------ T1.1 log-sum-exp
def t1_1_logsumexp() -> None:
    print("T1.1 log-sum-exp bound: max mu <= (1/lam)[log m + log mean_i e^{lam mu_i}]")
    lam, m = sp.symbols("lambda m", positive=True)

    # (i) symbolic identity: (1/lam)[log m + log((1/m) S)] == (1/lam) log S  for S > 0
    S = sp.symbols("S", positive=True)
    ident = sp.simplify((sp.log(m) + sp.log(S / m)) / lam - sp.log(S) / lam)
    check("identity (1/lam)[log m + log(S/m)] = (1/lam) log S", ident == 0)

    # (ii) residual argument: (1/lam) log sum_i e^{lam (mu_i - max)} >= 0 since the max term
    # contributes e^0 = 1 and every term is positive => sum >= 1 => log >= 0.
    mu1, mu2, mu3 = sp.symbols("mu1 mu2 mu3", real=True)
    mus = [mu1, mu2, mu3]
    mx = sp.Max(*mus)
    resid = sum(sp.exp(lam * (mu - mx)) for mu in mus)
    # each exponent is <= 0, one is exactly 0; check sum >= 1 numerically over random draws
    rng = np.random.default_rng(0)
    ok = True
    f = sp.lambdify((lam, mu1, mu2, mu3), resid, "numpy")
    for _ in range(20000):
        vals = rng.normal(0, 2, size=3)
        L = float(rng.uniform(0.05, 8.0))
        if f(L, *vals) < 1.0 - 1e-12:
            ok = False
            break
    check("residual sum_i e^{lam(mu_i - max)} >= 1 (20k random draws)", ok)

    # (iii) end-to-end numeric: the bound itself, vectorized over random instances
    ok = True
    worst = 0.0
    for _ in range(20000):
        mvec = rng.normal(0.3, 0.5, size=int(rng.integers(2, 30)))
        L = float(rng.uniform(0.05, 12.0))
        bound = (np.log(len(mvec)) + np.log(np.mean(np.exp(L * mvec)))) / L
        gap = mvec.max() - bound
        worst = max(worst, gap)
        if gap > 1e-10:
            ok = False
            break
    check("max mu <= bound on 20k random instances", ok, f"max violation {worst:.2e}")


# ------------------------------------------------------------------ T1.2 deconvolution
def t1_2_deconvolution() -> None:
    print("T1.2 noise deconvolution: E e^{lam X} = E e^{lam mu} * e^{lam^2 sigma^2/2}")
    lam, sigma = sp.symbols("lambda sigma", positive=True)
    eta = sp.symbols("eta", real=True)

    # (i) Gaussian MGF, symbolically: integral of e^{lam eta} * N(0, sigma^2) deta
    gauss = sp.exp(-(eta**2) / (2 * sigma**2)) / (sigma * sp.sqrt(2 * sp.pi))
    mgf = sp.simplify(sp.integrate(sp.exp(lam * eta) * gauss, (eta, -sp.oo, sp.oo)))
    check(
        "Gaussian MGF E e^{lam eta} = e^{lam^2 sigma^2/2} (symbolic integral)",
        sp.simplify(mgf - sp.exp(lam**2 * sigma**2 / 2)) == 0,
    )

    # (ii) factorization under independence: E e^{lam(mu(L)+eta)} = E e^{lam mu(L)} E e^{lam eta}
    # for a discrete L (mixture) and independent eta -- check by direct expansion on a 3-leaf
    # symbolic mixture with weights 1/3.
    mu1, mu2, mu3 = sp.symbols("mu1 mu2 mu3", real=True)
    lhs = sp.Rational(1, 3) * sum(
        sp.integrate(sp.exp(lam * (mu + eta)) * gauss, (eta, -sp.oo, sp.oo))
        for mu in (mu1, mu2, mu3)
    )
    rhs = (
        sp.Rational(1, 3)
        * sum(sp.exp(lam * mu) for mu in (mu1, mu2, mu3))
        * sp.exp(lam**2 * sigma**2 / 2)
    )
    check(
        "factorization E e^{lam X} = M_mu(lam) * M_eta(lam) (symbolic, 3-leaf mixture)",
        sp.simplify(lhs - rhs) == 0,
    )

    # (iii) the rearrangement used by the certificate: M_mu = E e^{lam X} * e^{-lam^2 s^2/2}
    EX = sp.symbols("EX", positive=True)
    Mmu = EX * sp.exp(-(lam**2) * sigma**2 / 2)
    check(
        "rearrangement M_mu = E e^{lam X} e^{-lam^2 sigma^2/2}",
        sp.simplify(Mmu * sp.exp(lam**2 * sigma**2 / 2) - EX) == 0,
    )


# ------------------------------------------------------------------ T1.3 assembly
def t1_3_assembly() -> None:
    print("T1.3 assembly: B(v) <= (1/lam)[log m + log G_up - lam^2 s^2/2] - (Xbar - eps)")
    rng = np.random.default_rng(1)
    ok = True
    worst = -np.inf
    for _ in range(20000):
        m = int(rng.integers(2, 40))
        mu = rng.uniform(0, 1, size=m)  # leaf means in [0,1] as assumed
        sigma = float(rng.uniform(0.01, 0.5))
        lam = float(rng.uniform(0.1, 8.0))
        f_v = mu.mean()
        B_true = mu.max() - f_v  # the aggregation bias
        E_expX = np.mean(np.exp(lam * mu)) * np.exp(lam**2 * sigma**2 / 2)  # exact E e^{lam X}
        # any valid upper confidence G_up >= E e^{lam X}, any valid lower conf f >= Xbar - eps:
        G_up = E_expX * float(rng.uniform(1.0, 3.0))
        eps = float(rng.uniform(0, 0.3))
        Xbar = f_v + float(rng.uniform(-1, 1)) * eps  # any Xbar with |Xbar - f| <= eps
        bound = (np.log(m) + np.log(G_up) - lam**2 * sigma**2 / 2) / lam - (Xbar - eps)
        gap = B_true - bound
        worst = max(worst, gap)
        if gap > 1e-10:
            ok = False
            break
    check("B(v) <= assembled bound on 20k exact random instances", ok, f"max violation {worst:.2e}")


# ------------------------------------------------------------------ T1.4 sub-Gaussian rate
def t1_4_subgaussian_rate() -> None:
    print("T1.4 sub-Gaussian recovery: min_lam [log m/lam + lam s_w^2/2] = s_w sqrt(2 log m)")
    lam = sp.symbols("lambda", positive=True)
    m, sw = sp.symbols("m sigma_w", positive=True)
    expr = sp.log(m) / lam + lam * sw**2 / 2
    lam_star = sp.solve(sp.diff(expr, lam), lam)
    lam_star = [s for s in lam_star if s.is_positive is not False]
    val = sp.simplify(expr.subs(lam, lam_star[0]))
    target = sw * sp.sqrt(2 * sp.log(m))
    check(
        "optimizer lam* = sqrt(2 log m)/sigma_w",
        sp.simplify(lam_star[0] - sp.sqrt(2 * sp.log(m)) / sw) == 0,
    )
    check("optimal value = sigma_w sqrt(2 log m)", sp.simplify(val - target) == 0)
    # second derivative = 2 log(m) / lam^3, positive iff m > 1; the paper's m is a leaf count
    # >= branching >= 2, so encode m = 1 + q with q > 0 to give sympy the needed assumption.
    q = sp.symbols("q", positive=True)
    second = sp.simplify(sp.diff(expr, lam, 2))
    check("second derivative = 2 log(m)/lam^3", sp.simplify(second - 2 * sp.log(m) / lam**3) == 0)
    check(
        "second derivative > 0 for m >= 2 (a minimum)",
        bool(sp.simplify(second.subs(m, 1 + q)).is_positive),
    )


# ------------------------------------------------------------------ T2.4 budget split
def t2_4_budget_split() -> None:
    print("T2.4 budget-split algebra: equalized exponents => both = B / H_edge")
    B, c0, H = sp.symbols("B c_0 H_cert", positive=True)
    Bs, Bc = sp.symbols("B_smooth B_cert", positive=True)
    sol = sp.solve([sp.Eq(Bs / c0, Bc / H), sp.Eq(Bs + Bc, B)], [Bs, Bc], dict=True)[0]
    ratio_s = sp.simplify(sol[Bs] / c0 - B / (c0 + H))
    ratio_c = sp.simplify(sol[Bc] / H - B / (c0 + H))
    check("B_smooth/c0 = B/(c0+H_cert)", ratio_s == 0)
    check("B_cert/H_cert = B/(c0+H_cert)", ratio_c == 0)


# ------------------------------------------------------------------ T3.2 jump-cell count
def t3_2_jump_cells() -> None:
    print("T3.2 dispersion bookkeeping: |V_jump| <= K * D on random trees")
    rng = np.random.default_rng(2)
    ok = True
    for _ in range(2000):
        b = int(rng.integers(2, 5))
        D = int(rng.integers(2, 7))
        n = b**D
        K = int(rng.integers(1, min(20, n)))
        boundaries = rng.choice(n, size=K, replace=False)  # K discontinuity leaf positions
        total = 0
        for level in range(1, D + 1):
            span = b ** (D - level)
            cells = set(int(x) // span for x in boundaries)
            total += len(cells)
            if len(cells) > K:  # per-level count can never exceed K
                ok = False
        if total > K * D:
            ok = False
        if not ok:
            break
    check("per-level jump cells <= K and total <= K*D (2k random trees)", ok)


def main() -> None:
    print(
        "Mechanical verification of the certificate algebra "
        "(symbolic where exact, randomized where an inequality)\n"
    )
    t1_1_logsumexp()
    t1_2_deconvolution()
    t1_3_assembly()
    t1_4_subgaussian_rate()
    t2_4_budget_split()
    t3_2_jump_cells()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print(
        "ALL CHECKS PASSED. Scope note: concentration steps (empirical Bernstein) and the "
        "composed bandit analyses are cited results, NOT verified here; Theorems 2-3 are "
        "proof sketches by design."
    )


if __name__ == "__main__":
    main()
