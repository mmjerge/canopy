"""Online / regret-minimization mode for the hierarchical bandit.

This is the *regret* counterpart to the pure-exploration top-k code. Each round the
learner commits to a node and follows a random path down (``env.play``), receiving a
reward whose mean is that node's subtree average. Competing against the best single
leaf, the per-round regret of committing at level ``l`` is at most ``spread(l)`` -- the
bias of mean-backup at that resolution.

The central design question (Suman): when do you keep refining a node's estimate vs.
expand it into children (trading bias for memory + new statistical uncertainty)?

Strategies:
  * :func:`run_fixed_depth` -- commit to a fixed resolution ``d``: run UCB over all
    ``b**d`` nodes at that depth. Memory ``b**d``; asymptotic per-round regret = the
    bias ``mu* - max_{depth-d node} f(node)``. ``d=0`` never expands (play the root);
    ``d=depth`` is full leaf resolution.
  * :func:`run_adaptive` -- optimistic descent with the regret-optimal expansion rule:
    refine a node while it is statistically limited and expand it the moment it becomes
    bias limited, i.e. when its confidence radius ``r(v) <= spread(level)``. This is the
    HOO/HCT-style trigger; it concentrates memory near the optimum.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from canopy.bandits.maxmean import mgf_bound_from_moments
from canopy.bandits.tree import Node, TreeBandit


@dataclass
class RegretResult:
    """Per-round cumulative (expected) regret and peak memory for one run."""

    cum_regret: np.ndarray  # shape (horizon,), cumulative expected regret
    memory: int  # peak number of nodes for which statistics were maintained
    label: str

    @property
    def final_regret(self) -> float:
        return float(self.cum_regret[-1])


def run_fixed_depth(
    env: TreeBandit,
    depth: int,
    horizon: int,
    rng: np.random.Generator,
    c: float = 0.5,
) -> RegretResult:
    """UCB over all nodes at a fixed resolution ``depth`` (never expands)."""
    nodes = [Node(depth, i) for i in range(env.branching**depth)]
    m = len(nodes)
    true_f = np.array([env.true_value(nd) for nd in nodes])
    mu_star = env.best_leaf_value()
    counts = np.zeros(m)
    means = np.zeros(m)
    cum = 0.0
    cum_regret = np.empty(horizon)
    for t in range(horizon):
        if t < m:
            a = t  # initialize each arm once
        else:
            bonus = c * np.sqrt(np.log(t + 1) / counts)
            a = int(np.argmax(means + bonus))
        reward = env.play(nodes[a], rng)
        counts[a] += 1
        means[a] += (reward - means[a]) / counts[a]
        cum += mu_star - true_f[a]
        cum_regret[t] = cum
    return RegretResult(cum_regret=cum_regret, memory=m, label=f"fixed-depth {depth}")


def run_adaptive(
    env: TreeBandit,
    horizon: int,
    spread: Callable[[int], float],
    rng: np.random.Generator,
    c: float = 0.5,
    warm_start: bool = True,
) -> RegretResult:
    """Optimistic descent that expands a node when ``r(v) <= spread(level)``."""
    n_nodes = sum(env.branching**level for level in range(env.depth + 1))
    log_term = math.log(n_nodes * horizon)
    mu_star = env.best_leaf_value()

    def n_star(level: int) -> float:
        s = spread(level)
        if s <= 1e-9:
            return math.inf  # leaves / zero-spread: never expand
        return (c * c) * log_term / (s * s)

    # stats[node] = [count, mean]; frontier = currently playable nodes.
    frontier: list[Node] = list(env.children(env.root()))
    stats: dict[Node, list[float]] = {nd: [0.0, 0.0] for nd in frontier}

    cum = 0.0
    cum_regret = np.empty(horizon)
    peak_memory = len(stats)
    for t in range(horizon):
        # pick the frontier node with the highest optimistic value
        best, best_u = frontier[0], -math.inf
        for v in frontier:
            count, mean = stats[v]
            if count == 0:
                u = math.inf
            else:
                u = mean + c * math.sqrt(log_term / count) + spread(v.level)
            if u > best_u:
                best_u, best = u, v
        v = best

        reward = env.play(v, rng)
        count, mean = stats[v]
        count += 1
        mean += (reward - mean) / count
        stats[v] = [count, mean]

        cum += mu_star - env.true_value(v)
        cum_regret[t] = cum

        # regret-optimal expansion: statistically limited -> bias limited
        if v.level < env.depth and count >= n_star(v.level):
            frontier.remove(v)
            for child in env.children(v):
                # warm-start children from the parent mean (it is their average:
                # a linear constraint), giving one pseudo-observation.
                stats[child] = [1.0, mean] if warm_start else [0.0, 0.0]
                frontier.append(child)
            peak_memory = max(peak_memory, len(stats))

    return RegretResult(cum_regret=cum_regret, memory=len(stats), label="adaptive")


def run_adaptive_variance(
    env: TreeBandit,
    horizon: int,
    rng: np.random.Generator,
    c: float = 0.5,
    n_min: int = 12,
    warm_start: bool = True,
) -> RegretResult:
    """Variance-aware adaptive expansion (the novel variant).

    Two changes from :func:`run_adaptive`:

    1. The confidence radius is empirical-Bernstein: it uses each node's *measured*
       reward variance instead of a fixed noise scale. Playing a node via random-path
       has variance ``sigma_within(v)^2 + noise^2``, so heterogeneous (coarse) nodes
       are correctly given wider intervals.
    2. The bias term is no longer an assumed ``spread(level)``. We estimate the
       within-subtree heterogeneity ``sigma_within(v)`` from data (subtracting the known
       observation noise) and use ``sigma_within(v) * sqrt(2 log|L(v)|)`` -- the expected
       gap between the max of ``|L(v)|`` leaves and their mean -- as a data-driven bias
       proxy. Expansion fires when the statistical radius drops below this estimated
       bias, so homogeneous subtrees stay coarse (saving memory) while heterogeneous
       ones get drilled.
    """
    log_term = math.log(sum(env.branching**level for level in range(env.depth + 1)) * horizon)
    mu_star = env.best_leaf_value()
    noise_var = env.noise_std**2

    # stats[node] = [count, mean, M2]  (Welford running variance)
    frontier: list[Node] = list(env.children(env.root()))
    stats: dict[Node, list[float]] = {nd: [0.0, 0.0, 0.0] for nd in frontier}

    def total_var(node: Node) -> float:
        count, _, m2 = stats[node]
        return m2 / (count - 1) if count >= 2 else float("inf")

    def within_std(node: Node) -> float:
        return math.sqrt(max(0.0, total_var(node) - noise_var))

    def leaves_under(node: Node) -> int:
        return env.leaves_per_node(node.level)

    def radius(node: Node) -> float:
        count, _, _ = stats[node]
        if count < 2:
            return math.inf
        v = total_var(node)
        # empirical-Bernstein style: variance term + lower-order term
        return c * math.sqrt(2.0 * v * log_term / count) + log_term / count

    def bias_proxy(node: Node) -> float:
        # data-driven analogue of spread(level): expected max-minus-mean of |L| leaves
        return within_std(node) * math.sqrt(2.0 * math.log(max(2, leaves_under(node))))

    cum = 0.0
    cum_regret = np.empty(horizon)
    for t in range(horizon):
        best, best_u = frontier[0], -math.inf
        for v in frontier:
            count, mean, _ = stats[v]
            u = math.inf if count < 2 else mean + radius(v) + bias_proxy(v)
            if u > best_u:
                best_u, best = u, v
        v = best

        reward = env.play(v, rng)
        count, mean, m2 = stats[v]
        count += 1
        delta = reward - mean
        mean += delta / count
        m2 += delta * (reward - mean)
        stats[v] = [count, mean, m2]

        cum += mu_star - env.true_value(v)
        cum_regret[t] = cum

        if v.level < env.depth and count >= n_min and radius(v) <= bias_proxy(v):
            frontier.remove(v)
            for child in env.children(v):
                stats[child] = [1.0, mean, 0.0] if warm_start else [0.0, 0.0, 0.0]
                frontier.append(child)

    return RegretResult(cum_regret=cum_regret, memory=len(stats), label="adaptive-variance")


def run_adaptive_mgf(
    env: TreeBandit,
    horizon: int,
    rng: np.random.Generator,
    c: float = 0.5,
    n_min: int = 12,
    delta: float = 0.05,
    lambdas: np.ndarray | None = None,
    warm_start: bool = True,
) -> RegretResult:
    """Self-certifying adaptive expansion using the deconvolved empirical-MGF bound.

    Like :func:`run_adaptive_variance`, but the bias term is the *high-probability*
    ``max - mean`` bound from :func:`canopy.bandits.maxmean.mgf_bound_from_moments` rather
    than the soft ``sigma_within * sqrt(2 log m)`` heuristic. Both the statistical radius
    and the bias bound are now data-driven and valid w.h.p. -- no assumed spread schedule
    and no light-tail heuristic. Per node we keep O(len(lambdas)) running MGF moments.
    """
    if lambdas is None:
        lambdas = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    log_term = math.log(sum(env.branching**lvl for lvl in range(env.depth + 1)) * horizon)
    mu_star = env.best_leaf_value()

    # stats[node] = [count, mean, M2, sum_exp(K,), sum_exp2(K,)]
    frontier: list[Node] = list(env.children(env.root()))
    stats: dict[Node, list] = {
        nd: [0.0, 0.0, 0.0, np.zeros(len(lambdas)), np.zeros(len(lambdas))] for nd in frontier
    }

    def radius(node: Node) -> float:
        count, _, m2, _, _ = stats[node]
        if count < 2:
            return math.inf
        var = m2 / (count - 1)
        return c * math.sqrt(2.0 * var * log_term / count) + log_term / count

    def bias(node: Node) -> float:
        count, mean, _, se, se2 = stats[node]
        if count < n_min:
            return math.inf
        return mgf_bound_from_moments(
            int(count),
            mean,
            se,
            se2,
            lambdas,
            env.leaves_per_node(node.level),
            env.noise_std,
            delta=delta,
        )

    cum = 0.0
    cum_regret = np.empty(horizon)
    for t in range(horizon):
        best, best_u = frontier[0], -math.inf
        for v in frontier:
            count = stats[v][0]
            if count < 2:
                u = math.inf
            else:
                b = bias(v)
                b = 1.0 if math.isinf(b) else b  # optimistic while bias not yet certified
                u = stats[v][1] + radius(v) + b
            if u > best_u:
                best_u, best = u, v
        v = best

        reward = env.play(v, rng)
        count, mean, m2, se, se2 = stats[v]
        count += 1
        d = reward - mean
        mean += d / count
        m2 += d * (reward - mean)
        se = se + np.exp(lambdas * reward)
        se2 = se2 + np.exp(2.0 * lambdas * reward)
        stats[v] = [count, mean, m2, se, se2]

        cum += mu_star - env.true_value(v)
        cum_regret[t] = cum

        if v.level < env.depth and count >= n_min and radius(v) <= bias(v):
            frontier.remove(v)
            for child in env.children(v):
                if warm_start:
                    stats[child] = [
                        1.0,
                        mean,
                        0.0,
                        np.exp(lambdas * mean),
                        np.exp(2.0 * lambdas * mean),
                    ]
                else:
                    stats[child] = [0.0, 0.0, 0.0, np.zeros(len(lambdas)), np.zeros(len(lambdas))]
                frontier.append(child)

    return RegretResult(cum_regret=cum_regret, memory=len(stats), label="adaptive-mgf")


def run_hoo(
    env: TreeBandit,
    horizon: int,
    spread: Callable[[int], float],
    rng: np.random.Generator,
    c: float = 0.5,
    memory_bounded: bool = False,
) -> RegretResult:
    """Provably regret-optimal optimistic algorithm (HOO / HCT style).

    The UCB-style index is

        U(v) = mean_hat(v) + c * sqrt(2 ln t / T(v)) + spread(level(v)),

    where the additive ``spread(level)`` is the bias bonus (the max-minus-mean bound of
    mean-backup at that resolution). Optimism is tightened by a B-value backup

        B(v) = min( U(v), max_{child} B(child) ),

    and each round descends by B-value to a tree-leaf, plays it (commit + random path),
    and updates statistics along the whole path. Under local smoothness near the optimum
    (cell "diameter" spread(h) -> 0), this attains the X-armed-bandit regret
    O~(n^{(d+1)/(d+2)}) with d the near-optimality dimension.

    ``memory_bounded=False`` expands a played tree-leaf after its first visit (HOO;
    memory grows ~O(n)). ``memory_bounded=True`` expands only once a node is bias-limited,
    ``T(v) >= c^2 log / spread(level)^2`` (HCT-style), giving the same regret with memory
    governed by the near-optimality dimension rather than the horizon.
    """
    mu_star = env.best_leaf_value()
    log_term = math.log(sum(env.branching**lvl for lvl in range(env.depth + 1)) * horizon)
    count: dict[Node, float] = {}
    mean: dict[Node, float] = {}
    bval: dict[Node, float] = {}
    expanded: set[Node] = set()
    root = env.root()
    count[root], mean[root], bval[root] = 0.0, 0.0, math.inf

    def n_star(level: int) -> float:
        s = spread(level)
        return math.inf if s <= 1e-9 else (c * c) * log_term / (s * s)

    cum = 0.0
    cum_regret = np.empty(horizon)
    for t in range(1, horizon + 1):
        # descend by B-value to a leaf of the current tree
        path = [root]
        v = root
        while v in expanded:
            v = max(env.children(v), key=lambda ch: bval[ch])
            path.append(v)

        reward = env.play(v, rng)
        for nd in path:
            count[nd] += 1
            mean[nd] += (reward - mean[nd]) / count[nd]

        cum += mu_star - env.true_value(v)
        cum_regret[t - 1] = cum

        threshold = 1.0 if not memory_bounded else n_star(v.level)
        if v.level < env.depth and v not in expanded and count[v] >= threshold:
            for child in env.children(v):
                count[child], mean[child], bval[child] = 0.0, 0.0, math.inf
            expanded.add(v)

        # refresh U and B bottom-up along the path
        for nd in reversed(path):
            n = count[nd]
            u = (
                math.inf
                if n == 0
                else mean[nd] + c * math.sqrt(2.0 * math.log(t + 1) / n) + spread(nd.level)
            )
            if nd in expanded:
                bval[nd] = min(u, max(bval[ch] for ch in env.children(nd)))
            else:
                bval[nd] = u

    label = "hoo-bounded" if memory_bounded else "hoo"
    return RegretResult(cum_regret=cum_regret, memory=len(count), label=label)


def run_hybrid(
    env: TreeBandit,
    horizon: int,
    spread: Callable[[int], float],
    rng: np.random.Generator,
    c: float = 0.5,
    n_min: int = 12,
    jump_factor: float = 1.5,
    warm_start: bool = True,
) -> RegretResult:
    """Lipschitz floor + variance-based jump detection (best of both worlds).

    The bias term is the tight (assumed) Lipschitz ``spread(level)`` in cells that look
    smooth, and is inflated to the data-driven ``sigma_within * sqrt(2 log m)`` only in
    cells where the *measured* within-cell variance exceeds the Lipschitz prediction by a
    factor ``jump_factor`` -- i.e. where a discontinuity is statistically detected. This
    keeps the cheap, low-regret behavior of assumed smoothness on smooth / wide-jump cells
    while staying robust on narrow / hidden jumps, instead of trading one for the other.
    """
    log_term = math.log(sum(env.branching**lvl for lvl in range(env.depth + 1)) * horizon)
    mu_star = env.best_leaf_value()
    noise_var = env.noise_std**2

    frontier: list[Node] = list(env.children(env.root()))
    stats: dict[Node, list[float]] = {nd: [0.0, 0.0, 0.0] for nd in frontier}

    def total_var(node: Node) -> float:
        count, _, m2 = stats[node]
        return m2 / (count - 1) if count >= 2 else float("inf")

    def within_std(node: Node) -> float:
        return math.sqrt(max(0.0, total_var(node) - noise_var))

    def radius(node: Node) -> float:
        count, _, _ = stats[node]
        if count < 2:
            return math.inf
        return c * math.sqrt(2.0 * total_var(node) * log_term / count) + log_term / count

    def bias(node: Node) -> float:
        count, _, _ = stats[node]
        if count < n_min:
            return math.inf
        lip = spread(node.level)
        sw = within_std(node)
        if sw > jump_factor * lip:  # measured variance contradicts Lipschitz -> jump
            m = max(2, env.leaves_per_node(node.level))
            return max(lip, sw * math.sqrt(2.0 * math.log(m)))
        return lip  # looks Lipschitz: use the tight floor

    cum = 0.0
    cum_regret = np.empty(horizon)
    for t in range(horizon):
        best, best_u = frontier[0], -math.inf
        for v in frontier:
            count, mean, _ = stats[v]
            u = math.inf if count < 2 else mean + radius(v) + bias(v)
            if u > best_u:
                best_u, best = u, v
        v = best

        reward = env.play(v, rng)
        count, mean, m2 = stats[v]
        count += 1
        d = reward - mean
        mean += d / count
        m2 += d * (reward - mean)
        stats[v] = [count, mean, m2]

        cum += mu_star - env.true_value(v)
        cum_regret[t] = cum

        if v.level < env.depth and count >= n_min and radius(v) <= bias(v):
            frontier.remove(v)
            for child in env.children(v):
                stats[child] = [1.0, mean, 0.0] if warm_start else [0.0, 0.0, 0.0]
                frontier.append(child)

    return RegretResult(cum_regret=cum_regret, memory=len(stats), label="hybrid")


def run_local_lipschitz(
    env: TreeBandit,
    horizon: int,
    rng: np.random.Generator,
    rho: float | None = None,
    prior_lipschitz: float = 1.0,
    c: float = 0.5,
    n_min: int = 12,
    warm_start: bool = True,
) -> RegretResult:
    """Locally-adaptive Lipschitz expansion (Suman's tighter-localization direction).

    Each node carries its own Lipschitz estimate ``L(v)``, so subtrees that are smoother
    get a tighter (smaller) constant -- hence a smaller bias and cheaper resolution --
    while rough subtrees get a larger one. The estimate is *propagated*: a node inherits
    its parent's ``L`` as a prior and refines it from its own measured within-cell spread
    (``L(v) = sigma_within(v) / rho**level``) once it has ``n_min`` samples. On expansion,
    children inherit the (possibly tight) ``L(v)``, so a smooth region's small constant
    flows downward and the algorithm stops over-exploring it.

    bias(v) = L(v) * rho**level   (the local Lipschitz bound on max - mean within v).
    """
    if rho is None:
        rho = 1.0 / env.branching
    log_term = math.log(sum(env.branching**lvl for lvl in range(env.depth + 1)) * horizon)
    mu_star = env.best_leaf_value()
    noise_var = env.noise_std**2

    # stats[node] = [count, mean, M2, L_est]
    frontier: list[Node] = list(env.children(env.root()))
    stats: dict[Node, list[float]] = {nd: [0.0, 0.0, 0.0, prior_lipschitz] for nd in frontier}

    def within_std(node: Node) -> float:
        count, _, m2, _ = stats[node]
        if count < 2:
            return float("inf")
        return math.sqrt(max(0.0, m2 / (count - 1) - noise_var))

    def local_lipschitz(node: Node) -> float:
        count, _, _, l_prior = stats[node]
        if count < n_min:
            return l_prior  # inherited prior until enough data
        measured = within_std(node) / (rho**node.level)
        return measured

    def radius(node: Node) -> float:
        count, _, m2, _ = stats[node]
        if count < 2:
            return math.inf
        var = m2 / (count - 1)
        return c * math.sqrt(2.0 * var * log_term / count) + log_term / count

    def bias(node: Node) -> float:
        return local_lipschitz(node) * (rho**node.level)

    cum = 0.0
    cum_regret = np.empty(horizon)
    for t in range(horizon):
        best, best_u = frontier[0], -math.inf
        for v in frontier:
            count = stats[v][0]
            u = math.inf if count < 2 else stats[v][1] + radius(v) + bias(v)
            if u > best_u:
                best_u, best = u, v
        v = best

        reward = env.play(v, rng)
        count, mean, m2, l_prior = stats[v]
        count += 1
        d = reward - mean
        mean += d / count
        m2 += d * (reward - mean)
        stats[v] = [count, mean, m2, l_prior]

        cum += mu_star - env.true_value(v)
        cum_regret[t] = cum

        if v.level < env.depth and count >= n_min and radius(v) <= bias(v):
            child_prior = local_lipschitz(v)  # propagate the refined local constant
            frontier.remove(v)
            for child in env.children(v):
                init_mean = mean if warm_start else 0.0
                init_count = 1.0 if warm_start else 0.0
                stats[child] = [init_count, init_mean, 0.0, child_prior]
                frontier.append(child)

    return RegretResult(cum_regret=cum_regret, memory=len(stats), label="local-lipschitz")


@dataclass
class ViolationReport:
    """Data-driven Lipschitz-violation detection at a fixed resolution.

    Attributes:
        level: the resolution at which cells were tested.
        detected: indices of level-``level`` cells flagged as violations (their estimated
            within-cell spread exceeds the Lipschitz floor).
        within_std: the estimated within-cell spread for every level-``level`` cell.
        floor: the Lipschitz floor ``spread(level)`` the spreads were compared against.
    """

    level: int
    detected: list[int]
    within_std: np.ndarray
    floor: float

    @property
    def count(self) -> int:
        return len(self.detected)


def detect_violations(
    env: TreeBandit,
    level: int,
    spread: Callable[[int], float],
    rng: np.random.Generator,
    n_samples_per_cell: int = 200,
    jump_factor: float = 1.5,
) -> ViolationReport:
    """Estimate the number of Lipschitz violations from data (no assumed constants).

    For each level-``level`` cell we draw ``n_samples_per_cell`` random-path plays, estimate
    the within-cell spread by deconvolving the known observation noise
    (``sqrt(max(0, Var_hat - noise^2))``), and flag the cell as a violation when that spread
    exceeds ``jump_factor`` times the Lipschitz floor ``spread(level)``. In a genuinely
    Lipschitz cell the within-cell spread is ``<= spread(level)``; a cell straddling a jump
    has anomalously large spread, so the count of flagged cells is a consistent, data-driven
    estimate of the number of violations ``K`` -- the quantity the regret bound is
    parameterized by. The algorithm never needs ``K`` (or the Lipschitz constant) a priori.
    """
    if not 0 <= level <= env.depth:
        raise ValueError("level must be in [0, depth]")
    noise_var = env.noise_std**2
    floor = float(spread(level))
    n_cells = env.branching**level
    within = np.empty(n_cells)
    detected: list[int] = []
    for idx in range(n_cells):
        node = Node(level, idx)
        samples = np.array([env.play(node, rng) for _ in range(n_samples_per_cell)])
        var_hat = float(samples.var(ddof=1)) if n_samples_per_cell >= 2 else 0.0
        within[idx] = math.sqrt(max(0.0, var_hat - noise_var))
        if within[idx] > jump_factor * floor:
            detected.append(idx)
    return ViolationReport(level=level, detected=detected, within_std=within, floor=floor)


@dataclass
class MultiscaleEdgeMap:
    """Multiscale (scale-adaptive) edge map of a tree function, estimated from data.

    A single resolution forces a bad tradeoff: a coarse level detects an edge but localizes
    it only to a large cell, while a fine level localizes tightly but misses wide / diluted
    features (and probes every ``b**level`` cell). This map estimates the within-cell spread
    at several levels with a *per-level, data-driven* floor (a multiple of that level's median
    spread -- no assumed Lipschitz constant), so it catches violations at whatever scale they
    live and localizes each at the finest level that still detects it.

    Attributes:
        levels: the resolutions probed.
        within_std: ``level -> array`` of deconvolved within-cell spreads.
        floor: ``level -> float`` detection floor (``factor * median`` spread at that level).
        leaf_score: per-leaf max-over-scales anomaly ratio ``within_std / floor`` -- the
            edge score used to drive targeted sampling.
        flagged: ``(level, index)`` cells whose spread exceeds the level floor.
    """

    levels: list[int]
    within_std: dict
    floor: dict
    leaf_score: np.ndarray
    flagged: list

    def finest_ranges(self) -> list[tuple[int, int]]:
        """Leaf ranges of the flagged cells, keeping only the finest (smallest) per region.

        Suitable to pass as ``relaxed_ranges`` to :class:`~canopy.bandits.topk.HierarchicalTopK`
        so expensive evaluations concentrate on the localized edges.
        """
        n_leaves = self.leaf_score.size
        # finest (smallest-cell) flags first, so coarser overlapping flags are dropped
        kept: list[tuple[int, int]] = []
        for lvl, idx in sorted(self.flagged, key=lambda li: li[0], reverse=True):
            span = n_leaves // len(self.within_std[lvl])
            start, end = idx * span, idx * span + span
            if not any(s <= start and end <= e for s, e in kept):
                kept.append((start, end))
        return kept


def multiscale_edge_map(
    env: TreeBandit,
    rng: np.random.Generator,
    levels: list[int] | None = None,
    n_samples_per_cell: int = 40,
    factor: float = 2.0,
) -> MultiscaleEdgeMap:
    """Estimate a scale-adaptive edge map from random-path probes (no assumed constants).

    For each level in ``levels`` (default ``1..depth-1``), estimate every cell's within-cell
    spread by deconvolving the known observation noise, set a data-driven floor at
    ``factor`` times that level's median spread, and flag the outlier cells. The per-leaf
    ``leaf_score`` is the maximum, over levels, of the normalized anomaly
    ``within_std / floor`` -- large wherever the function has a sharp edge at *any* scale.
    """
    if levels is None:
        levels = list(range(1, env.depth))
    within_std: dict[int, np.ndarray] = {}
    floor: dict[int, float] = {}
    flagged: list[tuple[int, int]] = []
    noise_var = env.noise_std**2
    # the within-std estimate of a truly-smooth cell still fluctuates at this scale, so the
    # floor is never allowed below it (keeps the per-level floor robust when most cells are
    # smooth and clamp to ~0, where a median-based floor would collapse to 0).
    min_floor = env.noise_std / math.sqrt(2.0 * max(1, n_samples_per_cell))
    n_leaves = env.n_leaves
    leaf_score = np.zeros(n_leaves)

    for level in levels:
        n_cells = env.branching**level
        ws = np.empty(n_cells)
        for c in range(n_cells):
            node = Node(level, c)
            samples = np.array([env.play(node, rng) for _ in range(n_samples_per_cell)])
            var_hat = float(samples.var(ddof=1)) if n_samples_per_cell >= 2 else 0.0
            ws[c] = math.sqrt(max(0.0, var_hat - noise_var))
        within_std[level] = ws
        # robust per-level floor: a multiple of the 75th-percentile (smooth-background) spread
        lvl_floor = factor * max(float(np.quantile(ws, 0.75)), min_floor)
        floor[level] = lvl_floor
        span = env.branching ** (env.depth - level)
        norm = ws / lvl_floor
        leaf_score = np.maximum(leaf_score, np.repeat(norm, span))
        flagged.extend((level, c) for c in range(n_cells) if ws[c] > lvl_floor)

    return MultiscaleEdgeMap(
        levels=list(levels),
        within_std=within_std,
        floor=floor,
        leaf_score=leaf_score,
        flagged=flagged,
    )
