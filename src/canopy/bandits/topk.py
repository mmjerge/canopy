"""Top-k leaf identification on a :class:`~canopy.bandits.tree.TreeBandit`.

Both strategies operate under a fixed COST ``budget`` and return the empirical top-k.
Leaf evaluations cost ``env.leaf_cost`` (expensive); internal probes cost
``env.probe_cost`` (cheap). The structure-blind baselines can only evaluate leaves, so
they pay full cost per observation; :class:`HierarchicalTopK` can spend many cheap
probes to localize promising regions before paying for expensive leaf evaluations.

* :class:`UniformTopK` -- splits the cost budget evenly across all leaves.

* :class:`HierarchicalTopK` -- descends best-first using cheap internal probes to prune
  whole regions that provably cannot contain a top-k leaf, then spends the remaining
  budget on expensive leaf evaluations to separate the survivors.

Confidence intervals use a sub-Gaussian (Hoeffding-style) radius with a union bound
over all nodes. The subtree pruning is sound whenever the supplied ``spread`` bound is
valid (see :class:`HierarchicalTopK`).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from canopy.bandits.tree import Node, TreeBandit


@dataclass
class TopKResult:
    """Outcome of a top-k identification run."""

    leaves: list[int]  # identified leaf indices, best estimated reward first
    estimates: dict[int, float]  # leaf index -> empirical mean
    n_pulls: int  # total observations taken
    certified: bool  # True if the top-k was separated with the target confidence
    cost: float = 0.0  # total cost consumed
    recovered: list[int] = field(default_factory=list)  # filled by evaluate()

    def evaluate(self, env: TreeBandit, k: int) -> float:
        """Fraction of the true top-k leaves that were recovered (in [0, 1])."""
        truth = set(env.top_k_leaves(k))
        self.recovered = sorted(truth & set(self.leaves))
        return len(self.recovered) / k


class _Stats:
    """Running mean / count for a single node."""

    __slots__ = ("n", "mean")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        self.mean += (x - self.mean) / self.n


class UniformTopK:
    """Baseline: split the cost ``budget`` evenly across all leaves, return top-k."""

    def __init__(self, budget: float) -> None:
        if budget <= 0:
            raise ValueError("budget must be > 0")
        self.budget = float(budget)

    def run(self, env: TreeBandit, k: int) -> TopKResult:
        # Spend (at most) `budget` cost on leaf evaluations. When the affordable number
        # of leaf samples is < n_leaves, only a subset of leaves is sampled at all; the
        # rest stay unknown (-inf) and cannot be selected.
        n = env.n_leaves
        affordable = int(self.budget // env.leaf_cost)
        per_leaf, remainder = divmod(affordable, n)
        means: dict[int, float] = {}
        for i, leaf in enumerate(env.leaf_nodes()):
            pulls = per_leaf + (1 if i < remainder else 0)
            if pulls == 0:
                means[leaf.index] = float("-inf")  # never sampled
                continue
            st = _Stats()
            for _ in range(pulls):
                st.add(env.sample(leaf))
            means[leaf.index] = st.mean
        order = sorted(means, key=means.get, reverse=True)  # type: ignore[arg-type]
        return TopKResult(
            leaves=order[:k],
            estimates=means,
            n_pulls=env.n_pulls,
            certified=False,
            cost=env.total_cost,
        )


class HierarchicalTopK:
    """Structure-exploiting top-k leaf identification under a try budget.

    Bounds for a node ``v`` at level ``l`` with empirical value and confidence radius
    ``r(v)``:

    * Since ``max(leaves) >= mean(leaves) == value(v)``, a *lower* bound on the best
      leaf under ``v`` is ``LB(v) = mean_hat(v) - r(v)``.
    * Assuming any leaf deviates from its level-``l`` ancestor's average by at most
      ``spread(l)``, an *upper* bound on the best leaf under ``v`` is
      ``UB(v) = mean_hat(v) + r(v) + spread(l)``.

    A subtree ``v`` is discarded when at least ``k`` other (disjoint) surviving subtrees
    ``u`` each satisfy ``LB(u) > UB(v)``: those guarantee ``k`` distinct leaves better
    than anything under ``v``. The pruning is sound whenever ``spread`` is a valid
    bound; with the conservative default ``spread = 1`` no internal pruning happens and
    the method reduces to a leaf-level race.

    Args:
        budget: Total COST available (leaf evals cost ``env.leaf_cost``, internal
            probes cost ``env.probe_cost``).
        confidence: Target error probability ``delta`` for the union bound / early stop.
        spread: ``level -> max leaf deviation from that level's subtree average``.
            Defaults to ``1.0`` everywhere (no smoothness assumption).
        descent_fraction: Fraction of the cost budget reserved for the coarse descent.
        samples_per_round: Observations per active node in each round.
        beam_width: If set, after the (sound) pruning at each descent level keep only
            the ``beam_width`` most promising nodes (highest optimistic value). This
            best-first focusing lets cheap probes concentrate on good regions and keeps
            the candidate-leaf set small, at the cost of the worst-case soundness
            guarantee. ``None`` (default) keeps every survivor (sound but can explode
            the frontier when probes are cheap).
    """

    def __init__(
        self,
        budget: float,
        confidence: float = 0.05,
        spread: Callable[[int], float] | float | None = None,
        descent_fraction: float = 0.4,
        samples_per_round: int = 4,
        beam_width: int | None = None,
        relaxed_ranges: list[tuple[int, int]] | None = None,
    ) -> None:
        if budget <= 0:
            raise ValueError("budget must be > 0")
        if not 0 < confidence < 1:
            raise ValueError("confidence must be in (0, 1)")
        if not 0 < descent_fraction < 1:
            raise ValueError("descent_fraction must be in (0, 1)")
        if samples_per_round < 1:
            raise ValueError("samples_per_round must be >= 1")
        if beam_width is not None and beam_width < 1:
            raise ValueError("beam_width must be >= 1 or None")
        self.budget = float(budget)
        self.confidence = confidence
        self.descent_fraction = descent_fraction
        self.samples_per_round = samples_per_round
        self.beam_width = beam_width
        self._relaxed_ranges = list(relaxed_ranges) if relaxed_ranges else []
        if spread is None:
            self._spread: Callable[[int], float] = lambda _l: 1.0
        elif callable(spread):
            self._spread = spread
        else:
            value = float(spread)
            self._spread = lambda _l: value

    def run(self, env: TreeBandit, k: int) -> TopKResult:
        if not 1 <= k <= env.n_leaves:
            raise ValueError("k must satisfy 1 <= k <= n_leaves")
        n_nodes = sum(env.branching**level for level in range(env.depth + 1))
        log_term = math.log(2.0 * n_nodes / self.confidence)
        stats: dict[Node, _Stats] = {}

        def radius(node: Node) -> float:
            st = stats.get(node)
            if st is None or st.n == 0:
                return math.inf
            return env.noise_for(node) * math.sqrt(2.0 * log_term / st.n)

        def mean_of(node: Node) -> float:
            st = stats.get(node)
            return st.mean if st is not None else 0.0

        def sample(node: Node, times: int) -> None:
            st = stats.setdefault(node, _Stats())
            for _ in range(times):
                st.add(env.sample(node))

        def prune(nodes: list[Node], spread_of: Callable[[Node], float]) -> list[Node]:
            lb = {nd: mean_of(nd) - radius(nd) for nd in nodes}
            ub = {nd: mean_of(nd) + radius(nd) + spread_of(nd) for nd in nodes}
            sorted_lb = sorted(lb.values(), reverse=True)
            survivors = []
            for nd in nodes:
                # Count distinct other subtrees provably better than nd's best leaf.
                better = sum(1 for v in sorted_lb if v > ub[nd])
                if lb[nd] > ub[nd]:  # don't count nd against itself
                    better -= 1
                if better < k:
                    survivors.append(nd)
            return survivors

        # --- phase 1: coarse descent, pruning hopeless subtrees via cheap probes ---
        def spread_for(node: Node) -> float:
            # Relax the smooth bound (no structural pruning) on detected-violation cells:
            # where a jump was found, the Lipschitz max-mean bound is invalid, so we fall
            # back to leaf-level certification there instead of (wrongly) pruning.
            if self._relaxed_ranges:
                s, e = env.leaf_range(node)
                if any(s < re and rs < e for rs, re in self._relaxed_ranges):
                    return 1.0
            return self._spread(node.level)

        descent_cap = self.budget * self.descent_fraction
        frontier = env.children(env.root())
        while frontier and not env.is_leaf(frontier[0]):
            for nd in frontier:
                if env.total_cost >= descent_cap:
                    break
                sample(nd, self.samples_per_round)
            frontier = prune(frontier, spread_for)
            if self.beam_width is not None and len(frontier) > self.beam_width:
                # best-first focusing: keep the most promising nodes only
                frontier = sorted(
                    frontier,
                    key=lambda nd: mean_of(nd) + radius(nd) + spread_for(nd),
                    reverse=True,
                )[: self.beam_width]
            if env.total_cost >= descent_cap:
                break
            frontier = [child for nd in frontier for child in env.children(nd)]

        candidates = [nd for nd in frontier if env.is_leaf(nd)] or env.leaf_nodes()

        # --- phase 2: spend the rest of the budget on expensive leaf evaluations ---
        active = list(candidates)
        for nd in active:
            if nd not in stats and env.total_cost < self.budget:
                sample(nd, self.samples_per_round)
        certified = False
        while env.total_cost < self.budget:
            survivors = prune(active, lambda _nd: 0.0)
            if len(survivors) <= k:
                certified = len(survivors) == k
                if survivors:
                    active = survivors
                break
            active = survivors
            for nd in active:
                if env.total_cost >= self.budget:
                    break
                sample(nd, self.samples_per_round)

        ranked = sorted(active, key=lambda nd: mean_of(nd), reverse=True)
        chosen = ranked[:k]
        estimates = {nd.index: mean_of(nd) for nd in active}
        return TopKResult(
            leaves=[nd.index for nd in chosen],
            estimates=estimates,
            n_pulls=env.n_pulls,
            certified=certified and len(chosen) == k,
            cost=env.total_cost,
        )
