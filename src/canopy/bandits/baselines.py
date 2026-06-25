"""Structure-blind top-k baselines for fair comparison against HierarchicalTopK.

:class:`SuccessiveEliminationTopK` is a strong (not naive) baseline: it uses the exact
same Hoeffding confidence intervals and elimination rule as the leaf-certification phase
of :class:`~canopy.bandits.topk.HierarchicalTopK`, but it only ever samples leaves -- it
cannot probe internal-node averages. Comparing the two at equal try budget therefore
isolates the value of the tree descent itself, rather than comparing against an
artificially weak uniform sampler.
"""

from __future__ import annotations

import math

from canopy.bandits.topk import TopKResult, _Stats
from canopy.bandits.tree import TreeBandit


class SuccessiveEliminationTopK:
    """Top-k identification by successive elimination over the leaves (no tree use).

    Round-robins samples over the undecided leaves, shrinking their confidence
    intervals until at most ``k`` remain plausible. A leaf is eliminated once ``k``
    other leaves are provably better (their lower bound exceeds its upper bound).

    Args:
        budget: Total COST available (this method only evaluates leaves, so it spends
            ``env.leaf_cost`` per observation).
        confidence: Target error probability ``delta`` for the union bound.
        samples_per_round: Observations per active leaf each round.
    """

    def __init__(self, budget: float, confidence: float = 0.05, samples_per_round: int = 1) -> None:
        if budget <= 0:
            raise ValueError("budget must be > 0")
        if not 0 < confidence < 1:
            raise ValueError("confidence must be in (0, 1)")
        if samples_per_round < 1:
            raise ValueError("samples_per_round must be >= 1")
        self.budget = float(budget)
        self.confidence = confidence
        self.samples_per_round = samples_per_round

    def run(self, env: TreeBandit, k: int) -> TopKResult:
        leaves = env.leaf_nodes()
        n = len(leaves)
        if not 1 <= k <= n:
            raise ValueError("k must satisfy 1 <= k <= n_leaves")
        log_term = math.log(2.0 * n / self.confidence)
        stats = {leaf: _Stats() for leaf in leaves}

        def radius(leaf) -> float:
            m = stats[leaf].n
            return env.noise_std * math.sqrt(2.0 * log_term / m) if m else math.inf

        active = list(leaves)
        for leaf in active:  # one sample each to start
            if env.total_cost >= self.budget:
                break
            stats[leaf].add(env.sample(leaf))

        certified = False
        while env.total_cost < self.budget and len(active) > k:
            lb = {leaf: stats[leaf].mean - radius(leaf) for leaf in active}
            ub = {leaf: stats[leaf].mean + radius(leaf) for leaf in active}
            sorted_lb = sorted(lb.values(), reverse=True)
            survivors = []
            for leaf in active:
                better = sum(1 for v in sorted_lb if v > ub[leaf])
                if lb[leaf] > ub[leaf]:  # don't count leaf against itself
                    better -= 1
                if better < k:
                    survivors.append(leaf)
            active = survivors if survivors else active
            if len(active) <= k:
                certified = len(active) == k
                break
            for leaf in active:  # sample another round
                if env.total_cost >= self.budget:
                    break
                for _ in range(self.samples_per_round):
                    if env.total_cost >= self.budget:
                        break
                    stats[leaf].add(env.sample(leaf))

        ranked = sorted(active, key=lambda leaf: stats[leaf].mean, reverse=True)
        chosen = ranked[:k]
        return TopKResult(
            leaves=[leaf.index for leaf in chosen],
            estimates={leaf.index: stats[leaf].mean for leaf in active},
            n_pulls=env.n_pulls,
            certified=certified and len(chosen) == k,
            cost=env.total_cost,
        )
