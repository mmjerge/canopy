"""Reasoning-tree search: value-guided (edge-following) descent vs. best-of-N.

This is the bridge from the theory to a concrete LLM use: test-time-compute search over a
reasoning tree. A leaf is a complete reasoning trace; an internal node is a partial trace; a
leaf's reward is the fraction of *decision steps* it gets right, and a fully correct answer
(reward $1$) requires getting all ``n_decisions`` of them right. The value of an internal
node (its subtree-average reward) is the process/value-model (PRM) signal, and it is smooth
except at the decision steps, where the value sharply differs between children -- the edges.

Two test-time strategies at a fixed oracle budget (number of cheap rollouts / evaluations):

* ``best_of_n`` -- structure-blind: sample whole traces and return the best-scoring one. To
  hit a fully correct trace it must *sample* one, which needs $\\sim b^{n\\_decisions}$ draws.
* ``value_guided_search`` -- edge-following: descend the tree, at each step probing the
  children with cheap rollouts and following the higher-value child (the value edge at a
  decision step). It resolves the ``n_decisions`` decisions one at a time, so its cost is
  polynomial in the number of decisions.

The separation is exponential in the number of decision steps: value-guided search finds the
correct trace at a budget where best-of-N cannot. This is a *search / compute-allocation*
result -- it requires that a correct trace be reachable and that the value signal be
informative; it does not add capability the model lacks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def reasoning_tree_rewards(
    depth: int,
    n_decisions: int | None = None,
    branching: int = 2,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Leaf rewards for a synthetic reasoning tree.

    ``n_decisions`` (default ``depth``) levels are *pivotal*: at each there is a single
    correct token. A leaf's reward is the fraction of pivotal levels at which its path took
    the correct token, so reward ``1`` means every decision was correct. Non-pivotal levels
    do not affect the reward (smooth steps). Returns ``(reward, decision_levels)``.
    """
    rng = rng or np.random.default_rng()
    if n_decisions is None:
        n_decisions = depth
    if not 1 <= n_decisions <= depth:
        raise ValueError("n_decisions must satisfy 1 <= n_decisions <= depth")
    n = branching**depth
    decision_levels = np.sort(rng.choice(depth, size=n_decisions, replace=False))
    correct = rng.integers(0, branching, size=n_decisions)
    leaves = np.arange(n)
    hits = np.zeros(n)
    for k, d in enumerate(decision_levels):
        token_d = (leaves // branching ** (depth - 1 - d)) % branching
        hits += token_d == correct[k]
    return hits / n_decisions, decision_levels.astype(np.int64)


def _subtree_range(level: int, index: int, depth: int, branching: int) -> tuple[int, int]:
    span = branching ** (depth - level)
    start = index * span
    return start, start + span


def best_of_n(
    reward: NDArray[np.float64], n_samples: int, sigma: float, rng: np.random.Generator
) -> tuple[bool, float]:
    """Sample ``n_samples`` whole traces, return the best by noisy score.

    Returns ``(success, reward_of_pick)`` where ``success`` is whether the returned trace is
    fully correct (reward $1$). Cost is ``n_samples`` evaluations.
    """
    n = reward.size
    idx = rng.integers(0, n, size=n_samples)
    obs = reward[idx] + rng.normal(0.0, sigma, size=n_samples)
    pick = int(idx[np.argmax(obs)])
    return bool(np.isclose(reward[pick], 1.0)), float(reward[pick])


def value_guided_search(
    reward: NDArray[np.float64],
    depth: int,
    probes_per_child: int,
    sigma: float,
    rng: np.random.Generator,
    branching: int = 2,
) -> tuple[bool, float]:
    """Edge-following descent: at each step follow the higher-value child.

    At each level, probe every child with ``probes_per_child`` cheap rollouts (a rollout =
    the noisy reward of a uniformly random leaf in the child's subtree, the multi-fidelity
    probe) and descend into the highest mean. Returns ``(success, reward_of_leaf)``. Cost is
    ``depth * branching * probes_per_child`` probes.
    """
    node = 0
    for level in range(depth):
        best_child, best_val = node * branching, -np.inf
        for j in range(branching):
            child = node * branching + j
            start, end = _subtree_range(level + 1, child, depth, branching)
            rollouts = reward[rng.integers(start, end, size=probes_per_child)]
            val = float((rollouts + rng.normal(0.0, sigma, size=probes_per_child)).mean())
            if val > best_val:
                best_val, best_child = val, child
        node = best_child
    return bool(np.isclose(reward[node], 1.0)), float(reward[node])


def success_rate(
    method: str,
    depth: int,
    n_decisions: int,
    budget: int,
    sigma: float,
    seeds: int = 200,
    branching: int = 2,
) -> float:
    """Fraction of instances on which ``method`` returns a fully-correct trace at ``budget``.

    ``method`` is ``"best_of_n"`` or ``"value_guided"``; the budget is the number of oracle
    calls (trace evaluations for best-of-N; rollout probes for value-guided).
    """
    out = []
    for s in range(seeds):
        reward, _ = reasoning_tree_rewards(
            depth, n_decisions, branching, rng=np.random.default_rng(s)
        )
        if method == "best_of_n":
            ok, _ = best_of_n(reward, budget, sigma, np.random.default_rng(10_000 + s))
        elif method == "value_guided":
            m = max(1, budget // (depth * branching))
            ok, _ = value_guided_search(
                reward, depth, m, sigma, np.random.default_rng(20_000 + s), branching
            )
        else:
            raise ValueError("method must be 'best_of_n' or 'value_guided'")
        out.append(ok)
    return float(np.mean(out))
