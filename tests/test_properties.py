"""Property-based tests for the structural invariants of the tree bandit.

These exercise the mathematical guarantees the algorithms rely on, across many random
trees/shapes, rather than a few hand-picked cases:

  * an internal node's value is exactly the average of its children (and of its leaves);
  * a subtree average never exceeds its best leaf;
  * ground-truth top-k matches a brute-force ranking;
  * leaf-index ranges tile the tree exactly;
  * the hierarchical-Gaussian construction preserves the root value (zero-sum);
  * the max-minus-mean bounds are non-negative.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from canopy.bandits import (
    PrefixTreeRouting,
    geometric_sigma,
    hierarchical_gaussian_leaf_means,
    make_routing_scenario,
)
from canopy.bandits.maxmean import samuelson_bound, subgaussian_bound
from canopy.bandits.tree import Node, TreeBandit

# Keep trees small enough to enumerate every node, but varied in shape.
branchings = st.integers(min_value=2, max_value=4)
depths = st.integers(min_value=1, max_value=4)
seeds = st.integers(min_value=0, max_value=2**32 - 1)

ATOL = 1e-9


def _tree(branching: int, depth: int, seed: int) -> TreeBandit:
    return TreeBandit(branching, depth, rng=np.random.default_rng(seed))


def _all_nodes(env: TreeBandit) -> list[Node]:
    return [
        Node(level, idx) for level in range(env.depth + 1) for idx in range(env.branching**level)
    ]


@settings(max_examples=50, deadline=None)
@given(branchings, depths, seeds)
def test_internal_node_is_average_of_children(branching, depth, seed):
    env = _tree(branching, depth, seed)
    for node in _all_nodes(env):
        if env.is_leaf(node):
            continue
        children = env.children(node)
        child_mean = np.mean([env.true_value(c) for c in children])
        assert math.isclose(env.true_value(node), child_mean, abs_tol=ATOL)


@settings(max_examples=50, deadline=None)
@given(branchings, depths, seeds)
def test_root_is_global_mean_and_leaf_is_its_own_mean(branching, depth, seed):
    env = _tree(branching, depth, seed)
    assert math.isclose(env.true_value(env.root()), float(env._leaf_means.mean()), abs_tol=ATOL)
    for i in range(env.n_leaves):
        assert math.isclose(
            env.true_value(Node(env.depth, i)), float(env._leaf_means[i]), abs_tol=ATOL
        )


@settings(max_examples=50, deadline=None)
@given(branchings, depths, seeds)
def test_subtree_average_never_exceeds_best_leaf(branching, depth, seed):
    env = _tree(branching, depth, seed)
    best = env.best_leaf_value()
    for node in _all_nodes(env):
        start, end = env.leaf_range(node)
        assert env.true_value(node) <= float(env._leaf_means[start:end].max()) + ATOL
        assert env.true_value(node) <= best + ATOL


@settings(max_examples=50, deadline=None)
@given(branchings, depths, seeds, st.integers(min_value=1, max_value=8))
def test_top_k_matches_bruteforce(branching, depth, seed, k):
    env = _tree(branching, depth, seed)
    k = min(k, env.n_leaves)
    chosen = env.top_k_leaves(k)
    assert len(chosen) == k
    # tie-robust: the selected values equal the k largest values overall
    selected = np.sort(env._leaf_means[chosen])
    expected = np.sort(env._leaf_means)[-k:]
    assert np.allclose(selected, expected)
    # returned in non-increasing order of mean
    vals = env._leaf_means[chosen]
    assert np.all(vals[:-1] >= vals[1:])


@settings(max_examples=50, deadline=None)
@given(branchings, depths, seeds)
def test_leaf_ranges_tile_the_tree(branching, depth, seed):
    env = _tree(branching, depth, seed)
    for node in _all_nodes(env):
        if env.is_leaf(node):
            continue
        parent_start, parent_end = env.leaf_range(node)
        cursor = parent_start
        for child in env.children(node):
            cs, ce = env.leaf_range(child)
            assert cs == cursor  # contiguous, no gaps/overlap
            cursor = ce
        assert cursor == parent_end  # children exactly cover the parent


@settings(max_examples=50, deadline=None)
@given(branchings, depths, seeds, st.floats(min_value=-2.0, max_value=2.0))
def test_hierarchical_gaussian_preserves_root_value(branching, depth, seed, root_value):
    means = hierarchical_gaussian_leaf_means(
        branching,
        depth,
        geometric_sigma(),
        root_value=root_value,
        rng=np.random.default_rng(seed),
    )
    assert means.shape == (branching**depth,)
    # zero-sum construction => overall mean equals the root value
    assert math.isclose(float(means.mean()), root_value, abs_tol=1e-6)


@settings(max_examples=50, deadline=None)
@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=2, max_size=64),
    st.integers(min_value=2, max_value=64),
    st.floats(min_value=0.0, max_value=0.5),
)
def test_maxmean_bounds_are_nonnegative(rewards, n_leaves, noise_std):
    arr = np.asarray(rewards, dtype=np.float64)
    for bound in (
        samuelson_bound(arr, n_leaves, noise_std),
        subgaussian_bound(arr, n_leaves, noise_std),
    ):
        assert bound >= 0.0
        assert math.isfinite(bound)


@settings(max_examples=30, deadline=None)
@given(branchings, st.integers(min_value=2, max_value=4), seeds)
def test_routing_region_in_range_and_oracle_is_best(branching, depth, seed):
    rng = np.random.default_rng(seed)
    quality, costs = make_routing_scenario(branching, depth, rng)
    env = PrefixTreeRouting(branching, depth, quality, costs, rng=rng)
    for leaf in range(env.n_leaves):
        for resolution in range(env.depth + 1):
            r = env.region(leaf, resolution)
            assert 0 <= r < env.branching**resolution
        # the per-prompt oracle utility dominates every model's utility
        assert env.oracle_utility[leaf] >= env.utility[:, leaf].max() - 1e-12
