"""Tests for the TreeBandit environment and its structural invariants."""

from __future__ import annotations

import numpy as np
import pytest

from canopy.bandits import TreeBandit
from canopy.bandits.tree import Node


def test_structure_basic():
    env = TreeBandit(branching=3, depth=2, leaf_means=np.arange(9.0) / 9, noise_std=0.0)
    assert env.n_leaves == 9
    root = env.root()
    assert not env.is_leaf(root)
    assert env.is_leaf(Node(2, 0))
    assert len(env.children(root)) == 3
    assert env.children(Node(2, 0)) == []  # leaves have no children


def test_leaf_ranges_partition_the_leaves():
    env = TreeBandit(branching=2, depth=3, leaf_means=np.zeros(8))
    # The leaf ranges of all nodes at a given level tile [0, n_leaves) exactly.
    for level in range(env.depth + 1):
        n_nodes = env.branching**level
        covered = []
        for idx in range(n_nodes):
            start, end = env.leaf_range(Node(level, idx))
            covered.extend(range(start, end))
        assert sorted(covered) == list(range(env.n_leaves))


def test_internal_node_value_is_subtree_leaf_average():
    rng = np.random.default_rng(0)
    leaf_means = rng.uniform(size=27)
    env = TreeBandit(branching=3, depth=3, leaf_means=leaf_means, noise_std=0.0)
    for level in range(env.depth + 1):
        for idx in range(env.branching**level):
            node = Node(level, idx)
            start, end = env.leaf_range(node)
            assert env.true_value(node) == pytest.approx(leaf_means[start:end].mean())


def test_parent_equals_mean_of_children():
    env = TreeBandit.from_hierarchical_gaussian(branching=4, depth=3, rng=np.random.default_rng(1))
    for level in range(env.depth):
        for idx in range(env.branching**level):
            parent = Node(level, idx)
            child_vals = [env.true_value(c) for c in env.children(parent)]
            assert env.true_value(parent) == pytest.approx(np.mean(child_vals))


def test_top_k_leaves_matches_argsort():
    leaf_means = np.array([0.2, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.4])
    env = TreeBandit(branching=2, depth=3, leaf_means=leaf_means)
    assert env.top_k_leaves(3) == [1, 6, 4]


def test_sample_counts_tries_and_is_unbiased():
    env = TreeBandit(
        branching=2,
        depth=2,
        leaf_means=np.array([0.0, 0.0, 0.0, 1.0]),
        noise_std=0.1,
        rng=np.random.default_rng(3),
    )
    leaf = Node(2, 3)
    obs = [env.sample(leaf) for _ in range(5000)]
    assert env.n_pulls == 5000
    assert np.mean(obs) == pytest.approx(1.0, abs=0.02)


def test_cost_accounting_distinguishes_probes_and_leaves():
    env = TreeBandit(
        branching=2,
        depth=2,
        leaf_means=np.zeros(4),
        leaf_cost=1.0,
        probe_cost=0.1,
        rng=np.random.default_rng(0),
    )
    env.sample(env.root())  # internal probe -> 0.1
    env.sample(Node(1, 0))  # internal probe -> 0.1
    env.sample(Node(2, 0))  # leaf eval     -> 1.0
    assert env.n_pulls == 3
    assert env.total_cost == pytest.approx(1.2)


def test_invalid_construction():
    with pytest.raises(ValueError):
        TreeBandit(branching=1, depth=2)
    with pytest.raises(ValueError):
        TreeBandit(branching=2, depth=0)
    with pytest.raises(ValueError):
        TreeBandit(branching=2, depth=2, leaf_means=np.zeros(3))  # wrong length
