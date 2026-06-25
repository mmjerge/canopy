"""Tests for the hierarchical Gaussian reward model and spread bound."""

from __future__ import annotations

import numpy as np
import pytest

from canopy.bandits import (
    TreeBandit,
    adversarial_spike_leaf_means,
    geometric_sigma,
    hierarchical_gaussian_leaf_means,
    hierarchical_spread,
)
from canopy.bandits.tree import Node


def test_leaf_means_shape():
    means = hierarchical_gaussian_leaf_means(
        branching=3, depth=3, sigma=geometric_sigma(), rng=np.random.default_rng(0)
    )
    assert means.shape == (27,)


def test_overall_mean_equals_root_value():
    # By the zero-sum construction, the mean of all leaves equals the root value.
    means = hierarchical_gaussian_leaf_means(
        branching=4,
        depth=4,
        sigma=geometric_sigma(base=0.6, decay=0.7),
        root_value=0.5,
        rng=np.random.default_rng(2),
    )
    assert means.mean() == pytest.approx(0.5)


def test_geometric_sigma_decays():
    sigma = geometric_sigma(base=0.4, decay=0.5)
    assert sigma(0) == pytest.approx(0.4)
    assert sigma(1) == pytest.approx(0.2)
    assert sigma(2) == pytest.approx(0.1)


def test_spread_is_positive_and_decreasing():
    sigma = geometric_sigma(base=0.5, decay=0.6)
    depth, branching = 5, 3
    spread = hierarchical_spread(sigma, depth, branching, z=3.0)
    vals = [spread(level) for level in range(depth)]
    assert all(v > 0 for v in vals)
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))


def test_spread_bound_holds_empirically():
    # The (high-probability) spread bound should cover the actual max leaf deviation
    # from each ancestor's value across many random instances.
    branching, depth = 3, 4
    sigma = geometric_sigma(base=0.5, decay=0.6)
    spread = hierarchical_spread(sigma, depth, branching, z=3.0)
    rng = np.random.default_rng(7)
    violations = 0
    trials = 200
    for _ in range(trials):
        env = TreeBandit.from_hierarchical_gaussian(branching, depth, sigma=sigma, rng=rng)
        for level in range(depth):
            for idx in range(branching**level):
                node = Node(level, idx)
                start, end = env.leaf_range(node)
                dev = np.max(np.abs(env._leaf_means[start:end] - env.true_value(node)))
                if dev > spread(level):
                    violations += 1
    # z=3 should keep violations rare.
    assert violations / (trials * sum(branching**level for level in range(depth))) < 0.02


def test_adversarial_spikes_structure():
    means = adversarial_spike_leaf_means(
        branching=4,
        depth=3,
        n_spikes=3,
        low=0.2,
        high=0.95,
        jitter=0.0,
        rng=np.random.default_rng(0),
    )
    assert means.shape == (64,)
    assert np.sum(means > 0.5) == 3  # exactly the spikes are high
    assert np.allclose(means[means < 0.5], 0.2)
