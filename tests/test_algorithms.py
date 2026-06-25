"""Tests for OGD and strongly-convex OGD (Chapters 2-3)."""

from __future__ import annotations

import numpy as np
import pytest

from canopy.algorithms import OnlineGradientDescent, StronglyConvexOGD


def test_ogd_first_step():
    ogd = OnlineGradientDescent(dim=2, learning_rate=0.5)
    x = ogd.update(np.array([2.0, -4.0]))
    # x_1 = 0 - 0.5 * g
    assert x == pytest.approx([-1.0, 2.0])
    assert ogd.t == 1


def test_ogd_projection_applied():
    box = lambda v: np.clip(v, -1.0, 1.0)  # noqa: E731
    ogd = OnlineGradientDescent(dim=2, learning_rate=10.0, projection=box)
    x = ogd.update(np.array([1.0, -1.0]))
    assert np.all(x <= 1.0) and np.all(x >= -1.0)


def test_strongly_convex_ogd_converges_to_optimum():
    # f_t(x) = 0.5 ||x - theta||^2 is 1-strongly convex; gradient at x is (x - theta).
    theta = np.array([1.0, -2.0, 0.5])
    sc = StronglyConvexOGD(dim=3, strong_convexity=1.0)
    for _ in range(2000):
        sc.update(sc.predict() - theta)
    assert sc.predict() == pytest.approx(theta, abs=1e-2)


def test_strongly_convex_requires_positive_alpha():
    with pytest.raises(ValueError):
        StronglyConvexOGD(dim=2, strong_convexity=0.0)
