"""Starter online convex optimization algorithms.

This module provides a minimal Online Gradient Descent (OGD) implementation to get
experiments going. Extend it with Follow-the-Regularized-Leader, online Newton step,
mirror descent, etc.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]


class OnlineGradientDescent:
    """Online Gradient Descent over a (optionally projected) convex domain.

    At each round t the learner plays ``x_t``, observes a convex loss ``f_t``, and
    updates using the (sub)gradient ``g_t = grad f_t(x_t)``:

        x_{t+1} = Proj( x_t - eta_t * g_t )

    Args:
        dim: Dimensionality of the decision vector.
        learning_rate: Either a constant step size or a callable ``t -> eta_t``
            (``t`` is 1-indexed). Defaults to the standard ``1 / sqrt(t)`` schedule.
        projection: Optional projection onto the feasible convex set. Defaults to
            identity (unconstrained).
    """

    def __init__(
        self,
        dim: int,
        learning_rate: float | Callable[[int], float] | None = None,
        projection: Callable[[Vector], Vector] | None = None,
    ) -> None:
        self.dim = dim
        self.x: Vector = np.zeros(dim, dtype=np.float64)
        self.t = 0
        self._projection = projection or (lambda v: v)

        if learning_rate is None:
            self._lr: Callable[[int], float] = lambda t: 1.0 / np.sqrt(t)
        elif callable(learning_rate):
            self._lr = learning_rate
        else:
            rate = float(learning_rate)
            self._lr = lambda _t: rate

    def predict(self) -> Vector:
        """Return the current decision vector ``x_t``."""
        return self.x.copy()

    def update(self, gradient: Vector) -> Vector:
        """Apply one OGD step given the loss (sub)gradient at the current point."""
        self.t += 1
        eta = self._lr(self.t)
        self.x = self._projection(self.x - eta * np.asarray(gradient, dtype=np.float64))
        return self.predict()


class StronglyConvexOGD(OnlineGradientDescent):
    """OGD for strongly convex losses with logarithmic regret (Hazan, Section 3.3.1).

    When every loss ``f_t`` is ``alpha``-strongly convex, using the step size
    ``eta_t = 1 / (alpha * t)`` yields ``O(log T)`` regret instead of the
    ``O(sqrt(T))`` of vanilla OGD.

    Args:
        dim: Dimensionality of the decision vector.
        strong_convexity: The strong convexity modulus ``alpha > 0``.
        projection: Optional projection onto the feasible convex set.
    """

    def __init__(
        self,
        dim: int,
        strong_convexity: float,
        projection: Callable[[Vector], Vector] | None = None,
    ) -> None:
        if strong_convexity <= 0:
            raise ValueError("strong_convexity (alpha) must be > 0")
        self.strong_convexity = float(strong_convexity)
        super().__init__(
            dim,
            learning_rate=lambda t: 1.0 / (self.strong_convexity * t),
            projection=projection,
        )
