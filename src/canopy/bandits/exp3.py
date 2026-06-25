"""EXP3: adversarial multi-armed bandit (Hazan, Algorithm 21, Chapter 6).

EXP3 runs Hedge on importance-weighted loss estimates. Only the loss of the played
arm is observed (bandit feedback); an unbiased estimate of the full loss vector is
reconstructed via inverse-propensity weighting.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]


class EXP3:
    """EXP3 for adversarial bandits with losses in ``[0, 1]``.

    With ``eps = sqrt(log(n) / (T * n))`` the expected regret is ``O(sqrt(T n log n))``.

    Args:
        n_arms: Number of arms ``n``.
        learning_rate: The step size ``eps > 0``.
    """

    def __init__(self, n_arms: int, learning_rate: float) -> None:
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        self.n_arms = n_arms
        self.learning_rate = float(learning_rate)
        self.weights: Vector = np.ones(n_arms, dtype=np.float64)
        self.t = 0

    def distribution(self) -> Vector:
        """Return the current sampling distribution ``x_t`` over arms."""
        return self.weights / self.weights.sum()

    def select(self, rng: np.random.Generator) -> tuple[int, Vector]:
        """Sample an arm ``i_t ~ x_t``.

        Returns:
            A tuple ``(arm, distribution)`` where ``distribution`` is ``x_t`` (needed
            to build the unbiased loss estimate in :meth:`update`).
        """
        dist = self.distribution()
        arm = int(rng.choice(self.n_arms, p=dist))
        return arm, dist

    def update(self, arm: int, loss: float, distribution: Vector) -> Vector:
        """Update weights from the observed bandit loss of the played arm.

        Args:
            arm: The played arm ``i_t``.
            loss: Observed loss ``l_t(i_t)`` in ``[0, 1]``.
            distribution: The distribution ``x_t`` returned by :meth:`select`.

        Returns:
            The next-round distribution ``x_{t+1}``.
        """
        self.t += 1
        estimate = loss / distribution[arm]  # inverse-propensity unbiased estimate
        self.weights[arm] *= np.exp(-self.learning_rate * estimate)
        return self.distribution()
