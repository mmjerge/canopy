"""Prediction from expert advice (Hazan, Chapter 1).

Implements the Hedge algorithm (Algorithm 1) and the closely related Randomized
Weighted Majority update. Both maintain a distribution over ``N`` experts and apply
multiplicative weight updates against observed per-expert losses.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]


class Hedge:
    """Hedge / multiplicative weights (Hazan, Algorithm 1, Theorem 1.5).

    Maintains weights ``W_t(i)`` over ``N`` experts. At each round the learner plays
    the normalized distribution ``x_t(i) = W_t(i) / sum_j W_t(j)``, observes the full
    loss vector ``l_t in [0, 1]^N``, and updates ``W_{t+1}(i) = W_t(i) * exp(-eps * l_t(i))``.

    With ``eps = sqrt(log(N) / T)`` and non-negative losses, the expected regret is
    ``O(sqrt(T log N))``.

    Args:
        n_experts: Number of experts ``N``.
        learning_rate: The step size ``eps > 0``.
    """

    def __init__(self, n_experts: int, learning_rate: float) -> None:
        if n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        self.n_experts = n_experts
        self.learning_rate = float(learning_rate)
        self.weights: Vector = np.ones(n_experts, dtype=np.float64)
        self.t = 0

    def distribution(self) -> Vector:
        """Return the current play distribution ``x_t`` over experts."""
        return self.weights / self.weights.sum()

    def expected_loss(self, losses: Vector) -> float:
        """Expected loss ``x_t^T l_t`` the learner incurs this round."""
        return float(self.distribution() @ np.asarray(losses, dtype=np.float64))

    def update(self, losses: Vector) -> Vector:
        """Apply the multiplicative-weights update for one round.

        Args:
            losses: Full-information loss vector ``l_t`` of length ``N`` (non-negative).

        Returns:
            The next-round distribution ``x_{t+1}``.
        """
        loss_vec = np.asarray(losses, dtype=np.float64)
        if loss_vec.shape != (self.n_experts,):
            raise ValueError(f"losses must have shape ({self.n_experts},)")
        if np.any(loss_vec < 0):
            raise ValueError("Hedge requires non-negative losses")
        self.t += 1
        self.weights *= np.exp(-self.learning_rate * loss_vec)
        return self.distribution()


class RandomizedWeightedMajority(Hedge):
    """Randomized Weighted Majority (Hazan, Section 1.3.2).

    Algorithmically identical multiplicative-weights update to :class:`Hedge`; exposed
    separately for readability and to provide explicit expert sampling.
    """

    def sample(self, rng: np.random.Generator) -> int:
        """Sample an expert index ``i_t ~ x_t`` from the current distribution."""
        return int(rng.choice(self.n_experts, p=self.distribution()))
