"""Tree-structured bandit environment for top-k leaf identification (multi-fidelity).

Problem (per the project brief, in the multi-fidelity / test-time-compute framing):
    * Nodes are arranged as a complete ``branching``-ary tree of a given ``depth``.
    * Each leaf has a true mean reward.
    * The value of any internal node is the AVERAGE reward of every leaf in its
      subtree (so ``value(parent) = mean(value(children))``).
    * Sampling a node returns a noisy observation of that node's value. Crucially,
      probing an INTERNAL node is a *cheap but biased* signal about the leaves below
      it (its average underestimates its best leaf, with the bias shrinking toward the
      leaves) -- think of a value model / verifier scoring a partial rollout. Probing a
      LEAF is an *expensive but unbiased* full evaluation. This is the multi-fidelity
      structure studied in modern best-arm / best-action identification.
    * Budgets are measured in COST, not raw pulls: a leaf evaluation costs
      ``leaf_cost``; an internal probe costs ``probe_cost`` (typically much smaller).
    * Goal: identify and localize the top-k highest-reward leaves at the lowest cost.

Because leaves are ordered left-to-right, every node owns a CONTIGUOUS range of
leaf indices, which makes subtree values and ground-truth top-k cheap to compute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from oco.bandits.rewards import SigmaSchedule


@dataclass(frozen=True)
class Node:
    """A node in the complete k-ary tree, identified by ``(level, index)``.

    ``level`` 0 is the root; ``level == depth`` are leaves. ``index`` is the
    0-based position within the level (left to right).
    """

    level: int
    index: int


class TreeBandit:
    """A complete ``branching``-ary tree whose internal nodes report subtree averages.

    Args:
        branching: Number of children per internal node (``b >= 2``).
        depth: Number of edges from root to a leaf (``depth >= 1``). There are
            ``branching ** depth`` leaves.
        leaf_means: Optional true leaf means, length ``branching ** depth``. If
            omitted, leaf means are drawn uniformly from ``[0, 1]``.
        noise_std: Std of the Gaussian noise on a leaf (full-evaluation) observation.
        leaf_cost: Cost charged for one leaf evaluation (the expensive fidelity).
        probe_cost: Cost charged for one internal-node probe (the cheap fidelity).
        probe_noise_std: Std of the noise on an internal probe. Defaults to
            ``noise_std``. (The internal probe is biased regardless, since a subtree
            average underestimates its best leaf.)
        rng: Optional NumPy generator for reproducibility.
    """

    def __init__(
        self,
        branching: int,
        depth: int,
        leaf_means: NDArray[np.float64] | None = None,
        noise_std: float = 0.1,
        leaf_cost: float = 1.0,
        probe_cost: float = 0.1,
        probe_noise_std: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if branching < 2:
            raise ValueError("branching must be >= 2")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if leaf_cost <= 0 or probe_cost <= 0:
            raise ValueError("leaf_cost and probe_cost must be > 0")
        self.branching = branching
        self.depth = depth
        self.noise_std = float(noise_std)
        self.probe_noise_std = float(probe_noise_std) if probe_noise_std is not None else float(noise_std)
        self.leaf_cost = float(leaf_cost)
        self.probe_cost = float(probe_cost)
        self.rng = rng or np.random.default_rng()

        self.n_leaves = branching**depth
        if leaf_means is None:
            self._leaf_means = self.rng.uniform(0.0, 1.0, size=self.n_leaves)
        else:
            leaf_means = np.asarray(leaf_means, dtype=np.float64)
            if leaf_means.shape != (self.n_leaves,):
                raise ValueError(f"leaf_means must have shape ({self.n_leaves},)")
            self._leaf_means = leaf_means

        # Prefix sums let us compute any contiguous subtree average in O(1).
        self._prefix = np.concatenate([[0.0], np.cumsum(self._leaf_means)])
        self.n_pulls = 0  # total number of observations taken
        self.total_cost = 0.0  # total cost consumed (the quantity to minimize)

    @classmethod
    def from_hierarchical_gaussian(
        cls,
        branching: int,
        depth: int,
        sigma: "SigmaSchedule | None" = None,
        root_value: float = 0.5,
        noise_std: float = 0.1,
        leaf_cost: float = 1.0,
        probe_cost: float = 0.1,
        probe_noise_std: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> "TreeBandit":
        """Build a tree whose leaf means come from a hierarchical Gaussian diffusion.

        Each internal node's value equals the exact average of its subtree leaves. See
        :mod:`oco.bandits.rewards`. If ``sigma`` is omitted, a geometric schedule
        ``0.4 * 0.5 ** level`` is used.
        """
        from oco.bandits.rewards import geometric_sigma, hierarchical_gaussian_leaf_means

        rng = rng or np.random.default_rng()
        sigma = sigma or geometric_sigma()
        leaf_means = hierarchical_gaussian_leaf_means(
            branching, depth, sigma, root_value=root_value, rng=rng
        )
        return cls(branching, depth, leaf_means=leaf_means, noise_std=noise_std,
                   leaf_cost=leaf_cost, probe_cost=probe_cost,
                   probe_noise_std=probe_noise_std, rng=rng)

    @classmethod
    def from_adversarial_spikes(
        cls,
        branching: int,
        depth: int,
        n_spikes: int,
        noise_std: float = 0.1,
        leaf_cost: float = 1.0,
        probe_cost: float = 0.1,
        probe_noise_std: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> "TreeBandit":
        """Build a tree with a flat baseline and randomly-placed high-reward spikes.

        Subtree averages are nearly uninformative here -- the adversarial case where
        trusting the tree structure can hurt. See :mod:`oco.bandits.rewards`.
        """
        from oco.bandits.rewards import adversarial_spike_leaf_means

        rng = rng or np.random.default_rng()
        leaf_means = adversarial_spike_leaf_means(branching, depth, n_spikes, rng=rng)
        return cls(branching, depth, leaf_means=leaf_means, noise_std=noise_std,
                   leaf_cost=leaf_cost, probe_cost=probe_cost,
                   probe_noise_std=probe_noise_std, rng=rng)

    @classmethod
    def from_piecewise_smooth(
        cls,
        branching: int,
        depth: int,
        n_jumps: int,
        jump_width: int | None = None,
        noise_std: float = 0.1,
        leaf_cost: float = 1.0,
        probe_cost: float = 0.1,
        probe_noise_std: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> "TreeBandit":
        """Build a mostly-smooth tree with a finite number of sharp jump discontinuities.

        Smooth base + ``n_jumps`` localized cliffs (the piecewise-Lipschitz / dispersion
        setting). See :mod:`oco.bandits.rewards`.
        """
        from oco.bandits.rewards import piecewise_smooth_leaf_means

        rng = rng or np.random.default_rng()
        leaf_means = piecewise_smooth_leaf_means(
            branching, depth, n_jumps, jump_width=jump_width, rng=rng
        )
        return cls(branching, depth, leaf_means=leaf_means, noise_std=noise_std,
                   leaf_cost=leaf_cost, probe_cost=probe_cost,
                   probe_noise_std=probe_noise_std, rng=rng)

    # --- structure helpers -------------------------------------------------

    def is_leaf(self, node: Node) -> bool:
        return node.level == self.depth

    def leaves_per_node(self, level: int) -> int:
        """How many leaves sit under any node at ``level``."""
        return self.branching ** (self.depth - level)

    def leaf_range(self, node: Node) -> tuple[int, int]:
        """Half-open ``[start, end)`` range of leaf indices under ``node``."""
        span = self.leaves_per_node(node.level)
        start = node.index * span
        return start, start + span

    def children(self, node: Node) -> list[Node]:
        """Return the children of an internal node (empty for a leaf)."""
        if self.is_leaf(node):
            return []
        base = node.index * self.branching
        return [Node(node.level + 1, base + j) for j in range(self.branching)]

    def root(self) -> Node:
        return Node(0, 0)

    def leaf_nodes(self) -> list[Node]:
        return [Node(self.depth, i) for i in range(self.n_leaves)]

    # --- ground truth (for evaluation only) --------------------------------

    def true_value(self, node: Node) -> float:
        """True value of a node = mean of leaf means in its subtree."""
        start, end = self.leaf_range(node)
        return float((self._prefix[end] - self._prefix[start]) / (end - start))

    def top_k_leaves(self, k: int) -> list[int]:
        """Ground-truth indices of the ``k`` highest-reward leaves (descending)."""
        order = np.argsort(-self._leaf_means, kind="stable")
        return order[:k].tolist()

    def best_leaf_value(self) -> float:
        """The highest leaf mean (the per-round benchmark for regret)."""
        return float(self._leaf_means.max())

    # --- interaction -------------------------------------------------------

    def play(self, node: Node, rng: np.random.Generator | None = None) -> float:
        """Online action: commit to ``node`` and follow a uniformly random path down.

        Returns the realized reward of a uniformly random leaf under ``node`` plus
        observation noise. Its expectation is the subtree average ``true_value(node)``,
        but its variance also reflects the within-subtree heterogeneity of leaf means --
        so coarse (shallow) nodes are both more biased and noisier. Used by the online /
        regret-minimization strategies.
        """
        rng = rng or self.rng
        start, end = self.leaf_range(node)
        leaf_idx = int(rng.integers(start, end))
        self.n_pulls += 1
        return float(self._leaf_means[leaf_idx]) + rng.normal(0.0, self.noise_std)

    def cost(self, node: Node) -> float:
        """Cost of one observation of ``node``: cheap for probes, expensive for leaves."""
        return self.leaf_cost if self.is_leaf(node) else self.probe_cost

    def noise_for(self, node: Node) -> float:
        """Observation-noise std for ``node`` (leaf vs internal probe)."""
        return self.noise_std if self.is_leaf(node) else self.probe_noise_std

    def sample(self, node: Node) -> float:
        """Observe ``node`` once and charge its cost.

        A leaf returns an unbiased, expensive observation of its mean. An internal
        probe returns a cheap observation of its subtree average -- a biased proxy for
        the best leaf below it (since an average underestimates the maximum). This is
        the only way an algorithm is meant to obtain information about the tree.
        """
        self.n_pulls += 1
        self.total_cost += self.cost(node)
        return self.true_value(node) + self.rng.normal(0.0, self.noise_for(node))
