"""Online Convex Optimization: a sandbox for OCO algorithms and experiments.

Modules:
    algorithms -- first-order online methods (OGD, strongly-convex OGD).  [Ch. 2-3]
    experts    -- prediction from expert advice (Hedge, RWM).             [Ch. 1]
    bandits    -- EXP3 and the tree-structured top-k identification task. [Ch. 6 + brief]
"""

from canopy.algorithms import OnlineGradientDescent, StronglyConvexOGD
from canopy.experts import Hedge, RandomizedWeightedMajority

__all__ = [
    "OnlineGradientDescent",
    "StronglyConvexOGD",
    "Hedge",
    "RandomizedWeightedMajority",
]
__version__ = "0.1.0"
