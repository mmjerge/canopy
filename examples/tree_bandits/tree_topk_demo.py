"""Quick multi-fidelity demo: top-k leaf identification at equal COST budget.

Internal-node probes are cheap but biased; leaf evaluations are expensive but unbiased.
Compares the hierarchical method (cheap probes to localize, then leaf evals to confirm)
against the strong structure-blind baseline and the weak uniform baseline, at one tight
budget. For the full fidelity/budget sweeps and charts, see examples/tree_bandits/benchmark.py.

Run with:  uv run python examples/tree_bandits/tree_topk_demo.py
"""

from __future__ import annotations

import numpy as np

from canopy.bandits import (
    HierarchicalTopK,
    SuccessiveEliminationTopK,
    TreeBandit,
    UniformTopK,
    geometric_sigma,
    hierarchical_spread,
)


def main() -> None:
    branching, depth, k = 4, 5, 5  # 1024 leaves
    budget, probe_cost, n_seeds = 1500.0, 0.05, 25
    sigma = geometric_sigma(base=0.5, decay=0.55)
    spread = hierarchical_spread(sigma, depth, branching, z=3.0)

    def mk(seed):
        return TreeBandit.from_hierarchical_gaussian(
            branching,
            depth,
            sigma=sigma,
            noise_std=0.1,
            probe_cost=probe_cost,
            rng=np.random.default_rng(seed),
        )

    hier, se, uni = [], [], []
    for seed in range(n_seeds):
        e = mk(seed)
        hier.append(
            HierarchicalTopK(budget=budget, spread=spread, beam_width=20).run(e, k).evaluate(e, k)
        )
        e = mk(seed)
        se.append(SuccessiveEliminationTopK(budget=budget).run(e, k).evaluate(e, k))
        e = mk(seed)
        uni.append(UniformTopK(budget=budget).run(e, k).evaluate(e, k))

    print(
        f"tree: {branching**depth} leaves, top-{k}, budget={budget:.0f} cost, "
        f"probe/leaf={probe_cost}, {n_seeds} seeds"
    )
    print(f"  HierarchicalTopK         mean recall = {np.mean(hier):.2f}")
    print(f"  SuccessiveEliminationTopK mean recall = {np.mean(se):.2f}  (strong baseline)")
    print(f"  UniformTopK              mean recall = {np.mean(uni):.2f}  (weak baseline)")


if __name__ == "__main__":
    main()
