"""Reasoning-tree search: value-guided (edge-following) descent vs. best-of-N.

A leaf is a reasoning trace; its reward is the fraction of ``K`` decision steps it gets right
(reward 1 = fully correct). The internal-node value (subtree-average reward) is the PRM
signal, smooth except at the decision steps (the edges). At a fixed oracle budget we compare:

  * best-of-N -- sample whole traces, return the best-scoring; must *sample* a fully-correct
    trace, which needs ~2^K draws; and
  * value-guided search -- descend the tree following the higher-value child at each step
    (the value edge at a decision), resolving the K decisions one at a time at polynomial
    cost.

Result: value-guided search finds the correct trace at a budget where best-of-N cannot --
an exponential separation in the number of decision steps. (This is a search /
compute-allocation result: it needs a reachable correct trace and an informative value
signal, and does not add capability the model lacks.)

Run with:  uv run python examples/reasoning/reasoning_search_demo.py
           uv run --extra plot python examples/reasoning/reasoning_search_demo.py   # + PNG
"""

from __future__ import annotations

from pathlib import Path

from canopy.bandits.reasoning import success_rate

SIGMA = 0.3
SEEDS = 200
K_VALUES = [4, 6, 8, 10, 12]
BUDGETS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
TARGET = 0.8


def budget_to_target(method: str, k: int) -> int | None:
    for b in BUDGETS:
        if (
            success_rate(method, depth=k, n_decisions=k, budget=b, sigma=SIGMA, seeds=SEEDS)
            >= TARGET
        ):
            return b
    return None


def main() -> None:
    print(
        f"reasoning tree (binary), reward = fraction of K decisions correct, noise {SIGMA}, "
        f"{SEEDS} seeds"
    )
    print(
        f"{'K':>3} {'best-of-N':>12} {'value-guided':>14}   (budget to reach "
        f"{int(TARGET*100)}% fully-correct)"
    )
    bo_budgets, vg_budgets = [], []
    for k in K_VALUES:
        b_bo = budget_to_target("best_of_n", k)
        b_vg = budget_to_target("value_guided", k)
        bo_budgets.append(b_bo)
        vg_budgets.append(b_vg)
        print(f"{k:3d} {str(b_bo):>12} {str(b_vg):>14}")
    print(
        "\nbest-of-N needs exponentially many samples (fails within budget for K>=6); "
        "value-guided\nsearch follows the value edges and succeeds at polynomial cost."
    )

    # success-vs-budget at a fixed K, for the curves panel
    k_fixed = 8
    curve_bo = [success_rate("best_of_n", k_fixed, k_fixed, b, SIGMA, SEEDS) for b in BUDGETS]
    curve_vg = [success_rate("value_guided", k_fixed, k_fixed, b, SIGMA, SEEDS) for b in BUDGETS]

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(install the 'plot' extra for the chart: uv run --extra plot ...)")
        return

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

    axA.semilogx(BUDGETS, curve_vg, "-o", color="#27ae60", lw=2, label="value-guided (edges)")
    axA.semilogx(BUDGETS, curve_bo, "-s", color="#c0392b", lw=2, label="best-of-N")
    axA.axhline(TARGET, color="#7f8c8d", ls="--", lw=1)
    axA.set_xlabel("oracle budget (rollouts / evaluations)")
    axA.set_ylabel("P(returns fully-correct trace)")
    axA.set_title(f"(A) Success vs budget at K={k_fixed} decisions", fontsize=11, loc="left")
    axA.set_ylim(0, 1.02)
    axA.grid(True, which="both", ls=":", alpha=0.4)
    axA.legend(loc="upper left", fontsize=9)

    cap = BUDGETS[-1] * 2
    bo_plot = [b if b is not None else cap for b in bo_budgets]
    vg_plot = [b if b is not None else cap for b in vg_budgets]
    axB.semilogy(K_VALUES, vg_plot, "-o", color="#27ae60", lw=2, label="value-guided (edges)")
    axB.semilogy(K_VALUES, bo_plot, "-s", color="#c0392b", lw=2, label="best-of-N")
    axB.axhline(BUDGETS[-1], color="#7f8c8d", ls=":", lw=1)
    axB.text(
        K_VALUES[0],
        cap,
        "best-of-N fails within budget →",
        color="#c0392b",
        fontsize=8,
        va="bottom",
    )
    axB.set_xlabel("number of decision steps K")
    axB.set_ylabel(f"budget to reach {int(TARGET*100)}% (log)")
    axB.set_title(
        "(B) Best-of-N is exponential in K; value-guided is polynomial", fontsize=11, loc="left"
    )
    axB.grid(True, which="both", ls=":", alpha=0.4)
    axB.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        "Value-guided reasoning-tree search finds the correct trace where best-of-N " "cannot",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(__file__).parent.parent / "images" / "tree_reasoning_search.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved chart to {out}")


if __name__ == "__main__":
    main()
