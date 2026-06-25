"""Long-horizon agentic search: rollout-guided planning vs. best-of-N (random shooting).

Lifts the reasoning-tree separation (``examples/reasoning_search_demo.py``) into a real
sequential decision process -- grid navigation with a genuine horizon. Both planners spend the
same number of simulator steps; the only difference is *how* they allocate that compute.

    uv run --extra plot python examples/agentic_search_demo.py

Produces ``examples/tree_agentic_search.png``:
  (1) example trajectories on the grid (rollout-guided walks to the goal; random shooting's
      best-of-N sample wanders);
  (2) goal-reaching rate vs. horizon -- best-of-N collapses as the horizon grows while
      rollout-guided planning holds up (the long-horizon separation);
  (3) goal-reaching rate vs. compute budget on a fixed long-horizon grid.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from oco.bandits.agentic import (
    GridWorld,
    best_of_n_plan,
    rollout_policy_plan,
    success_rates,
)


def _draw_grid(ax, env: GridWorld, traj, title: str, color: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(-0.5, env.size - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for r in range(env.size):
        for c in range(env.size):
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                       edgecolor="0.85", lw=0.5))
    xs = [s[1] for s in traj]
    ys = [s[0] for s in traj]
    ax.plot(xs, ys, "-o", color=color, ms=3, lw=1.5, alpha=0.8)
    ax.plot(env.start[1], env.start[0], "s", color="black", ms=9, label="start")
    ax.plot(env.goal[1], env.goal[0], "*", color="gold", ms=18,
            markeredgecolor="black", label="goal")
    ax.invert_yaxis()


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # --- panel 1: example trajectories on a long-horizon grid ---
    env = GridWorld.open_diagonal(size=10)
    vg = rollout_policy_plan(env, n_rollouts=10, rng=np.random.default_rng(0))
    n = max(1, round(vg.steps / env.horizon))
    bo = best_of_n_plan(env, n=n, rng=np.random.default_rng(1))
    ax = axes[0]
    _draw_grid(ax, env, bo.trajectory, "", "tab:orange")
    ax.plot([], [], "-o", color="tab:orange", ms=3, label=f"best-of-N (reached={bo.reached_goal})")
    ax.plot([s[1] for s in vg.trajectory], [s[0] for s in vg.trajectory],
            "-o", color="tab:blue", ms=3, lw=1.5, alpha=0.8,
            label=f"rollout-guided (reached={vg.reached_goal})")
    ax.set_title(f"Trajectories on a {env.size}x{env.size} grid "
                 f"(horizon {env.horizon}, ~{vg.steps} steps each)", fontsize=10)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)

    # --- panel 2: success vs horizon (grid size) at matched budget ---
    sizes = [4, 6, 8, 10, 12, 14]
    vg_rate, bo_rate, horizons = [], [], []
    for sz in sizes:
        e = GridWorld.open_diagonal(size=sz)
        r = success_rates(e, n_rollouts=8, seeds=60)
        vg_rate.append(r["value_guided"])
        bo_rate.append(r["best_of_n"])
        horizons.append(e.horizon)
    ax = axes[1]
    ax.plot(horizons, vg_rate, "-o", color="tab:blue", label="rollout-guided")
    ax.plot(horizons, bo_rate, "-s", color="tab:orange", label="best-of-N")
    ax.set_xlabel("horizon (longer task)")
    ax.set_ylabel("goal-reaching rate")
    ax.set_title("Success vs. horizon (matched step budget)", fontsize=10)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # --- panel 3: success vs budget on a fixed long-horizon grid ---
    env_b = GridWorld.open_diagonal(size=12)
    rollouts = [1, 2, 4, 8, 16]
    vg_b, bo_b, budgets = [], [], []
    for m in rollouts:
        r = success_rates(env_b, n_rollouts=m, seeds=60)
        vg_b.append(r["value_guided"])
        bo_b.append(r["best_of_n"])
        budgets.append(r["avg_budget_steps"])
    ax = axes[2]
    ax.plot(budgets, vg_b, "-o", color="tab:blue", label="rollout-guided")
    ax.plot(budgets, bo_b, "-s", color="tab:orange", label="best-of-N")
    ax.set_xscale("log")
    ax.set_xlabel("compute budget (simulator steps, log scale)")
    ax.set_ylabel("goal-reaching rate")
    ax.set_title(f"Success vs. budget ({env_b.size}x{env_b.size}, horizon {env_b.horizon})",
                 fontsize=10)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Long-horizon agentic search: where to spend test-time compute "
                 "(equal budget, the only difference is allocation)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = "examples/tree_agentic_search.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    print(f"  panel1: best-of-N reached={bo.reached_goal}, rollout-guided reached={vg.reached_goal}")
    print(f"  panel2 horizons={horizons}")
    print(f"          vg={[round(x,2) for x in vg_rate]}  bo={[round(x,2) for x in bo_rate]}")


if __name__ == "__main__":
    main()
