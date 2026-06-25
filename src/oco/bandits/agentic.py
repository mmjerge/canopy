"""Long-horizon agentic search: rollout-guided planning vs. best-of-N (random shooting).

This lifts the reasoning-tree result (``oco.bandits.reasoning``) from a static tree of leaf
rewards to a *real sequential decision process* with state and dynamics -- the agentic /
multi-step setting. It is the same multi-fidelity structure as :class:`oco.bandits.tree.TreeBandit`:

* an **expensive, unbiased** signal -- the terminal reward of a committed trajectory -- and
* a **cheap, biased** signal -- a short random rollout from a partial trajectory (the value
  probe), whose mean over-/under-estimates the best reachable leaf but is informative about
  which action makes progress.

Two planners are compared at a matched budget of simulator steps:

* :func:`best_of_n_plan` -- structure-blind: sample ``n`` whole random trajectories and keep
  the best by terminal reward. To *land on* a long correct trajectory it must sample one,
  which is exponentially unlikely as the horizon grows.
* :func:`rollout_policy_plan` -- edge-following: at each state, score each action by a few
  cheap random rollouts to the horizon (the multi-fidelity probe) and commit to the best,
  then advance. It resolves one decision at a time, so its cost grows polynomially.

The point of moving to an environment (vs. GSM8K) is that the task has a genuine horizon and
the value probe is informative at intermediate states -- exactly the regime where search beats
sampling, and exactly what a single votable final answer (GSM8K) does not have. Honest scope:
this is a *search / compute-allocation* result. It needs the value probe to be informative;
the included maze/local-optimum case shows where that assumption breaks and the gap closes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

State = tuple[int, int]

# 4-connected grid moves: up, down, left, right.
_MOVES: tuple[State, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class GridWorld:
    """A deterministic grid-navigation task: reach ``goal`` from ``start`` within ``horizon``.

    Actions are the four 4-connected moves; a move into a wall or off the grid leaves the
    agent in place (so the action space is always ``n_actions`` and trajectories never error).
    The terminal reward is ``1 - manhattan(state, goal) / d_max`` in ``[0, 1]`` -- a smooth
    shaping that is ``1`` exactly at the goal -- so a *random rollout* from a state carries a
    weak but real signal about that state's distance to the goal (the cheap value probe).
    """

    size: int
    goal: State
    start: State
    horizon: int
    walls: frozenset[State] = field(default_factory=frozenset)

    @property
    def n_actions(self) -> int:
        return len(_MOVES)

    def step(self, state: State, action: int) -> State:
        dr, dc = _MOVES[action]
        nr, nc = state[0] + dr, state[1] + dc
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            return state
        if (nr, nc) in self.walls:
            return state
        return (nr, nc)

    def is_goal(self, state: State) -> bool:
        return state == self.goal

    def _d_max(self) -> int:
        return max(1, 2 * (self.size - 1))

    def reward(self, state: State) -> float:
        d = abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1])
        return 1.0 - d / self._d_max()

    @classmethod
    def open_diagonal(cls, size: int, horizon: int | None = None) -> "GridWorld":
        """Open grid, start at one corner, goal at the opposite corner.

        Shortest path is ``2 * (size - 1)``; the default ``horizon`` gives just enough slack
        (``+ size``) to reach it, so random shooting must be efficient to succeed.
        """
        dist = 2 * (size - 1)
        return cls(size=size, goal=(size - 1, size - 1), start=(0, 0),
                   horizon=horizon if horizon is not None else dist + size)


@dataclass
class PlanResult:
    """Outcome of a planner: whether it reached the goal, terminal reward, and cost (steps)."""

    reached_goal: bool
    reward: float
    steps: int
    trajectory: list[State] = field(default_factory=list)


def best_of_n_plan(env: GridWorld, n: int, rng: np.random.Generator) -> PlanResult:
    """Sample ``n`` whole random trajectories; return the best by terminal reward.

    Cost is the total number of simulator transitions across all ``n`` rollouts (rollouts stop
    early on reaching the goal). This is the structure-blind baseline (random shooting).
    """
    total_steps = 0
    best: PlanResult | None = None
    for _ in range(n):
        s = env.start
        traj = [s]
        for _h in range(env.horizon):
            s = env.step(s, int(rng.integers(env.n_actions)))
            total_steps += 1
            traj.append(s)
            if env.is_goal(s):
                break
        r = env.reward(s)
        if best is None or r > best.reward:
            best = PlanResult(env.is_goal(s), r, 0, traj)
    assert best is not None
    best.steps = total_steps
    return best


def rollout_policy_plan(
    env: GridWorld, n_rollouts: int, rng: np.random.Generator,
) -> PlanResult:
    """Edge-following planner: at each state, pick the action with the best rollout value.

    For each of ``n_actions`` candidate actions, take the action then run ``n_rollouts`` random
    rollouts to the horizon and average the terminal reward (the cheap multi-fidelity probe);
    commit to the highest-mean action and advance. Cost is the total simulator transitions
    (probe rollouts + committed steps). This is one-step rollout policy improvement / a
    depth-one tree search with random-rollout value.
    """
    s = env.start
    traj = [s]
    total_steps = 0
    for h in range(env.horizon):
        if env.is_goal(s):
            break
        remaining = env.horizon - h
        best_a, best_q = 0, -np.inf
        for a in range(env.n_actions):
            s1 = env.step(s, a)
            total_steps += 1
            q_sum = 0.0
            for _ in range(n_rollouts):
                t = s1
                for _k in range(remaining - 1):
                    t = env.step(t, int(rng.integers(env.n_actions)))
                    total_steps += 1
                    if env.is_goal(t):
                        break
                q_sum += env.reward(t)
            q = q_sum / n_rollouts
            if q > best_q:
                best_q, best_a = q, a
        s = env.step(s, best_a)
        total_steps += 1
        traj.append(s)
    return PlanResult(env.is_goal(s), env.reward(s), total_steps, traj)


def compare_matched_budget(
    env: GridWorld, n_rollouts: int, seed: int,
) -> dict[str, PlanResult]:
    """Run both planners at a matched step budget; best-of-N gets value-guided's step count.

    Value-guided runs first with ``n_rollouts`` probes per action; best-of-N is then given
    ``n = round(vg_steps / horizon)`` random trajectories so both spend ~the same number of
    simulator transitions. Returns ``{"value_guided": ..., "best_of_n": ...}``.
    """
    vg = rollout_policy_plan(env, n_rollouts, np.random.default_rng(seed))
    n = max(1, round(vg.steps / env.horizon))
    bo = best_of_n_plan(env, n, np.random.default_rng(10_000 + seed))
    return {"value_guided": vg, "best_of_n": bo}


def success_rates(
    env: GridWorld, n_rollouts: int, seeds: int = 100,
) -> dict[str, float]:
    """Goal-reaching rate of each planner over ``seeds`` instances at a matched budget."""
    vg_ok = bo_ok = 0
    vg_steps = 0
    for s in range(seeds):
        res = compare_matched_budget(env, n_rollouts, seed=s)
        vg_ok += res["value_guided"].reached_goal
        bo_ok += res["best_of_n"].reached_goal
        vg_steps += res["value_guided"].steps
    return {
        "value_guided": vg_ok / seeds,
        "best_of_n": bo_ok / seeds,
        "avg_budget_steps": vg_steps / seeds,
    }
