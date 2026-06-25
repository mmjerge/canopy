"""Tests for the long-horizon agentic search environment and planners."""

from __future__ import annotations

import numpy as np

from canopy.bandits.agentic import (
    GridWorld,
    best_of_n_plan,
    compare_matched_budget,
    rollout_policy_plan,
    success_rates,
)


def test_gridworld_dynamics():
    env = GridWorld.open_diagonal(size=4)
    assert env.start == (0, 0)
    assert env.goal == (3, 3)
    # moving up/left from the corner is blocked (stays in place)
    assert env.step((0, 0), 0) == (0, 0)
    assert env.step((0, 0), 2) == (0, 0)
    # moving down/right makes progress
    assert env.step((0, 0), 1) == (1, 0)
    assert env.step((0, 0), 3) == (0, 1)
    # reward is 1 exactly at the goal and < 1 elsewhere
    assert env.reward(env.goal) == 1.0
    assert env.reward(env.start) < 1.0


def test_walls_block_movement():
    env = GridWorld(size=3, goal=(2, 2), start=(0, 0), horizon=10,
                    walls=frozenset({(0, 1)}))
    assert env.step((0, 0), 3) == (0, 0)  # blocked by wall to the right
    assert env.step((0, 0), 1) == (1, 0)  # down is open


def test_rollout_policy_beats_best_of_n_on_long_horizon():
    # On a grid large enough that random shooting rarely lands on the goal,
    # rollout-guided planning should reach it far more often at a matched budget.
    env = GridWorld.open_diagonal(size=8)
    rates = success_rates(env, n_rollouts=8, seeds=40)
    assert rates["value_guided"] > rates["best_of_n"]
    assert rates["value_guided"] > 0.7


def test_matched_budget_is_comparable():
    env = GridWorld.open_diagonal(size=6)
    res = compare_matched_budget(env, n_rollouts=6, seed=0)
    vg_steps = res["value_guided"].steps
    bo_steps = res["best_of_n"].steps
    # best-of-N's budget is set from value-guided's step count; allow one horizon of slack.
    assert abs(vg_steps - bo_steps) <= env.horizon


def test_results_are_deterministic_given_seed():
    env = GridWorld.open_diagonal(size=6)
    a = rollout_policy_plan(env, n_rollouts=5, rng=np.random.default_rng(123))
    b = rollout_policy_plan(env, n_rollouts=5, rng=np.random.default_rng(123))
    assert a.reached_goal == b.reached_goal
    assert a.trajectory == b.trajectory
    c = best_of_n_plan(env, n=20, rng=np.random.default_rng(7))
    d = best_of_n_plan(env, n=20, rng=np.random.default_rng(7))
    assert c.trajectory == d.trajectory
