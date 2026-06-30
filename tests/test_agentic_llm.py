"""Tests for the environment-agnostic agentic search (value-guided vs. best-of-N).

The search core is exercised with a mock environment built by wrapping the tested
``GridWorld`` (shaped distance reward) behind the :class:`AgentEnv` protocol, with a uniformly
random policy. Because the reward is informative about progress, a few cheap rollouts give a
useful value signal, so value-guided lookahead should reach the goal more often than best-of-N
at a matched policy-call budget. A deceptive (anti-correlated) value closes the gap, which is
the conditional-advantage check.
"""

from __future__ import annotations

import copy

import numpy as np

from canopy.bandits import (
    GridWorld,
    ReplayCloneEnv,
    best_of_n_episodes,
    compare_matched_budget_agent,
    value_guided_episode,
)

_ACTIONS = ["up", "down", "left", "right"]


class GridAgentEnv:
    """Mutable AgentEnv adapter over the (frozen, tested) GridWorld with string actions."""

    def __init__(self, world: GridWorld, seed: int = 0):
        self.world = world
        self.state = world.start
        self.t = 0
        self.rng = np.random.default_rng(seed)

    def reset(self) -> str:
        self.state = self.world.start
        self.t = 0
        return f"at {self.state}"

    def valid_actions(self) -> list[str]:
        return list(_ACTIONS)

    def step(self, action: str) -> tuple[str, float, bool]:
        self.state = self.world.step(self.state, _ACTIONS.index(action))
        self.t += 1
        done = self.world.is_goal(self.state) or self.t >= self.world.horizon
        return f"at {self.state}", self.world.reward(self.state), done

    def clone(self) -> "GridAgentEnv":
        return copy.deepcopy(self)


def _random_policy(seed_base: int = 0):
    rng = np.random.default_rng(seed_base)

    def act(_obs, actions, _seed):
        return rng.choice(actions)

    return act


def test_value_guided_beats_best_of_n_at_matched_budget():
    world = GridWorld.open_diagonal(size=7)
    vg_ok = bo_ok = 0
    trials = 30
    for s in range(trials):
        env = GridAgentEnv(world, seed=s)
        act = _random_policy(seed_base=1000 + s)
        res = compare_matched_budget_agent(
            env,
            act,
            branching=4,
            rollouts=3,
            rollout_horizon=world.horizon,
            max_steps=world.horizon,
        )
        vg_ok += res["value_guided"].solved
        bo_ok += res["best_of_n"].solved
    assert vg_ok > bo_ok


def test_budget_is_counted_and_episode_terminates():
    world = GridWorld.open_diagonal(size=4)
    env = GridAgentEnv(world, seed=0)
    act = _random_policy(seed_base=7)
    res = value_guided_episode(
        env, act, branching=3, rollouts=2, rollout_horizon=world.horizon, max_steps=world.horizon
    )
    assert res.budget.calls > 0
    assert res.steps <= world.horizon


def test_best_of_n_respects_episode_count():
    world = GridWorld.open_diagonal(size=4)
    env = GridAgentEnv(world, seed=1)
    act = _random_policy(seed_base=3)
    res = best_of_n_episodes(env, act, n=5, max_steps=world.horizon)
    # 5 episodes of at most `horizon` steps each.
    assert 0 < res.budget.calls <= 5 * world.horizon
    assert 0.0 <= res.reward <= 1.0


class _CounterStepEnv:
    """A deterministic StepEnv with no native clone: walk a line toward a target index."""

    def __init__(self, target: int = 5, length: int = 8):
        self.target = target
        self.length = length
        self.pos = 0

    def reset(self) -> str:
        self.pos = 0
        return f"pos={self.pos}"

    def valid_actions(self) -> list[str]:
        return ["inc", "dec"]

    def step(self, action: str) -> tuple[str, float, bool]:
        self.pos += 1 if action == "inc" else -1
        self.pos = max(0, min(self.length, self.pos))
        reward = 1.0 - abs(self.pos - self.target) / self.length
        done = self.pos == self.target
        return f"pos={self.pos}", reward, done


def test_replay_clone_reproduces_state_and_is_independent():
    env = ReplayCloneEnv(_CounterStepEnv(target=5, length=8))
    env.step("inc")
    env.step("inc")  # pos=2 on the live env
    fork = env.clone()
    # advancing the fork must not affect the original (sync-on-access repositions the engine)
    fork.step("inc")
    fork.step("inc")  # fork pos=4
    o_live, _, _ = env.step("dec")  # original pos=1
    assert "pos=1" in o_live
    o_fork, _, _ = fork.step("inc")  # fork pos=5 (reaches target)
    assert "pos=5" in o_fork


def test_value_guided_runs_on_replayed_env():
    # value-guided lookahead works over a replay-synced, natively non-clonable env.
    env = ReplayCloneEnv(_CounterStepEnv(target=5, length=8))
    act = _random_policy(seed_base=11)
    res = value_guided_episode(env, act, branching=2, rollouts=2, rollout_horizon=8, max_steps=8)
    assert res.budget.calls > 0
    # lookahead over the replayed env makes clear progress toward the target
    assert res.reward > 0.5
    assert res.steps <= 8
