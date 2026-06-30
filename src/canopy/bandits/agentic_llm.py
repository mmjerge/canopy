"""Real-agent tree search: value-guided lookahead vs. best-of-N rollout, environment-agnostic.

This is the long-horizon counterpart to ``canopy.bandits.reasoning_llm``. Where that harness
searches a tree of reasoning steps for a single votable answer, this one searches the action
tree of a *sequential decision process* with state and dynamics: each node is a partial
trajectory, its children are the admissible next actions, and the reward arrives only at the
end of an episode. This is exactly the regime ALFWorld / tau-bench live in, and the one the
multi-fidelity contributions target (which flat routing benchmarks cannot exercise).

It is deliberately decoupled from any specific environment or model. An environment implements
the tiny :class:`AgentEnv` protocol (``reset`` / ``valid_actions`` / ``step`` / ``clone``); a
policy is any callable ``act(observation, valid_actions, seed) -> action``. So the search runs
against a mock grid environment in tests and against a real benchmark with an LLM policy (via
``canopy.llm``) in ``examples/agentic/taubench_search.py``.

Two strategies, compared at a matched budget of policy calls:

* :func:`best_of_n_episodes` -- structure-blind: run ``n`` whole episodes and keep the best by
  terminal reward. Landing on a long correct trajectory by sampling is exponentially unlikely
  as the horizon grows.
* :func:`value_guided_episode` -- edge-following: at each state sample a few candidate actions,
  score each by cheap rollouts to the horizon (the multi-fidelity probe), commit to the best,
  and advance. It resolves one decision at a time, so cost grows polynomially in the horizon.

Honest scope: this is a search / compute-allocation method. It wins only when the rollout
value is informative at the pivotal states; a flat or deceptive value closes the gap, which is
the same conditional-advantage story as the rest of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from canopy.bandits.reasoning_llm import Budget

ActFn = Callable[[str, Sequence[str], int], str]
"""A policy: ``(observation, valid_actions, seed) -> chosen action`` (one of valid_actions)."""


class AgentEnv(Protocol):
    """Minimal stateful environment interface for the agentic search.

    Implementations are mutable: ``step`` advances internal state. ``reset`` returns to the
    same task's initial state (gym semantics), and ``clone`` returns an independent deep copy
    of the *current* state so the search can look ahead without disturbing the live episode.
    """

    def reset(self) -> str:
        """Reset to the task's initial state and return the initial observation."""
        ...

    def valid_actions(self) -> list[str]:
        """Admissible actions in the current state (non-empty until the episode is done)."""
        ...

    def step(self, action: str) -> tuple[str, float, bool]:
        """Apply ``action``; return ``(observation, reward, done)``."""
        ...

    def clone(self) -> "AgentEnv":
        """Return an independent copy of the current state (for lookahead rollouts)."""
        ...


class StepEnv(Protocol):
    """A plain reset/step environment with no native cloning (e.g. a TextWorld game)."""

    def reset(self) -> str: ...

    def valid_actions(self) -> list[str]: ...

    def step(self, action: str) -> tuple[str, float, bool]: ...


class ReplayCloneEnv:
    """Make a deterministic non-clonable environment forkable via a single shared engine.

    Many real environments (ALFWorld / TextWorld games) cannot be deep-copied mid-episode, and
    rebuilding a fresh instance is far too expensive (ALFWorld rescans its whole game set and
    spawns a subprocess per env). For a *deterministic* environment we can fork it with a single
    persistent engine instead: each logical clone is just an action history, and the engine is
    repositioned to a clone's state on demand by ``reset`` + replaying that history (the classic
    "replay to node" trick from search over deterministic simulators).

    All clones share one underlying engine and a marker of where it currently sits; any access
    that needs a different position triggers one ``reset`` and a replay of the prefix. So only a
    single engine is ever built, and forks cost O(prefix length) env steps rather than a full
    rebuild.

    Args:
        engine: a single :class:`StepEnv` whose ``reset`` returns to the same task (for ALFWorld,
            one built over a single pinned game file). Determinism of the engine is what makes
            replay exact.
    """

    def __init__(
        self,
        engine: StepEnv,
        _history: list[str] | None = None,
        _shared: dict | None = None,
    ) -> None:
        self._engine = engine
        self._history: list[str] = list(_history or [])
        # Shared across all clones: the action history the engine currently reflects.
        self._shared = _shared if _shared is not None else {"cur": None}
        self._obs = ""
        self._reward = 0.0
        self._done = False
        self._sync()

    def _sync(self) -> None:
        """Reposition the shared engine to this clone's history if it is somewhere else."""
        if self._shared["cur"] == self._history:
            return
        obs = self._engine.reset()
        reward, done = 0.0, False
        for a in self._history:
            obs, reward, done = self._engine.step(a)
        self._obs, self._reward, self._done = obs, reward, done
        self._shared["cur"] = list(self._history)

    def reset(self) -> str:
        self._history = []
        self._obs = self._engine.reset()
        self._reward, self._done = 0.0, False
        self._shared["cur"] = []
        return self._obs

    def valid_actions(self) -> list[str]:
        self._sync()
        return self._engine.valid_actions()

    def step(self, action: str) -> tuple[str, float, bool]:
        self._sync()
        obs, reward, done = self._engine.step(action)
        self._history.append(action)
        self._shared["cur"] = list(self._history)
        self._obs, self._reward, self._done = obs, reward, done
        return obs, reward, done

    def clone(self) -> "ReplayCloneEnv":
        return ReplayCloneEnv(self._engine, list(self._history), self._shared)


@dataclass
class EpisodeResult:
    """Outcome of a planner on one task: success, terminal reward, steps, and budget spent."""

    solved: bool
    reward: float
    steps: int
    budget: Budget = field(default_factory=Budget)


def _rollout(
    env: AgentEnv,
    act: ActFn,
    budget: Budget,
    horizon: int,
    seed0: int,
) -> float:
    """Roll a cheap policy out to ``horizon`` from ``env``'s current state; return last reward.

    This is the cheap, biased value probe: a short self-play rollout whose terminal reward
    estimates the value of the state it started from. Charges one policy call per step.
    """
    reward = 0.0
    for k in range(horizon):
        actions = env.valid_actions()
        if not actions:
            break
        a = act("", actions, seed0 + k)
        budget.charge(a)
        _obs, reward, done = env.step(a)
        if done:
            break
    return reward


def best_of_n_episodes(
    env: AgentEnv,
    act: ActFn,
    n: int,
    max_steps: int,
    success_threshold: float = 1.0,
    progress: Callable[[str], None] | None = None,
) -> EpisodeResult:
    """Run ``n`` independent episodes; keep the best terminal reward (structure-blind baseline).

    Each episode resets the env and follows ``act`` greedily for up to ``max_steps``. The
    result is ``solved`` if any episode reaches ``success_threshold``. Budget counts every
    policy call across all episodes. ``progress`` (if given) is called with a short status line.
    """
    budget = Budget()
    best_reward = 0.0
    solved = False
    total_steps = 0
    for i in range(n):
        obs = env.reset()
        reward = 0.0
        for t in range(max_steps):
            actions = env.valid_actions()
            if not actions:
                break
            a = act(obs, actions, 1000 * i + t)
            budget.charge(a)
            obs, reward, done = env.step(a)
            total_steps += 1
            if done:
                break
        best_reward = max(best_reward, reward)
        solved = solved or reward >= success_threshold
        if progress:
            progress(f"episode {i + 1}/{n} reward={reward:.2f} calls={budget.calls}")
    return EpisodeResult(solved=solved, reward=best_reward, steps=total_steps, budget=budget)


def value_guided_episode(
    env: AgentEnv,
    act: ActFn,
    branching: int = 3,
    max_steps: int = 20,
    rollouts: int = 2,
    rollout_horizon: int = 8,
    success_threshold: float = 1.0,
    progress: Callable[[str], None] | None = None,
) -> EpisodeResult:
    """One episode with one-step value-guided lookahead (depth-one tree search per state).

    At each live step it samples up to ``branching`` distinct candidate actions from ``act``,
    scores each by ``rollouts`` cheap rollouts of horizon ``rollout_horizon`` from a cloned
    state (the multi-fidelity probe, averaging the realized terminal reward), commits to the
    highest-value action, and advances. Budget counts candidate-proposal calls plus all
    rollout calls. ``progress`` (if given) is called with a short status line per step.
    """
    budget = Budget()
    obs = env.reset()
    reward = 0.0
    total_steps = 0
    for step in range(max_steps):
        actions = env.valid_actions()
        if not actions:
            break
        # Candidate actions: enumerate the admissible set when it is small (the env provides
        # it for free); otherwise sample distinct proposals from the policy.
        candidates: list[str] = []
        if len(actions) <= branching:
            candidates = list(actions)
        else:
            for c in range(branching):
                a = act(obs, actions, 7000 + 100 * step + c)
                budget.charge(a)
                if a not in candidates:
                    candidates.append(a)
        # Score each candidate by cheap rollouts from a cloned state.
        best_a, best_val = candidates[0], -float("inf")
        for ci, a in enumerate(candidates):
            sim = env.clone()
            _o, r_imm, done = sim.step(a)
            if done:
                val = r_imm
            else:
                acc = 0.0
                for rr in range(rollouts):
                    roll_env = sim.clone()
                    acc += _rollout(
                        roll_env,
                        act,
                        budget,
                        rollout_horizon,
                        seed0=90_000 + 1000 * (step * branching + ci) + 50 * rr,
                    )
                val = acc / max(1, rollouts)
            if val > best_val:
                best_val, best_a = val, a
        obs, reward, done = env.step(best_a)
        total_steps += 1
        if progress:
            progress(
                f"step {step + 1}/{max_steps} action={best_a!r} "
                f"reward={reward:.2f} calls={budget.calls}"
            )
        if done:
            break
    return EpisodeResult(
        solved=reward >= success_threshold, reward=reward, steps=total_steps, budget=budget
    )


def compare_matched_budget(
    env: AgentEnv,
    act: ActFn,
    branching: int = 3,
    rollouts: int = 2,
    rollout_horizon: int = 8,
    max_steps: int = 20,
    success_threshold: float = 1.0,
    progress: Callable[[str], None] | None = None,
) -> dict[str, EpisodeResult]:
    """Run value-guided search, then best-of-N at the same policy-call budget.

    Value-guided runs first; best-of-N is given ``n = round(vg_calls / max_steps)`` episodes so
    both spend approximately the same number of policy calls. Returns
    ``{"value_guided": ..., "best_of_n": ...}``. ``progress`` is forwarded with a phase prefix.
    """

    def vg_progress(m: str) -> None:
        if progress:
            progress(f"[value-guided] {m}")

    def bo_progress(m: str) -> None:
        if progress:
            progress(f"[best-of-N] {m}")

    vg = value_guided_episode(
        env,
        act,
        branching=branching,
        max_steps=max_steps,
        rollouts=rollouts,
        rollout_horizon=rollout_horizon,
        success_threshold=success_threshold,
        progress=vg_progress,
    )
    n = max(1, round(vg.budget.calls / max(1, max_steps)))
    bo = best_of_n_episodes(
        env,
        act,
        n=n,
        max_steps=max_steps,
        success_threshold=success_threshold,
        progress=bo_progress,
    )
    return {"value_guided": vg, "best_of_n": bo}
