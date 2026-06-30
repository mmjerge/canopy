"""Real-LLM agentic search on a self-contained text navigation task.

This is the runnable, dependency-free demonstration of the agentic tree search
(``canopy.bandits.agentic_llm``): value-guided lookahead vs. best-of-N rollout at a matched
policy-call budget, with a real LLM as the policy. The environment is a small text-rendered
grid (a wrapper over the tested :class:`canopy.bandits.GridWorld`), so it is deterministic and
cheaply cloneable, which is exactly what value-guided lookahead needs and what published agent
benchmarks make hard (see the note at the bottom of this file).

Why a self-contained env first: it de-risks the LLM-policy path (prompt, parse, budget,
caching) on a task we fully control, with the same search code that will drive ALFWorld later.
Run it with ``--mock`` to smoke-test the whole pipeline with a random policy and no credentials.

Run (real model):
    uv run --extra llm python examples/agentic/textgrid_search.py --size 6
Run (no creds, logic smoke test):
    uv run python examples/agentic/textgrid_search.py --mock --size 6
"""

from __future__ import annotations

import argparse
import copy

import numpy as np

from canopy.bandits import GridWorld, compare_matched_budget_agent

_ACTIONS = ["up", "down", "left", "right"]


class TextGridEnv:
    """Text-rendered, cloneable AgentEnv over GridWorld (deterministic transitions)."""

    def __init__(self, world: GridWorld):
        self.world = world
        self.state = world.start

    def reset(self) -> str:
        self.state = self.world.start
        return self._obs()

    def _obs(self) -> str:
        return (
            f"You are on a {self.world.size}x{self.world.size} grid at row {self.state[0]}, "
            f"column {self.state[1]}. The goal is row {self.world.goal[0]}, column "
            f"{self.world.goal[1]}. Rows increase downward, columns increase rightward."
        )

    def valid_actions(self) -> list[str]:
        return list(_ACTIONS)

    def step(self, action: str) -> tuple[str, float, bool]:
        self.state = self.world.step(self.state, _ACTIONS.index(action))
        done = self.world.is_goal(self.state)
        return self._obs(), self.world.reward(self.state), done

    def clone(self) -> "TextGridEnv":
        return copy.deepcopy(self)


def make_llm_act(client, model: str):
    """Build an act(obs, valid_actions, seed) policy backed by a real LLM through canopy.llm."""

    def act(obs: str, actions, seed: int) -> str:
        prompt = (
            f"{obs}\nChoose the single best next move to reach the goal.\n"
            f"Reply with exactly one word from: {', '.join(actions)}."
        )
        try:
            text, _, _ = client.generate(model, prompt, temperature=0.7, max_tokens=4)
        except Exception:  # noqa: BLE001 -- fall back to a valid move on any provider hiccup
            return actions[seed % len(actions)]
        low = text.strip().lower()
        for a in actions:
            if a in low:
                return a
        return actions[seed % len(actions)]

    return act


def make_mock_act(seed: int = 0):
    rng = np.random.default_rng(seed)

    def act(_obs, actions, _seed):
        return rng.choice(actions)

    return act


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=6)
    ap.add_argument("--branching", type=int, default=4)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--model", default="us.meta.llama3-1-70b-instruct-v1:0")
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--max-spend", type=float, default=None)
    ap.add_argument("--cache", default="examples/.cache/textgrid_search.jsonl")
    ap.add_argument("--mock", action="store_true", help="use a random policy (no credentials)")
    args = ap.parse_args()

    world = GridWorld.open_diagonal(size=args.size)

    if args.mock:
        act = make_mock_act(seed=0)
    else:
        try:
            from canopy.llm import BedrockClient, CachingLLMClient

            client = CachingLLMClient(
                BedrockClient(region=args.region, max_tokens=4),
                args.cache,
                max_calls=args.max_calls,
                max_spend_usd=args.max_spend,
            )
            act = make_llm_act(client, args.model)
        except Exception as e:  # noqa: BLE001
            print(f"Could not init the real-LLM policy ({e}); rerun with --mock for a smoke test.")
            return

    env = TextGridEnv(world)
    res = compare_matched_budget_agent(
        env,
        act,
        branching=args.branching,
        rollouts=args.rollouts,
        rollout_horizon=world.horizon,
        max_steps=world.horizon,
    )
    print(f"text grid {args.size}x{args.size}, horizon {world.horizon}, model {args.model}")
    for name in ("value_guided", "best_of_n"):
        r = res[name]
        print(
            f"  {name:13s} solved={r.solved!s:5s} reward={r.reward:.3f} "
            f"steps={r.steps} calls={r.budget.calls}"
        )
    if not args.mock:
        print(f"  budget: {client.stats()}")


if __name__ == "__main__":
    main()
