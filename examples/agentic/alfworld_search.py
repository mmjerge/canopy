"""Path A: value-guided tree search vs. best-of-N on ALFWorld (real long-horizon agent).

ALFWorld text games are the right fit for the agentic search in ``canopy.bandits.agentic_llm``:
the action space is the enumerable ``admissible_commands`` set and transitions are
deterministic, so a state can be forked for lookahead by replaying the action prefix
(:class:`canopy.bandits.ReplayCloneEnv`). We compare, at a matched policy-call budget:

* best-of-N: run N whole episodes, succeed if any solves the task;
* value-guided: at each step score candidate commands by cheap rollouts (the multi-fidelity
  probe) and commit to the best.

This is the experiment that exercises tree-depth and multi-fidelity, which the routing
benchmarks cannot. The search logic is verified by tests/test_agentic_llm.py; this file is the
real-environment glue.

Setup (separate env; see the install notes we discussed):
    conda create -n canopy-agentic python=3.10 && conda activate canopy-agentic
    pip install alfworld && alfworld-download           # sets ALFWORLD_DATA in ~/.cache/alfworld
    pip install -e /path/to/canopy[llm]
Run (needs the ALFWorld base_config.yaml and Bedrock creds):
    python examples/agentic/alfworld_search.py --config /path/to/alfworld/configs/base_config.yaml
Smoke test the pipeline without alfworld or credentials:
    python examples/agentic/alfworld_search.py --mock
"""

from __future__ import annotations

import argparse
import os
import re

from canopy.bandits import ReplayCloneEnv, compare_matched_budget_agent


class _AlfworldStepEnv:
    """StepEnv adapter over an ALFWorld batched text env (batch_size=1).

    NOTE: ALFWorld's API varies by version; the three marked lines are the integration points
    to verify against your install. Everything else (search, budgeting, replay) is generic.
    """

    def __init__(self, env):
        self._env = env
        self._info: dict = {}

    def reset(self) -> str:
        obs, info = self._env.reset()  # <-- ALFWorld reset returns (obs, info)
        self._info = info
        return obs[0]

    def valid_actions(self) -> list[str]:
        return list(self._info["admissible_commands"][0])  # <-- per-batch admissible set

    def step(self, action: str) -> tuple[str, float, bool]:
        obs, scores, dones, infos = self._env.step([action])  # <-- batched step
        self._info = infos
        return obs[0], float(scores[0]), bool(dones[0])


def load_alfworld(config_path: str, split: str):
    """Construct AlfredTWEnv once (the expensive game scan) and return (alfred, sorted_files).

    The YAML config is loaded directly (ALFWorld's ``generic.load_config`` parses ``sys.argv``,
    which would clash with our argparse). Set ``ALFWORLD_DATA`` (done by ``alfworld-download``)
    so the data paths resolve.
    """
    import yaml
    from alfworld.agents.environment import get_environment

    with open(config_path) as f:
        config = yaml.safe_load(f)
    env_type = config["env"]["type"]  # 'AlfredTWEnv' for the text-only games
    alfred = get_environment(env_type)(config, train_eval=split)  # game scan happens here only
    return alfred, sorted(getattr(alfred, "game_files", []) or [])


def engine_for_task(alfred, files, task_index: int) -> _AlfworldStepEnv:
    """Pin AlfredTWEnv to one game and build a fresh text engine for it (deterministic reset)."""
    if files:
        alfred.game_files = [files[task_index % len(files)]]
    return _AlfworldStepEnv(alfred.init_env(batch_size=1))


# --- a tiny deterministic stand-in so the pipeline is runnable without alfworld -----------


class _MockAlfworldStepEnv:
    """Toy text task: 'go to' then 'take' a target object; deterministic, admissible commands."""

    def __init__(self) -> None:
        self.stage = 0

    def reset(self) -> str:
        self.stage = 0
        return "You are in a room. Task: take the mug from the shelf."

    def valid_actions(self) -> list[str]:
        if self.stage == 0:
            return ["go to shelf", "go to fridge", "open fridge"]
        if self.stage == 1:
            return ["take mug from shelf", "go to fridge", "examine shelf"]
        return ["look", "inventory"]

    def step(self, action: str) -> tuple[str, float, bool]:
        if self.stage == 0 and action == "go to shelf":
            self.stage = 1
            return "You are at the shelf. You see a mug.", 0.0, False
        if self.stage == 1 and action == "take mug from shelf":
            self.stage = 2
            return "You take the mug. Task complete.", 1.0, True
        return "Nothing happens.", 0.0, False


def make_llm_act(client, model: str):
    def act(obs: str, actions, seed: int) -> str:
        numbered = "\n".join(f"{i}. {a}" for i, a in enumerate(actions))
        prompt = (
            f"{obs}\nChoose the single best next action to complete the task.\n"
            f"{numbered}\nReply with only the number of the chosen action."
        )
        try:
            text, _, _ = client.generate(model, prompt, temperature=0.7, max_tokens=6)
        except Exception:  # noqa: BLE001
            return actions[seed % len(actions)]
        m = re.search(r"\d+", text)
        if m and int(m.group()) < len(actions):
            return actions[int(m.group())]
        low = text.strip().lower()
        for a in actions:
            if a.lower() in low:
                return a
        return actions[seed % len(actions)]

    return act


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-index", type=int, default=0, help="first task (game) index")
    ap.add_argument("--num-tasks", type=int, default=1, help="number of tasks to average over")
    ap.add_argument("--split", default="eval_out_of_distribution")
    ap.add_argument(
        "--config",
        default=os.environ.get("ALFWORLD_CONFIG"),
        help="path to ALFWorld base_config.yaml (or set ALFWORLD_CONFIG)",
    )
    ap.add_argument("--branching", type=int, default=5)
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument(
        "--rollout-horizon",
        type=int,
        default=0,
        help="rollout length for the value probe; 0 means use --max-steps (full-length rollouts, "
        "needed for the sparse ALFWorld reward to ever be non-zero)",
    )
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--model", default="us.anthropic.claude-opus-4-7")
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--max-spend", type=float, default=None)
    ap.add_argument("--cache", default="examples/.cache/alfworld_search.jsonl")
    ap.add_argument("--mock", action="store_true", help="use a toy env + random policy, no deps")
    args = ap.parse_args()

    rollout_horizon = args.rollout_horizon or args.max_steps

    if args.mock:
        import numpy as np

        rng = np.random.default_rng(0)

        def act(_obs, actions, _seed):
            return rng.choice(actions)

        client = None

        def get_engine(_idx):
            return _MockAlfworldStepEnv()

        task_indices = list(range(args.num_tasks))
    else:
        if not args.config or not os.path.exists(args.config):
            print(
                "ALFWorld needs --config pointing to base_config.yaml (or set ALFWORLD_CONFIG). "
                "It ships in the alfworld repo under configs/base_config.yaml. Use --mock to "
                "smoke-test the search pipeline without alfworld."
            )
            return
        try:
            from canopy.llm import BedrockClient, CachingLLMClient

            alfred, files = load_alfworld(args.config, args.split)
            client = CachingLLMClient(
                BedrockClient(region=args.region, max_tokens=6),
                args.cache,
                max_calls=args.max_calls,
                max_spend_usd=args.max_spend,
            )
            act = make_llm_act(client, args.model)

            def get_engine(idx):
                return engine_for_task(alfred, files, idx)

            task_indices = list(range(args.task_index, args.task_index + args.num_tasks))
        except Exception as e:  # noqa: BLE001
            print(
                f"Could not initialize ALFWorld + LLM ({type(e).__name__}: {e}).\n"
                "Install alfworld in a separate env (see the header), run alfworld-download, "
                "and configure Bedrock. Use --mock to smoke-test the search pipeline."
            )
            return

    label = "mock" if args.mock else args.model
    print(f"ALFWorld {args.split}: {len(task_indices)} task(s), model {label}")
    agg = {"value_guided": [0, 0.0], "best_of_n": [0, 0.0]}  # [solved_count, reward_sum]
    try:
        for ti in task_indices:
            env = ReplayCloneEnv(get_engine(ti))
            res = compare_matched_budget_agent(
                env,
                act,
                branching=args.branching,
                rollouts=args.rollouts,
                rollout_horizon=rollout_horizon,
                max_steps=args.max_steps,
                progress=lambda m, _ti=ti: print(f"  task {_ti}: {m}", flush=True),
            )
            for name in ("value_guided", "best_of_n"):
                agg[name][0] += int(res[name].solved)
                agg[name][1] += res[name].reward
            print(
                f"  task {ti}: value_guided solved={res['value_guided'].solved} "
                f"best_of_n solved={res['best_of_n'].solved}"
            )
    except Exception as e:  # noqa: BLE001
        from canopy.llm import BudgetError

        if not isinstance(e, BudgetError):
            raise
        print(f"[budget stop] {e}")

    m = max(1, len(task_indices))
    print(f"\nALFWorld results over {len(task_indices)} task(s), model {label}:")
    for name in ("value_guided", "best_of_n"):
        print(f"  {name:13s} success={agg[name][0] / m:.2f}  avg_reward={agg[name][1] / m:.3f}")
    if client is not None:
        print(f"  budget: {client.stats()}")


if __name__ == "__main__":
    main()
