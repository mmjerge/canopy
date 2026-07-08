"""Value-guided tree search vs. best-of-N on ALFWorld (real long-horizon agent).

ALFWorld text games are the right fit for the agentic search in ``canopy.bandits.agentic_llm``:
the action space is the enumerable ``admissible_commands`` set and transitions are
deterministic, so a state can be forked for lookahead by replaying the action prefix
(:class:`canopy.bandits.ReplayCloneEnv`). At a *matched* policy-call budget we compare

* best-of-N: run N whole episodes, succeed if any solves the task;
* value-guided: at each step score candidate commands by cheap rollouts (the multi-fidelity
  probe) and commit to the best.

This is the long-horizon, tree-depth + multi-fidelity experiment that the flat routing
benchmarks cannot exercise -- the agentic counterpart to the GSM8K reasoning-search flagship.
Per-task outcomes are recorded so we can report bootstrap confidence bands, and results
(JSON + LaTeX table + figure) are written to ``paper/figures/`` like the other experiments.

Setup (separate env; see the install notes we discussed):
    conda create -n canopy-agentic python=3.10 && conda activate canopy-agentic
    pip install alfworld && alfworld-download           # sets ALFWORLD_DATA in ~/.cache/alfworld
    pip install -e /path/to/canopy[llm]
Run (needs the ALFWorld base_config.yaml and Bedrock creds), resumable:
    python examples/agentic/alfworld_search.py --config /path/to/configs/base_config.yaml \
        --num-tasks 50 --max-spend 20 --resume
Smoke-test the pipeline (search, budget, replay, checkpoint, table, figure) with no deps/creds:
    python examples/agentic/alfworld_search.py --mock --num-tasks 12
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from canopy.bandits import ReplayCloneEnv, compare_matched_budget_agent

FIGDIR = Path(__file__).resolve().parents[2] / "paper" / "figures"


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
    """Construct AlfredTWEnv once (the expensive game scan) and return (alfred, sorted_files)."""
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

    def __init__(self, seed: int = 0) -> None:
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
            # Pass the seed so distinct candidate actions at the same state are independent
            # samples (the cache keys on seed); without it, all candidates would collapse to one
            # cached response and value-guided search would see no action diversity.
            text, _, _ = client.generate(model, prompt, temperature=0.7, max_tokens=6, seed=seed)
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


def _bootstrap_ci(hits, iters=2000, seed=0):
    """95% bootstrap CI for the mean of a 0/1 list; returns (mean, lo, hi)."""
    import numpy as np

    a = np.asarray(hits, dtype=float)
    if a.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(iters, a.size))].mean(axis=1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _write_outputs(per_task: dict, model: str, quiet=False):
    """per_task maps task_index(str) -> {'value_guided': {solved,reward,calls}, 'best_of_n': {...}}."""
    if not per_task:
        return
    FIGDIR.mkdir(parents=True, exist_ok=True)
    (FIGDIR / "alfworld_search_results.json").write_text(
        json.dumps({"model": model, "per_task": per_task}, indent=2)
    )

    def _agg(key):
        hits = [v[key]["solved"] for v in per_task.values()]
        calls = [v[key]["calls"] for v in per_task.values()]
        m, lo, hi = _bootstrap_ci(hits, seed=0 if key == "best_of_n" else 1)
        avg_calls = sum(calls) / max(1, len(calls))
        return m, lo, hi, avg_calls

    vg_m, vg_lo, vg_hi, vg_calls = _agg("value_guided")
    bo_m, bo_lo, bo_hi, bo_calls = _agg("best_of_n")
    n = len(per_task)

    tex = (
        "% ALFWorld value-guided vs best-of-N (auto-generated by alfworld_search.py)\n"
        "\\begin{tabular}{lccr}\n\\toprule\n"
        "Planner & Success [95\\% CI] & Avg.\\ calls \\\\\n\\midrule\n"
        f"best-of-N & {bo_m:.3f} [{bo_lo:.2f}, {bo_hi:.2f}] & {bo_calls:.0f} \\\\\n"
        f"\\textbf{{value-guided (ours)}} & \\textbf{{{vg_m:.3f}}} [{vg_lo:.2f}, {vg_hi:.2f}] "
        f"& {vg_calls:.0f} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
    )
    (FIGDIR / "alfworld_search_table.tex").write_text(tex)

    if quiet:
        return
    print(f"\nALFWorld ({n} tasks), model {model}, matched policy-call budget:")
    print(f"  best-of-N    success={bo_m:.3f} [{bo_lo:.2f},{bo_hi:.2f}]  avg calls={bo_calls:.0f}")
    print(f"  value-guided success={vg_m:.3f} [{vg_lo:.2f},{vg_hi:.2f}]  avg calls={vg_calls:.0f}")
    try:
        out = _plot(vg_m, vg_lo, vg_hi, bo_m, bo_lo, bo_hi, n, model)
        print(f"\nwrote json+table to {FIGDIR} and figure to {out} (+ .png)")
    except Exception as e:  # noqa: BLE001
        print(f"\nwrote json+table to {FIGDIR} (figure skipped: {type(e).__name__}: {e})")


def _plot(vg_m, vg_lo, vg_hi, bo_m, bo_lo, bo_hi, n, model):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _plotstyle import PALETTE, save_figure, set_style

    set_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    labels = ["best-of-N", "value-guided\n(ours)"]
    means = [bo_m, vg_m]
    lo = [bo_m - bo_lo, vg_m - vg_lo]
    hi = [bo_hi - bo_m, vg_hi - vg_m]
    ax.bar(labels, means, yerr=[lo, hi], capsize=6,
           color=[PALETTE["orange"], PALETTE["red"]], width=0.6)
    ax.set_ylabel("task success")
    ax.set_ylim(0, 1)
    ax.set_title(f"ALFWorld at matched compute ({n} tasks)")
    fig.tight_layout()
    return str(save_figure(fig, "alfworld_search"))


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
    ap.add_argument("--model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--max-spend", type=float, default=None)
    ap.add_argument("--cache", default="examples/.cache/alfworld_search.jsonl")
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--resume", action="store_true",
                    help="skip tasks already in paper/figures/alfworld_search_results.json")
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
        model_label = "mock"
    else:
        if not args.config or not os.path.exists(args.config):
            print(
                "ALFWorld needs --config pointing to base_config.yaml (or set ALFWORLD_CONFIG). "
                "It ships in the alfworld repo under configs/base_config.yaml. Use --mock to "
                "smoke-test the search pipeline without alfworld."
            )
            return
        try:
            from canopy.llm import BedrockClient, BudgetError, CachingLLMClient

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
            model_label = args.model
        except Exception as e:  # noqa: BLE001
            print(
                f"Could not initialize ALFWorld + LLM ({type(e).__name__}: {e}).\n"
                "Install alfworld in a separate env (see the header), run alfworld-download, "
                "and configure Bedrock. Use --mock to smoke-test the search pipeline."
            )
            return

    print(f"ALFWorld {args.split if not args.mock else 'mock'}: {len(task_indices)} task(s), "
          f"model {model_label}")

    per_task: dict[str, dict] = {}
    if args.resume:
        rp = FIGDIR / "alfworld_search_results.json"
        if rp.exists():
            per_task = json.loads(rp.read_text()).get("per_task", {})
            if per_task:
                print(f"resuming: {len(per_task)} tasks already done")

    start = time.monotonic()
    try:
        from canopy.llm import BudgetError
    except Exception:  # noqa: BLE001
        class BudgetError(Exception):
            pass

    try:
        for i, ti in enumerate(task_indices):
            if str(ti) in per_task:
                continue
            env = ReplayCloneEnv(get_engine(ti))
            try:
                res = compare_matched_budget_agent(
                    env, act, branching=args.branching, rollouts=args.rollouts,
                    rollout_horizon=rollout_horizon, max_steps=args.max_steps,
                )
            except BudgetError as e:
                print(f"[budget stop] {e}")
                break
            except Exception as e:  # noqa: BLE001
                print(f"  [task {ti} skipped] {type(e).__name__}: {str(e)[:140]}")
                continue
            per_task[str(ti)] = {
                name: {
                    "solved": int(res[name].solved),
                    "reward": float(res[name].reward),
                    "calls": res[name].budget.calls,
                }
                for name in ("value_guided", "best_of_n")
            }
            done = len(per_task)
            if done % max(1, args.checkpoint_every) == 0:
                _write_outputs(per_task, model_label, quiet=True)
                el = (time.monotonic() - start) / 60
                print(f"  [ckpt] {done} tasks done ({el:.1f}m)")
    except KeyboardInterrupt:
        print("\n[interrupted] writing partial results")

    _write_outputs(per_task, model_label)
    if client is not None:
        print(f"  budget: {client.stats()}")


if __name__ == "__main__":
    main()
