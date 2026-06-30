"""Path B (real benchmark): per-call model routing on tau-bench, with a Bedrock model pool.

tau-bench (Sierra) emulates a customer-service agent talking to a simulated user and calling
domain API tools (retail / airline). Each agent turn is a model call; most are easy, a few are
pivotal. We run a faithful Act-style agent loop against the real ``tau_bench`` environment and,
at each turn, route the call to one model from a cost-tiered Bedrock pool. Success is tau-bench's
task reward; cost is the realized agent spend.

We report every fixed single model (the cost/quality "cloud") plus a tiered routing policy that
starts on the cheapest model and escalates up the pool when a turn looks hard (tool error or no
progress). The story mirrors the RouterBench / MMLU routing experiments: the router recovers
most of the strongest model's task success at a fraction of its cost, dominating the fixed-model
cloud, the same cost-vs-quality routing decision, made per call inside a long-horizon trajectory.

Setup (tau-bench needs its own install; the user simulator runs through LiteLLM, which supports
Bedrock):
    pip install git+https://github.com/sierra-research/tau-bench.git
    pip install -e /path/to/canopy[llm]
Run (override the pool, cheapest first):
    python examples/agentic/taubench_routing.py --env retail --num-tasks 40 \
        --models amazon.nova-lite-v1:0,us.meta.llama3-1-70b-instruct-v1:0,\
us.anthropic.claude-sonnet-4-5-20250929-v1:0
Smoke test without tau-bench or credentials:
    python examples/agentic/taubench_routing.py --mock
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001 -- progress bar is optional
    tqdm = None

# paper/figures (canopy root is two levels up from examples/agentic/).
FIGDIR = Path(__file__).resolve().parents[2] / "paper" / "figures"

# Default ordered pool (cheapest first). Override with --models. Edit to whatever you can invoke.
DEFAULT_POOL = [
    "amazon.nova-lite-v1:0",
    "us.meta.llama3-1-70b-instruct-v1:0",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
]
# Mock pool (two tiers) for the dependency-free smoke test.
MOCK_CHEAP = "amazon.nova-lite-v1:0"
MOCK_FRONTIER = "us.anthropic.claude-opus-4-7"


class TieredRouter:
    """Stateful per-call router: start on the cheapest model, climb the pool when struggling.

    Escalates one tier after ``patience`` consecutive "hard" turns (a tool error or no progress),
    and stays there. Reset at the start of each episode. This is the multi-model generalization
    of cheap-by-default / escalate-on-difficulty.
    """

    def __init__(self, pool: list[str], patience: int = 1):
        self.pool = pool
        self.patience = patience
        self.tier = 0
        self.bad = 0

    def reset(self) -> None:
        self.tier = 0
        self.bad = 0

    def __call__(self, obs: str, step: int, stuck: int) -> str:
        hard = any(k in obs.lower() for k in ("error", "cannot", "invalid", "not found"))
        if hard or stuck >= 1:
            self.bad += 1
            if self.bad >= self.patience and self.tier < len(self.pool) - 1:
                self.tier += 1
                self.bad = 0
        return self.pool[self.tier]


def _short(model: str) -> str:
    return model.split(".")[-1].replace(":0", "")


def _system_prompt(wiki: str, rules, tools_info) -> str:
    tool_lines = []
    for t in tools_info:
        fn = t.get("function", t)
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = json.dumps(fn.get("parameters", {}).get("properties", {}))
        tool_lines.append(f"- {name}: {desc} params={params}")
    rules_text = "\n".join(rules) if isinstance(rules, (list, tuple)) else str(rules or "")
    return (
        "You are a customer-service agent. Follow the domain policy strictly.\n\n"
        f"POLICY:\n{wiki}\n{rules_text}\n\n"
        "TOOLS (call at most one per turn):\n" + "\n".join(tool_lines) + "\n\n"
        "CRITICAL: To make ANY change (return, exchange, cancel, modify an order, update an "
        "address or payment), you MUST call the corresponding tool. Telling the user you will "
        "do it does NOT perform it and counts as failure. Authenticate the user, look up the "
        "ids you need, then CALL the action tool to execute the change. Do not end the "
        "conversation until every requested change has actually been executed via a tool call.\n"
        "You CAN and SHOULD look up product and order details with the tools (get/list product "
        "details, get order details) to find items, prices, and ids; never refuse a lookup. To "
        "cancel or modify only SOME items in an order, use the modify-items tool, not the "
        "whole-order cancel tool. Be efficient: do not re-fetch data you already have.\n\n"
        "Reply with ONLY a single JSON object, no prose. To call a tool:\n"
        '  {"name": "<tool_name>", "kwargs": {...}}\n'
        "To reply to the user:\n"
        '  {"name": "respond", "kwargs": {"content": "<message>"}}\n'
    )


def _first_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` object in ``text`` (ignoring braces inside strings).

    Models sometimes emit one valid action then hallucinate the rest of the conversation; we
    want only that first action, not a greedy match spanning several objects.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_action(text: str, make_action):
    """Parse the model's text into an Action (fallback: respond with the raw text)."""
    blob = _first_json_object(text)
    if blob is not None:
        try:
            obj = json.loads(blob)
            name = obj.get("name", "respond")
            kwargs = obj.get("kwargs", {}) or {}
            if name == "respond" and "content" not in kwargs:
                kwargs = {"content": text}
            return make_action(name, kwargs)
        except (json.JSONDecodeError, TypeError):
            pass
    return make_action("respond", {"content": text})


def run_taubench_episode(
    env, choose_model, generate, max_steps, make_action, task_index=None, trace=None
):
    """Run one tau-bench task with a routed Act agent; return (reward, cost, steps)."""
    if hasattr(choose_model, "reset"):
        choose_model.reset()
    # Reset to the SPECIFIC task (tau-bench's no-arg reset picks a random, sometimes
    # out-of-range, index via randint(0, len(tasks))).
    reset = env.reset(task_index=task_index)
    obs = reset.observation
    sys_prompt = _system_prompt(env.wiki, getattr(env, "rules", []), env.tools_info)
    history = [f"User: {obs}"]
    cost = 0.0
    reward = 0.0
    stuck = 0
    t = 0
    if trace:
        trace(f"  TASK: {obs[:200]}")
    for t in range(max_steps):
        model = choose_model(obs, t, stuck)
        prompt = sys_prompt + "\n\nConversation so far:\n" + "\n".join(history) + "\nAssistant:"
        text, call_cost = generate(model, prompt)
        cost += call_cost
        action = _parse_action(text, make_action)
        resp = env.step(action)
        new_obs = resp.observation
        reward, done = resp.reward, resp.done
        stuck = stuck + 1 if new_obs == obs else 0
        history.append(f"Assistant: {text}")
        history.append(f"[{resp.info.source}] {new_obs}")
        if trace:
            trace(
                f"  [{_short(model)} step {t+1}] {action.name}({action.kwargs}) -> {new_obs[:120]}"
            )
        obs = new_obs
        if done:
            break
    if trace:
        trace(f"  DONE reward={reward} steps={t+1} cost=${cost:.4f}")
    return reward, cost, t + 1


def make_llm_generate(client):
    seen_errors = {"n": 0}

    def _call(model, prompt, with_temp):
        if with_temp:
            return client.generate(model, prompt, temperature=0.0, max_tokens=512)
        return client.generate(model, prompt, max_tokens=512)

    def generate(model: str, prompt: str):
        try:
            try:
                text, it, ot = _call(model, prompt, with_temp=True)
            except Exception as e:  # noqa: BLE001
                # Some reasoning models (e.g. Claude Opus 4.x) reject `temperature`; retry without.
                if "temperature" in str(e).lower():
                    text, it, ot = _call(model, prompt, with_temp=False)
                else:
                    raise
            in_p, out_p = client.price_per_1k(model)
            return text, in_p * it / 1000 + out_p * ot / 1000
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ == "BudgetError":  # spend cap hit -> stop the whole run
                raise
            if seen_errors["n"] < 5:
                print(f"  [generate error] {model}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                seen_errors["n"] += 1
            return '{"name": "respond", "kwargs": {"content": "..."}}', 0.0

    return generate


def _write_outputs(results: dict, env_name: str, n_tasks: int, quiet: bool = False) -> None:
    """Console table + LaTeX table + figure + JSON. results: name -> (success, cost)."""
    if not results:
        return
    if not quiet:
        print(f"\ntau-bench {env_name}: {n_tasks} tasks")
        print(f"  {'policy':28s} {'success':>8s} {'avg cost ($)':>13s}")
        for name, (succ, cost) in results.items():
            print(f"  {name:28s} {succ:8.3f} {cost:13.5f}")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _plotstyle import FIGURE_DIR, PALETTE, save_figure, set_style

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIGURE_DIR / "taubench_routing_results.json").write_text(
        json.dumps({"env": env_name, "n_tasks": n_tasks, "results": results}, indent=2)
    )

    rows = []
    for name, (succ, cost) in results.items():
        bold = name.startswith("routed")
        cell = f"\\textbf{{{name}}}" if bold else _short(name)
        s = f"\\textbf{{{succ:.3f}}}" if bold else f"{succ:.3f}"
        c = f"\\textbf{{{cost:.5f}}}" if bold else f"{cost:.5f}"
        rows.append(f"{cell} & {s} & {c} \\\\")
    tex = (
        "% tau-bench agentic routing (auto-generated by taubench_routing.py)\n"
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "Policy & Task success & Avg.\\ cost/task (\\$) \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIGURE_DIR / "taubench_routing_table.tex").write_text(tex)

    set_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for name, (succ, cost) in results.items():
        if name.startswith("routed"):
            continue
        ax.scatter([cost], [succ], color=PALETTE["orange"], s=40, zorder=4)
        ax.annotate(
            _short(name), (cost, succ), fontsize=7, xytext=(4, 3), textcoords="offset points"
        )
    routed = sorted((c, s) for n, (s, c) in results.items() if n.startswith("routed"))
    if len(routed) > 1:
        ax.plot(
            [c for c, _ in routed],
            [s for _, s in routed],
            "-*",
            color=PALETTE["red"],
            markersize=13,
            zorder=5,
            label="routed (ours)",
        )
    elif routed:
        ax.scatter(
            [routed[0][0]],
            [routed[0][1]],
            color=PALETTE["red"],
            s=160,
            marker="*",
            zorder=5,
            label="routed (ours)",
        )
    ax.set_xscale("log")
    ax.set_xlabel("avg cost per task (USD)")
    ax.set_ylabel("task success")
    ax.set_title(f"tau-bench {env_name}: per-call routing vs. fixed models ({n_tasks} tasks)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = save_figure(fig, "taubench_routing")
    if not quiet:
        print(
            f"\nwrote table to {FIGURE_DIR}/taubench_routing_table.tex "
            f"and figure to {out} (+ .png)"
        )


# --- mock tau-bench-like env + models (deterministic; runnable without tau_bench) ---------


class _MockAction:
    def __init__(self, name, kwargs):
        self.name = name
        self.kwargs = kwargs


class _MockInfo:
    source = "user"


class _MockResp:
    def __init__(self, observation, reward, done):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = _MockInfo()


class _MockTauEnv:
    """Toy 3-call task: look up, then resolve an 'invalid' (pivotal) case, then confirm."""

    wiki = "Help the user. Use tools. Resolve issues before confirming."
    rules = ["Never confirm before resolving."]
    tools_info = [
        {"function": {"name": "lookup", "description": "look up account", "parameters": {}}},
        {"function": {"name": "resolve", "description": "resolve an issue", "parameters": {}}},
        {"function": {"name": "confirm", "description": "confirm resolution", "parameters": {}}},
    ]

    def __init__(self):
        self.i = 0
        self.script = ["lookup", "resolve", "confirm"]

    def reset(self, task_index=None):
        self.i = 0
        return _MockResp("The account shows an invalid charge. Please fix it.", 0.0, False)

    def step(self, action):
        want = self.script[self.i] if self.i < len(self.script) else None
        if action.name == want:
            self.i += 1
            if self.i >= len(self.script):
                return _MockResp("Done. Thank you. ###STOP###", 1.0, True)
            return _MockResp(f"Step {self.i} ok. Continue.", 0.0, False)
        return _MockResp("Error: that did not work.", 0.0, False)


def make_mock_generate(rng):
    def generate(model, prompt):
        cost = 0.0006 if model == MOCK_FRONTIER else 0.00002
        n_ok = prompt.count("ok. Continue")
        want = ["lookup", "resolve", "confirm"][min(n_ok, 2)]
        if want == "resolve" and model == MOCK_CHEAP and rng.random() > 0.2:
            return '{"name": "respond", "kwargs": {"content": "could you clarify?"}}', cost
        return json.dumps({"name": want, "kwargs": {}}), cost

    return generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="retail", choices=["retail", "airline"])
    ap.add_argument("--num-tasks", type=int, default=-1, help="number of tasks; -1 = full split")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument(
        "--trials", type=int, default=1, help="repeats per task (averages user-sim RNG)"
    )
    ap.add_argument("--task-split", default="test")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--user-model", default="bedrock/amazon.nova-pro-v1:0")
    ap.add_argument(
        "--models",
        default=",".join(DEFAULT_POOL),
        help="comma-separated model pool, cheapest first",
    )
    ap.add_argument(
        "--patience", type=int, default=1, help="hard turns before the router escalates"
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="dump a progress JSON + log line every N episodes (0 = off)",
    )
    ap.add_argument(
        "--patience-sweep",
        default="",
        help="comma-separated router patience values to trace a curve, e.g. '0,1,2,3'",
    )
    ap.add_argument("--cache", default="examples/.cache/taubench_routing.jsonl")
    ap.add_argument("--max-spend", type=float, default=None)
    ap.add_argument("--mock", action="store_true", help="toy env + mock models, no deps")
    ap.add_argument("--verbose", action="store_true", help="trace the first task of each policy")
    args = ap.parse_args()

    if args.mock:
        rng = np.random.default_rng(0)
        generate = make_mock_generate(rng)
        client = None
        make_action = _MockAction
        pool = [MOCK_CHEAP, MOCK_FRONTIER]

        def make_env(_i):
            return _MockTauEnv()

        task_indices = list(range(args.num_tasks if args.num_tasks > 0 else 5))
    else:
        try:
            from canopy.llm import BedrockClient, CachingLLMClient
            from tau_bench.envs import get_env
            from tau_bench.types import Action

            client = CachingLLMClient(
                BedrockClient(region=args.region, max_tokens=512),
                args.cache,
                max_spend_usd=args.max_spend,
            )
            generate = make_llm_generate(client)
            pool = [m.strip() for m in args.models.split(",") if m.strip()]

            def make_action(name, kwargs):
                return Action(name=name, kwargs=kwargs)

            def make_env(task_index):
                return get_env(
                    env_name=args.env,
                    user_strategy="llm",
                    user_model=args.user_model,
                    user_provider="bedrock",
                    task_split=args.task_split,
                    task_index=task_index,
                )

            if args.num_tasks > 0:
                task_indices = list(range(args.start_index, args.start_index + args.num_tasks))
            else:
                total = len(make_env(args.start_index).tasks)  # full split
                task_indices = list(range(args.start_index, total))
        except Exception as e:  # noqa: BLE001
            print(f"Could not init tau-bench + Bedrock ({type(e).__name__}: {e}); try --mock.")
            return

    # One policy per fixed model (the cloud) plus the tiered router (optionally swept).
    policies: dict = {}
    for model in pool:
        policies[_short(model)] = lambda obs, t, stuck, mdl=model: mdl
    if args.patience_sweep.strip():
        for p in [int(x) for x in args.patience_sweep.split(",") if x.strip() != ""]:
            policies[f"routed(p={p})"] = TieredRouter(pool, patience=p)
    else:
        policies["routed"] = TieredRouter(pool, patience=args.patience)

    print(
        f"tau-bench {args.env}: {len(task_indices)} tasks x {args.trials} trial(s) "
        f"over {len(pool)} models"
    )
    print("  pool (cheap->expensive): " + ", ".join(_short(m) for m in pool))

    env = make_env(args.start_index)  # build once; reset(task_index=...) selects the task

    total_eps = len(policies) * len(task_indices) * args.trials
    use_bar = tqdm is not None and not args.verbose
    bar = tqdm(total=total_eps, unit="ep", desc="tau-bench") if use_bar else None

    def log(msg: str) -> None:
        if bar is not None:
            bar.write(msg)
        else:
            print(msg, flush=True)

    results: dict = {}
    stopped = False
    start = time.monotonic()
    ep_done = 0
    try:
        for name, choose in policies.items():
            succ = cost = 0.0
            runs = 0
            for k, ti in enumerate(task_indices):
                for trial in range(args.trials):
                    tr = (
                        (lambda m: print(m, flush=True))
                        if (args.verbose and k == 0 and trial == 0)
                        else None
                    )
                    try:
                        r, c, _ = run_taubench_episode(
                            env,
                            choose,
                            generate,
                            args.max_steps,
                            make_action,
                            task_index=ti,
                            trace=tr,
                        )
                    except Exception as e:  # noqa: BLE001
                        if type(e).__name__ == "BudgetError":
                            log(f"\n[budget stop] {e}")
                            stopped = True
                            break
                        log(
                            f"  [task {ti} trial {trial} skipped] "
                            f"{type(e).__name__}: {str(e)[:140]}"
                        )
                        r, c = 0.0, 0.0
                    succ += r
                    cost += c
                    runs += 1
                    ep_done += 1
                    if bar is not None:
                        bar.update(1)
                        bar.set_postfix_str(f"{name} succ={succ / runs:.2f} ${cost / runs:.4f}")
                    if args.checkpoint_every and ep_done % args.checkpoint_every == 0:
                        elapsed = time.monotonic() - start
                        rate = ep_done / max(1e-9, elapsed)
                        eta_h = (total_eps - ep_done) / max(1e-9, rate) / 3600
                        snap = dict(results)
                        snap[name] = (succ / runs, cost / runs)
                        FIGDIR.mkdir(parents=True, exist_ok=True)
                        (FIGDIR / "taubench_routing_progress.json").write_text(
                            json.dumps(
                                {
                                    "env": args.env,
                                    "episodes_done": ep_done,
                                    "episodes_total": total_eps,
                                    "elapsed_min": round(elapsed / 60, 1),
                                    "eta_hours": round(eta_h, 2),
                                    "results_so_far": snap,
                                    "in_progress": {
                                        "policy": name,
                                        "tasks_done": runs,
                                        "success": round(succ / runs, 4),
                                        "avg_cost": round(cost / runs, 6),
                                    },
                                },
                                indent=2,
                            )
                        )
                        log(
                            f"  [ckpt {ep_done}/{total_eps}] {name}: {runs} eps "
                            f"succ={succ / runs:.3f} ${cost / runs:.4f} | "
                            f"elapsed={elapsed / 60:.0f}m eta={eta_h:.1f}h"
                        )
                if stopped:
                    break
            if runs:
                results[name] = (succ / runs, cost / runs)
                log(f"  done {name}: success={succ / runs:.3f} avg_cost=${cost / runs:.5f}")
                if client is not None:
                    log(f"    budget so far: {client.stats()}")
                _write_outputs(results, args.env, len(task_indices), quiet=True)  # checkpoint
            if stopped:
                break
    except KeyboardInterrupt:
        log("\n[interrupted] writing partial results")
    finally:
        if bar is not None:
            bar.close()

    _write_outputs(results, args.env, len(task_indices))


if __name__ == "__main__":
    main()
