"""Real benchmark: learned per-call model routing on tau-bench, using the paper's bandit.

tau-bench (Sierra) emulates a customer-service agent talking to a simulated user and calling
domain API tools (retail / airline). Each agent turn is a model call; most are easy, a few are
pivotal. We run a faithful Act-style agent loop against the real ``tau_bench`` environment and,
at each turn, route the call to one model from a cost-tiered Bedrock pool -- but the router is
the paper's **online cost-aware contextual UCB bandit** (``canopy.bandits.ContextualUCBRouter``),
NOT a hand-coded heuristic. This is the long-horizon test of the actual method.

How the bandit learns here: the context (region) of a turn is a small discrete feature of its
difficulty/phase (whether the agent is stuck or hit a tool error, and early vs. late in the
episode); the arm is a model; the quality signal is the episode's terminal task reward, credited
back to every (region, model) pull in that episode (Monte-Carlo credit assignment). The learner
picks the arm maximizing ``quality - lam*cost + UCB bonus`` per region, so it discovers -- online,
from reward -- which model tier to use in which kind of turn. We compare against (i) the same
learner with a single region (``flat``, structure-blind: same information, no context) and (ii)
every fixed single model (the cost/quality cloud). Regional beating flat is the evidence that the
tree/region structure -- the paper's contribution -- helps.

Honest limitation: terminal-reward credit assignment is high-variance (a trivial turn in a
successful episode still gets credit); it optimizes exactly the served metric (task success per
cost) and is the standard episodic-bandit choice, but a per-turn value signal would be tighter.

Setup (tau-bench needs its own install; the user simulator runs through LiteLLM -> Bedrock):
    pip install git+https://github.com/sierra-research/tau-bench.git
    pip install -e /path/to/canopy[llm]
Run (override the pool, cheapest first; ALWAYS set a spend cap):
    python examples/agentic/taubench_routing.py --env retail --num-tasks 80 --trials 1 \
        --max-spend 30 --resume
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

from canopy.bandits import ContextualUCBRouter

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001 -- progress bar is optional
    tqdm = None

FIGDIR = Path(__file__).resolve().parents[2] / "paper" / "figures"

# Default ordered pool (cheapest first). Override with --models.
DEFAULT_POOL = [
    "amazon.nova-lite-v1:0",
    "us.meta.llama3-1-70b-instruct-v1:0",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
]
MOCK_CHEAP = "amazon.nova-lite-v1:0"
MOCK_FRONTIER = "us.anthropic.claude-opus-4-7"

N_REGIONS = 4  # difficulty (easy/hard) x phase (early/late)

# User-simulator spend is billed through LiteLLM (tau-bench's user model), separately from the
# agent's CachingLLMClient. We accumulate it via a LiteLLM success callback so the reported
# budget reflects the TRUE Bedrock bill, and so the spend cap can cover it too.
_user_sim_cost = {"usd": 0.0, "calls": 0}


def _register_user_sim_cost_tracker() -> None:
    try:
        import litellm

        def _cb(kwargs, completion_response, start_time, end_time):  # noqa: ANN001
            try:
                c = kwargs.get("response_cost")
                if c is None:
                    c = litellm.completion_cost(completion_response=completion_response)
                _user_sim_cost["usd"] += float(c or 0.0)
                _user_sim_cost["calls"] += 1
            except Exception:  # noqa: BLE001 -- cost unknown for this model; skip
                pass

        litellm.success_callback = [_cb]
    except Exception:  # noqa: BLE001 -- litellm absent; nothing to track
        pass


def _agent_spend(client) -> float:
    if client is None:
        return 0.0
    try:
        return float(client.stats().get("est_spend_usd", 0.0))
    except Exception:  # noqa: BLE001
        return 0.0


def _total_spend(client) -> float:
    return _agent_spend(client) + _user_sim_cost["usd"]


def _router_snapshot(router):
    if router is None:
        return None
    return {"counts": router.counts.tolist(), "sums": router.sums.tolist(), "t": router.t}


def _restore_router(router, snap) -> None:
    if router is None or not snap:
        return
    router.counts = np.asarray(snap["counts"], dtype=float)
    router.sums = np.asarray(snap["sums"], dtype=float)
    router.t = int(snap["t"])


OUT_SUFFIX = ""  # set from --out-tag so replicate runs write to separate files


def _episodes_file() -> str:
    return f"taubench_routing{OUT_SUFFIX}_episodes.json"


def _save_episode_log(in_progress: dict) -> None:
    """Persist per-episode outcomes + router state for in-progress policies (per-episode resume)."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    (FIGDIR / _episodes_file()).write_text(json.dumps({"in_progress": in_progress}))


def _load_episode_log() -> dict:
    p = FIGDIR / _episodes_file()
    if p.exists():
        try:
            return json.loads(p.read_text()).get("in_progress", {})
        except Exception:  # noqa: BLE001
            return {}
    return {}


_HARD_KEYS = ("error", "cannot", "invalid", "not found")


def region_of(step: int, stuck: int, obs: str, n_regions: int) -> int:
    """Discrete context for a turn: difficulty (stuck/tool-error) x phase (early/late).

    With ``n_regions == 1`` this collapses to the flat (structure-blind) learner.
    """
    if n_regions <= 1:
        return 0
    hard = stuck >= 1 or any(k in obs.lower() for k in _HARD_KEYS)
    phase = 1 if step >= 5 else 0
    return (2 if hard else 0) + phase  # 0..3


def _short(model: str) -> str:
    return model.split(".")[-1].replace(":0", "")


def cost_vector(pool, client) -> np.ndarray:
    """Relative per-model cost (in+out price per 1k), normalized to max=1, for the UCB utility."""
    if client is None:
        # mock: cheap vs frontier
        raw = [0.6 if m == MOCK_FRONTIER else 0.02 for m in pool]
    else:
        raw = []
        for m in pool:
            try:
                in_p, out_p = client.price_per_1k(m)
            except Exception:  # noqa: BLE001
                in_p, out_p = 0.001, 0.001
            raw.append(in_p + out_p)
    arr = np.asarray(raw, dtype=float)
    return arr / max(arr.max(), 1e-9)


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
        "You CAN and SHOULD look up product and order details with the tools. To cancel or "
        "modify only SOME items in an order, use the modify-items tool, not the whole-order "
        "cancel tool. Be efficient: do not re-fetch data you already have.\n\n"
        "Reply with ONLY a single JSON object, no prose. To call a tool:\n"
        '  {"name": "<tool_name>", "kwargs": {...}}\n'
        "To reply to the user:\n"
        '  {"name": "respond", "kwargs": {"content": "<message>"}}\n'
    )


def _first_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` object in ``text`` (ignoring braces inside strings)."""
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
                if "temperature" in str(e).lower():
                    text, it, ot = _call(model, prompt, with_temp=False)
                else:
                    raise
            in_p, out_p = client.price_per_1k(model)
            return text, in_p * it / 1000 + out_p * ot / 1000
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ == "BudgetError":
                raise
            if seen_errors["n"] < 5:
                print(f"  [generate error] {model}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                seen_errors["n"] += 1
            return '{"name": "respond", "kwargs": {"content": "..."}}', 0.0

    return generate


def run_episode(env, generate, max_steps, make_action, task_index, *, pool, router=None,
                fixed_model=None, n_regions=1, trace=None):
    """Run one tau-bench task. If ``router`` is given, route per turn with the learned bandit and
    credit the terminal reward back to each (region, model) pull; else use ``fixed_model``.
    Returns (reward, cost, steps)."""
    reset = env.reset(task_index=task_index)
    obs = reset.observation
    sys_prompt = _system_prompt(env.wiki, getattr(env, "rules", []), env.tools_info)
    history = [f"User: {obs}"]
    cost = 0.0
    reward = 0.0
    stuck = 0
    t = 0
    pulls: list[tuple[int, int]] = []
    for t in range(max_steps):
        if router is not None:
            region = region_of(t, stuck, obs, n_regions)
            m_idx = router.select(region)
            model = pool[m_idx]
            pulls.append((region, m_idx))
        else:
            model = fixed_model
        prompt = sys_prompt + "\n\nConversation so far:\n" + "\n".join(history) + "\nAssistant:"
        text, call_cost = generate(model, prompt)
        cost += call_cost
        action = _parse_action(text, make_action)
        resp = env.step(action)
        new_obs, reward, done = resp.observation, resp.reward, resp.done
        stuck = stuck + 1 if new_obs == obs else 0
        history.append(f"Assistant: {text}")
        history.append(f"[{resp.info.source}] {new_obs}")
        if trace:
            trace(f"  [{_short(model)} step {t+1}] {action.name}({action.kwargs}) -> {new_obs[:100]}")
        obs = new_obs
        if done:
            break
    if router is not None:  # Monte-Carlo credit assignment: terminal reward to every pull
        for region, m_idx in pulls:
            router.update(region, m_idx, reward)
    return reward, cost, t + 1


def _bootstrap_ci(hits, iters: int = 2000, seed: int = 0):
    """Bootstrap mean and 95% CI of a 0/1 success list (error bars over tasks, no extra runs)."""
    a = np.asarray(hits, dtype=float)
    if a.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(iters, a.size))].mean(axis=1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _success_ci(results: dict, name: str, curves: dict):
    """Return (mean, lo, hi) success for a policy, bootstrapping over its per-episode hits."""
    hits = results.get(name, {}).get("hits") or curves.get(name)
    if hits:
        return _bootstrap_ci(hits, seed=hash(name) & 0xFFFF)
    s = results.get(name, {}).get("success", 0.0)
    return s, s, s


def _write_outputs(results: dict, curves: dict, env_name: str, n_tasks: int, quiet=False) -> None:
    """results: name -> {success, cost, hits?}. curves: learned-name -> [per-episode 0/1].

    Error bars are a bootstrap over the per-task 0/1 outcomes (valid from a single trial), so we
    report a 95% CI on task success without extra, expensive runs.
    """
    if not results:
        return
    FIGDIR.mkdir(parents=True, exist_ok=True)
    (FIGDIR / f"taubench_routing{OUT_SUFFIX}_results.json").write_text(
        json.dumps({"env": env_name, "n_tasks": n_tasks, "results": results,
                    "learning_curves": curves}, indent=2)
    )
    order = sorted(results, key=lambda k: (not k.startswith("routed"), results[k]["cost"]))
    rows = []
    for name in order:
        r = results[name]
        bold = name.startswith("routed")
        cell = f"\\textbf{{{name}}}" if bold else _short(name)
        m, lo, hi = _success_ci(results, name, curves)
        sval = f"{m:.3f} [{lo:.2f}, {hi:.2f}]"
        s = f"\\textbf{{{sval}}}" if bold else sval
        c = f"\\textbf{{{r['cost']:.5f}}}" if bold else f"{r['cost']:.5f}"
        rows.append(f"{cell} & {s} & {c} \\\\")
    tex = (
        "% tau-bench learned per-call routing (95% CI bootstrapped over tasks; "
        "auto-generated by taubench_routing.py)\n"
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "Policy & Task success [95\\% CI] & Avg.\\ cost/task (\\$) \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (FIGDIR / f"taubench_routing{OUT_SUFFIX}_table.tex").write_text(tex)
    if quiet:
        return
    print(f"\ntau-bench {env_name}: {n_tasks} tasks (95% CI bootstrapped over tasks)")
    print(f"  {'policy':28s} {'success [95% CI]':>26s} {'avg cost ($)':>13s}")
    for name in order:
        m, lo, hi = _success_ci(results, name, curves)
        print(f"  {name:28s} {f'{m:.3f} [{lo:.2f}, {hi:.2f}]':>26s} "
              f"{results[name]['cost']:13.5f}")
    try:
        out = _plot(results, curves, env_name, n_tasks)
        print(f"\nwrote json+table to {FIGDIR} and figure to {out} (+ .png)")
    except Exception as e:  # noqa: BLE001
        print(f"\nwrote json+table to {FIGDIR} (figure skipped: {type(e).__name__}: {e})")


def _plot(results: dict, curves: dict, env_name: str, n_tasks: int) -> str:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _plotstyle import PALETTE, save_figure, set_style

    set_style()
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, r in results.items():
        if name.startswith("routed"):
            continue
        m, lo, hi = _success_ci(results, name, curves)
        axA.errorbar([r["cost"]], [m], yerr=[[m - lo], [hi - m]], fmt="o", color=PALETTE["orange"],
                     ms=6, capsize=2, zorder=3)
        axA.annotate(_short(name), (r["cost"], m), fontsize=7,
                     xytext=(4, 3), textcoords="offset points")
    marks = {"routed-regional (ours)": (PALETTE["red"], "*", 220),
             "routed-flat": (PALETTE["blue"], "s", 90)}
    for name, (col, mk, sz) in marks.items():
        if name in results:
            m, lo, hi = _success_ci(results, name, curves)
            axA.errorbar([results[name]["cost"]], [m], yerr=[[m - lo], [hi - m]], fmt=mk,
                         color=col, ms=np.sqrt(sz), capsize=3, zorder=5, label=name)
    axA.set_xscale("log")
    axA.set_xlabel("avg cost per task (USD)")
    axA.set_ylabel("task success")
    axA.set_title(f"tau-bench {env_name}: learned routing vs. fixed models")
    axA.legend(loc="lower right")
    for name, (col, mk, sz) in marks.items():
        if name in curves and curves[name]:
            hits = np.asarray(curves[name], dtype=float)
            roll = np.cumsum(hits) / np.arange(1, len(hits) + 1)
            axB.plot(np.arange(1, len(hits) + 1), roll, color=col, label=name)
    axB.set_xlabel("episodes seen")
    axB.set_ylabel("task success (running mean)")
    axB.set_title("Online learning curve: regional vs. flat")
    axB.legend(loc="lower right")
    fig.tight_layout()
    return str(save_figure(fig, f"taubench_routing{OUT_SUFFIX}"))


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
    """Toy 3-call task: look up, resolve an 'invalid' (pivotal) case, then confirm."""

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
        # the pivotal 'resolve' step needs the frontier model; cheap model often flubs it
        if want == "resolve" and model == MOCK_CHEAP and rng.random() > 0.2:
            return '{"name": "respond", "kwargs": {"content": "could you clarify?"}}', cost
        return json.dumps({"name": want, "kwargs": {}}), cost

    return generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="retail", choices=["retail", "airline"])
    ap.add_argument("--num-tasks", type=int, default=-1, help="number of tasks; -1 = full split")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--trials", type=int, default=1, help="repeats per task (more data to learn)")
    ap.add_argument("--task-split", default="test")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--user-model", default="bedrock/amazon.nova-pro-v1:0")
    ap.add_argument("--models", default=",".join(DEFAULT_POOL), help="pool, cheapest first")
    ap.add_argument("--lam", type=float, default=0.3, help="cost weight in the UCB net utility")
    ap.add_argument("--ucb-c", type=float, default=0.4, help="UCB exploration constant")
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--cache", default="examples/.cache/taubench_routing.jsonl")
    ap.add_argument("--max-spend", type=float, default=None)
    ap.add_argument("--mock", action="store_true", help="toy env + mock models, no deps")
    ap.add_argument("--resume", action="store_true",
                    help="skip policies already in paper/figures/taubench_routing_results.json")
    ap.add_argument("--verbose", action="store_true", help="trace the first task of each policy")
    ap.add_argument("--learners-only", action="store_true",
                    help="run only the two learned routers (skip the fixed-model cloud), for "
                         "seed-variance replicates where the baselines are already measured")
    ap.add_argument("--out-tag", default="",
                    help="suffix for output/episode filenames, e.g. 'rep2', so replicate runs "
                         "do not overwrite the main results")
    args = ap.parse_args()

    global OUT_SUFFIX
    OUT_SUFFIX = f"_{args.out_tag}" if args.out_tag else ""

    if args.mock:
        rng = np.random.default_rng(0)
        generate = make_mock_generate(rng)
        client = None
        make_action = _MockAction
        pool = [MOCK_CHEAP, MOCK_FRONTIER]

        def make_env(_i):
            return _MockTauEnv()

        task_indices = list(range(args.num_tasks if args.num_tasks > 0 else 40))
    else:
        try:
            from canopy.llm import BedrockClient, CachingLLMClient
            from tau_bench.envs import get_env
            from tau_bench.types import Action

            try:
                import litellm

                litellm.request_timeout = 60
                litellm.num_retries = 2
            except Exception:  # noqa: BLE001
                pass
            _register_user_sim_cost_tracker()

            client = CachingLLMClient(
                BedrockClient(region=args.region, max_tokens=512), args.cache,
                max_spend_usd=args.max_spend,
            )
            generate = make_llm_generate(client)
            pool = [m.strip() for m in args.models.split(",") if m.strip()]

            def make_action(name, kwargs):
                return Action(name=name, kwargs=kwargs)

            def make_env(task_index):
                return get_env(env_name=args.env, user_strategy="llm", user_model=args.user_model,
                               user_provider="bedrock", task_split=args.task_split,
                               task_index=task_index)

            if args.num_tasks > 0:
                task_indices = list(range(args.start_index, args.start_index + args.num_tasks))
            else:
                total = len(make_env(args.start_index).tasks)
                task_indices = list(range(args.start_index, total))
        except Exception as e:  # noqa: BLE001
            print(f"Could not init tau-bench + Bedrock ({type(e).__name__}: {e}); try --mock.")
            return


    costs = cost_vector(pool, client)

    # Policies: fixed models (the cloud) + two ONLINE LEARNED routers (regional vs flat).
    def make_regional():
        return ContextualUCBRouter(len(pool), costs, n_regions=N_REGIONS, lam=args.lam, c=args.ucb_c)

    def make_flat():
        return ContextualUCBRouter(len(pool), costs, n_regions=1, lam=args.lam, c=args.ucb_c)

    policies: dict[str, dict] = {}
    # Learned routers FIRST so the method result is protected if the spend cap trips during the
    # (expensive) fixed-model cloud sweep that follows.
    policies["routed-regional (ours)"] = {"kind": "router", "make": make_regional, "nreg": N_REGIONS}
    policies["routed-flat"] = {"kind": "router", "make": make_flat, "nreg": 1}
    if not args.learners_only:
        for m in pool:
            policies[_short(m)] = {"kind": "fixed", "model": m}

    print(f"tau-bench {args.env}: {len(task_indices)} tasks x {args.trials} trial(s) "
          f"over {len(pool)} models; learned routing via ContextualUCBRouter")
    print("  pool (cheap->expensive): " + ", ".join(_short(m) for m in pool))

    env = make_env(args.start_index)

    results: dict = {}
    curves: dict = {}
    if args.resume:
        rp = FIGDIR / f"taubench_routing{OUT_SUFFIX}_results.json"
        if rp.exists():
            loaded = json.loads(rp.read_text())
            results = {k: v for k, v in loaded.get("results", {}).items() if k in policies}
            curves = {k: v for k, v in loaded.get("learning_curves", {}).items() if k in policies}
            if results:
                print(f"resuming: {len(results)} policies done, skipping: {list(results)}")
    episode_log: dict = _load_episode_log() if args.resume else {}
    # drop any partial log for policies already fully done
    episode_log = {k: v for k, v in episode_log.items() if k in policies and k not in results}

    remaining = [p for p in policies if p not in results]
    total_eps = len(remaining) * len(task_indices) * args.trials
    prior_eps = sum(len(episode_log.get(p, {}).get("episodes", [])) for p in remaining)
    use_bar = tqdm is not None and not args.verbose
    bar = tqdm(total=total_eps, initial=prior_eps, unit="ep", desc="tau-bench") if use_bar else None

    def log(msg: str) -> None:
        (bar.write if bar is not None else print)(msg)

    def budget_line() -> str:
        return (f"agent ${_agent_spend(client):.4f} + user-sim ${_user_sim_cost['usd']:.4f} "
                f"({_user_sim_cost['calls']} calls) = total ${_total_spend(client):.4f}")

    stopped = False
    start = time.monotonic()
    ep_done = prior_eps
    try:
        for name, spec in policies.items():
            if name in results:
                continue
            router = spec["make"]() if spec["kind"] == "router" else None
            entry = episode_log.get(name, {"episodes": [], "router": None})
            done_pairs = {(e[0], e[1]) for e in entry["episodes"]}
            succ = sum(e[2] for e in entry["episodes"])
            cost = sum(e[3] for e in entry["episodes"])
            runs = len(entry["episodes"])
            hits = [int(e[2] >= 1.0) for e in entry["episodes"]] if router is not None else []
            if router is not None:
                _restore_router(router, entry.get("router"))
            if runs:
                log(f"  resuming {name}: {runs} episodes already done")
            for k, ti in enumerate(task_indices):
                for trial in range(args.trials):
                    if (ti, trial) in done_pairs:
                        continue
                    tr = (lambda m: print(m, flush=True)) if (args.verbose and k == 0 and trial == 0) else None
                    try:
                        r, c, _ = run_episode(
                            env, generate, args.max_steps, make_action, task_index=ti, pool=pool,
                            router=router, fixed_model=spec.get("model"),
                            n_regions=spec.get("nreg", 1), trace=tr,
                        )
                    except Exception as e:  # noqa: BLE001
                        if type(e).__name__ == "BudgetError":
                            log(f"\n[budget stop] {e}")
                            stopped = True
                            break
                        log(f"  [task {ti} trial {trial} skipped] {type(e).__name__}: {str(e)[:120]}")
                        r, c = 0.0, 0.0
                    succ += r
                    cost += c
                    runs += 1
                    ep_done += 1
                    entry["episodes"].append([ti, trial, r, c])
                    if router is not None:
                        hits.append(int(r >= 1.0))
                        entry["router"] = _router_snapshot(router)
                    episode_log[name] = entry
                    _save_episode_log(episode_log)  # per-episode resume (cheap, small file)
                    if bar is not None:
                        bar.update(1)
                        bar.set_postfix_str(f"{name} succ={succ/runs:.2f} ${cost/runs:.4f}")
                    if args.max_spend is not None and _total_spend(client) >= args.max_spend:
                        log(f"\n[spend cap] total est ${_total_spend(client):.2f} "
                            f">= ${args.max_spend:.2f} ({budget_line()})")
                        stopped = True
                        break
                    if args.checkpoint_every and ep_done % args.checkpoint_every == 0:
                        snap = dict(results)
                        snap[name] = {
                            "success": succ / runs, "cost": cost / runs,
                            "hits": [int(e[2] >= 1.0) for e in entry["episodes"]],
                        }
                        snap_c = dict(curves)
                        if router is not None:
                            snap_c[name] = hits
                        _write_outputs(snap, snap_c, args.env, len(task_indices), quiet=True)
                        el = (time.monotonic() - start) / 60
                        log(f"  [ckpt {ep_done}/{total_eps}] {name}: succ={succ/runs:.3f} "
                            f"${cost/runs:.4f} | {el:.0f}m | {budget_line()}")
                if stopped:
                    break
            if runs and not stopped:  # policy finished: promote to results, clear its episode log
                results[name] = {
                    "success": succ / runs, "cost": cost / runs,
                    "hits": [int(e[2] >= 1.0) for e in entry["episodes"]],  # per-task, for CI
                }
                if router is not None:
                    curves[name] = hits
                episode_log.pop(name, None)
                _save_episode_log(episode_log)
                log(f"  done {name}: success={succ/runs:.3f} avg_cost=${cost/runs:.5f}")
                if client is not None:
                    log(f"    budget so far: {budget_line()}")
                _write_outputs(results, curves, args.env, len(task_indices), quiet=True)
            elif runs and stopped:
                log(f"  {name} partial ({runs} eps) saved; rerun with --resume to continue it")
            if stopped:
                break
    except KeyboardInterrupt:
        log("\n[interrupted] partial progress saved; rerun with --resume to continue")
    finally:
        if bar is not None:
            bar.close()

    _write_outputs(results, curves, args.env, len(task_indices))
    if client is not None:
        print(f"final budget: {budget_line()}")


if __name__ == "__main__":
    main()
