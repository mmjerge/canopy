# Canopy

[![CI](https://github.com/mmjerge/canopy/actions/workflows/ci.yml/badge.svg)](https://github.com/mmjerge/canopy/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mmjerge/canopy/branch/dev/graph/badge.svg)](https://codecov.io/gh/mmjerge/canopy)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Exploiting smooth tree priors for bandits.** A multi-fidelity bandit over a complete tree
where an internal node's value is the *average* reward of its leaf-subtree. Probing an internal
node is a **cheap but biased** signal (its average underestimates its best leaf); evaluating a
leaf is **expensive but unbiased**. The question this repo answers: *when does trusting the tree
structure (the smoothness prior) let you identify good leaves, or minimize regret, at lower cost
than a structure-blind baseline?*

## Install

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check src/ tests/ examples/
```

Optional extras (stack as many as you need, e.g. `uv sync --extra llm --extra bench --extra plot`):

| Extra | Adds | Needed for |
| --- | --- | --- |
| `plot` | `matplotlib` | any demo that renders a chart |
| `llm` | `boto3` | Amazon Bedrock model calls |
| `bench` | `boto3`, `datasets`, `tiktoken` | pulling real benchmarks (MMLU, MATH, GSM8K, GPQA, SWE-bench, real prompt corpora) |
| `openai` | `openai` | `canopy.llm.OpenAIClient` (set `OPENAI_API_KEY`) |
| `anthropic` | `anthropic` | `canopy.llm.AnthropicClient` (set `ANTHROPIC_API_KEY`) |
| `dev` | `pytest`, `ruff`, `mypy`, ... | running the test suite / linting |

**Setting up on a new machine with AWS access:** the Bedrock client uses boto3's default
credential chain — `aws configure` (or exported `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/
`AWS_SESSION_TOKEN`, or an SSO profile) is all that's needed; every script also accepts
`--region` (default `us-east-1`) and confirm your account has **model access enabled** in the
Bedrock console for whichever model IDs you pass via `--model`.

**Every real-model script shares the same conventions**, so once you know one you know them
all:
- `--cache PATH` — an on-disk JSONL response cache. Re-running a script re-serves already-seen
  (model, prompt) pairs for free; only new calls cost money.
- `--max-spend USD` / `--max-calls N` — hard caps. The script stops cleanly and reports what
  finished; re-run the same command to resume.
- `--mock` / `--dry-run` — exercises the full pipeline (search logic, resume, output writing)
  with a deterministic fake model and no network/credentials. **Always try this first** on a new
  machine to confirm the harness itself works before spending anything.
- Outputs (figures, `.tex` tables, raw JSON) are written to `paper/figures/`.

Start with a small `--n-problems`/`--n-instances` and a tight `--max-spend` on any new script.

## Core idea

* `TreeBandit` (`canopy.bandits.tree`) — the multi-fidelity environment. `leaf_cost` vs
  `probe_cost`; budgets are measured in **cost**. Scenarios: `from_hierarchical_gaussian`,
  `from_adversarial_spikes`, `from_piecewise_smooth`.
* Two regimes: **pure exploration** (identify the top-k leaves at least cost) and
  **regret minimization** (commit each round, compete with the best leaf).
* A data-driven **certificate** (`canopy.bandits.maxmean`) replaces the classically-assumed
  smoothness schedule with a bound estimated online from probes, and a **detect-and-relax**
  hybrid (`canopy.bandits.online.run_hybrid`, `detect_violations`) targets expensive evaluations
  at the cells where that certificate flags a Lipschitz violation.

## Applications

The same tree-bandit machinery instantiates every application below, each mapping a real
problem onto the tree (leaf, arm/region, and budget).

### Serving-stack (depth-one / structural)

| Application | Mapping | Code |
| --- | --- | --- |
| **LLM routing** | leaf = prompt, arm = model, region = subject / token-prefix | `canopy.bandits.routing`; `examples/llm_routing/{mmlu,routerbench,llmrouterbench,routereval,bedrock}_routing.py` |
| **Prefix caching** | cache = ancestor-closed subtree of the token trie, storage = memory budget | `canopy.bandits.prefix_cache`; `examples/llm_routing/prefix_cache{,_demo}.py` — the cleanest instantiation, since the prefix tree **is** the actual cache data structure |
| **Prompt trimming** | arm = trim level, region = subject | `examples/llm_routing/prompt_optimization.py`, `bbh_trim.py`, `longbench_trim.py` |

```bash
uv run --extra plot python examples/llm_routing/llm_routing_demo.py       # routing, synthetic
uv run --extra plot python examples/llm_routing/prefix_cache_demo.py      # caching under drift, synthetic
uv run --extra bench --extra plot python examples/llm_routing/routerbench_routing.py   # real, offline (no API calls)
uv run --extra llm --extra bench --extra plot python examples/llm_routing/mmlu_routing.py --mock  # try --mock first
```

### Reasoning & code search (the multi-fidelity flagship)

leaf = complete solution trace, internal node = partial trace, cheap probe = a rollout / partial
grade, expensive eval = the full grader. `examples/reasoning/reasoning_search.py` runs
value-guided search vs. best-of-N at matched compute across `--benchmark {math, gsm8k, gpqa,
gpqa_diamond, humaneval, mbpp}`; `combine_reasoning_models.py` sweeps a model capability ladder;
`examples/analysis/{theory_link,reasoning_tree_lipschitz}.py` measure whether the value function
is actually tree-Lipschitz on real traces.

```bash
uv run --extra plot python examples/reasoning/reasoning_search_demo.py    # synthetic mechanism check
uv run --extra llm --extra bench python examples/reasoning/reasoning_search.py \
    --benchmark math --mock --n-problems 12                              # try --mock first
```

### The code-generation benchmark ladder

Single-function code (HumanEval/MBPP) is a **predicted null** for this method — no cost gap
between a cheap probe and a full sample, near-saturated for a strong model, and a one-bit probe
signal. The theory predicts the advantage should show up at the *repository* level instead,
where there's a real cost gradient and an informative execution-graded probe. The ladder below
tests that prediction end to end; see **`docs/code_benchmarks.md`** for the full reasoning and
`docs/review_changes_and_runbook.md` for measured results so far.

| # | Benchmark | Script | Needs | Status |
| - | --- | --- | --- | --- |
| 1 | HumanEval / MBPP (one-assert probe) | `reasoning_search.py --benchmark humaneval\|mbpp` | `llm`, `bench` | run — the control/null |
| 2 | MBPP, multi-test execution probe | `mbpp_probe_check.py` | `llm`, `bench` | ready to run |
| 3 | SWE-bench Lite, BM25 localization gate | `analysis/swebench_stage1.py` | `bench` (offline, no API) | **measured**: 33% gold-file recall, MRR 0.12 |
| 4 | SWE-bench, patch-level cheap-vs-true characterization | `analysis/swebench_tree_lipschitz.py` | `llm`, Docker + `pip install swebench` | pipeline verified (`--mock`); real run pending |
| 5 | SWE-bench, repo-level value-guided vs. best-of-N | `reasoning/swebench_search.py`, sweep: `reasoning/run_swebench_sweep.sh` | `llm`, Docker (~120GB disk for prebuilt images) | pipeline verified (`--mock`); real run pending |
| 6 | Terminal-Bench 2.x (Harbor) | `reasoning/terminalbench_search.py` | separate Python **3.12** Harbor venv | pipeline verified (`--mock`); real run pending |

```bash
# always smoke-test first (no deps/creds/Docker needed):
python examples/analysis/swebench_stage1.py            # already measured; reruns free (offline HF dataset)
python examples/reasoning/swebench_search.py --mock --n-instances 12
python examples/reasoning/terminalbench_search.py --mock --n-tasks 12

# real runs (rungs 4-5), once Docker is available and `pip install swebench` succeeds:
python examples/reasoning/swebench_search.py \
    --dataset princeton-nlp/SWE-bench_Verified --n-instances 20 \
    --model us.anthropic.claude-sonnet-4-5-20250929-v1:0 --branching 3 --depth 1 --resume

# the 5-model x 3-depth capability sweep (long-running; run in tmux, resumable):
tmux new-session -d -s swesweep \
    'bash examples/reasoning/run_swebench_sweep.sh 2>&1 | tee logs/swebench_sweep.log'
```

Rung 4 reuses the response cache from rung 5's run, so characterizing the value function is
nearly free once the race itself is underway. **Read the rung-3 gate before spending on rungs
4–5**: if a sharper localization/retrieval probe doesn't clear a similar bar, the theory predicts
the repo-level race won't pay either — fix the probe first.

### Agentic (long-horizon, real environments)

| Demo | What it shows | Needs |
| --- | --- | --- |
| `agentic/taubench_routing.py` | Learned per-call model routing on real τ-bench (retail/airline): the online contextual UCB router (regional) vs. flat learner vs. fixed models | `llm`, `tau_bench` |
| `agentic/alfworld_search.py` | Value-guided tree search vs. best-of-N on real ALFWorld | `llm`, `alfworld` |
| `agentic/textgrid_search.py` | Value-guided vs. best-of-N on a text grid world, no external deps | — |

```bash
python examples/agentic/taubench_routing.py --mock
python examples/agentic/alfworld_search.py --mock --num-tasks 12
```

### Systems (real serving stack, GPU)

| Demo | What it shows | Needs |
| --- | --- | --- |
| `systems/vllm_prefix_cache_eval.py` | Real vLLM serving, prefix caching ON vs. OFF: TTFT, throughput, hit rate, KV memory (+ optional popularity `--shift`) | `vllm`, NVIDIA GPU |
| `systems/vllm_policy_eval.py` | GPU-calibrated eviction policy comparison: adaptive vs. LRU/LFU in realized prefill time | `vllm`, NVIDIA GPU |

```bash
python examples/systems/vllm_prefix_cache_eval.py --dry-run     # validate harness, no GPU
python examples/systems/vllm_policy_eval.py --dry-run
```

See `examples/systems/README.md` for the full runbook.

## What's in the repo

| Module | Role |
| --- | --- |
| `canopy.bandits.tree` | `TreeBandit` multi-fidelity environment |
| `canopy.bandits.topk` | `HierarchicalTopK`, `UniformTopK` (top-k identification) |
| `canopy.bandits.baselines` | `SuccessiveEliminationTopK` (strong structure-blind baseline) |
| `canopy.bandits.online` | regret mode: `run_hoo`, `run_adaptive(_variance)`, `run_fixed_depth`, `run_hybrid`, `run_local_lipschitz`; `detect_violations` |
| `canopy.bandits.maxmean` | noise-deconvolved high-probability bound on `max − mean` (certifies the bias term) |
| `canopy.bandits.rewards` | reward families (hierarchical-Gaussian, adversarial spikes, piecewise-smooth, violation) |
| `canopy.bandits.routing`, `canopy.bandits.prefix_cache` | applied bandit instantiations (LLM routing, prefix caching) |
| `canopy.bandits.reasoning`, `canopy.bandits.reasoning_llm` | synthetic and real-LLM value-guided reasoning search |
| `canopy.bandits.code_eval` | execution-graded code probes (HumanEval/MBPP grading, the continuous multi-test probe, the Stage-1 gate helpers) |
| `canopy.bandits.swebench_eval`, `canopy.bandits.terminalbench_eval` | repo-level and terminal-task grading harnesses |
| `canopy.bandits.agentic`, `canopy.bandits.agentic_llm`, `canopy.bandits.harbor_agent` | long-horizon agentic search (synthetic and real) |
| `canopy.bandits.exp3`, `canopy.experts`, `canopy.algorithms` | classical OCO/MAB baselines (Hazan) |

## Synthetic results (deterministic, reproducible)

These use fixed seeds and are the same every run — a good place to sanity-check a fresh install.

**Multi-fidelity top-k — the prior pays off only with cheap probes.** 1024 leaves, top-5, 30
seeds, budget 1500:

| probe/leaf cost | Hierarchical | SuccElim (strong) |
| --- | --- | --- |
| 1.0 (no advantage) | 0.29 | 0.58 |
| 0.5 | 0.80 | 0.58 |
| 0.05 | 0.84 | 0.58 |

When an internal probe costs as much as a leaf, the descent is wasted overhead and the tree
loses; it overtakes the strong baseline once probes are ≥2× cheaper. (This table reflects the
best-first search variant; the certified-sound variant only *matches* the baseline on this
instance family — see the open item in `docs/review_changes_and_runbook.md`.)

**Regret/memory — adaptive matches full resolution at a fraction of the memory.** 256 leaves,
12k rounds: adaptive reaches near full-leaf regret at ~1/3 the memory (variance-aware: ~6×
less), sitting below the fixed-depth tradeoff frontier.

**Graceful degradation in the number of violations.** The detect-and-relax hybrid dominates
both assume-smooth and structure-blind while violations are sparse, converging to the
structure-blind floor only as violations fill the tree — the `H_edge → H_blind` prediction made
concrete (`docs/violation_regret.md`).

**Certificate coverage.** The data-driven bound on `max − mean` holds at its stated confidence
level across independent calibration trials (`tests/test_maxmean.py`).

```bash
uv run --extra plot python examples/tree_bandits/benchmark.py            # fidelity + budget sweep
uv run --extra plot python examples/tree_bandits/regret_storage_demo.py  # regret vs memory
uv run --extra plot python examples/tree_bandits/violation_regret_demo.py
uv run --extra plot python examples/tree_bandits/lipschitz_ablation.py   # estimated vs. assumed L
```

Real-model results (routing, reasoning, code, agentic, systems) depend on model availability,
pricing, and dataset versions at run time, so they aren't hardcoded here — regenerate them with
the scripts above (they write figures and `.tex` tables into `paper/figures/`) and check
**`docs/review_changes_and_runbook.md`** for the latest measured numbers and what's still
pending.

## Docs

`applications.md` (the applications above), `fixed_budget_bound.md`, `ucb_optimal.md`,
`infinite_tree.md`, `lipschitz_regret.md`, `maxmean_bound.md`, `regret_storage_note.md`,
`violation_regret.md`, `llm_routing.md`, `reasoning_search.md`, `agentic_search.md`,
`targeted_sampling.md`, **`code_benchmarks.md`** (the code-generation ladder in depth),
**`review_changes_and_runbook.md`** (everything found in review, what changed, and the
step-by-step runbook for every experiment still pending).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development
setup and PR workflow, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community
expectations. Security reports: see [SECURITY.md](SECURITY.md). Release notes live in
[CHANGELOG.md](CHANGELOG.md).
