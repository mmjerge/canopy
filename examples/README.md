# Examples

Runnable demos for `canopy`, grouped by topic. Each script is self-contained and
prints a short summary; the ones that draw charts save a PNG into `examples/images/`.

## Layout

```
examples/
├── tree_bandits/   # core multi-fidelity tree-bandit theory (synthetic, numpy-only)
├── llm_routing/    # applied LLM routing, prefix caching, prompt trimming
├── reasoning/      # reasoning-tree and long-horizon agentic search
└── images/         # generated charts (not committed; regenerated on run)
```

## Optional dependencies

Most demos need only `numpy`. Some need optional extras (see `pyproject.toml`):

- `plot` — `matplotlib`, for the demos that render charts.
- `llm` — `boto3`, for demos that call Amazon Bedrock.
- `openai` — `openai`, to use `canopy.llm.OpenAIClient` (set `OPENAI_API_KEY`) in place of Bedrock.
- `anthropic` — `anthropic`, to use `canopy.llm.AnthropicClient` (set `ANTHROPIC_API_KEY`).
- `bench` — `boto3` + `datasets`, for demos that pull a real benchmark (MMLU, GSM8K).

The `llm`/`bench` demos make real, paid model calls and require AWS credentials with
Bedrock access. Start small (small `--n-problems`, small budgets). The demos go through
the provider-agnostic `canopy.llm.LLMClient` interface, so swapping Bedrock for another
provider (e.g. `OpenAIClient`) needs no algorithm changes.

## tree_bandits — core theory (synthetic)

| Demo | What it shows |
| --- | --- |
| `tree_topk_demo.py` | Quick top-k leaf identification at an equal cost budget. |
| `benchmark.py` | When does exploiting the tree beat a strong baseline? (fidelity + budget sweeps) |
| `regret_storage_demo.py` | Online/regret mode: the regret-vs-storage tradeoff and adaptive expansion. |
| `infinite_depth_demo.py` | Infinite-depth case: finite-state compression of the regret-optimal algorithm. |
| `local_lipschitz_demo.py` | Locally-adaptive Lipschitz: tighter constants in smoother subtrees. |
| `jump_robustness_demo.py` | Robustness to jump discontinuities: data-driven bias vs. assumed smoothness. |
| `multiscale_edge_demo.py` | Multiscale (scale-adaptive) tree edge map. |
| `violation_regret_demo.py` | Graceful degradation in the number of Lipschitz violations. |
| `targeted_sampling_demo.py` | Spectral edge isolation → targeted sampling: the sample-efficiency payoff. |

```bash
uv run python examples/tree_bandits/tree_topk_demo.py            # prints, no chart
uv run --extra plot python examples/tree_bandits/benchmark.py    # fidelity + budget sweep
uv run --extra plot python examples/tree_bandits/regret_storage_demo.py
```

## llm_routing — applied (routing, caching, trimming)

| Demo | What it shows | Needs |
| --- | --- | --- |
| `llm_routing_demo.py` | Cost-aware LLM routing over a prefix tree (synthetic). | `plot` |
| `prefix_cache_demo.py` | Prefix-cache management as online tree selection (synthetic). | `plot` |
| `prefix_cache.py` | Prefix caching on a **real** prompt stream (real tokenizer): adaptive vs LRU/LFU/offline. Writes to `paper/figures/`; `--mock` for offline. | `plot` (+`bench` for HF datasets) |
| `mmlu_routing.py` | Real MMLU-by-subject routing over diverse Bedrock models. Writes to `paper/figures/`. | `bench`, `plot`, AWS |
| `prompt_optimization.py` | Redundant-prefix prompt trimming on real Bedrock + MMLU. Resumable; writes to `paper/figures/`; `--mock` for offline. | `bench`, `plot`, AWS |
| `bedrock_routing.py` | Real-LLM routing over Bedrock. | `llm`, AWS |
| `bedrock_benchmark.py` | Real Bedrock benchmark for prefix-tree routing. | `llm`, `plot`, AWS |

```bash
uv run --extra plot python examples/llm_routing/llm_routing_demo.py
uv run --extra plot python examples/llm_routing/prefix_cache.py --corpus-file prompts.txt
uv run --extra bench --extra plot python examples/llm_routing/mmlu_routing.py        # real Bedrock
uv run --extra bench --extra plot python examples/llm_routing/prompt_optimization.py # real Bedrock
```

## reasoning — reasoning-tree and agentic search

| Demo | What it shows | Needs |
| --- | --- | --- |
| `reasoning_search_demo.py` | Value-guided (edge-following) descent vs. best-of-N. | `plot` |
| `agentic_search_demo.py` | Long-horizon agentic search: rollout-guided planning vs. best-of-N. | `plot` |
| `probe_scope_demo.py` | Synthetic phase diagram over (saturation × probe informativeness) — reproduces the sign/size of both the real MATH win and the real MBPP null from one model. | `plot` |
| `reasoning_search.py` | **Flagship**: real-LLM value-guided search vs. best-of-N at matched compute over `--benchmark {math, gsm8k, gpqa, gpqa_diamond, humaneval, mbpp}`; sweeps a budget curve with bootstrap CIs. Resumable; writes to `paper/figures/`; `--mock` for offline. | `llm`, `bench` |
| `combine_reasoning_models.py` | Capability-ladder sweep: the paired value-guided − best-of-N gain across a model ladder. | `llm`, `bench` |
| `mbpp_probe_check.py` | Code ladder rung 2: is a continuous multi-test execution probe informative on MBPP, vs. the one-assert null? Reports the Stage-1 gate correlation before any race. | `llm`, `bench` |
| `swebench_search.py` | Code ladder rung 5: repo-level value-guided search vs. best-of-N on real SWE-bench issues, graded in Docker. `--mock` for offline. | `llm`, Docker + `swebench` |
| `run_swebench_sweep.sh` | Five-model × three-depth SWE-bench capability sweep (long-running; run in `tmux`, resumable). | same as above |
| `terminalbench_search.py` | Code ladder rung 6: the same value-guided machinery on Terminal-Bench 2.x via Harbor, graded in a fresh container. `--mock` for offline. | separate Python 3.12 Harbor venv |
| `gsm8k_diagnostic.py` | Matched-budget GSM8K diagnostic (self-consistency vs. oracle value). | `llm`, `bench` |

See `examples/analysis/` below for the accompanying tree-Lipschitz characterizations
(rung 3–4 of the code ladder) and `docs/code_benchmarks.md` for the full ladder writeup.

```bash
uv run --extra plot python examples/reasoning/reasoning_search_demo.py
uv run --extra plot python examples/reasoning/agentic_search_demo.py
uv run --extra plot python examples/reasoning/probe_scope_demo.py
uv run --extra llm --extra bench python examples/reasoning/reasoning_search.py --benchmark math --mock  # try --mock first
python examples/reasoning/swebench_search.py --mock --n-instances 12
python examples/reasoning/terminalbench_search.py --mock --n-tasks 12
```

## analysis — measuring the prior directly on real data

| Demo | What it shows | Needs |
| --- | --- | --- |
| `theory_link.py` | Sublinear-regret exponent on a synthetic tree-Lipschitz function, and a variance decomposition of the real RouterBench value function by tree resolution. | `bench`, `plot` |
| `reasoning_tree_lipschitz.py` | Is the reasoning value function almost tree-$K$-Lipschitz on real MATH/GPQA traces? Cheap-vs-true node-value correlation + pivotal-step count. | `llm`, `bench` |
| `swebench_stage1.py` | Code ladder rung 3: offline BM25 localization gate on SWE-bench Lite — no API calls, no Docker. | `bench` |
| `swebench_tree_lipschitz.py` | Code ladder rung 4: the same characterization as above, directly on generated *patches* rather than file retrieval. `--mock` for offline. Reuses `swebench_search.py`'s response cache. | `llm`, Docker + `swebench` |

```bash
uv run --extra bench --extra plot python examples/analysis/theory_link.py
python examples/analysis/swebench_stage1.py                       # already measured; free to rerun
python examples/analysis/swebench_tree_lipschitz.py --mock --n-instances 12
```

## agentic — long-horizon real-benchmark search and routing

| Demo | What it shows | Needs |
| --- | --- | --- |
| `taubench_routing.py` | Learned per-call model routing on real tau-bench (retail/airline): the online contextual UCB bandit (regional) vs. flat learner vs. fixed models. Resumable; writes to `paper/figures/`; `--mock` for offline. | `llm`, `tau_bench`, AWS |
| `alfworld_search.py` | Value-guided tree search vs. best-of-N on real ALFWorld (multi-fidelity + tree depth). Resumable; writes to `paper/figures/`; `--mock` for offline. | `llm`, `alfworld`, AWS |
| `textgrid_search.py` | Value-guided vs. best-of-N on a text grid world (no external deps). | `plot` |

```bash
python examples/agentic/taubench_routing.py --mock          # smoke-test, no deps
python examples/agentic/alfworld_search.py --mock --num-tasks 12
```

## systems — real serving-stack eval (GPU)

| Demo | What it shows | Needs |
| --- | --- | --- |
| `systems/vllm_prefix_cache_eval.py` | Real vLLM serving with prefix caching ON vs OFF: TTFT, throughput, hit rate, KV memory on a shared-prefix workload (+ optional popularity `--shift`). Writes to `paper/figures/`; `--dry-run` for offline. | `vllm`, NVIDIA GPU |
| `systems/vllm_policy_eval.py` | GPU-calibrated eviction policy comparison: measures real per-token prefill savings, then replays adaptive/LRU/LFU/offline over a real shared-prefix workload with a popularity shift. `--dry-run` for offline. | `vllm`, NVIDIA GPU |

```bash
python examples/systems/vllm_prefix_cache_eval.py --dry-run          # validate harness, no GPU
python examples/systems/vllm_prefix_cache_eval.py --model Qwen/Qwen2.5-7B-Instruct --shift  # on a GPU
python examples/systems/vllm_policy_eval.py --dry-run
```
See `examples/systems/README.md` for the full runbook and the Tier-2 (in-engine adaptive policy) sketch.

## Generated artifacts

Charts (`examples/images/*.png`) and benchmark caches (`*.npz`) are produced by running
the demos and are git-ignored — they are not committed. Re-run the relevant demo to
regenerate them.
