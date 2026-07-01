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
| `reasoning_search.py` | **Flagship**: real-LLM value-guided search vs. best-of-N at matched compute, over **MATH** (headline) or **GSM8K** (`--benchmark`); sweeps a budget curve with bootstrap CIs. Resumable; writes to `paper/figures/`; `--mock` for offline. | `llm`, `bench`, AWS |
| `gsm8k_diagnostic.py` | Matched-budget GSM8K diagnostic (self-consistency vs. oracle value). | `llm`, `bench`, AWS |

```bash
uv run --extra plot python examples/reasoning/reasoning_search_demo.py
uv run --extra plot python examples/reasoning/agentic_search_demo.py
uv run --extra llm --extra bench python examples/reasoning/reasoning_search.py --benchmark math --resume  # real Bedrock
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

```bash
python examples/systems/vllm_prefix_cache_eval.py --dry-run          # validate harness, no GPU
python examples/systems/vllm_prefix_cache_eval.py --model Qwen/Qwen2.5-7B-Instruct --shift  # on a GPU
```
See `examples/systems/README.md` for the full runbook and the Tier-2 (in-engine adaptive policy) sketch.

## Generated artifacts

Charts (`examples/images/*.png`) and benchmark caches (`*.npz`) are produced by running
the demos and are git-ignored — they are not committed. Re-run the relevant demo to
regenerate them.
