# Systems eval — vLLM prefix caching

Turns the token-reuse proxy (`examples/llm_routing/prefix_cache.py`) into a **real serving
measurement**: launches a vLLM server with automatic prefix caching ON vs OFF, drives the same
shared-prefix request trace against each, and records TTFT, latency, throughput, prefix-cache hit
rate, and KV-cache memory into `paper/figures/`.

## Requirements
- An NVIDIA GPU (A10/L4 for ~7–8B; A100 for larger).
- `pip install vllm requests matplotlib` (matplotlib only for the figure).
- HF access/token if the model is gated (default `Qwen/Qwen2.5-7B-Instruct` is open).

## Run (Tier 1 — the premise: reuse → real latency/memory)
```bash
python examples/systems/vllm_prefix_cache_eval.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-prompts 2000 --n-templates 16 --pad-tokens 200 \
    --request-rate 20 --max-tokens 64 --shift
```
Knobs: `--pad-tokens` (longer shared prefixes → larger prefill savings), `--request-rate`
(offered load; 0 = as fast as possible), `--shift` (permute template popularity at the midpoint
for the non-stationary case), `--corpus-file prompts.txt` (use a real prompt log instead of the
templated workload), `--extra-arg` (pass through any `vllm serve` flag, repeatable).

Validate the harness with **no GPU** (mocked timings; exercises trace + aggregation + outputs):
```bash
python examples/systems/vllm_prefix_cache_eval.py --dry-run
```

Outputs: `vllm_prefix_cache_results.json`, `vllm_prefix_cache_table.tex`,
`vllm_prefix_cache.{pdf,png}`.

## What this does and does NOT show
- **Does**: prefix reuse in a real engine reduces TTFT and raises throughput on a shared-prefix
  workload (the systems payoff the paper's proxy stands in for). Uses vLLM's built-in eviction.
- **Does NOT**: test *our adaptive eviction policy* vs LRU — vLLM and SGLang both evict KV blocks
  by LRU and don't expose a pluggable policy. That is Tier 2.

## Tier 2 — testing the adaptive policy in-engine (scoped, not yet built)
To back the paper's specific claim (adaptive beats LRU/LFU under drift) inside a real stack, patch
the engine's evictor and A/B it on stationary vs shifted workloads:
- **vLLM**: the block/KV manager that owns block reuse and eviction (v1: the KV-cache manager and
  its free-block ordering). Replace the LRU free-list ordering with a pluggable score (recency,
  frequency, EWMA-recency = our adaptive) keyed per cached block/prefix.
- **SGLang**: the RadixAttention radix-tree evictor (the LRU leaf-eviction on the radix tree).
  Swap the eviction key for the same adaptive score.
Then rerun this harness with each policy build and compare TTFT/throughput/hit-rate under a
mid-stream popularity shift. This is engine-internals work (a fork + a few hundred lines), so
scope it against the deadline; Tier 1 + the honest proxy already discharge the Limitations note.
