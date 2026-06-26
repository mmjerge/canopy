# Canopy applications: three real LLM use cases

All three map a real LLM problem onto the multi-fidelity tree (a leaf, an arm/region, and a
storage/cost budget) and run on the same `TreeBandit` machinery. These are the empirical
contributions of the ICLR/ICML paper.

## 1. LLM routing

- **Mapping.** Leaf = prompt; arm = model; region = subject / token-prefix. The LCA
  ultrametric (longest-common-prefix distance) lets a hierarchical router generalize routing
  decisions across prompts that share a prefix.
- **Result.** Real MMLU + 6 Bedrock models: 0.84 quality vs 0.69 for the best fixed model,
  matching always-largest at ~½ the cost (≈12× lower regret than the best fixed policy).
- **Code.** `canopy.bandits.routing`; `examples/llm_routing/mmlu_routing.py`,
  `examples/llm_routing/llm_routing_demo.py`, `examples/llm_routing/bedrock_routing.py`. Real models via
  `canopy.llm` (`BedrockClient` / `OpenAIClient` / `AnthropicClient`; needs the matching extra + credentials).
- **Docs.** `docs/llm_routing.md`.

## 2. Prefix caching

- **Mapping.** Cache = ancestor-closed subtree of the token trie; storage = memory budget.
  Caching a prefix saves recompute for every prompt that passes through it.
- **Result.** Matches LFU and the hindsight optimum on a stationary stream, and beats both LFU
  and the best static cache under a popularity shift (it tracks the drift).
- **Code.** `canopy.bandits.prefix_cache`; `examples/llm_routing/prefix_cache_demo.py`.
- **Why it's the cleanest.** The prefix tree *is* the actual cache data structure (vLLM APC /
  SGLang RadixAttention), so there is no tree-alignment assumption, unlike routing/trimming
  where the tree must align with capability structure.

## 3. Prompt optimization (trimming)

- **Mapping.** Arm = trim level; region = subject. Adaptive per-subject prompt trimming.
- **Result.** Real MMLU + nova-lite: adaptive trim beats the best fixed trim on accuracy
  (0.77 vs 0.75) and tokens (97 vs 101).
- **Code.** `examples/llm_routing/prompt_optimization.py`.
- **Honest caveat.** This run used an 8-token output cap that penalized verbose prompts; it is
  the weakest of the three and needs a re-run at a larger output budget to be airtight.

## Planned: RouterBench (a fourth, real-data routing testbed)

RouterBench (precomputed quality + cost for 11 LLMs over ~36K prompts) is a natural offline
environment for the bandit router — no API calls. The Canopy use of it is the **pure-bandit**
one: run the hierarchical bandit router on RouterBench's real quality/cost and report
regret/cost vs the per-prompt oracle and best-single baselines.

(The *harmonic* slice-reconstruction approach to routing is a separate result and lives in the
Horizon repo; the two are kept distinct because they demonstrate different things — online
regret minimization here vs. offline angular-sparsity reconstruction there.)

Not yet implemented; see the project notes.
