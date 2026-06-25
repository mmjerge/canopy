# Paper outline: online optimization over token prefix trees

Working thesis: **routing, prefix-caching, and prompt-trimming for LLMs are all instances
of online optimization over a token prefix tree with aggregate feedback under a
storage/cost budget** — one framework, one regret/storage theory, three real applications.

## Contributions

1. **Framework.** A hierarchical bandit on a (token) tree where an internal node's value
   is the aggregate (average) of its subtree's leaves, feedback is multi-fidelity
   (cheap/biased internal probes vs. expensive/unbiased leaf evaluations), and the cost is
   a storage/compute budget. (`canopy.bandits.tree`)
2. **Regret-optimal algorithm + the expand-vs-refine rule.** UCB index
   `mean + sqrt(2 ln t / T) + spread(level)` with B-value backup (HOO/HCT); the
   per-node crossover "refine while statistically-limited, expand when `r(v) <= spread`"
   gives near-optimal regret at memory governed by the near-optimality dimension, not the
   tree size (finite-state compression in the infinite-depth limit). (`run_hoo`,
   `docs/ucb_optimal.md`, `docs/infinite_tree.md`)
3. **Data-driven / certified smoothness.** A noise-deconvolved empirical-MGF bound on
   `max - mean` certifies the bias term from data instead of assuming `spread`; the
   variance-aware and Lipschitz-floor+jump-detection (`run_hybrid`) variants are robust to
   hidden jump discontinuities; the local-Lipschitz variant adapts the constant per
   subtree. (`docs/maxmean_bound.md`, `docs/lipschitz_regret.md`)
4. **Regret bound for the piecewise-Lipschitz case.** `R_n <= C1 n^{(d+1)/(d+2)} +
   C2 K depth B` for L-Lipschitz-with-K-dispersed-jumps (statement + proof sketch).
5. **Three empirical LLM applications** (below), all on the same machinery.

## Empirical applications (current status)

| application | mapping | result | code |
| --- | --- | --- | --- |
| **Routing** | leaf = prompt, arm = model, region = subject | real MMLU + 6 Bedrock models: 0.84 quality vs 0.69 best-fixed-model, beats always-largest at ~½ cost | `routing.py`, `examples/llm_routing/mmlu_routing.py` |
| **Prefix caching** | cache = ancestor-closed subtree, storage = memory | matches LFU/offline on stationary; beats both LFU and best static cache under popularity shift | `prefix_cache.py`, `examples/llm_routing/prefix_cache_demo.py` |
| **Prompt trimming** | arm = trim level, region = subject | real MMLU + nova-lite: adaptive per-subject trim beats best fixed trim on accuracy (0.77 vs 0.75) and tokens (97 vs 101); verbose prefix hurt | `examples/llm_routing/prompt_optimization.py` |

Caching is the cleanest: the prefix tree is the *actual* cache data structure, so there is
no tree-alignment assumption (unlike routing/trimming, where the tree must align with
capability structure).

## Related work to position against

X-armed / hierarchical bandits (HOO, HCT, SOO/StoSOO, POO); combinatorial pure exploration
with full-bandit feedback; multi-fidelity / two-fidelity-tree best-arm identification;
piecewise-Lipschitz online optimization / dispersion (Balcan–Dick–Vitercik); LLM routing
(RouteLLM etc.); prefix caching (SGLang RadixAttention, vLLM APC). The novel slice is the
**unified online-learning treatment with tree-aggregate value and a storage/cost budget**,
plus the data-driven certified-smoothness machinery, instantiated on three real LLM tasks.

## What's solid vs. what to harden for ICLR (honest)

Solid: the framework + algorithms run and behave as theory predicts; the routing result is
on real models and is a clear win; caching has a clean drift result; the simulations are
reproducible (61 tests).

Needs hardening:
- **Theory is statements + sketches**, not fully worked proofs (the piecewise-Lipschitz
  bound, the cache-eviction regret, the routing-regret composition).
- **Benchmarks are small** (32 MMLU items, few models/trims). Scale up for tight numbers.
- **Caching is simulated** with a 1-token-per-node model; a real vLLM/SGLang integration
  measuring latency/memory is the credible systems eval.
- **Confounds**: prompt-trimming used an 8-token output cap (penalizes verbose prompts);
  re-run at larger budget. Routing/trimming assume the tree aligns with capability regions
  (hand-arranged by subject) — needs a real token-prefix or embedding hierarchy.
- **Related-work pass**: confirm the novel slice against the crowded routing/caching and
  X-armed-bandit literature before claiming novelty.

## Suggested next steps

1. Pick one application to make airtight end-to-end (caching with a real serving stack is
   the strongest systems story; routing is the strongest "it works on real models" story).
2. Turn one regret bound into a complete proof.
3. Scale the benchmarks and remove the trimming output-budget confound.
4. Full related-work / novelty pass.
