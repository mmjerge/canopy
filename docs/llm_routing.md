# Theory + applied: LLM routing on a prefix tree

How the hierarchical-tree-bandit framework becomes a concrete, testable LLM-routing
method — and a plan to validate it on real data.

## The mapping

| framework object | routing instantiation |
| --- | --- |
| branching-ary tree | token prefix tree (trie): branching = vocab, depth = prompt length |
| leaf | a prompt / input |
| internal node | a prefix region (all prompts sharing that prefix) |
| leaf value f(leaf) | quality q_m(prompt) of model m (correctness, −error, ...) |
| node value (subtree avg) | average quality of model m over that prefix region |
| LCA ultrametric | longest-common-prefix distance between prompts |
| tree-Lipschitz | prompts sharing a long prefix get similar quality |
| jump discontinuity | a token flips model behavior (prompt-sensitivity cliff) |
| cheap probe (low fidelity) | small value-model / verifier score on a prefix or partial output |
| expensive leaf eval (high fidelity) | full generation + grader on the target model |
| per-model error function | a separate tree function f_m per LLM (replant per model) |

Net utility of routing a prompt to model m: `u_m(prompt) = q_m(prompt) − lam · cost_m`.
The per-prompt oracle routes `argmax_m u_m`; **routing regret** is the gap to that oracle.

## Why the tree structure helps (the theory)

Routing per individual prompt is hopeless (exponentially many). Tree-Lipschitz quality
means the best model is (mostly) constant within a prefix region, so a learner only needs
to resolve the routing boundary, not every prompt. Concretely, at region resolution r the
learner solves `b^r` small model-choice bandits; the smoothness bias `spread(r) = L·rho^r`
controls how wrong the regional decision can be, and discontinuities (region boundaries)
are handled by the same piecewise-Lipschitz / jump machinery (`docs/lipschitz_regret.md`).
This yields the same bias–variance–storage tradeoff in r: too coarse misses regional
structure, too fine learns slowly, a middle resolution generalizes. A routing-regret bound
follows from the per-region bandit regret plus the `O(K·depth)` jump term for the boundary
regions.

## Experiment (simulation; `examples/llm_routing_demo.py`)

Big model (quality ~0.85 everywhere, cost 1.0) vs. cheap model (strong only in a few
prefix regions with sharp boundaries, cost 0.1). 1024 prompts, 20k-prompt stream, 8 seeds,
lam = 0.3.

| policy | routing regret | cost / query | avg quality |
| --- | --- | --- | --- |
| oracle (per-prompt best) | 0 | 0.77 | 0.869 |
| **hierarchical router (r=2)** | **146** | **0.75** | **0.856** |
| hierarchical router (r=0, global) | 1777 | 0.99 | 0.845 |
| best single model (fixed) | 1750 | 1.00 | 0.850 |
| always-largest | 1750 | 1.00 | 0.850 |

The router learns which prefix regions the cheap model handles, **matching oracle quality
at ~25% lower cost than always using the big model, with ~12x lower regret** than the best
fixed policy. Resolution r=0 (one global choice) and r=full (per-prompt) both fail — the
same resolution tradeoff as the rest of the framework.

## Path to real validation

The algorithms are unchanged for real data — only `quality` changes. `canopy.bandits.bedrock`
provides this bridge: `BedrockClient` calls models via the Bedrock Converse API (uniform
across providers) and `measure_quality_matrix(prompts, model_ids, client, grade)` returns
the `(quality, costs)` arrays a `PrefixTreeRouting` consumes. See `examples/bedrock_routing.py`
and the Terraform Bedrock stack (`terraform/bedrock.tf`) for the IAM/logging setup.

1. **Drop in a router benchmark** (e.g. measured per-model correctness on a prompt set):
   build `PrefixTreeRouting` from real `q_m(prompt)` arrays and rerun. Tokenize prompts to
   define the prefix tree (or use an embedding-induced hierarchy if token prefixes are too
   sparse).
2. **Check the assumption**: measure how Lipschitz quality actually is in the LCA metric
   and how many/where the jumps are (dispersion). This is the empirical crux — if quality
   is wildly non-smooth, the hybrid (data-driven, jump-robust) variant is the one to use.
3. **Multi-fidelity for real**: use a cheap value model on prefixes as the low-fidelity
   probe and full generation as the high-fidelity eval; measure the real cost ratio and
   bias, and plug into the cost-aware router.

Honest caveats: token-prefix trees can be sparse/huge (an embedding hierarchy may be the
better tree); real prompt→quality is likely jumpy (hence the jump-robust hybrid); and the
routing-regret bound is so far a composition argument, not a fully worked theorem.
