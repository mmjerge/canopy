# Reasoning-tree search: value-guided (edge-following) vs. best-of-N

This is the first bridge from the theory to a concrete LLM use — test-time-compute search
over a reasoning tree. Module: `canopy.bandits.reasoning`.

## Setup

A leaf is a complete reasoning trace; an internal node is a partial trace. A leaf's reward
is the fraction of `K` *decision steps* it gets right, so a fully correct answer (reward 1)
requires getting all `K` decisions right. The internal-node value (subtree-average reward)
is the process/value-model (PRM) signal; it is smooth except at the decision steps, where the
value differs sharply between children — the **edges**. The cheap multi-fidelity probe is a
rollout (the noisy reward of a random completion of the prefix); the expensive evaluation is
a full trace + verifier.

## Two strategies at a fixed oracle budget

* **best-of-N** (`best_of_n`) — structure-blind: sample whole traces, return the best-scoring
  one. To return a fully-correct trace it must *sample* one, which needs `~b^K` draws.
* **value-guided search** (`value_guided_search`) — edge-following: descend the tree, probing
  each child with cheap rollouts and following the higher-value child (the value edge at a
  decision step). It resolves the `K` decisions one at a time, at polynomial cost.

## Result (binary tree, reward = fraction of K decisions correct, noise 0.3, 200 seeds)

Budget (oracle calls) to reach 80% fully-correct:

| K decisions | 4 | 6 | 8 | 10 | 12 |
| --- | --- | --- | --- | --- | --- |
| best-of-N | 4096 | — | — | — | — |
| **value-guided** | **128** | **512** | **1024** | **2048** | **4096** |

`—` means it did not reach 80% within a budget of 32768. best-of-N is **exponential** in the
number of decision steps (it must sample a fully-correct trace) and fails outright for
`K ≥ 6` within budget; value-guided search follows the value edges and succeeds at
**polynomial** cost. At `K=8`, value-guided reaches ~0.9 success by budget 1024 while
best-of-N stays near its `~K·2^{-K}` selection rate.

## Honest scope

This is a **search / compute-allocation** result, not a capability boost. It requires:

* a **reachable** correct trace (it cannot solve a problem the model's sampling distribution
  never produces), and
* an **informative value signal** — the whole method rests on the value being smooth with
  detectable edges at the decision steps, and on the verifier recognizing correctness.

The central modeling assumption — that a real PRM's value actually jumps at the pivotal
steps — is exactly what to validate next on a real model + PRM (e.g. GSM8K/MATH): does the
energy-spike detector on PRM scores localize the steps where verifier-correctness flips, and
does edge-targeted search beat best-of-N at equal compute? The synthetic result here is the
controlled bridge to that experiment.

```bash
uv run python examples/reasoning/reasoning_search_demo.py
uv run --extra plot python examples/reasoning/reasoning_search_demo.py   # success-vs-budget + K-scaling
```

## The scope map: when the advantage exists at all

The two conditions above are *quantitative* knobs, and real benchmarks differ on exactly
them. `scoped_success_rate` / `value_guided_search_scoped` add the two knobs to the
synthetic model:

* **saturation `s`** — fraction of instances any method solves (no headroom / small
  effective `K`): GSM8K and HumanEval/MBPP for a strong model are near-saturated;
* **probe informativeness `q`** — probability a cheap probe batch actually reflects the
  child it scores: a single public assert that plausible-but-wrong code passes is a
  low-`q` probe; execution-graded probes on repo-level tasks (patch applies / imports /
  targeted tests) are high-`q`, and come with a genuine cost gradient
  (`c_p/c_l ~ 1e-2 .. 1e-3`).

The phase diagram (`examples/reasoning/probe_scope_demo.py`, depth 8, K=8, budget 1024,
200 seeds/cell) shows the paired value-guided − best-of-N gap is **+0.81** in the
unsaturated/informative corner, **≈ 0** along the saturated edge, and **−0.04 to −0.07**
along the uninformative edge — the search overhead actively hurts when the value edge
carries no signal, reproducing the sign and magnitude of the measured MBPP null. The
HumanEval/MBPP null and the MATH win are therefore the *same* theory evaluated at
different `(s, q)`; repo-level coding (SWE-bench / RepoBench style: a real
file → function → edit decision tree, execution probes, unsaturated for a capable agent)
sits in the winning corner and is the predicted next instantiation. The scope predictions
are encoded as tests in `tests/test_reasoning.py` (gain requires both knobs; collapses
proportionally with saturation; reverses sign as `q → 0`).

```bash
uv run --extra plot python examples/reasoning/probe_scope_demo.py
```

## From synthetic to a real benchmark (no synthetic shortcuts)

The result above is **synthetic** — no LLM is involved; the "reward" and "value" are formulas.
It confirms the *mechanism and the math* (value-guided search beats best-of-N when the value
signal is informative at the decision steps), not that it helps a real model.

`canopy.bandits.reasoning_llm` is the generator-agnostic harness for the real test: GSM8K-style
answer extraction/grading, `best_of_n` (self-consistency) and `value_guided_search` (beam/tree
search over reasoning steps with a cheap-rollout value), and a `Budget` counting generation
calls. The value signal is a pluggable `value_fn` — **self-consistency** by default (the
realistic, ground-truth-free test-time signal) or a verifier/PRM-style value. The search logic
is verified now with a mock LLM (`tests/test_reasoning_llm.py`): given an informative
(graded/PRM-style) value, value-guided beats best-of-N at matched budget, confirming the
machinery; whether self-consistency is informative enough on real traces is the open question.

`examples/reasoning/gsm8k_reasoning_search.py` runs it on real GSM8K via the Bedrock client used by the
routing experiments (matched generation-call budget, accuracy reported):

```bash
uv sync --extra llm --extra bench
uv run --extra llm --extra bench python examples/reasoning/gsm8k_reasoning_search.py --n-problems 20
# (needs AWS creds + Bedrock model access; makes real paid calls)
```

The honest framing: a negative result is also informative — it would say the bottleneck is the
*value signal*, not the search. See `docs/reasoning_search.md`.
