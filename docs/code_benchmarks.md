# Code benchmarks: why the single-function null happened, and the exact protocol that should work

Status: protocol implemented as a three-rung ladder; localization gate **measured**.
The single-function (HumanEval/MBPP) experiments produced a null; this document
explains why that is *predicted* by the theory, and specifies the repo-level
experiment (SWE-bench / RepoBench style) that the theory predicts should work — with
the soundness gates that must pass before any compute is spent on the headline race.

## The ladder (implemented)

| Rung | Benchmark + probe | Script | Status |
| --- | --- | --- | --- |
| 1 | MBPP/HumanEval, one-assert probe | (the pre-registered null) | control; keep as the scoped null |
| 2 | MBPP, multi-test execution probe | `examples/reasoning/mbpp_probe_check.py` (probe in `canopy.bandits.code_llm`) | built; needs AWS to run |
| 3 | SWE-bench Lite, localization tree | `examples/analysis/swebench_stage1.py` | **gate measured** (below) |

**Rung-3 gate, measured (offline, 300 Lite instances):** with the standard BM25-13K
retrieval as the cheap localization probe, `P(>=1 gold-patch file in context) = 0.33`,
MRR 0.12 — far above the few-percent random-file baseline (informative), far below
reliable. Reading: BM25-only localization caps value-guided repo search at a third of
instances, so the probe must be upgraded (embedding retrieval / agentic navigation /
import-graph walk) before the Stage-3 race is worth running. This is the gate doing
its job — a measured no-go for the naive probe, with a concrete number to beat,
instead of another ungated null. (Caveat: within-context ordering may not be BM25 rank
— the set-level 0.33 is the robust statistic.)

## Why HumanEval/MBPP sit outside the theory's conditions

The identification bound (`H_edge`, paper §3.4) makes the advantage conditional on
four things. Single-function benchmarks with a one-public-assert probe violate three:

| Condition | HumanEval/MBPP (1 assert) | Repo-level (SWE-bench / RepoBench) |
| --- | --- | --- |
| Cost asymmetry `c_p << c_l` | absent: a "cheap rollout" of a partial function costs the same tokens as a full sample; the hidden suite is also cheap. `c_p/c_l ~ 1` — the regime where Figure 1 shows the tree *loses* | real: patch-applies/imports/lint ~ ms, targeted test ~ s, full suite in container ~ min. `c_p/c_l ~ 1e-2..1e-3` |
| Headroom (reachable-but-unreliable) | near-saturated for a strong model (pass@N ≈ 0.8) → small effective `K`; behaves like the GSM8K control | curated splits (SWE-bench Lite/Verified) with a capable agent sit mid-ladder. Caveat: a weak model near 0% is the *unreachable* regime — expect a null there too (cf. llama-8B on MATH, Δ = −0.147) |
| Informative probe at pivotal steps | a single Bernoulli bit; plausible-but-wrong completions often pass the public example test | execution-graded and continuous: fraction of a test subset passing, compile/apply gates — causally tied to the true objective |
| A real decision tree | a 10-line body has no meaningful tree; "steps" are entangled tokens | root → module → file → function → edit region; LCA = path overlap. RepoBench-R *is* the routing instantiation; RepoBench-P is the pipeline tree |

The synthetic phase diagram (`examples/reasoning/probe_scope_demo.py`) quantifies
this: the value-guided − best-of-N gap is +0.81 in the unsaturated/informative corner,
≈ 0 on the saturated edge, and −0.04..−0.07 on the uninformative edge — the measured
MBPP null's sign and magnitude. The scope predictions are unit-tested
(`tests/test_reasoning.py`).

## Pre-registered protocol for the repo-level experiment

The discipline that made the MATH result sound was: **measure the prior first, then
predict the outcome, then run the race.** Same three stages here.

### Stage 1 — measure the prior (gate; cheap)

Code analog of `examples/analysis/reasoning_tree_lipschitz.py`. On ~50 SWE-bench Lite
instances, instrument the localization/edit tree and log, per internal node:

* **cheap value**: retrieval/embedding score at localization nodes; patch-applies +
  imports + targeted-test fraction at edit nodes;
* **true value**: probability a completion from this node resolves the issue
  (estimated by grading a few full rollouts against fail-to-pass tests).

Report the same two statistics as Table 8: cheap-vs-true correlation (backbone), and
pivotal-step fraction / edge-following hit rate at pivotal steps (violations).

**Gate:** proceed to Stage 3 only if the pivotal hit rate beats chance with margin
(MATH reference: 0.73 vs 0.33). If it doesn't, the probe — not the search — is the
bottleneck; fix the probe and re-gate. This is exactly the check the one-assert probe
would have failed, saving the null run.

### Stage 2 — probe repair for single-function benchmarks (optional)

If HumanEval/MBPP results are still wanted: replace the one-assert probe with a
continuous execution probe — all public examples + `k` model-generated tests filtered
for mutual consistency; value = fraction passed. Report probe↔hidden-suite correlation
next to any Δ. Prediction from the phase diagram: this moves `q` up but not `s`, so
expect the gap to move from slightly negative toward ~0, not to a win — saturation
still binds for a strong model.

### Stage 3 — the race (only after Stage 1 passes)

* **Tree:** root → candidate files (retrieval) → candidate regions/functions →
  candidate patches. Internal probes as in Stage 1; leaf eval = full fail-to-pass +
  pass-to-pass suite in the container.
* **Baseline:** best-of-N whole patches (sample N end-to-end attempts, grade all,
  submit the best by cheap score), at **matched measured cost** — tokens *and*
  execution seconds both counted; calls are not a fair unit here because probe and
  leaf costs differ by orders of magnitude (that asymmetry is the point, and it must
  be in the budget accounting on both sides).
* **Stats:** paired per-instance bootstrap, exactly as the MATH tables.
* **Model × split:** pick the pair mid-ladder (agent resolves ~20–60% at best-of-N).
  Include one weaker model as the predicted-null control (unreachable regime) — the
  capability-ladder pattern from MATH should reproduce.

## Expected outcomes and what they would mean

* Stage 1 passes + Stage 3 win → the advisor's expectation confirmed, in the regime
  the theory says it should hold; the single-function null stands as the scoped
  contrast, not a contradiction.
* Stage 1 fails (probe uninformative even with execution signals) → the honest
  conclusion is that repo-level value functions violate the tree-Lipschitz backbone;
  that is a real finding about the prior, and the phase diagram says don't run the
  race.
* Stage 1 passes but Stage 3 null → the interesting case: informative probe, headroom,
  no gain — would point at the tree construction (localization hierarchy misaligned
  with the objective), the `K`-structure (violations not sparse), or budget accounting.
