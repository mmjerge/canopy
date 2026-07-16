# Review findings, changes made, and the remaining experiment runbook

This document records (1) the issues found in the original theory and code during the
July 2026 review, (2) exactly what was changed and why, and (3) the step-by-step
runbook for the remaining AWS/GPU-gated experiments. Companion docs:
`docs/code_benchmarks.md` (the code-benchmark ladder and its gates) and
`paper/README.md` (the LaTeX directory; every deviation from the prior draft is
marked with `% [FIXED IN REVIEW]` / `% [ADDED IN REVIEW]` comments in the sources).

---

## 1. Issues found in the original theory

### T1 — Theorem 1 (the certificate) subtracted an unestimated mean *(fixed)*
The stated bound was `B(v) <= min_lambda {(1/lambda)[log m + log G_up(lambda) -
lambda^2 sigma^2 / 2]} - X_bar`. But `B(v)` subtracts `f(v) = E[X]`, and `X_bar` only
*estimates* it: whenever `X_bar > f(v)` (probability ~1/2) the stated quantity
under-covers, so it was not a valid `1 - delta` bound. The appendix had a dangling
`+,` and a missing symbol where the confidence radius was clearly intended.
**Fix:** one slice of `delta` now buys an empirical-Bernstein *lower* confidence
`f(v) >= X_bar - eps_n`; the bound subtracts `X_bar - eps_n` and `delta` is split
over `|Lambda| + 1` events. Statement, proof (Appendix A, Step 4), and implementation
all updated. Empirically the omission rarely bit (other slack dominates), so no
synthetic results shifted — this was a validity fix, not a results fix.

### T2 — Theorem 3's additive term claimed a horizon-independent constant *(fixed)*
The regret bound claimed `R_n <= C1 n^{(d+1)/(d+2)} + C2 K D` with `C2` constant. The
proof asserted each jump cell is visited `O(1/Delta^2)` times with "bounded aggregate
regret." Standard UCB accounting gives `O(log n / Delta_v^2)` visits at per-visit
regret `Delta_v`, i.e. `O(log n / Delta_v)` per cell — both the `log n` factor and
the inverse-gap dependence were dropped without justification.
**Fix:** the statement now reads `... + C2 K D log(n) / Delta_min` and the proof
carries the terms explicitly. The qualitative story survives (the jump penalty grows
only logarithmically, so it never changes the polynomial rate).

### T3 — Theorem 2's detection step assumed away detectability *(fixed)*
The proof claimed the detector flags *all* violations and *no* smooth cells "by
setting the confidence level appropriately." A one-sided confidence bound controls
false flags; catching every true violation additionally requires the jump's
within-cell spread to exceed the threshold by a margin at the detection sample size.
**Fix:** stated explicitly as Assumption 1 (detectability margin `gamma`); the
budget-equalization step is now pinned to plug-in estimates from the detection phase.

### T4 — Bounded-range Bernstein under unbounded Gaussian noise *(documented)*
The Bernstein step needs bounded `e^{lambda X}`; the code hard-coded a +-3 sigma range
without accounting for escapes (each probe escapes w.p. ~0.27%).
**Fix:** the truncation is now a parameter (`trunc_z`) and the realized coverage
`1 - delta - 2n*Phi(-z)` is stated in the code and the paper (new Remark). The exact
sub-exponential treatment remains open (limitations section).

### T5 — Manuscript defects *(fixed in the rebuilt paper/ directory)*
Garbled Eq. (4); ~8 citations rendering as `(?)`; Table 4's caption (tau-bench
routing) sitting over best-of-N/value-guided *reasoning* columns (copy/paste error);
a floating GPQA header with no rows in Table 6; several `[pending]` figures/tables;
unresolved `??` references for the code-benchmark tables.

---

## 2. Issues found in the original code / experiments

### C1 — Two probe channels; identification used the wrong one *(disclosed, not yet unified)*
Paper Eq. (5) defines the probe as `mu(random leaf) + noise` (mixture). The regret
engine (`env.play`) implements exactly that, but the identification experiments
(`HierarchicalTopK` via `env.sample`) observe `f(v) + noise` — same mean, strictly
less variance. Now disclosed in the paper (Remark 1, "direct-average probes"); a
random-path variant of the top-k benchmark remains a recommended robustness run.

### C2 — The certificate's advertised role was inverted *(disclosed)*
The paper says: explore with the tight heuristic, certify pruning/stopping with
Theorem 1. In the code, the rigorous bound is only wired in as the optimism bonus of
`run_adaptive_mgf` (the configuration the paper advises against), and
`HierarchicalTopK` prunes with the assumed schedule. Flagged inline in the paper
(TODO); wiring Theorem 1 into the pruning path is future work.

### C3 — The claimed learned router did not exist *(fixed)*
Sec. 4.2 claimed a learned `ContextualUCBRouter` ("the learned bandit — not a
heuristic"); the repo implemented `TieredRouter`, a hand-coded escalate-on-error
heuristic. **Fix:** `ContextualUCBRouter` now exists
(`canopy.bandits.routing`), with episodic Monte-Carlo credit assignment, and is wired
into `taubench_routing.py` as `learned (regional)` + `learned (flat)` policies.
Building it surfaced a genuine identifiability failure: with deterministic
tie-breaking, regions explore in lockstep, choices are perfectly correlated across
regions, and the shared terminal reward cannot identify per-region means (they pin at
the grand mean). Fixed with a small epsilon-randomization that decorrelates the
per-region designs; documented in the class docstring and covered by tests.

### C4 — Figure 3 gave the hybrid free budget *(fixed)*
The detect-and-relax hybrid's detection pass (~32% of the 400-unit budget) ran on a
separate environment and was never charged. **Fix:** the script now subtracts the
detection cost from the hybrid's identification budget. Regenerated under fair
accounting the hybrid still dominates at sparse K (0.97/0.90/0.70 at K=1/2/4) but
now visibly converges to the baselines at large K — which is exactly the
`H_edge -> H_blind` prediction, so the paper's "strictly dominating" was softened to
the measured claim.

### C5 — Experimental-protocol disclosures *(added to the paper)*
- Figures 1 and 3 use the best-first *beam* variant of the hierarchical search
  (explicitly unsound worst-case) — now disclosed in the captions.
- MMLU routing: the model pool was curated so no single model dominates after cost
  (per a code comment), and the 6k-round bandit run replays a once-measured 32-question
  quality matrix with synthetic Gaussian noise — both now disclosed.
- Reasoning search: "matched compute" clarified to matched generation *calls*
  (candidate steps use half the tokens of a full trace).
- Stale README numbers (older runs) contradicted the draft's tables.

### C6 — ~8 scripts named in the paper did not exist *(all but one now committed)*
`lipschitz_ablation.py`, `theory_link.py`, `reasoning_search.py`,
`reasoning_tree_lipschitz.py`, `prefix_cache.py` (real-prompt),
`vllm_prefix_cache_eval.py`, `vllm_policy_eval.py` — all now committed (several also
*run*; see Section 3). Remaining: `combine_reasoning_models.py` is a thin loop over
`reasoning_search.py` per-model runs (regenerate Fig 9 / Table 6 from those).

---

## 3. What was added, and what the regenerated numbers showed

**New machinery (all tested; suite went from 115 to 169 passing tests):**
- Certificate fix + `trunc_z` + a *coverage calibration test* (the test that would
  have caught T1).
- Scope knobs for the synthetic reasoning model (`scoped_success_rate`,
  `value_guided_search_scoped`): saturation x probe-informativeness, with the theory's
  scope predictions encoded as unit tests, and the phase-diagram demo
  (`probe_scope_demo.py`). The map quantitatively reproduces the real pattern:
  +0.81 in the unsaturated/informative corner, ~0 on the saturated edge (GSM8K), and
  -0.04..-0.07 on the uninformative edge — the measured MBPP null's sign *and*
  magnitude.
- Execution-graded code probe (`canopy.bandits.code_llm`: continuous multi-test
  value, subprocess-isolated) + `mbpp_probe_check.py` + `swebench_stage1.py` — the
  three-rung code ladder (see `docs/code_benchmarks.md`).
- `instrumented_value_guided_search` + `reasoning_tree_lipschitz.py` (Fig 11/Table 8).
- MATH/GPQA/GSM8K `reasoning_search.py` with paired per-problem bootstrap CIs.
- Real-prompt prefix cache (alpaca + dolly corpora, tiktoken) with the post-shift
  *transient* metric (the honest window: LFU re-learns given enough stream).
- `ContextualUCBRouter` + tau-bench integration.
- vLLM live-server measurement script + GPU-calibrated trace-driven eviction
  comparison.

**Key regenerated results (all synthetic/offline claims verified):**
| Claim | Number |
| --- | --- |
| Tree loses at `c_p = c_l`, wins at `<= 0.5x` | 0.27 vs 0.57 -> 0.78 vs 0.57 |
| Sublinear rate (d ~ 0 => alpha ~ 0.5) | alpha = 0.548, d_hat ~ 0.2 |
| Full-resolution regret at ~6x less memory | 935 @ 42 nodes vs 936 @ 256 |
| Estimated L vs oracle-tuned fixed L | **1750 vs 3003** (stronger than the draft) |
| Certificate coverage | 0 violations / 400 trials at delta = 0.1 |
| RouterBench variance decomposition | 30.7% category / 69.3% within (draft: 31.3/68.7 — reproduced) |
| Real cache post-shift transient | adaptive 5.23 vs LFU 3.32 (alpaca); 5.70 vs 3.66 (dolly) |
| Eviction in realized ms (1.06 ms/tok) | adaptive 6.3 ~ LFU 6.4 stationary; post-shift 5.5 vs LFU 3.5 |
| SWE-bench BM25 localization gate | P(gold file retrieved) = 0.33 — informative but insufficient; upgrade the probe before racing |

---

## 4. Runbook: the remaining gated experiments

All scripts cache responses on disk (JSONL) and support `--max-calls` /
`--max-spend`, so runs are resumable and capped. Do a small smoke run first
(`--n-problems 10 --max-spend 1`), inspect, then scale.

### A. AWS Bedrock (needs credentials with model access; `uv sync --extra llm --extra bench --extra plot`)

1. **Flagship: MATH sweep (Fig 8 / Table 5).**
   ```bash
   uv run --extra llm --extra bench --extra plot \
     python examples/reasoning/reasoning_search.py --benchmark math \
     --n-problems 300 --max-spend 25
   ```
   Produces `paper/figures/reasoning_search_math.{pdf,tex}`. Then the controls:
   `--benchmark gsm8k` (should show ~0 gap) and `--benchmark gpqa_diamond`
   (Fig 10 / Table 7; the HF dataset is gated — accept its terms and `huggingface-cli
   login` first). Rough cost: dominated by ~5 budget points x 300 problems x ~100
   calls of a 70B model; set `--max-spend` to your comfort and let the checkpointing
   resume across runs.

2. **Capability ladder (Fig 9 / Table 6).** Re-run step 1 with `--model` set to each
   of the five pool models (the response cache makes repeats cheap), then assemble the
   per-model results JSONs (`examples/.cache/reasoning_math_results.json`) into the
   ladder figure. (This replaces the never-committed `combine_reasoning_models.py`.)

3. **The prior, measured (Fig 11 / Table 8).** Shares the response cache with step 1,
   so much of it is free after the sweep:
   ```bash
   uv run --extra llm --extra bench --extra plot \
     python examples/analysis/reasoning_tree_lipschitz.py --benchmark math --n-problems 300
   ```
   Gate to check: pivotal-step hit rate clearly above 1/branching (MATH reference:
   0.73 vs 0.33) and a small mean K.

4. **Code ladder rung 2 (MBPP probe repair).**
   ```bash
   uv run --extra llm --extra bench \
     python examples/reasoning/mbpp_probe_check.py --n-problems 50 --n-samples 8 --max-spend 5
   ```
   Read the Stage-1 gate first (probe-vs-hidden Spearman); only if clearly positive is
   a matched-compute code race worth running (prediction: probe improves, saturation
   still binds — expect the gap to move toward 0, not a win).

5. **tau-bench learned router (Table 4 / Fig 7).** Needs `tau-bench` installed and a
   user-sim model:
   ```bash
   uv run python examples/agentic/taubench_routing.py --env retail \
     --num-tasks 80 --max-spend 30
   ```
   Now includes `learned (regional)` and `learned (flat)` alongside the tiered
   heuristic and fixed models. The paper's claim rests on regional > flat; if the
   80-task budget is tight, prioritize those two policies plus the best fixed model.
   Note the learned routers improve over the task stream — report the online curve,
   and consider `--trials 2+` for variance.

### B. GPU host (vLLM; any 24 GB card works for Qwen2.5-7B)

6. **Prefix caching ON/OFF + calibration (Fig 13 / Table 10).**
   ```bash
   vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000          # caching ON (default)
   uv run python examples/systems/vllm_prefix_cache_eval.py --label on
   uv run python examples/systems/vllm_prefix_cache_eval.py --calibrate   # ms/token

   vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --no-enable-prefix-caching
   uv run python examples/systems/vllm_prefix_cache_eval.py --label off
   ```

7. **KV-budget sweep (Fig 14).** Relaunch the server per point with
   `--num-gpu-blocks-override {500,1000,2000,4000}` and record `--label kvN`; then
   `--plot` renders both figures and Table 10.

8. **Recalibrated eviction comparison (Fig 15 / Table 11).** Re-run the (offline)
   `vllm_policy_eval.py --ms-per-token <measured>` with the calibration from step 6
   to replace the default-calibration numbers currently in the paper.

### C. Free robustness runs that would make the claims more ironclad (no API/GPU)

- **Sound-variant check (RUN — important finding):** with `beam_width=None` (the
  sound algorithm Theorem 2 analyzes), the hierarchical search *matches but does not
  beat* successive elimination on the hierarchical-Gaussian family at any tested
  budget (0.61 vs 0.58 at budget 1500 / cheap probes; 0.82/0.90/0.97 vs
  0.79/0.90/0.96 at 3000/6000/12000), while the beam-20 variant reaches 0.84 at
  budget 1500. **The empirical Figure-1 advantage is carried by the beam focusing,
  not by the sound pruning** (the conservative z=3 spread bound prunes too little and
  the frontier explodes). The paper must either (a) show both variants in Figure 1
  and attach the empirical win to the beam variant with the sound one as the
  never-worse safety net, or (b) find an instance family / tighter certified bound
  where sound pruning separates. Do not leave the current caption implying the
  analyzed algorithm produces the plotted curves.
- **Random-path probes for identification (closes C1):** a `TreeBandit` flag to route
  `sample()` on internal nodes through the mixture channel, then rerun Fig 1.
- **Error-rate collapse for Theorem 2:** sweep budget B on synthetic instances with
  different measured `H_edge` and plot log error vs `B / H_edge` — all instances
  should collapse onto one line, validating the bound's functional form (not just its
  direction).
- **Certificate calibration breadth:** extend the coverage test across instance
  families (spikes, heavy tails) and small n.
- **MMLU replay noise model:** replace Gaussian noise on the 0/1 matrix with
  Bernoulli resampling; rerun Table 3 (no new API calls needed).
- **RouterBench non-stationary routing:** a mid-stream region-distribution shift
  (analog of the cache experiment) to show the online router tracks drift on real
  data where the frozen policy cannot.
