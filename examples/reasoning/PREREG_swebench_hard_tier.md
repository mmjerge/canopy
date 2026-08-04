# Pre-registration: SWE-bench Verified "1-4 hours" tier

Committed before any run on this tier. The commit timestamp of this file is the
pre-registration date; results will be reported in the paper regardless of outcome.

## Motivation and directional hypothesis

The paper's theory predicts the value-guided-vs-best-of-N gain grows with decision-chain
length (RQ3: the gain is a sample-efficiency effect, and best-of-N degrades faster when the
solution requires longer chains). The "1-4 hours" difficulty tier of SWE-bench Verified has
longer expected fix times than the "15 min - 1 hour" tier used in the main sweep.

Hypothesis (directional): at matched budget, the relative improvement (value-guided resolved
rate / best-of-N resolved rate) on the "1-4 hours" tier is at least as large as on the
"15 min - 1 hour" tier for the same model and budget (reference points from the audited
sweep: 1.80x at B=6, 2.00x at B=9 for claude-sonnet-4.5).

## Protocol (fixed in advance, identical to the main sweep except the tier)

- Dataset: princeton-nlp/SWE-bench_Verified, difficulty == "1-4 hours".
- Instances: ALL 42 instances in the tier. No selection, no filtering beyond the tier
  label. Instances whose evaluation harness errors are dropped from BOTH arms (same rule
  as the main sweep).
- Model: us.anthropic.claude-sonnet-4-5-20250929-v1:0 (the flagship model; the theory
  predicts the effect only for capable models, and one model keeps the spend bounded).
- Cells: depth 1 and depth 2 with branching 3 (matched budgets 6 and 9 generation calls),
  via examples/reasoning/swebench_search.py with the same flags as run_swebench_sweep.sh.
- Grading: official SWE-bench harness, resolved criterion. run-id canopy_swe_hard.
- Primary endpoint: paired per-instance Delta resolved with 95% bootstrap CI (the
  pipeline's standard estimator), per budget.
- Secondary endpoints: relative ratio vg/bo, discordant-pair counts, McNemar exact p
  (examples/analysis/audit_swebench_results.py methodology).

## Pre-stated limitations

- n = 42 is small: CIs will be wide, and this is stated wherever the result is reported.
- At low base rates the ratio is unstable. If best-of-N resolves fewer than 5 of 42
  instances in a cell, we will report the ratio as "unstable (fewer than 5 baseline
  resolutions)" and lead with the paired Delta and CI only.
- No further cells, models, budgets, or tiers will be added to this experiment after
  seeing results. If the hypothesis fails, that is reported as a bound on the effect.
