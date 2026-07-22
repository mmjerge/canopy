# Experimentation Plan — Multi-Fidelity Tree Bandits

Theory → synthetic → real-world status, the code-generation benchmark ladder, and the
remaining runbook. Companion to `docs/review_changes_and_runbook.md` (full technical
detail) and `docs/code_benchmarks.md` (the ladder in depth) — this document is the
condensed version for status discussions.

**At a glance:** 6/6 theory issues fixed and reproven · 169+ tests passing · 6
code-generation benchmarks staged (3 run, 3 built and pipeline-verified, pending
infra) · 1 open risk item requiring a decision.

---

## 1. Where things stand

The review found five issues in the theorems and six in the code/experiments.

### Theory (all fixed)

| Issue | What was wrong | Status |
| --- | --- | --- |
| Certificate (Theorem 1) | Subtracted the raw probe mean instead of a confidence bound on it — under-covered on ~half of draws | Fixed + calibration test |
| Regret bound (Theorem 3) | Claimed a horizon-independent additive constant; standard accounting needs a `log n` factor | Fixed |
| Detection (Theorem 2) | Assumed the detector catches every violation without a minimum-margin condition | Fixed (new explicit assumption) |
| Truncation caveat | Bounded-range step under unbounded Gaussian noise had an undocumented escape probability | Documented |
| Manuscript defects | Garbled equation, ~8 broken citations, a table caption over the wrong columns | Fixed |

### Code / experiments

Five of six are fixed, disclosed, or superseded by independent work already on `main`.
One is now a flagged decision (see §5) rather than a bug — the sound-vs-beam gap in
Figures 1 and 3.

---

## 2. Synthetic validation

Every regenerated synthetic number is consistent with what the corrected theory
predicts, with one flagged exception.

| Claim | Prediction | Measured | Status |
| --- | --- | --- | --- |
| Tree wins only when probes are cheap | Loses at cost parity, wins as probes cheapen | 0.29 → 0.84 recall as ratio drops (1024 leaves, top-5) | Confirmed |
| Memory savings (expand-vs-refine) | Near full-resolution regret at a fraction of the memory | 935 regret @ 42 nodes vs. 936 @ 256 | Confirmed |
| Graceful degradation (Fig. 3, budget now fairly charged) | Hybrid dominates while *K* sparse, converges as *K* fills the tree | 0.97 → 0.13 vs. baselines' 0.50 → 0.17 | Confirmed |
| Certificate coverage | ≤ δ violation rate over many trials | 0/400 trials at δ = 0.1 | Confirmed |
| Estimated vs. oracle-tuned *L* | Local estimate matches or beats any single global constant | 1750 (estimated) vs. 3003 (oracle-tuned) | Confirmed |
| Scope map (saturation × probe informativeness) | Reproduces sign *and* size of both the MATH win and the MBPP null from one model | +0.81 / ≈0 / −0.04…−0.07 | Confirmed |
| Sound pruning vs. baseline (Fig. 1, non-beam variant) | Should beat the structure-blind baseline (the theorem) | Only ties it — the beam heuristic carries the win | **Open — see §5** |

---

## 3. Real-world validation, run so far

Three real datasets, entirely offline — no API spend.

| Dataset | What it tests | Result |
| --- | --- | --- |
| RouterBench (11 real LLMs, 36k prompts) | Regional routing prior on measured quality/cost | Router frontier dominates every fixed model; category tree explains 31% of value variance |
| Alpaca + Dolly (real prompts, tiktoken) | Adaptive cache under a real popularity shift | Adaptive holds the post-shift transient; LFU collapses |
| SWE-bench Lite (300 real instances, offline) | Stage-1 gate: is BM25 localization informative? | **33%** gold-file retrieval, MRR 0.12 — informative, not sufficient |

---

## 4. The code-generation benchmark ladder

Six benchmarks, staged so each rung isolates one condition the theory requires before
the next rung is worth running. Rungs 4–6 already have complete, working pipelines
(verified end to end in `--mock` mode); what's missing is Docker/Harbor infrastructure
and model spend, not code.

| # | Benchmark | Script | Needs | Status |
| - | --- | --- | --- | --- |
| 1 | HumanEval / MBPP, one-assert probe | `reasoning_search.py --benchmark humaneval\|mbpp` | AWS | **Run — the control.** Near-saturated, no cost gap, a one-bit probe. The theory predicts a null; that's what was measured. |
| 2 | MBPP, multi-test execution probe | `mbpp_probe_check.py` | AWS | Built, ready to run. Prediction: informativeness improves but saturation still binds — expect the gap to move toward zero, not to a win. |
| 3 | SWE-bench Lite, BM25 localization | `analysis/swebench_stage1.py` | offline only | **Gate measured.** P(gold file retrieved) = 0.33, MRR = 0.12. |
| 4 | SWE-bench, patch-level characterization | `analysis/swebench_tree_lipschitz.py` | AWS + Docker | Built, mock-verified. A sharper Stage-1 gate than rung 3 — cheap-vs-true correlation directly on generated patches. Reuses rung 5's response cache. |
| 5 | SWE-bench, repo-level race | `reasoning/swebench_search.py` + `run_swebench_sweep.sh` | AWS + Docker (~120GB disk) | Built, mock-verified. The actual value-guided-vs-best-of-N result at matched compute, graded by SWE-bench's official criterion; the sweep script runs a 5-model × 3-depth capability ladder. |
| 6 | Terminal-Bench 2.x (Harbor) | `reasoning/terminalbench_search.py` | separate Python 3.12 Harbor venv | Built, mock-verified. Second independent agentic benchmark, fresh-container grading (avoids the ALFWorld state-cloning issue). |

**Read the rung-3 gate before spending on rungs 4–5.** If a sharper probe doesn't clear
a similar bar, the theory predicts the repo-level race won't pay either.

Setup, on a machine with AWS access:

```bash
uv sync --extra llm --extra bench --extra plot

# smoke-test every real-run script first — no deps/creds/Docker needed:
python examples/analysis/swebench_stage1.py                      # already measured; free to rerun
python examples/reasoning/swebench_search.py --mock --n-instances 12
python examples/reasoning/terminalbench_search.py --mock --n-tasks 12

# rung 1 (the control), start small:
uv run --extra llm --extra bench python examples/reasoning/reasoning_search.py \
    --benchmark mbpp --n-problems 20 --max-spend 5

# rung 5, once Docker + `pip install swebench` are set up:
python examples/reasoning/swebench_search.py \
    --dataset princeton-nlp/SWE-bench_Verified --n-instances 20 \
    --model us.anthropic.claude-sonnet-4-5-20250929-v1:0 --branching 3 --depth 1 --resume

# the long-running capability sweep, in tmux (resumable):
tmux new-session -d -s swesweep \
    'bash examples/reasoning/run_swebench_sweep.sh 2>&1 | tee logs/swebench_sweep.log'
```

---

## 5. Decisions needed

### Fig. 1 / Fig. 3 presentation

The empirical win in these figures currently comes from a best-first heuristic
(`beam_width=20`), not the algorithm the theorem analyzes — which only *ties* the
structure-blind baseline on the tested instance family.

**Recommendation:** show both variants honestly in the paper — beam as the practical
method, sound pruning as the certified never-worse floor. Still a strong, defensible
story; it just needs to be told accurately.

### SWE-bench rung 3, go/no-go

The measured 33% localization gate means BM25 alone caps the repo-level race well
below where it's clearly worth running at full scale.

**Recommendation:** run rung 4 (the sharper patch-level gate) alongside a small rung-5
pilot rather than waiting — rung 4 shares the pilot's response cache, so the marginal
cost of checking the gate properly is low.

---

## 6. What's left, ordered by value per unit cost

### Free — no API spend, no GPU, run anytime

1. **H<sub>edge</sub> error-rate collapse.** Sweep budget across instances with
   different measured *H*<sub>edge</sub>; log-error vs. *B*/*H*<sub>edge</sub> should
   collapse onto one line. Validates the bound's functional form, not just its
   direction — the strongest theory-to-data test not yet run.
2. **Resolve the sound-vs-beam presentation** (§5).
3. **Random-path probe variant of Fig. 1** — closes the gap between the two probe
   channels used in the paper's math vs. its identification experiments.
4. **RouterBench non-stationary routing** — the real-data analog of the cache-shift
   result.

### Gated — needs AWS Bedrock credentials

5. MATH / GPQA / GSM8K sweep (the flagship result).
6. Capability ladder (5 models) and the tree-Lipschitz characterization (Table 8) —
   the latter shares its response cache with the sweep above.
7. Code ladder rungs 1–2, 4–5 (§4), and τ-bench with the now-fixed learned router.

### Gated — needs Docker / Harbor / one GPU host

8. Code ladder rungs 4–6 (Docker + `swebench`, and a separate Harbor venv).
9. vLLM live-server measurements (prefix-cache ON/OFF, the KV-budget frontier, the
   GPU-calibrated eviction comparison) — any 24GB+ card, a few hours wall-clock.

---

## 7. Proposed phases

1. **Free robustness runs** — no dependency, can start immediately, run in parallel
   with the next phase.
2. **AWS Bedrock sweep** — MATH → GPQA → GSM8K → capability ladder → tree-Lipschitz
   characterization → code ladder rungs 1–2 → τ-bench, in that order (each shares
   cache/infrastructure with the one before it).
3. **Docker / Harbor / GPU phase** — code ladder rungs 4–6, then vLLM systems
   validation.
4. **Assemble and submit** — fold every regenerated number into the paper, resolve or
   explicitly scope the sound-vs-beam item, finalize the rung-3 recommendation.

---

Full technical detail — every issue found, exactly what changed, and the complete
runbook with commands — is in `docs/review_changes_and_runbook.md` and
`docs/code_benchmarks.md`.
