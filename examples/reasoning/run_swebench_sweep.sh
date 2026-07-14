#!/usr/bin/env bash
# SWE-bench model x compute-budget sweep for the repo-level value-guided-vs-best-of-N result.
#
# Runs value_guided vs best-of-N over a common subset of SWE-bench Verified "15 min - 1 hour"
# issues, across a five-model capability ladder and three search depths (matched budgets 6/9/12).
# Cells are ordered so the highest-value results land first: Sonnet's full budget curve, then all
# models at the headline depth (=2), then the remaining depths. A shared --run-id caches the
# per-repo Docker env images across cells, so only the first cell pays the full build cost. Every
# cell is --resume-safe (per-instance checkpoint), so this whole script can be re-run to continue.
#
# Run on the box, in tmux:
#   tmux new-session -d -s swesweep 'bash examples/reasoning/run_swebench_sweep.sh 2>&1 | tee logs/swebench_sweep.log'
set -u

PY=~/canopy/.venv/bin/python
DATASET=princeton-nlp/SWE-bench_Verified
DIFF="15 min - 1 hour"
N=${N:-80}                # instances per cell (common subset; override with N=...)
BRANCH=3
WORKERS=${WORKERS:-4}
RUNID=canopy_swe_sweep    # shared so env images / harness reports cache across cells

# model_id:short_label
MODELS=(
  "us.anthropic.claude-sonnet-4-5-20250929-v1:0:sonnet45"
  "us.amazon.nova-pro-v1:0:nova-pro"
  "us.meta.llama3-1-70b-instruct-v1:0:llama70b"
  "mistral.mistral-large-2402-v1:0:mistral-large"
  "us.meta.llama3-1-8b-instruct-v1:0:llama8b"
)

run_cell () {  # $1=model_id $2=short $3=depth
  local model="$1" short="$2" depth="$3"
  local tag="swebench_sweep_${short}_d${depth}"
  echo "=== cell: model=$short depth=$depth (budget=$((BRANCH*(depth+1)))) tag=$tag $(date -u) ==="
  "$PY" examples/reasoning/swebench_search.py \
    --dataset "$DATASET" --difficulty "$DIFF" --n-instances "$N" \
    --model "$model" --branching "$BRANCH" --depth "$depth" \
    --harness-workers "$WORKERS" --run-id "$RUNID" --tag "$tag" --resume
}

cd ~/canopy || exit 1

# Priority 1: Sonnet full budget curve (depths 2,1,3 -> budgets 9,6,12).
run_cell "us.anthropic.claude-sonnet-4-5-20250929-v1:0" "sonnet45" 2
run_cell "us.anthropic.claude-sonnet-4-5-20250929-v1:0" "sonnet45" 1
run_cell "us.anthropic.claude-sonnet-4-5-20250929-v1:0" "sonnet45" 3

# Priority 2: capability sweep at the headline depth (=2) for the other four models.
for mm in "${MODELS[@]:1}"; do
  run_cell "${mm%:*}" "${mm##*:}" 2   # %: strips from the LAST colon (model ids contain ':0')
done

# Priority 3: remaining budget points (depths 1 and 3) for the other four models.
for depth in 1 3; do
  for mm in "${MODELS[@]:1}"; do
    run_cell "${mm%:*}" "${mm##*:}" "$depth"
  done
done

echo "=== sweep complete $(date -u) ==="
