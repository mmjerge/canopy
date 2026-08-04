#!/usr/bin/env bash
# Pre-registered SWE-bench Verified "1-4 hours" tier run (see PREREG_swebench_hard_tier.md).
# Protocol identical to run_swebench_sweep.sh except the difficulty tier: all 42 tier
# instances, claude-sonnet-4.5, depths 1 and 2 (matched budgets 6 and 9), branching 3.
#
# Run on the box, in tmux:
#   tmux new-session -d -s swehard 'bash examples/reasoning/run_swebench_hard_tier.sh 2>&1 | tee logs/swebench_hard.log'
set -u

PY=~/canopy/.venv/bin/python
DATASET=princeton-nlp/SWE-bench_Verified
DIFF="1-4 hours"
N=42
BRANCH=3
WORKERS=${WORKERS:-4}
RUNID=canopy_swe_hard
MODEL="us.anthropic.claude-sonnet-4-5-20250929-v1:0"

cd ~/canopy || exit 1

for depth in 2 1; do  # headline budget (9) first
  tag="swebench_hard_sonnet45_d${depth}"
  echo "=== cell: 1-4h tier depth=$depth (budget=$((BRANCH*(depth+1)))) tag=$tag $(date -u) ==="
  "$PY" examples/reasoning/swebench_search.py \
    --dataset "$DATASET" --difficulty "$DIFF" --n-instances "$N" \
    --model "$MODEL" --branching "$BRANCH" --depth "$depth" \
    --harness-workers "$WORKERS" --run-id "$RUNID" --tag "$tag" --resume
done

echo "=== hard-tier run complete $(date -u) ==="
