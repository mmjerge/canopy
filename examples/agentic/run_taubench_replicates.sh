#!/usr/bin/env bash
# Seed-variance replicates for the tau-bench learned-routing result.
#
# Runs 4 additional independent replicates (rep2..rep5) of ONLY the two learned routers
# (regional vs flat) over the same 80 retail tasks as the main run. The fixed-model
# baselines are not re-run: the claim under scrutiny is the regional-vs-flat learner
# comparison, which was previously a single trial. Each replicate uses its own LLM cache
# and tagged output files, so nothing overwrites the main artifacts.
#
# Waits for the pre-registered SWE-bench hard-tier tmux session (swehard) to finish first,
# because both workloads call claude-sonnet-4.5 on Bedrock and would contend for quota.
#
# Run on the box, in tmux:
#   tmux new-session -d -s taureps 'bash examples/agentic/run_taubench_replicates.sh 2>&1 | tee logs/taubench_replicates.log'
set -u

PY=~/canopy/.venv/bin/python
cd ~/canopy || exit 1

while tmux has-session -t swehard 2>/dev/null; do
  echo "waiting for swehard to finish... $(date -u)"
  sleep 300
done

for rep in rep2 rep3 rep4 rep5; do
  echo "=== tau-bench replicate $rep $(date -u) ==="
  "$PY" examples/agentic/taubench_routing.py --env retail --num-tasks 80 --trials 1 \
    --learners-only --out-tag "$rep" \
    --cache "examples/.cache/taubench_routing_${rep}.jsonl" --resume
done

echo "=== replicates complete $(date -u) ==="
