#!/usr/bin/env bash
# Expand the SWE-bench tree-Lipschitz instrumentation (RQ4) from 20 to 60 issues.
#
# Precision improvement only: same measurement, same protocol, more instances, to grow the
# pivotal-step count beyond the current n=8 (whose rate CI is degenerate). Resumes over the
# existing 20 instances (their generations are cached), so only the 40 new issues pay cost.
# Whatever the expanded correlation/pivotal-rate numbers are, they replace the current ones.
#
# Waits for the taureps tmux session (which itself waits for swehard) to avoid Bedrock
# contention. Run on the box, in tmux:
#   tmux new-session -d -s treelip 'bash examples/analysis/run_treelip_expand.sh 2>&1 | tee logs/treelip_expand.log'
set -u

PY=~/canopy/.venv/bin/python
cd ~/canopy || exit 1

while tmux has-session -t swehard 2>/dev/null || tmux has-session -t taureps 2>/dev/null; do
  echo "waiting for swehard/taureps to finish... $(date -u)"
  sleep 300
done

"$PY" examples/analysis/swebench_tree_lipschitz.py \
  --dataset princeton-nlp/SWE-bench_Verified --difficulty "15 min - 1 hour" \
  --n-instances 60 \
  --model us.anthropic.claude-sonnet-4-5-20250929-v1:0 --branching 3 --depth 2 --resume

echo "=== tree-lipschitz expansion complete $(date -u) ==="
