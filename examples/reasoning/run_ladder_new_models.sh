#!/bin/bash
# Capability-ladder runs for one new model, matching the published ladder protocol:
# MATH n=300, GSM8K n=200 (near-saturated control), GPQA-Diamond n=198.
#
# Usage: run_ladder_new_models.sh <bedrock_model_id> <tag>
#   e.g. run_ladder_new_models.sh global.anthropic.claude-sonnet-5 sonnet5
#
# Idempotent: --resume skips completed budget levels and the response cache makes
# already-generated completions free, so after any credential expiry just re-run
# the same command (mwinit first).
set -uo pipefail
MODEL="$1"; TAG="$2"
cd "$(dirname "$0")/../.."
unset AWS_BEARER_TOKEN_BEDROCK 2>/dev/null || true
export AWS_PROFILE=mjerge-Admin
PY=.venv/bin/python

creds_ok() {
  aws sts get-caller-identity --output text >/dev/null 2>&1
}

run() {
  local bench=$1 n=$2
  if ! creds_ok; then
    echo "!!! $TAG / $bench NOT STARTED: credentials dead (run mwinit, then re-run this script)"
    exit 1
  fi
  echo "=== $TAG / $bench (n=$n) started $(date '+%F %T') ==="
  $PY examples/reasoning/reasoning_search.py \
    --benchmark "$bench" --n-problems "$n" --model "$MODEL" --tag "$TAG" \
    --max-tokens 2048 --workers 8 --max-spend 300 --resume \
    2>&1 | tee -a "logs/reasoning_${bench}_${TAG}.log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "!!! $TAG / $bench FAILED (exit $rc) $(date '+%F %T') -- stopping; fix creds and re-run"
    exit "$rc"
  fi
  # a level saved with skipped problems is a biased subset, not a result
  if tail -50 "logs/reasoning_${bench}_${TAG}.log" | grep -q "skipped after retries"; then
    echo "!!! $TAG / $bench had SKIPPED PROBLEMS -- its levels are suspect; strip them and re-run"
    exit 1
  fi
  echo "=== $TAG / $bench finished clean $(date '+%F %T') ==="
}

run math 300
run gsm8k 200
run gpqa_diamond 198
echo "=== $TAG ladder COMPLETE $(date '+%F %T') ==="
