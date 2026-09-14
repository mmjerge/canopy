#!/bin/bash
# Relaunch the capability-ladder runs for the two new models, in parallel.
#
#   mwinit && ./examples/reasoning/run_new_models.sh
#
# Runs MATH (n=300) and GSM8K (n=200) for claude-sonnet-5 and gpt-5.6-sol, four
# jobs at once, each appending to its own log. Safe to re-run at any time:
# --resume skips completed budget levels and the response cache makes
# already-generated completions free, so after a credential expiry just run it
# again. GPQA-Diamond is deliberately excluded (slowest benchmark, and MATH +
# GSM8K already give the unsaturated-gain and saturated-null pair); add it with
# --benchmark gpqa_diamond --n-problems 198 if wanted.
set -uo pipefail
cd "$(dirname "$0")/../.."
unset AWS_BEARER_TOKEN_BEDROCK 2>/dev/null || true
export AWS_PROFILE=mjerge-Admin
PY=.venv/bin/python

if ! aws sts get-caller-identity --output text >/dev/null 2>&1; then
  echo "!!! credentials are dead -- run 'mwinit' first, then re-run this script"
  exit 1
fi
echo "credentials ok: $(aws sts get-caller-identity --query Arn --output text)"

launch() {
  local bench=$1 n=$2 model=$3 tag=$4
  local log="logs/reasoning_${bench}_${tag}.log"
  nohup $PY examples/reasoning/reasoning_search.py \
    --benchmark "$bench" --n-problems "$n" --model "$model" --tag "$tag" \
    --max-tokens 2048 --workers 8 --max-spend 300 --resume \
    >> "$log" 2>&1 &
  echo "  launched pid $! : $tag / $bench (n=$n) -> $log"
}

echo "launching four runs:"
launch math  300 global.anthropic.claude-sonnet-5 sonnet5
launch gsm8k 200 global.anthropic.claude-sonnet-5 sonnet5
launch math  300 global.openai.gpt-5.6-sol        gpt56sol
launch gsm8k 200 global.openai.gpt-5.6-sol        gpt56sol

cat <<'EOF'

All four are running in the background; you can close this terminal.
Check progress:   tail -f logs/reasoning_math_sonnet5.log
Count them:       ps aux | grep -c "[r]easoning_search.py"
If they die on a credential expiry: mwinit, then re-run this same script.
EOF
