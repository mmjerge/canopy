#!/usr/bin/env bash
# run_experiments.sh -- launch all Canopy paper experiments (backgrounded, logged, resumable).
#
#   Group A (no AWS):   synthetic tree-bandit figures + RouterBench + real-prompt caching.
#   Group B (Bedrock):  reasoning search (MATH + GSM8K), MMLU routing, prompt trimming, tau-bench.
#   Group C (alfworld): ALFWorld agentic search (only if ALFWORLD_CONFIG is set; separate env).
#
# Each job runs under nohup+caffeinate, writes logs/<name>.log, and drops its figure/table into
# paper/figures/. Bedrock jobs are spend-capped and --resume-safe (kill & rerun to continue).
#
# Usage:
#   bash run_experiments.sh            # launch everything
#   GROUP=A bash run_experiments.sh    # only the no-AWS jobs
#   GROUP=B bash run_experiments.sh    # only the Bedrock jobs
#   GROUP=C bash run_experiments.sh    # only ALFWorld
#
# Override anything via env vars, e.g.:
#   PY=/path/to/python CAP_TAUBENCH=20 ALFWORLD_CONFIG=/path/base_config.yaml bash run_experiments.sh
set -uo pipefail
cd "$(dirname "$0")"

PY="${PY:-/opt/anaconda3/envs/canopy-taubench/bin/python}"
ALFWORLD_PY="${ALFWORLD_PY:-/opt/anaconda3/envs/canopy-alfworld/bin/python}"
GROUP="${GROUP:-ALL}"
MODELS="${MODELS:-amazon.nova-micro-v1:0,amazon.nova-lite-v1:0,us.meta.llama3-1-8b-instruct-v1:0,us.meta.llama3-1-70b-instruct-v1:0,mistral.mistral-large-2402-v1:0,us.anthropic.claude-sonnet-4-5-20250929-v1:0}"
USER_MODEL="${USER_MODEL:-bedrock/amazon.nova-pro-v1:0}"

# Spend caps (USD), overridable per job.
CAP_REASON_MATH="${CAP_REASON_MATH:-60}"
CAP_REASON_GSM8K="${CAP_REASON_GSM8K:-30}"
CAP_MMLU="${CAP_MMLU:-8}"
CAP_TRIM="${CAP_TRIM:-8}"
CAP_TAUBENCH="${CAP_TAUBENCH:-30}"
CAP_ALFWORLD="${CAP_ALFWORLD:-20}"
# Sample sizes + reasoning model (bumped for robust results; override freely).
N_MATH="${N_MATH:-300}"
N_GSM8K="${N_GSM8K:-200}"
REASON_MODEL="${REASON_MODEL:-us.meta.llama3-1-70b-instruct-v1:0}"
# vLLM systems eval (Group D): needs a GPU + vllm in $VLLM_PY.
VLLM_PY="${VLLM_PY:-$PY}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
# ALFWorld model: default to a known-good Bedrock id (the one tau-bench uses), overridable.
ALF_MODEL="${ALF_MODEL:-us.anthropic.claude-sonnet-4-5-20250929-v1:0}"

mkdir -p logs
PIDS_FILE="logs/pids.txt"
: > "$PIDS_FILE"

launch() {  # launch <name> <cmd...>  -> nohup+caffeinate, logged, PID recorded
  local name="$1"; shift
  nohup caffeinate -dimsu "$@" > "logs/$name.log" 2>&1 &
  local pid=$!
  echo "$pid  $name" | tee -a "$PIDS_FILE"
}

# --- preflight -----------------------------------------------------------------------------
if [ ! -x "$PY" ]; then
  echo "ERROR: interpreter not found: $PY  (set PY=/path/to/python)"; exit 1
fi
if ! "$PY" -c "import canopy" 2>/dev/null; then
  echo "ERROR: 'canopy' not importable in $PY. Install it (pip install -e .) in that env."; exit 1
fi
if { [ "$GROUP" = "ALL" ] || [ "$GROUP" = "A" ] || [ "$GROUP" = "B" ]; } \
   && ! "$PY" -c "import datasets, pandas, matplotlib" 2>/dev/null; then
  echo "ERROR: missing deps in $PY. Run:"
  echo "  $PY -m pip install datasets pandas huggingface_hub matplotlib tiktoken"; exit 1
fi

echo "interpreter: $PY"
echo "group:       $GROUP"
echo "logs:        ./logs/<name>.log   (pids in $PIDS_FILE)"
echo

# --- Group A: no AWS ------------------------------------------------------------------------
if [ "$GROUP" = "ALL" ] || [ "$GROUP" = "A" ]; then
  echo "== Group A (no AWS) =="
  launch tree_topk       "$PY" examples/tree_bandits/benchmark.py
  launch tree_regret     "$PY" examples/tree_bandits/regret_storage_demo.py
  launch tree_violation  "$PY" examples/tree_bandits/violation_regret_demo.py
  launch tree_infinite   "$PY" examples/tree_bandits/infinite_depth_demo.py
  launch routerbench     "$PY" examples/llm_routing/routerbench_routing.py
  launch prefix_cache    "$PY" examples/llm_routing/prefix_cache.py --dataset tatsu-lab/alpaca --n-prompts 20000
fi

# --- Group B: Bedrock (spend-capped, resumable) ---------------------------------------------
if [ "$GROUP" = "ALL" ] || [ "$GROUP" = "B" ]; then
  echo "== Group B (Bedrock) =="
  launch reasoning_math  "$PY" examples/reasoning/reasoning_search.py \
    --benchmark math --model "$REASON_MODEL" --n-problems "$N_MATH" \
    --depth-sweep 2,4,6,8,10 --max-spend "$CAP_REASON_MATH" --resume
  launch reasoning_gsm8k "$PY" examples/reasoning/reasoning_search.py \
    --benchmark gsm8k --model "$REASON_MODEL" --n-problems "$N_GSM8K" \
    --depth-sweep 2,3,4,5 --max-spend "$CAP_REASON_GSM8K" --resume
  launch mmlu            "$PY" examples/llm_routing/mmlu_routing.py --max-spend "$CAP_MMLU"
  launch prompt_trim     "$PY" examples/llm_routing/prompt_optimization.py \
    --n-subjects 16 --q-per 32 --max-spend "$CAP_TRIM"
  launch taubench        "$PY" examples/agentic/taubench_routing.py \
    --env retail --num-tasks 80 --trials 1 --max-spend "$CAP_TAUBENCH" --resume \
    --models "$MODELS" --user-model "$USER_MODEL"
fi

# --- Group C: ALFWorld (separate env; only if configured) -----------------------------------
if [ "$GROUP" = "ALL" ] || [ "$GROUP" = "C" ]; then
  echo "== Group C (ALFWorld) =="
  if [ -n "${ALFWORLD_CONFIG:-}" ] && [ -x "$ALFWORLD_PY" ]; then
    launch alfworld "$ALFWORLD_PY" examples/agentic/alfworld_search.py \
      --config "$ALFWORLD_CONFIG" --model "$ALF_MODEL" \
      --num-tasks 50 --max-spend "$CAP_ALFWORLD" --resume
  else
    echo "  skipping ALFWorld: set ALFWORLD_CONFIG=/path/to/base_config.yaml and ensure"
    echo "  \$ALFWORLD_PY exists ($ALFWORLD_PY). See examples/agentic/alfworld_search.py header."
  fi
fi

# --- Group D: vLLM prefix-caching systems eval (needs an NVIDIA GPU + vllm) ------------------
if [ "$GROUP" = "ALL" ] || [ "$GROUP" = "D" ]; then
  echo "== Group D (vLLM systems eval) =="
  if command -v nvidia-smi >/dev/null 2>&1 && "$VLLM_PY" -c "import vllm" >/dev/null 2>&1; then
    launch vllm_systems "$VLLM_PY" examples/systems/vllm_prefix_cache_eval.py \
      --model "$VLLM_MODEL" --num-prompts 2000 --pad-tokens 200 \
      --request-rate 20 --max-tokens 64 --shift
  else
    echo "  skipping vLLM systems eval: needs an NVIDIA GPU (nvidia-smi) and vllm importable"
    echo "  in \$VLLM_PY ($VLLM_PY). Run 'GROUP=D VLLM_PY=/gpu/env/bin/python bash run_experiments.sh'"
    echo "  on the GPU box, or run examples/systems/vllm_prefix_cache_eval.py directly."
  fi
fi

echo
echo "all jobs launched (they keep running if you close this terminal)."
echo "monitor:   tail -f logs/reasoning_math.log      (or any logs/<name>.log)"
echo "list:      cat $PIDS_FILE"
echo "stop all:  kill \$(awk '{print \$1}' $PIDS_FILE)"
echo "figures/tables appear in paper/figures/ as each job finishes."
