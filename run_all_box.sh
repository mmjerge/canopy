#!/usr/bin/env bash
# run_all_box.sh -- run the full Canopy experiment suite on the GPU box (Linux), sequentially,
# logged, and resumable. Each script parallelizes internally (--workers / seeds), so we run one
# job at a time to avoid Bedrock throttling and keep the output readable.
#
# Launch inside tmux so it survives disconnects:
#   tmux new -s canopy
#   cd ~/canopy && bash run_all_box.sh            # everything
#   GROUP=reasoning bash run_all_box.sh           # just the reasoning sweep
#   GROUP=synthetic bash run_all_box.sh           # just the no-AWS synthetic figures
#   GROUP=routing   bash run_all_box.sh           # routerbench + mmlu + tau-bench
#   GROUP=systems   bash run_all_box.sh           # vLLM (needs GPU + vllm)
# Detach: Ctrl-b then d.  Reattach: tmux attach -t canopy.
#
# Override: PY=/path/python WORKERS=16 N_MATH=300 N_GSM8K=200 bash run_all_box.sh
set -uo pipefail
cd "$(dirname "$0")"

PY="${PY:-$HOME/canopy/.venv/bin/python}"
VLLM_PY="${VLLM_PY:-$PY}"
WORKERS="${WORKERS:-16}"
GROUP="${GROUP:-all}"
N_MATH="${N_MATH:-300}"
N_GSM8K="${N_GSM8K:-200}"
N_MMLU_TRIALS="${N_MMLU_TRIALS:-20}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
mkdir -p logs

# Reasoning model pool as "tag|model_id" (portable; no bash-4 associative arrays needed). The
# headline (llama-70b) uses tag "headline" -> EMPTY --tag, so it writes the paper's canonical file
# names (reasoning_search_<bench>.*); the rest get a tag so their outputs don't collide.
REASON=(
  "headline|us.meta.llama3-1-70b-instruct-v1:0"
  "llama8b|us.meta.llama3-1-8b-instruct-v1:0"
  "sonnet45|us.anthropic.claude-sonnet-4-5-20250929-v1:0"
  "mistral-large|mistral.mistral-large-2402-v1:0"
  "nova-pro|us.amazon.nova-pro-v1:0"
)
HEADLINE_MODEL="us.meta.llama3-1-70b-instruct-v1:0"

want() { [ "$GROUP" = "all" ] || [ "$GROUP" = "$1" ]; }
run() {  # run <name> <cmd...> : timestamped, tee'd to logs/<name>.log, never aborts the suite
  local name="$1"; shift
  echo ">>> $name  ($(date '+%H:%M:%S'))"
  "$@" 2>&1 | tee "logs/$name.log" || echo "!!! $name failed (continuing)"
}

if ! "$PY" -c "import canopy" 2>/dev/null; then
  echo "ERROR: canopy not importable in $PY (run: $PY -m pip install -e .)"; exit 1
fi
echo "interpreter=$PY  workers=$WORKERS  group=$GROUP  logs=./logs/"

# --- Synthetic (no AWS): deterministic tree-bandit figures + theory link ---------------------
if want synthetic; then
  run tree_topk      "$PY" examples/tree_bandits/benchmark.py
  run tree_regret    "$PY" examples/tree_bandits/regret_storage_demo.py
  run tree_violation "$PY" examples/tree_bandits/violation_regret_demo.py
  run tree_infinite  "$PY" examples/tree_bandits/infinite_depth_demo.py
  run ablation       "$PY" examples/tree_bandits/lipschitz_ablation.py
  run theory_link    "$PY" examples/analysis/theory_link.py
  run prefix_cache   "$PY" examples/llm_routing/prefix_cache.py \
                       --dataset tatsu-lab/alpaca --n-prompts 20000
fi

# --- Reasoning (Bedrock): MATH + GSM8K across the model pool + mechanism + combine -----------
if want reasoning; then
  for entry in "${REASON[@]}"; do
    tag="${entry%%|*}"; model="${entry#*|}"
    tagflag=(); [ "$tag" != "headline" ] && tagflag=(--tag "$tag")
    run "reason_math_${tag}" "$PY" examples/reasoning/reasoning_search.py \
      --benchmark math  --model "$model" "${tagflag[@]}" \
      --n-problems "$N_MATH" --depth-sweep 2,4,6,8,10 --workers "$WORKERS" --resume
    run "reason_gsm8k_${tag}" "$PY" examples/reasoning/reasoning_search.py \
      --benchmark gsm8k --model "$model" "${tagflag[@]}" \
      --n-problems "$N_GSM8K" --depth-sweep 2,3,4,5 --workers "$WORKERS" --resume
  done
  # mechanism characterization on the headline model (the "almost tree-K-Lipschitz" evidence)
  run reason_tree "$PY" examples/analysis/reasoning_tree_lipschitz.py \
    --benchmark math --model "$HEADLINE_MODEL" \
    --n-problems "$N_MATH" --branching 3 --n-steps 6 --rollouts 8 --workers "$WORKERS"
  # combine the per-model runs into the capability-sweep figures
  run combine_math  "$PY" examples/reasoning/combine_reasoning_models.py --benchmark math
  run combine_gsm8k "$PY" examples/reasoning/combine_reasoning_models.py --benchmark gsm8k
fi

# --- Routing (offline RouterBench + Bedrock MMLU + tau-bench) --------------------------------
if want routing; then
  run routerbench "$PY" examples/llm_routing/routerbench_routing.py
  run mmlu        "$PY" examples/llm_routing/mmlu_routing.py --trials "$N_MMLU_TRIALS"
  run prompt_trim "$PY" examples/llm_routing/prompt_optimization.py --n-subjects 16 --q-per 32
  run taubench    "$PY" examples/agentic/taubench_routing.py \
    --env retail --num-tasks 80 --trials 1 --resume \
    --models "amazon.nova-micro-v1:0,amazon.nova-lite-v1:0,us.meta.llama3-1-8b-instruct-v1:0,us.meta.llama3-1-70b-instruct-v1:0,mistral.mistral-large-2402-v1:0,us.anthropic.claude-sonnet-4-5-20250929-v1:0" \
    --user-model "bedrock/amazon.nova-pro-v1:0"
fi

# --- Systems (vLLM prefix caching; needs an NVIDIA GPU + vllm importable) --------------------
if want systems; then
  if command -v nvidia-smi >/dev/null 2>&1 && "$VLLM_PY" -c "import vllm" >/dev/null 2>&1; then
    run vllm_onoff  "$VLLM_PY" examples/systems/vllm_prefix_cache_eval.py \
      --model "$VLLM_MODEL" --num-prompts 2000 --pad-tokens 200 --request-rate 20 \
      --max-tokens 64 --shift
    run vllm_budget "$VLLM_PY" examples/systems/vllm_prefix_cache_eval.py \
      --model "$VLLM_MODEL" --num-prompts 2000 --pad-tokens 200 \
      --kv-block-sweep 500,1000,2000,4000
    run vllm_policy "$VLLM_PY" examples/systems/vllm_policy_eval.py \
      --model "$VLLM_MODEL" --num-prompts 4000 --pad-tokens 200 --kv-budget 64
  else
    echo "  skipping systems: needs nvidia-smi and vllm importable in $VLLM_PY"
  fi
fi

echo "=== done ($(date '+%H:%M:%S')). figures/tables in paper/figures/, logs in logs/ ==="
