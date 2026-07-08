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
N_CODE="${N_CODE:-164}"   # HumanEval has 164 tasks; MBPP test split is larger
N_GPQA_D="${N_GPQA_D:-198}"   # GPQA-Diamond has 198 questions (all of them)
N_GPQA_MAIN="${N_GPQA_MAIN:-200}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
# ALFWorld runs in its OWN env (alfworld deps conflict with the main venv). Set ALFWORLD_CONFIG
# and point ALFWORLD_PY at that env's python; that env must have canopy reinstalled (pip install
# -e .) so it has the seed-keyed cache fix. ALFWORLD_DATA must be exported (alfworld-download).
ALFWORLD_PY="${ALFWORLD_PY:-$HOME/alfworld-venv/bin/python}"
ALFWORLD_CONFIG="${ALFWORLD_CONFIG:-}"
ALF_MODEL="${ALF_MODEL:-us.anthropic.claude-sonnet-4-5-20250929-v1:0}"
N_ALF="${N_ALF:-50}"
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

SKIP="${SKIP:-}"          # comma-separated groups to skip, e.g. SKIP=systems
want() {                  # run group $1 unless it's in SKIP and unless GROUP selects another
  case ",$SKIP," in *",$1,"*) return 1 ;; esac
  [ "$GROUP" = "all" ] || [ "$GROUP" = "$1" ]
}
run() {  # run <name> <cmd...> : timestamped, tee'd to logs/<name>.log, never aborts the suite
  local name="$1"; shift
  echo ">>> $name  ($(date '+%H:%M:%S'))"
  "$@" 2>&1 | tee "logs/$name.log" || echo "!!! $name failed (continuing)"
}

if ! "$PY" -c "import canopy" 2>/dev/null; then
  echo "ERROR: canopy not importable in $PY (run: $PY -m pip install -e .)"; exit 1
fi
echo "interpreter=$PY  workers=$WORKERS  group=$GROUP  skip=${SKIP:-none}  logs=./logs/"

# --- status: inventory what results already exist (no runs, no spend) -----------------------
if [ "$GROUP" = "status" ]; then
  echo "=== results present in paper/figures/ (what's already done) ==="
  "$PY" - <<'PYEOF'
import glob, json, os
files = sorted(glob.glob("paper/figures/*_results.json"))
if not files:
    print("  (none yet)")
for f in files:
    try:
        d = json.load(open(f))
        lv = d.get("levels")
        n = len(lv) if isinstance(lv, dict) else "-"
        print(f"  {os.path.basename(f):52s} levels={n}  n_problems={d.get('n_problems','?')}")
    except Exception as e:  # noqa: BLE001
        print(f"  {os.path.basename(f)}: unreadable ({e})")
PYEOF
  exit 0
fi

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
    # GPQA-Diamond (graduate science, hard for all models -> strong reachable-but-unreliable case)
    run "reason_gpqad_${tag}" "$PY" examples/reasoning/reasoning_search.py \
      --benchmark gpqa_diamond --model "$model" "${tagflag[@]}" \
      --n-problems "$N_GPQA_D" --workers "$WORKERS" --resume
  done
  # code domain (second flagship): HumanEval + MBPP on the headline model. Grades by EXECUTING
  # model-generated code in a sandboxed subprocess -- safe only on this disposable box.
  run reason_humaneval_headline "$PY" examples/reasoning/reasoning_search.py \
    --benchmark humaneval --model "$HEADLINE_MODEL" --n-problems "$N_CODE" \
    --workers "$WORKERS" --resume
  run reason_mbpp_headline "$PY" examples/reasoning/reasoning_search.py \
    --benchmark mbpp --model "$HEADLINE_MODEL" --n-problems "$N_CODE" \
    --workers "$WORKERS" --resume
  # GPQA main split on the headline model (broader science set beyond Diamond)
  run reason_gpqa_headline "$PY" examples/reasoning/reasoning_search.py \
    --benchmark gpqa --model "$HEADLINE_MODEL" --n-problems "$N_GPQA_MAIN" \
    --workers "$WORKERS" --resume
  # mechanism characterization on the headline model (the "almost tree-K-Lipschitz" evidence)
  run reason_tree "$PY" examples/analysis/reasoning_tree_lipschitz.py \
    --benchmark math --model "$HEADLINE_MODEL" \
    --n-problems "$N_MATH" --branching 3 --n-steps 6 --rollouts 8 --workers "$WORKERS"
  # combine the per-model runs into the capability-sweep figures
  run combine_math  "$PY" examples/reasoning/combine_reasoning_models.py --benchmark math
  run combine_gsm8k "$PY" examples/reasoning/combine_reasoning_models.py --benchmark gsm8k
  run combine_gpqad "$PY" examples/reasoning/combine_reasoning_models.py --benchmark gpqa_diamond
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

# --- ALFWorld (separate env; opt-in). Re-examines the earlier negative result now that the ------
# candidate-action sampling collapse (seedless cache) is fixed -- alfworld_search.py now threads
# the seed, so value-guided finally sees diverse candidate actions.
if want alfworld; then
  if [ -n "$ALFWORLD_CONFIG" ] && [ -x "$ALFWORLD_PY" ]; then
    run alfworld "$ALFWORLD_PY" examples/agentic/alfworld_search.py \
      --config "$ALFWORLD_CONFIG" --model "$ALF_MODEL" --num-tasks "$N_ALF" --resume
  else
    echo "  skipping alfworld: set ALFWORLD_CONFIG=/path/base_config.yaml, ensure \$ALFWORLD_PY"
    echo "  ($ALFWORLD_PY) exists with canopy reinstalled (pip install -e .), and export"
    echo "  ALFWORLD_DATA (run alfworld-download). See examples/agentic/alfworld_search.py header."
  fi
fi

echo "=== done ($(date '+%H:%M:%S')). figures/tables in paper/figures/, logs in logs/ ==="
