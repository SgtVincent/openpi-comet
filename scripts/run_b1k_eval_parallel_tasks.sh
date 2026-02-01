#!/usr/bin/env bash
set -euo pipefail

# Parallel BEHAVIOR-1K evaluation launcher for OpenPi-Comet (multi-task).
#
# What it does:
# - Runs OpenPi policy servers + BEHAVIOR evaluators in parallel
# - Assigns ONE task per (server, eval) pair (i.e., per eval simulator process)
#
# Pairing modes:
# - split:    servers use GPU[0..half-1], evaluators use GPU[half..NUM_GPUS-1] (default; safest for VRAM)
# - colocate: each pair shares the same GPU id for server + evaluator (enables 8 pairs on 8 GPUs)
#
# Why one-task-per-server:
# - The served policy wrapper keeps per-episode rollout state (action queue / step counter).
# - BEHAVIOR websocket client sends explicit reset messages between episodes.
# - If multiple evaluators share one server, their resets + rollout state will collide.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Comma-separated list of task names.
# Example:
#   TASKS="turning_on_radio,picking_up_trash,tidying_bedroom" NUM_GPUS=8 ./scripts/run_b1k_eval_parallel_tasks.sh
# TASKS="${TASKS:-turning_on_radio}"

# Most difficult 8 tasks from task_skill_complexity.ipynb:
# TASKS="${TASKS:-make_pizza,cook_cabbage,carrying_in_groceries,clean_up_your_desk,canning_food,cook_bacon,can_meat,chop_an_onion}"

# Easiest 8 tasks from task_skill_complexity.ipynb:
TASKS="${TASKS:-putting_shoes_on_rack,picking_up_trash,setting_mousetraps,turning_on_radio,hiding_Easter_eggs,bringing_in_wood,moving_boxes_to_storage,sorting_vegetables}"

# Pairing mode:
# - split:    servers use GPU[0..half-1], evaluators use GPU[half..NUM_GPUS-1] (default; safest for VRAM)
# - colocate: each pair shares the same GPU id for server + evaluator (enables 8 pairs on 8 GPUs)
PAIR_MODE="${PAIR_MODE:-colocate}"  # split | colocate

# Optional: explicitly choose which GPU ids to use (comma-separated).
# If set, the script will ONLY use these GPUs (ignores NUM_GPUS for placement).
# Example: GPU_IDS="4,5,6,7" SLOTS_PER_GPU=2  -> 8 pairs total, 2 pairs per GPU.
GPU_IDS="${GPU_IDS:-}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-1}"

NUM_GPUS="${NUM_GPUS:-8}"
PORT_BASE="${PORT_BASE:-8000}"

# How many BEHAVIOR *test-instance indices* (0..9) to run per task.
# NOTE: This is NOT "episodes" in the RL sense; it selects entries from the public test split.
EVAL_EPISODES="${EVAL_EPISODES:-10}"

# Websocket/health-check host used by the BEHAVIOR evaluator.
# Use IPv4 localhost to avoid IPv6/proxy edge cases.
MODEL_HOST="${MODEL_HOST:-127.0.0.1}"

# Ensure localhost health checks never go through an HTTP proxy.
export NO_PROXY="localhost,127.0.0.1,::1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="localhost,127.0.0.1,::1${no_proxy:+,$no_proxy}"

# JAX/XLA GPU memory behavior for the policy servers.
# See: https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
# If unset, choose a safer default when multiple servers share a GPU.
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-}"
# Default to a conservative fraction to reduce the chance of VRAM contention
# when running policy server + simulator on the same GPU (PAIR_MODE=colocate).
XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

BEHAVIOR_DIR="${BEHAVIOR_DIR:-/home/ubuntu/repo/BEHAVIOR-1K}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/checkpoints/openpi_comet/pi05-b1kpt50-cs32}"
OPENPI_CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_b1k-base}"

CONTROL_MODE="${CONTROL_MODE:-receeding_horizon}"
MAX_LEN="${MAX_LEN:-32}"

WRITE_VIDEO="${WRITE_VIDEO:-true}"
HEADLESS="${HEADLESS:-true}"

# Optional: cap max_steps to keep quick sanity runs short.
# Example: MAX_STEPS=200
MAX_STEPS="${MAX_STEPS:-}"

# How long to wait after launching servers before starting evaluators.
# Some checkpoints take longer to load on busy machines.
SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-10}"

# Evaluator entrypoint:
# - eval.py:        upstream BEHAVIOR evaluator
# - eval_custom.py: OpenPi-Comet custom evaluator (requires files copied into OmniGibson/omnigibson/learning)
EVAL_ENTRYPOINT="${EVAL_ENTRYPOINT:-eval.py}"  # eval.py | eval_custom.py

# To reproduce OpenPi-Comet settings, default to using the full-resolution RGB wrapper.
# This is safe for both eval.py and eval_custom.py.
ENV_WRAPPER_TARGET="${ENV_WRAPPER_TARGET:-omnigibson.learning.wrappers.RGBWrapper}"

RUN_TAG="${RUN_TAG:-parallel_tasks_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/eval_logs/pi05-b1kpt50-cs32/$RUN_TAG}"

# DISPLAY
export DISPLAY=:10.0

mkdir -p "$OUT_DIR"

if [[ ! -d "$BEHAVIOR_DIR" ]]; then
  echo "ERROR: BEHAVIOR_DIR not found: $BEHAVIOR_DIR" >&2
  exit 1
fi

if [[ ! -d "$CKPT_DIR" ]]; then
  echo "ERROR: CKPT_DIR not found: $CKPT_DIR" >&2
  exit 1
fi

case "$PAIR_MODE" in
  split|colocate) ;;
  *)
    echo "ERROR: PAIR_MODE must be 'split' or 'colocate' (got: $PAIR_MODE)" >&2
    exit 1
    ;;
esac

if ! [[ "$SLOTS_PER_GPU" =~ ^[0-9]+$ ]] || (( SLOTS_PER_GPU < 1 )); then
  echo "ERROR: SLOTS_PER_GPU must be an integer >= 1 (got: $SLOTS_PER_GPU)" >&2
  exit 1
fi

if [[ "$PAIR_MODE" == "split" ]]; then
  if (( NUM_GPUS < 2 )); then
    echo "ERROR: NUM_GPUS must be >= 2 in split mode" >&2
    exit 1
  fi

  half=$((NUM_GPUS / 2))
  if (( half < 1 )); then
    echo "ERROR: computed half=$half" >&2
    exit 1
  fi

  NUM_PAIRS=${NUM_PAIRS:-$half}
  if (( NUM_PAIRS > half )); then
    echo "ERROR: NUM_PAIRS=$NUM_PAIRS exceeds available server GPUs ($half)" >&2
    exit 1
  fi
  if (( NUM_PAIRS > (NUM_GPUS - half) )); then
    echo "ERROR: NUM_PAIRS=$NUM_PAIRS exceeds available eval GPUs ($((NUM_GPUS-half)))" >&2
    exit 1
  fi

  SERVER_GPU_BASE=0
  EVAL_GPU_BASE=$half
else
  if (( NUM_GPUS < 1 )); then
    echo "ERROR: NUM_GPUS must be >= 1" >&2
    exit 1
  fi

  NUM_PAIRS=${NUM_PAIRS:-$NUM_GPUS}
  if (( NUM_PAIRS > NUM_GPUS )); then
    echo "ERROR: NUM_PAIRS=$NUM_PAIRS exceeds available GPUs ($NUM_GPUS) in colocate mode" >&2
    exit 1
  fi

  SERVER_GPU_BASE=0
  EVAL_GPU_BASE=0
fi

if (( EVAL_EPISODES > 10 )); then
  echo "ERROR: EVAL_EPISODES=$EVAL_EPISODES exceeds BEHAVIOR test instance count (10)" >&2
  exit 1
fi

# Parse tasks into an array.
IFS=',' read -r -a TASK_ARRAY <<< "$TASKS"

# If GPU_IDS is provided, we run one colocated (server+eval) pair per task,
# distributed across the provided GPU ids.
declare -a GPU_ID_ARRAY
USE_GPU_IDS=false
if [[ -n "$GPU_IDS" ]]; then
  USE_GPU_IDS=true
  IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
  if (( ${#GPU_ID_ARRAY[@]} == 0 )); then
    echo "ERROR: GPU_IDS provided but empty" >&2
    exit 1
  fi
  for g in "${GPU_ID_ARRAY[@]}"; do
    if ! [[ "$g" =~ ^[0-9]+$ ]]; then
      echo "ERROR: GPU_IDS must be comma-separated integers (got: $GPU_IDS)" >&2
      exit 1
    fi
  done
fi

if [[ -z "$XLA_PYTHON_CLIENT_MEM_FRACTION" ]]; then
  if [[ "$USE_GPU_IDS" == "true" ]] && (( SLOTS_PER_GPU > 1 )); then
    XLA_PYTHON_CLIENT_MEM_FRACTION="0.20"
  else
    XLA_PYTHON_CLIENT_MEM_FRACTION="0.40"
  fi
fi

echo "Writing outputs to: $OUT_DIR"
echo "Tasks: $TASKS"
echo "GPUs: $NUM_GPUS   Port base: $PORT_BASE   Per-task eval_instance_ids: 0..$((EVAL_EPISODES-1))"
echo "Model host: $MODEL_HOST"
if [[ "$USE_GPU_IDS" == "true" ]]; then
  echo "GPU_IDS: $GPU_IDS (SLOTS_PER_GPU=$SLOTS_PER_GPU)"
fi
if [[ "$PAIR_MODE" == "split" ]]; then
  echo "Server GPUs: 0..$((half-1))"
  echo "Eval GPUs:   $half..$((NUM_GPUS-1))"
else
  echo "GPUs (colocate): 0..$((NUM_GPUS-1)) (each runs 1 server + 1 evaluator)"
fi

if [[ "$USE_GPU_IDS" == "true" ]]; then
  NUM_PAIRS=${NUM_PAIRS:-$(( ${#GPU_ID_ARRAY[@]} * SLOTS_PER_GPU ))}
  if (( NUM_PAIRS > ${#TASK_ARRAY[@]} )); then
    NUM_PAIRS=${#TASK_ARRAY[@]}
  fi
fi

echo "Pairs:       $NUM_PAIRS"
echo "Server XLA: PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION ALLOCATOR=$XLA_PYTHON_CLIENT_ALLOCATOR"

# Build eval_instance_ids string, e.g. "[0,1,2]".
eval_ids="["
for ((i=0; i<EVAL_EPISODES; i++)); do
  if (( i > 0 )); then
    eval_ids+="," 
  fi
  eval_ids+="$i"
done
eval_ids+="]"

declare -a SERVER_PIDS
declare -a EVAL_PIDS
cleanup() {
  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      # Kill the entire process group so wrappers (e.g., uv / conda) don't leak children.
      kill -- -"$pid" 2>/dev/null || true
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${EVAL_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -- -"$pid" 2>/dev/null || true
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

echo -e "\n== Launching policy servers (one per task) =="
for ((p=0; p<NUM_PAIRS; p++)); do
  task="${TASK_ARRAY[$p]:-}"
  if [[ -z "$task" ]]; then
    echo "- Pair $p: no task assigned; skipping"
    continue
  fi

  if [[ "$USE_GPU_IDS" == "true" ]]; then
    server_gpu="${GPU_ID_ARRAY[$((p % ${#GPU_ID_ARRAY[@]}))]}"
  else
    server_gpu=$((SERVER_GPU_BASE + p))
  fi
  port=$((PORT_BASE + p))
  server_log="$OUT_DIR/server_gpu${server_gpu}_p${port}_${task}.log"

  echo "- Pair $p: task=$task  server_gpu=$server_gpu  port=$port"

  setsid bash -c "
    set -euo pipefail
    cd \"$REPO_ROOT\"
    export CUDA_VISIBLE_DEVICES=\"$server_gpu\"
    export XLA_PYTHON_CLIENT_PREALLOCATE=\"$XLA_PYTHON_CLIENT_PREALLOCATE\"
    export XLA_PYTHON_CLIENT_MEM_FRACTION=\"$XLA_PYTHON_CLIENT_MEM_FRACTION\"
    export XLA_PYTHON_CLIENT_ALLOCATOR=\"$XLA_PYTHON_CLIENT_ALLOCATOR\"
    exec uv run scripts/serve_b1k.py \
      --task_name=\"$task\" \
      --control_mode=\"$CONTROL_MODE\" \
      --max_len=\"$MAX_LEN\" \
      --port=\"$port\" \
      policy:checkpoint \
      --policy.config=\"$OPENPI_CONFIG_NAME\" \
      --policy.dir=\"$CKPT_DIR\"
  " >"$server_log" 2>&1 &

  SERVER_PIDS+=("$!")
done

echo -e "\nWaiting ${SERVER_STARTUP_WAIT}s for servers to start..."
sleep "$SERVER_STARTUP_WAIT"

echo -e "\n== Launching BEHAVIOR evaluators (one task per eval process) =="
for ((p=0; p<NUM_PAIRS; p++)); do
  task="${TASK_ARRAY[$p]:-}"
  if [[ -z "$task" ]]; then
    continue
  fi

  if [[ "$USE_GPU_IDS" == "true" ]]; then
    eval_gpu="${GPU_ID_ARRAY[$((p % ${#GPU_ID_ARRAY[@]}))]}"
  else
    eval_gpu=$((EVAL_GPU_BASE + p))
  fi
  port=$((PORT_BASE + p))
  eval_log="$OUT_DIR/eval_gpu${eval_gpu}_p${port}_${task}.log"
  eval_out="$OUT_DIR/${task}/eval_gpu${eval_gpu}_p${port}"
  mkdir -p "$eval_out"

  setsid bash -c "
    set -euo pipefail
    cd \"$BEHAVIOR_DIR\"
    # Isaac / Omniverse warns that setting CUDA_VISIBLE_DEVICES can be problematic.
    # We select the GPU via OMNIGIBSON_GPU_ID instead.
    unset CUDA_VISIBLE_DEVICES
    export OMNIGIBSON_GPU_ID=\"$eval_gpu\"
    exec conda run -n behavior --no-capture-output \
      python \"OmniGibson/omnigibson/learning/${EVAL_ENTRYPOINT}\" \
        policy=websocket \
        task.name=\"$task\" \
        log_path=\"$eval_out\" \
        headless=$HEADLESS \
        write_video=$WRITE_VIDEO \
        env_wrapper._target_=\"$ENV_WRAPPER_TARGET\" \
        ${MAX_STEPS:+max_steps=$MAX_STEPS} \
        model.host=\"$MODEL_HOST\" \
        model.port=\"$port\" \
        eval_instance_ids=\"$eval_ids\"
  " >"$eval_log" 2>&1 &

  EVAL_PIDS+=("$!")
  echo "- Started eval: pair=$p task=$task eval_gpu=$eval_gpu port=$port -> $eval_out"
done

if (( ${#EVAL_PIDS[@]} == 0 )); then
  echo "ERROR: No evaluators were started (check TASKS / NUM_PAIRS)." >&2
  exit 1
fi

echo -e "\nAll evaluators launched. Tail logs, e.g.:"
example_task="${TASK_ARRAY[0]:-}"
example_eval_gpu="$EVAL_GPU_BASE"
echo "  tail -f ${OUT_DIR}/eval_gpu${example_eval_gpu}_p${PORT_BASE}_${example_task}.log"
echo "  tail -f ${OUT_DIR}/server_gpu0_p${PORT_BASE}_${example_task}.log"

# Wait for evaluators to finish. Servers run forever and will be killed by the EXIT trap.
wait "${EVAL_PIDS[@]}"