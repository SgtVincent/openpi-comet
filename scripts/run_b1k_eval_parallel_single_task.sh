#!/usr/bin/env bash
set -euo pipefail

# Parallel BEHAVIOR-1K evaluation launcher for OpenPi-Comet.
#
# - Runs policy servers on the first half of GPUs (OpenPi / uv env)
# - Runs BEHAVIOR evaluators on the second half of GPUs (conda env: behavior)
# - Splits BEHAVIOR `eval_instance_ids` across pairs in round-robin
#
# Requirements:
# - `conda env list` includes an env named: behavior
# - BEHAVIOR-1K repo is present locally
# - Checkpoint directory exists locally (download with `proxy_on` enabled if needed)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TASK_NAME="${TASK_NAME:-turning_on_radio}"

# Pairing mode:
# - split:    servers use GPU[0..half-1], evaluators use GPU[half..NUM_GPUS-1] (safest for VRAM)
# - colocate: each pair shares the same GPU id for server + evaluator (maximizes throughput, higher VRAM risk)
PAIR_MODE="${PAIR_MODE:-split}"  # split | colocate

NUM_GPUS="${NUM_GPUS:-8}"
PORT_BASE="${PORT_BASE:-8000}"
# How many evaluation runs to launch.
# BEHAVIOR test split has 10 entries (indices 0..9). Each entry corresponds to 1 rollout episode.
EVAL_EPISODES="${EVAL_EPISODES:-10}"

# Websocket/health-check host used by the BEHAVIOR evaluator.
# Use IPv4 localhost to avoid IPv6/proxy edge cases.
MODEL_HOST="${MODEL_HOST:-127.0.0.1}"

# Ensure localhost health checks never go through an HTTP proxy.
# (The websocket client does `requests.get(http://<host>:<port>/healthz)`.)
export NO_PROXY="localhost,127.0.0.1,::1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="localhost,127.0.0.1,::1${no_proxy:+,$no_proxy}"

# JAX/XLA GPU memory behavior for the policy servers.
# These help prevent JAX from preallocating most of the GPU memory, leaving room for Isaac.
#
# See: https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
# Default to a conservative fraction to reduce the chance of VRAM contention.
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.40}"
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
SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-10}"

# Evaluator entrypoint:
# - eval.py:        upstream BEHAVIOR evaluator
# - eval_custom.py: OpenPi-Comet custom evaluator (requires files copied into OmniGibson/omnigibson/learning)
EVAL_ENTRYPOINT="${EVAL_ENTRYPOINT:-eval.py}"  # eval.py | eval_custom.py

# To reproduce OpenPi-Comet settings, default to using the full-resolution RGB wrapper.
ENV_WRAPPER_TARGET="${ENV_WRAPPER_TARGET:-omnigibson.learning.wrappers.RGBWrapper}"

# Server mode:
# - per_eval: start one policy server per evaluator process (highest compatibility, most GPU/VRAM usage)
# - shared:  start ONE server and let multiple evaluators connect concurrently
#            (recommended when simulation is the bottleneck and model inference is fast)
SERVER_MODE="${SERVER_MODE:-per_eval}"  # per_eval | shared

# Shared-server tuning:
# By default, the shared server reserves its GPU exclusively for the model.
# If you set this to true, one evaluator will also run on the server GPU (higher utilization, higher VRAM risk).
ALLOW_SERVER_GPU_AS_WORKER="${ALLOW_SERVER_GPU_AS_WORKER:-false}"  # true | false

# Optionally pin the shared server to a specific GPU id.
# Default: GPU 0 (or the first server GPU in split mode).
SERVER_GPU_ID="${SERVER_GPU_ID:-}"

RUN_TAG="${RUN_TAG:-parallel_${TASK_NAME}_$(date +%Y%m%d_%H%M%S)}"
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

echo "Writing outputs to: $OUT_DIR"
echo "Task: $TASK_NAME"
echo "GPUs: $NUM_GPUS   Port base: $PORT_BASE   Eval episodes: $EVAL_EPISODES"
echo "Model host: $MODEL_HOST"
echo "Server XLA: PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION ALLOCATOR=$XLA_PYTHON_CLIENT_ALLOCATOR"
echo "Pair mode: $PAIR_MODE"
echo "Server mode: $SERVER_MODE"
echo "ALLOW_SERVER_GPU_AS_WORKER: $ALLOW_SERVER_GPU_AS_WORKER"

case "$PAIR_MODE" in
  split|colocate) ;;
  *)
    echo "ERROR: PAIR_MODE must be 'split' or 'colocate' (got: $PAIR_MODE)" >&2
    exit 1
    ;;
esac

case "$SERVER_MODE" in
  per_eval|shared) ;;
  *)
    echo "ERROR: SERVER_MODE must be 'per_eval' or 'shared' (got: $SERVER_MODE)" >&2
    exit 1
    ;;
esac

if [[ "$PAIR_MODE" == "split" ]]; then
  if (( NUM_GPUS < 2 )); then
    echo "ERROR: NUM_GPUS must be >= 2 in split mode" >&2
    exit 1
  fi

  # Split GPUs: servers on [0..half-1], eval on [half..NUM_GPUS-1].
  half=$((NUM_GPUS / 2))
  if (( half < 1 )); then
    echo "ERROR: computed half=$half" >&2
    exit 1
  fi

  # Number of (server, eval) pairs.
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
  echo "Server GPUs: 0..$((half-1))"
  echo "Eval GPUs:   $half..$((NUM_GPUS-1))"
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
  if [[ "$SERVER_MODE" == "per_eval" ]]; then
    echo "GPUs (colocate): 0..$((NUM_GPUS-1)) (each runs 1 server + 1 evaluator)"
  else
    echo "GPUs (colocate): 0..$((NUM_GPUS-1)) (shared server + multiple evaluators)"
  fi
fi

echo "Pairs:       $NUM_PAIRS"

if (( EVAL_EPISODES > 10 )); then
  echo "ERROR: EVAL_EPISODES=$EVAL_EPISODES exceeds BEHAVIOR test instance count (10)" >&2
  exit 1
fi

declare -a WORKER_TO_EPISODES

if [[ "$SERVER_MODE" == "per_eval" ]]; then
  # Distribute indices [0..EVAL_EPISODES-1] across pairs in round-robin.
  for ((p=0; p<NUM_PAIRS; p++)); do
    WORKER_TO_EPISODES[p]=""
  done
  for ((idx=0; idx<EVAL_EPISODES; idx++)); do
    p=$((idx % NUM_PAIRS))
    if [[ -z "${WORKER_TO_EPISODES[p]}" ]]; then
      WORKER_TO_EPISODES[p]="$idx"
    else
      WORKER_TO_EPISODES[p]="${WORKER_TO_EPISODES[p]},$idx"
    fi
  done
else
  # Shared server: use ALL available eval GPUs as workers (excluding the server GPU).
  # Determine worker GPU ids.
  declare -a WORKER_GPU_IDS

  # Pick server GPU.
  if [[ -z "$SERVER_GPU_ID" ]]; then
    SERVER_GPU_ID="$SERVER_GPU_BASE"
  fi
  if ! [[ "$SERVER_GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SERVER_GPU_ID must be an integer (got: $SERVER_GPU_ID)" >&2
    exit 1
  fi
  if (( SERVER_GPU_ID < 0 || SERVER_GPU_ID >= NUM_GPUS )); then
    echo "ERROR: SERVER_GPU_ID=$SERVER_GPU_ID out of range for NUM_GPUS=$NUM_GPUS" >&2
    exit 1
  fi

  if [[ "$PAIR_MODE" == "split" ]]; then
    # Candidate workers are the eval half.
    for ((g=EVAL_GPU_BASE; g<NUM_GPUS; g++)); do
      if (( g == SERVER_GPU_ID )) && [[ "$ALLOW_SERVER_GPU_AS_WORKER" != "true" ]]; then
        continue
      fi
      WORKER_GPU_IDS+=("$g")
    done
  else
    # Candidate workers are all GPUs.
    for ((g=0; g<NUM_GPUS; g++)); do
      if (( g == SERVER_GPU_ID )) && [[ "$ALLOW_SERVER_GPU_AS_WORKER" != "true" ]]; then
        continue
      fi
      WORKER_GPU_IDS+=("$g")
    done
  fi

  if (( ${#WORKER_GPU_IDS[@]} == 0 )); then
    echo "ERROR: No worker GPUs available for evaluation (NUM_GPUS=$NUM_GPUS)." >&2
    exit 1
  fi

  NUM_WORKERS=${NUM_WORKERS:-${#WORKER_GPU_IDS[@]}}
  if (( NUM_WORKERS > ${#WORKER_GPU_IDS[@]} )); then
    echo "ERROR: NUM_WORKERS=$NUM_WORKERS exceeds available worker GPUs (${#WORKER_GPU_IDS[@]})." >&2
    exit 1
  fi

  # Distribute indices [0..EVAL_EPISODES-1] across workers.
  for ((w=0; w<NUM_WORKERS; w++)); do
    WORKER_TO_EPISODES[w]=""
  done
  for ((idx=0; idx<EVAL_EPISODES; idx++)); do
    w=$((idx % NUM_WORKERS))
    if [[ -z "${WORKER_TO_EPISODES[w]}" ]]; then
      WORKER_TO_EPISODES[w]="$idx"
    else
      WORKER_TO_EPISODES[w]="${WORKER_TO_EPISODES[w]},$idx"
    fi
  done
fi

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

echo "\n== Launching policy servers =="
if [[ "$SERVER_MODE" == "per_eval" ]]; then
  for ((p=0; p<NUM_PAIRS; p++)); do
    ids="${WORKER_TO_EPISODES[p]}"
    if [[ -z "$ids" ]]; then
      echo "- Pair $p: no eval episodes assigned; skipping server"
      continue
    fi

    server_gpu=$((SERVER_GPU_BASE + p))
    port=$((PORT_BASE + p))
    server_log="$OUT_DIR/server_gpu${server_gpu}_p${port}.log"
    echo "- Pair $p: server_gpu=$server_gpu  port=$port   eval_instance_ids=[$ids]"

    setsid bash -c "
      set -euo pipefail
      cd \"$REPO_ROOT\"
      export CUDA_VISIBLE_DEVICES=\"$server_gpu\"
      export XLA_PYTHON_CLIENT_PREALLOCATE=\"$XLA_PYTHON_CLIENT_PREALLOCATE\"
      export XLA_PYTHON_CLIENT_MEM_FRACTION=\"$XLA_PYTHON_CLIENT_MEM_FRACTION\"
      export XLA_PYTHON_CLIENT_ALLOCATOR=\"$XLA_PYTHON_CLIENT_ALLOCATOR\"
      exec uv run scripts/serve_b1k.py \
        --task_name=\"$TASK_NAME\" \
        --control_mode=\"$CONTROL_MODE\" \
        --max_len=\"$MAX_LEN\" \
        --port=\"$port\" \
        policy:checkpoint \
        --policy.config=\"$OPENPI_CONFIG_NAME\" \
        --policy.dir=\"$CKPT_DIR\"
    " >"$server_log" 2>&1 &

    SERVER_PIDS+=($!)
  done
else
  # Shared server: one server on SERVER_GPU_ID, port PORT_BASE.
  if [[ -z "$SERVER_GPU_ID" ]]; then
    SERVER_GPU_ID="$SERVER_GPU_BASE"
  fi
  server_gpu="$SERVER_GPU_ID"
  port="$PORT_BASE"
  server_log="$OUT_DIR/server_gpu${server_gpu}_p${port}.log"
  echo "- Shared server: server_gpu=$server_gpu  port=$port"

  setsid bash -c "
    set -euo pipefail
    cd \"$REPO_ROOT\"
    export CUDA_VISIBLE_DEVICES=\"$server_gpu\"
    export XLA_PYTHON_CLIENT_PREALLOCATE=\"$XLA_PYTHON_CLIENT_PREALLOCATE\"
    export XLA_PYTHON_CLIENT_MEM_FRACTION=\"$XLA_PYTHON_CLIENT_MEM_FRACTION\"
    export XLA_PYTHON_CLIENT_ALLOCATOR=\"$XLA_PYTHON_CLIENT_ALLOCATOR\"
    exec uv run scripts/serve_b1k.py \
      --task_name=\"$TASK_NAME\" \
      --control_mode=\"$CONTROL_MODE\" \
      --max_len=\"$MAX_LEN\" \
      --port=\"$port\" \
      policy:checkpoint \
      --policy.config=\"$OPENPI_CONFIG_NAME\" \
      --policy.dir=\"$CKPT_DIR\"
  " >"$server_log" 2>&1 &

  SERVER_PIDS+=($!)
fi

echo "\nWaiting ${SERVER_STARTUP_WAIT}s for servers to start..."
sleep "$SERVER_STARTUP_WAIT"

echo "\n== Launching BEHAVIOR evaluators =="
if [[ "$SERVER_MODE" == "per_eval" ]]; then
  for ((p=0; p<NUM_PAIRS; p++)); do
    ids="${WORKER_TO_EPISODES[p]}"
    if [[ -z "$ids" ]]; then
      continue
    fi

    eval_gpu=$((EVAL_GPU_BASE + p))
    port=$((PORT_BASE + p))
    eval_log="$OUT_DIR/eval_gpu${eval_gpu}_p${port}.log"
    eval_out="$OUT_DIR/eval_gpu${eval_gpu}_p${port}"
    mkdir -p "$eval_out"

    (
      cd "$BEHAVIOR_DIR"
      unset CUDA_VISIBLE_DEVICES
      OMNIGIBSON_GPU_ID="$eval_gpu" \
        conda run -n behavior --no-capture-output \
          python "OmniGibson/omnigibson/learning/${EVAL_ENTRYPOINT}" \
            policy=websocket \
            task.name="$TASK_NAME" \
            log_path="$eval_out" \
            headless=$HEADLESS \
            write_video=$WRITE_VIDEO \
            env_wrapper._target_="$ENV_WRAPPER_TARGET" \
            ${MAX_STEPS:+max_steps=$MAX_STEPS} \
            model.host="$MODEL_HOST" \
            model.port="$port" \
            eval_instance_ids="[$ids]"
    ) >"$eval_log" 2>&1 &

    EVAL_PIDS+=($!)
    echo "- Started eval: pair=$p eval_gpu=$eval_gpu port=$port -> $eval_out"
  done
else
  port=$PORT_BASE
  for ((w=0; w<NUM_WORKERS; w++)); do
    ids="${WORKER_TO_EPISODES[w]}"
    if [[ -z "$ids" ]]; then
      continue
    fi

    eval_gpu="${WORKER_GPU_IDS[$w]}"
    eval_log="$OUT_DIR/eval_gpu${eval_gpu}_p${port}.log"
    eval_out="$OUT_DIR/eval_gpu${eval_gpu}_p${port}"
    mkdir -p "$eval_out"

    setsid bash -c "
      set -euo pipefail
      cd \"$BEHAVIOR_DIR\"
      unset CUDA_VISIBLE_DEVICES
      export OMNIGIBSON_GPU_ID=\"$eval_gpu\"
      exec conda run -n behavior --no-capture-output \
        python \"OmniGibson/omnigibson/learning/${EVAL_ENTRYPOINT}\" \
          policy=websocket \
          task.name=\"$TASK_NAME\" \
          log_path=\"$eval_out\" \
          headless=$HEADLESS \
          write_video=$WRITE_VIDEO \
          env_wrapper._target_=\"$ENV_WRAPPER_TARGET\" \
          ${MAX_STEPS:+max_steps=$MAX_STEPS} \
          model.host=\"$MODEL_HOST\" \
          model.port=\"$port\" \
          eval_instance_ids=\"[$ids]\"
    " >"$eval_log" 2>&1 &

    EVAL_PIDS+=($!)
    echo "- Started eval: worker=$w eval_gpu=$eval_gpu port=$port eval_instance_ids=[$ids] -> $eval_out"
  done
fi

echo "\nAll evaluators launched. Tail logs, e.g.:"
if [[ "$PAIR_MODE" == "split" ]]; then
  example_eval_gpu="$EVAL_GPU_BASE"
else
  example_eval_gpu=1
fi
echo "  tail -f $OUT_DIR/eval_gpu${example_eval_gpu}_p${PORT_BASE}.log"
echo "  tail -f $OUT_DIR/server_gpu0_p${PORT_BASE}.log"

if (( ${#EVAL_PIDS[@]} == 0 )); then
  echo "ERROR: No evaluators were started (check BEHAVIOR_DIR and EVAL_EPISODES)." >&2
  exit 1
fi

# Wait for evaluators to finish. Servers run forever and will be killed by the EXIT trap.
wait "${EVAL_PIDS[@]}"