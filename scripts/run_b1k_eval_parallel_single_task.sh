#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_b1k_eval_parallel_single_task.sh [--dry-run] <ckpt_dir>
  bash scripts/run_b1k_eval_parallel_single_task.sh [--dry-run] <ckpt_list.txt>
  bash scripts/run_b1k_eval_parallel_single_task.sh [--dry-run] <ckpt_a> <ckpt_b> ...

Core behavior:
  - Single mode: launch one server + one evaluator per worker GPU.
  - Multi mode: batch checkpoints and split GPU pool across checkpoints.
  - Dry-run: print launch plan only.

Key env vars:
  TASK_NAME=turning_on_radio
  BEHAVIOR_DIR=/home/ubuntu/repo/BEHAVIOR-1K
  OPENPI_ENV=openpi-comet-nas
  BEHAVIOR_ENV=behavior

  GPU_IDS=0,1,2,3
  NUM_GPUS=8

  PORT_BASE=8000
  PORT_BASE_START=9700
  PORT_STRIDE=20

  EVAL_ENTRYPOINT=eval_custom.py    # eval.py|eval_custom.py
  EVAL_INSTANCE_IDS=0,1,2,3,4,5,6,7,8,9
  DRY_RUN=false
EOF
}

log() { echo "[Info] $*"; }
warn() { echo "[Warn] $*" >&2; }
die() { echo "[Error] $*" >&2; exit 1; }

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ----------------------------
# Args
# ----------------------------
DRY_RUN="${DRY_RUN:-false}"
declare -a POSITIONAL_ARGS=()
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      DRY_RUN=true
      ;;
    *)
      POSITIONAL_ARGS+=("$arg")
      ;;
  esac
done
set -- "${POSITIONAL_ARGS[@]}"

# ----------------------------
# Defaults
# ----------------------------
TASK_NAME="${TASK_NAME:-turning_on_radio}"

NUM_GPUS="${NUM_GPUS:-8}"
GPU_IDS="${GPU_IDS:-}"

PORT_BASE="${PORT_BASE:-8000}"
PORT_BASE_START="${PORT_BASE_START:-9700}"
PORT_STRIDE="${PORT_STRIDE:-20}"
STOP_STALE_ON_START="${STOP_STALE_ON_START:-true}"

MODEL_HOST="${MODEL_HOST:-127.0.0.1}"

OPENPI_ENV="${OPENPI_ENV:-openpi-comet-nas}"
OPENPI_PYTHON="${OPENPI_PYTHON:-}"
BEHAVIOR_ENV="${BEHAVIOR_ENV:-behavior}"
BEHAVIOR_DIR="${BEHAVIOR_DIR:-/home/ubuntu/repo/BEHAVIOR-1K}"

OPENPI_CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_b1k-base}"
CONTROL_MODE="${CONTROL_MODE:-receeding_horizon}"
MAX_LEN="${MAX_LEN:-32}"

EVAL_ENTRYPOINT="${EVAL_ENTRYPOINT:-eval_custom.py}"
EVAL_INSTANCE_IDS="${EVAL_INSTANCE_IDS:-0,1,2,3,4,5,6,7,8,9}"

HEADLESS="${HEADLESS:-true}"
WRITE_VIDEO="${WRITE_VIDEO:-false}"
MAX_STEPS="${MAX_STEPS:-}"
SIM_DISPLAY="${SIM_DISPLAY:-${DISPLAY:-:10.0}}"
SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-10}"
ENV_WRAPPER_TARGET="${ENV_WRAPPER_TARGET:-omnigibson.learning.wrappers.RGBWrapper}"

SAVE_ROLLOUT="${SAVE_ROLLOUT:-false}"
PERTURB_POSE="${PERTURB_POSE:-false}"
PERTURB_POSE_SEED="${PERTURB_POSE_SEED:-42}"
PARALLEL_EVALUATOR_START_IDX="${PARALLEL_EVALUATOR_START_IDX:-0}"
PARALLEL_EVALUATOR_END_IDX="${PARALLEL_EVALUATOR_END_IDX:-10}"

XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.40}"
XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

# ----------------------------
# Helpers
# ----------------------------
is_positive_int() {
  [[ "$1" =~ ^[0-9]+$ ]] && (( "$1" > 0 ))
}

parse_csv_ints() {
  local csv="$1"
  local -n out_ref="$2"
  out_ref=()
  local raw item
  IFS=',' read -r -a raw <<< "$csv"
  for item in "${raw[@]}"; do
    item="${item//[[:space:]]/}"
    [[ -z "$item" ]] && continue
    [[ "$item" =~ ^[0-9]+$ ]] || die "invalid integer: $item"
    out_ref+=("$item")
  done
  (( ${#out_ref[@]} > 0 )) || die "empty integer list from csv: $csv"
}

read_checkpoint_list_file() {
  local file="$1"
  while IFS= read -r line || [[ -n "$line" ]]; do
    local trimmed="${line#${line%%[![:space:]]*}}"
    trimmed="${trimmed%${trimmed##*[![:space:]]}}"
    [[ -z "$trimmed" || "$trimmed" == \#* ]] && continue
    MULTI_CKPTS+=("$trimmed")
  done < "$file"
}

resolve_checkpoint_dir() {
  local root="$1"
  [[ -n "$root" ]] || die "checkpoint path not provided"

  if [[ -f "$root/model.safetensors" || -d "$root/params" ]]; then
    echo "$root"
    return 0
  fi

  local latest_step
  latest_step="$({
    find "$root" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+' -print 2>/dev/null | sort -V
  } | while read -r d; do
    if [[ -f "$d/model.safetensors" || -d "$d/params" ]]; then
      echo "$d"
    fi
  done | tail -n 1)"

  if [[ -n "$latest_step" ]]; then
    echo "$latest_step"
  else
    echo "$root"
  fi
}

resolve_gpu_pool() {
  local -n gpu_pool_ref="$1"
  gpu_pool_ref=()

  if [[ -n "$GPU_IDS" ]]; then
    parse_csv_ints "$GPU_IDS" gpu_pool_ref
  else
    local detected
    detected="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "$detected" =~ ^[0-9]+$ ]] && (( detected > 0 )); then
      local g
      for ((g=0; g<detected; g++)); do
        gpu_pool_ref+=("$g")
      done
    fi
  fi

  if (( ${#gpu_pool_ref[@]} == 0 )); then
    is_positive_int "$NUM_GPUS" || die "NUM_GPUS must be >= 1"
    local g
    for ((g=0; g<NUM_GPUS; g++)); do
      gpu_pool_ref+=("$g")
    done
  fi

  is_positive_int "$NUM_GPUS" || die "NUM_GPUS must be >= 1"
  (( NUM_GPUS <= ${#gpu_pool_ref[@]} )) || die "NUM_GPUS exceeds resolved GPU count (${#gpu_pool_ref[@]})"
  gpu_pool_ref=("${gpu_pool_ref[@]:0:NUM_GPUS}")
}

resolve_openpi_runtime() {
  USE_CONDA_OPENPI=0
  if conda env list | awk '{print $1}' | grep -qx "$OPENPI_ENV"; then
    USE_CONDA_OPENPI=1
  fi

  if (( USE_CONDA_OPENPI == 0 )); then
    if [[ -z "$OPENPI_PYTHON" && -x "$REPO_ROOT/.venv/bin/python" ]]; then
      OPENPI_PYTHON="$REPO_ROOT/.venv/bin/python"
    fi
    [[ -x "$OPENPI_PYTHON" ]] || die "OPENPI_ENV '$OPENPI_ENV' not found and OPENPI_PYTHON is not executable"
    log "OpenPi runtime: OPENPI_PYTHON=$OPENPI_PYTHON"
  else
    log "OpenPi runtime: conda env '$OPENPI_ENV'"
  fi
}

assign_eval_ids() {
  local worker_count="$1"
  WORKER_TO_IDS=()
  local i
  for ((i=0; i<worker_count; i++)); do
    WORKER_TO_IDS[i]=""
  done
  for ((i=0; i<${#EVAL_IDS[@]}; i++)); do
    local w=$(( i % worker_count ))
    if [[ -z "${WORKER_TO_IDS[w]}" ]]; then
      WORKER_TO_IDS[w]="${EVAL_IDS[i]}"
    else
      WORKER_TO_IDS[w]="${WORKER_TO_IDS[w]},${EVAL_IDS[i]}"
    fi
  done
}

launch_server() {
  local gpu="$1"
  local port="$2"
  local log_file="$3"

  setsid bash -c "
    set -euo pipefail
    cd \"$REPO_ROOT\"
    export CUDA_VISIBLE_DEVICES=\"$gpu\"
    export XLA_PYTHON_CLIENT_PREALLOCATE=\"$XLA_PYTHON_CLIENT_PREALLOCATE\"
    export XLA_PYTHON_CLIENT_MEM_FRACTION=\"$XLA_PYTHON_CLIENT_MEM_FRACTION\"
    export XLA_PYTHON_CLIENT_ALLOCATOR=\"$XLA_PYTHON_CLIENT_ALLOCATOR\"
    if [[ \"$USE_CONDA_OPENPI\" == \"1\" ]]; then
      exec conda run -n \"$OPENPI_ENV\" --no-capture-output \
        python scripts/serve_b1k.py \
        --task_name=\"$TASK_NAME\" \
        --control_mode=\"$CONTROL_MODE\" \
        --max_len=\"$MAX_LEN\" \
        --port=\"$port\" \
        policy:checkpoint \
        --policy.config=\"$OPENPI_CONFIG_NAME\" \
        --policy.dir=\"$CKPT_DIR\"
    else
      exec \"$OPENPI_PYTHON\" scripts/serve_b1k.py \
        --task_name=\"$TASK_NAME\" \
        --control_mode=\"$CONTROL_MODE\" \
        --max_len=\"$MAX_LEN\" \
        --port=\"$port\" \
        policy:checkpoint \
        --policy.config=\"$OPENPI_CONFIG_NAME\" \
        --policy.dir=\"$CKPT_DIR\"
    fi
  " >"$log_file" 2>&1 &
  SERVER_PIDS+=("$!")
}

launch_eval() {
  local gpu="$1"
  local port="$2"
  local ids_csv="$3"

  local eval_log="$OUT_DIR/eval_gpu${gpu}_p${port}.log"
  local eval_out="$OUT_DIR/eval_gpu${gpu}_p${port}"
  mkdir -p "$eval_out"

  local max_steps_arg=""
  local env_wrapper_arg=""
  local eval_custom_args=""

  [[ -n "$MAX_STEPS" ]] && max_steps_arg="max_steps=$MAX_STEPS"
  [[ -n "$ENV_WRAPPER_TARGET" ]] && env_wrapper_arg="env_wrapper._target_=$ENV_WRAPPER_TARGET"

  if [[ "$EVAL_ENTRYPOINT" == "eval_custom.py" ]]; then
    eval_custom_args="use_parallel_evaluator=false save_rollout=$SAVE_ROLLOUT perturb_pose=$PERTURB_POSE perturb_pose_seed=$PERTURB_POSE_SEED parallel_evaluator_start_idx=$PARALLEL_EVALUATOR_START_IDX parallel_evaluator_end_idx=$PARALLEL_EVALUATOR_END_IDX"
  fi

  setsid bash -c "
    set -euo pipefail
    cd \"$BEHAVIOR_DIR\"
    export CUDA_VISIBLE_DEVICES=\"$gpu\"
    unset OMNIGIBSON_GPU_ID
    export OMNIGIBSON_HEADLESS=\"$HEADLESS\"
    if [[ -n \"$SIM_DISPLAY\" ]]; then
      export DISPLAY=\"$SIM_DISPLAY\"
    else
      unset DISPLAY
    fi
    exec conda run -n \"$BEHAVIOR_ENV\" --no-capture-output \
      python \"OmniGibson/omnigibson/learning/$EVAL_ENTRYPOINT\" \
      policy=websocket \
      task.name=\"$TASK_NAME\" \
      log_path=\"$eval_out\" \
      headless=$HEADLESS \
      write_video=$WRITE_VIDEO \
      model.host=\"$MODEL_HOST\" \
      model.port=\"$port\" \
      $max_steps_arg \
      $env_wrapper_arg \
      $eval_custom_args \
      eval_instance_ids=\"[$ids_csv]\"
  " >"$eval_log" 2>&1 &

  EVAL_PIDS+=("$!")
  log "Started eval: gpu=$gpu port=$port ids=[$ids_csv]"
}

cleanup() {
  local pid
  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
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

run_single_checkpoint_mode() {
  command -v conda >/dev/null 2>&1 || die "conda not found in PATH"
  [[ -d "$BEHAVIOR_DIR" ]] || die "BEHAVIOR_DIR not found: $BEHAVIOR_DIR"
  [[ "$EVAL_ENTRYPOINT" == "eval.py" || "$EVAL_ENTRYPOINT" == "eval_custom.py" ]] || die "EVAL_ENTRYPOINT must be eval.py|eval_custom.py"

  CKPT_DIR="$(resolve_checkpoint_dir "$CKPT_DIR")"
  [[ -d "$CKPT_DIR" ]] || die "CKPT_DIR not found: $CKPT_DIR"

  resolve_openpi_runtime

  WORKER_GPUS=()
  resolve_gpu_pool WORKER_GPUS

  parse_csv_ints "$EVAL_INSTANCE_IDS" EVAL_IDS
  local id
  for id in "${EVAL_IDS[@]}"; do
    (( id >= 0 && id <= 9 )) || die "eval instance id out of range [0,9]: $id"
  done

  assign_eval_ids "${#WORKER_GPUS[@]}"

  local run_name_ckpt="$(basename "$CKPT_DIR")"
  RUN_TAG="${RUN_TAG:-parallel_${TASK_NAME}_${run_name_ckpt}_$(date +%Y%m%d_%H%M%S)}"
  OUT_DIR="${OUT_DIR:-$REPO_ROOT/eval_logs/$RUN_TAG}"
  mkdir -p "$OUT_DIR"

  export NO_PROXY="localhost,127.0.0.1,::1${NO_PROXY:+,$NO_PROXY}"
  export no_proxy="localhost,127.0.0.1,::1${no_proxy:+,$no_proxy}"

  log "Writing outputs to: $OUT_DIR"
  log "Task=$TASK_NAME Checkpoint=$CKPT_DIR"
  log "Worker GPUs=[${WORKER_GPUS[*]}] Eval IDs=[${EVAL_IDS[*]}]"

  local i
  for ((i=0; i<${#WORKER_TO_IDS[@]}; i++)); do
    local worker_port=$((PORT_BASE + i))
    log "worker $i -> ids=[${WORKER_TO_IDS[i]}] gpu=${WORKER_GPUS[i]} port=$worker_port"
  done

  if [[ "$DRY_RUN" == "true" ]]; then
    log "Dry-run only: no process started. OUT_DIR=$OUT_DIR"
    return 0
  fi

  SERVER_PIDS=()
  EVAL_PIDS=()
  trap cleanup EXIT
  trap 'cleanup; exit 130' INT
  trap 'cleanup; exit 143' TERM

  log "Launching servers..."
  for ((i=0; i<${#WORKER_TO_IDS[@]}; i++)); do
    [[ -z "${WORKER_TO_IDS[i]}" ]] && continue
    local worker_port=$((PORT_BASE + i))
    local server_log="$OUT_DIR/server_gpu${WORKER_GPUS[i]}_p${worker_port}.log"
    launch_server "${WORKER_GPUS[i]}" "$worker_port" "$server_log"
  done

  log "Waiting ${SERVER_STARTUP_WAIT}s for server warm-up..."
  sleep "$SERVER_STARTUP_WAIT"

  log "Launching evaluators..."
  for ((i=0; i<${#WORKER_TO_IDS[@]}; i++)); do
    [[ -z "${WORKER_TO_IDS[i]}" ]] && continue
    local worker_port=$((PORT_BASE + i))
    launch_eval "${WORKER_GPUS[i]}" "$worker_port" "${WORKER_TO_IDS[i]}"
  done

  (( ${#EVAL_PIDS[@]} > 0 )) || die "no evaluators were started"
  log "Launched ${#SERVER_PIDS[@]} server(s) and ${#EVAL_PIDS[@]} evaluator(s)."

  local overall_rc=0
  local pid
  for pid in "${EVAL_PIDS[@]}"; do
    if ! wait "$pid"; then
      overall_rc=1
    fi
  done

  if (( overall_rc != 0 )); then
    die "one or more evaluators failed. Logs: $OUT_DIR"
  else
    log "All evaluators completed successfully. Logs: $OUT_DIR"
  fi
}

run_multi_checkpoint_mode() {
  resolve_gpu_pool MULTI_GPUS

  local total_models="${#MULTI_CKPTS[@]}"
  local total_gpus="${#MULTI_GPUS[@]}"
  (( total_models > 0 )) || die "no checkpoints resolved for multi mode"
  (( total_gpus > 0 )) || die "no GPUs resolved for multi mode"

  local ckpt
  for ckpt in "${MULTI_CKPTS[@]}"; do
    [[ -d "$ckpt" ]] || die "checkpoint dir not found: $ckpt"
  done

  log "Multi mode: models=$total_models gpus=$total_gpus ids=[${MULTI_GPUS[*]}] dry_run=$DRY_RUN"

  if [[ "$STOP_STALE_ON_START" == "true" && "$DRY_RUN" != "true" ]]; then
    pkill -f 'scripts/serve_b1k.py|OmniGibson/omnigibson/learning/eval_custom.py' >/dev/null 2>&1 || true
    sleep 2
  fi

  local model_index=0
  while (( model_index < total_models )); do
    local remaining=$(( total_models - model_index ))
    local batch_size=$(( remaining < total_gpus ? remaining : total_gpus ))
    local base_per_model=$(( total_gpus / batch_size ))
    local remainder=$(( total_gpus % batch_size ))
    local gpu_cursor=0

    declare -a batch_pids=()
    local slot
    for ((slot=0; slot<batch_size; slot++)); do
      ckpt="${MULTI_CKPTS[model_index]}"

      local per_model_gpus=$base_per_model
      (( slot < remainder )) && per_model_gpus=$(( per_model_gpus + 1 ))

      declare -a assigned_gpus=("${MULTI_GPUS[@]:gpu_cursor:per_model_gpus}")
      local assigned_gpu_csv
      assigned_gpu_csv="$(IFS=,; echo "${assigned_gpus[*]}")"
      gpu_cursor=$(( gpu_cursor + per_model_gpus ))

      local port_base=$(( PORT_BASE_START + model_index * PORT_STRIDE ))
      local run_tag="parallel_${TASK_NAME}_m$((model_index + 1))_$(basename "$ckpt")_$(date +%Y%m%d_%H%M%S)"

      log "Model $((model_index + 1))/$total_models ckpt=$ckpt gpus=$assigned_gpu_csv num_gpus=$per_model_gpus port_base=$port_base"

      if [[ "$DRY_RUN" == "true" ]]; then
        echo "  DRY-RUN: GPU_IDS=$assigned_gpu_csv NUM_GPUS=$per_model_gpus PORT_BASE=$port_base RUN_TAG=$run_tag bash $SCRIPT_PATH '$ckpt'"
      else
        (
          export DRY_RUN=false
          export GPU_IDS="$assigned_gpu_csv"
          export NUM_GPUS="$per_model_gpus"
          export PORT_BASE="$port_base"
          export RUN_TAG="$run_tag"
          bash "$SCRIPT_PATH" "$ckpt"
        ) &
        batch_pids+=("$!")
      fi

      model_index=$(( model_index + 1 ))
    done

    if [[ "$DRY_RUN" == "true" ]]; then
      continue
    fi

    local batch_failed=0
    local pid
    for pid in "${batch_pids[@]}"; do
      if ! wait "$pid"; then
        batch_failed=1
      fi
    done
    (( batch_failed == 0 )) || die "one or more model runs failed in batch"
  done

  if [[ "$DRY_RUN" == "true" ]]; then
    log "All multi-checkpoint runs planned (dry-run only)."
  else
    log "All multi-checkpoint runs completed successfully."
  fi
}

# ----------------------------
# Dispatch
# ----------------------------
declare -a MULTI_CKPTS=()
if (( $# >= 2 )); then
  MULTI_CKPTS=("$@")
  run_multi_checkpoint_mode
  exit 0
fi

if (( $# == 1 )) && [[ -f "$1" ]]; then
  read_checkpoint_list_file "$1"
  run_multi_checkpoint_mode
  exit 0
fi

if (( $# >= 1 )); then
  CKPT_DIR="$1"
else
  CKPT_DIR="${CKPT_DIR:-}"
fi

[[ -n "$CKPT_DIR" ]] || die "checkpoint path not provided (positional arg or CKPT_DIR)"
run_single_checkpoint_mode
