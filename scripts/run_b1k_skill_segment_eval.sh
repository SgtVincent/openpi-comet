#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_b1k_skill_segment_eval.sh [--dry-run] <full_run_dir> <ckpt_dir>

Key env vars:
  TASK_NAME=turning_on_radio
  BEHAVIOR_DIR=/home/ubuntu/repo/BEHAVIOR-1K
  DEMO_DATA_PATH=/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos
  RAWDATA_PATH=/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-rawdata
  GPU_IDS=2,3,4,5,6
  NUM_GPUS=5
  PORT_BASE=9900
  SEGMENT_LEVEL=skill
  SUCCESS_MODE=predicate_subgoal
  SEGMENT_LIMIT_PER_DEMO=1
  SEGMENT_INDICES=1,3        # optional: only run these segment_idx values
  INSTANCE_IDS=242,295
  SEGMENT_MAX_STEPS=200
  SEGMENT_DRY_RUN=false
EOF
}

log() { echo "[Info] $*"; }
warn() { echo "[Warn] $*" >&2; }
die() { echo "[Error] $*" >&2; exit 1; }

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

FULL_RUN_DIR="${FULL_RUN_DIR:-${1:-}}"
CKPT_DIR="${CKPT_DIR:-${2:-}}"

TASK_NAME="${TASK_NAME:-turning_on_radio}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_IDS="${GPU_IDS:-}"
PORT_BASE="${PORT_BASE:-9900}"
STOP_STALE_ON_START="${STOP_STALE_ON_START:-true}"
MODEL_HOST="${MODEL_HOST:-127.0.0.1}"

OPENPI_ENV="${OPENPI_ENV:-openpi-comet-nas}"
OPENPI_PYTHON="${OPENPI_PYTHON:-}"
OPENPI_PYTHONPATH="${OPENPI_PYTHONPATH:-}"
BEHAVIOR_ENV="${BEHAVIOR_ENV:-behavior}"
BEHAVIOR_PYTHONPATH="${BEHAVIOR_PYTHONPATH:-}"
BEHAVIOR_DIR="${BEHAVIOR_DIR:-/home/ubuntu/repo/BEHAVIOR-1K}"

OPENPI_CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_b1k-base}"
CONTROL_MODE="${CONTROL_MODE:-receeding_horizon}"
MAX_LEN="${MAX_LEN:-32}"

SEGMENT_LEVEL="${SEGMENT_LEVEL:-skill}"
SUCCESS_MODE="${SUCCESS_MODE:-predicate_subgoal}"
GROUNDING_TOPK="${GROUNDING_TOPK:-3}"
SEGMENT_MAX_STEPS="${SEGMENT_MAX_STEPS:-}"
SEGMENT_DRY_RUN="${SEGMENT_DRY_RUN:-false}"
SEGMENT_LIMIT_PER_DEMO="${SEGMENT_LIMIT_PER_DEMO:-}"
SEGMENT_INDICES="${SEGMENT_INDICES:-}"
INSTANCE_IDS="${INSTANCE_IDS:-}"
SEGMENT_EXTRA_OVERRIDES="${SEGMENT_EXTRA_OVERRIDES:-}"
ENV_WRAPPER_TARGET="${ENV_WRAPPER_TARGET:-omnigibson.learning.wrappers.RGBWrapper}"

DEMO_DATA_PATH="${DEMO_DATA_PATH:-/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos}"
RAWDATA_PATH="${RAWDATA_PATH:-/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-rawdata}"
WRITE_VIDEO="${WRITE_VIDEO:-false}"
HEADLESS="${HEADLESS:-true}"
OG_DEBUG_ASSISTED_RESTORE="${OG_DEBUG_ASSISTED_RESTORE:-0}"
SIM_DISPLAY="${SIM_DISPLAY:-}"
SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-120}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-$SERVER_STARTUP_WAIT}"
SERVER_READY_POLL_INTERVAL="${SERVER_READY_POLL_INTERVAL:-2}"
OMNIGIBSON_APPDATA_PATH_BASE="${OMNIGIBSON_APPDATA_PATH_BASE:-/tmp/omnigibson-appdata}"
OMNIGIBSON_APPDATA_PATH_MODE="${OMNIGIBSON_APPDATA_PATH_MODE:-per_gpu}"
OMNIGIBSON_DISABLE_EXTENSION_REGISTRY="${OMNIGIBSON_DISABLE_EXTENSION_REGISTRY:-0}"
OMNIGIBSON_DISABLE_DRIVER_VERSION_CHECK="${OMNIGIBSON_DISABLE_DRIVER_VERSION_CHECK:-1}"
MIN_GPU_FREE_MB="${MIN_GPU_FREE_MB:-20000}"

XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.40}"
XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

if [[ -z "$OPENPI_PYTHONPATH" && -d "$BEHAVIOR_DIR/OmniGibson" && -d "$BEHAVIOR_DIR/bddl3" ]]; then
  OPENPI_PYTHONPATH="$BEHAVIOR_DIR/joylo:$BEHAVIOR_DIR/OmniGibson:$BEHAVIOR_DIR/bddl3"
fi
if [[ -z "$BEHAVIOR_PYTHONPATH" && -d "$BEHAVIOR_DIR/OmniGibson" && -d "$BEHAVIOR_DIR/bddl3" ]]; then
  BEHAVIOR_PYTHONPATH="$BEHAVIOR_DIR/joylo:$BEHAVIOR_DIR/OmniGibson:$BEHAVIOR_DIR/bddl3"
fi

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

wait_for_port() {
  local host="$1"
  local port="$2"
  (echo >/dev/tcp/"$host"/"$port") >/dev/null 2>&1
}

wait_for_server_ready() {
  local port="$1"
  local log_file="$2"
  local timeout_s="$3"
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if [[ -f "$log_file" ]] && grep -qE "server listening on 0\\.0\\.0\\.0:${port}" "$log_file"; then
      return 0
    fi
    if wait_for_port 127.0.0.1 "$port"; then
      return 0
    fi
    local now
    now="$(date +%s)"
    if (( now - start_ts >= timeout_s )); then
      return 1
    fi
    sleep "$SERVER_READY_POLL_INTERVAL"
  done
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

validate_gpu_memory() {
  (( MIN_GPU_FREE_MB > 0 )) || return 0
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local mem_lines
  mem_lines="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)"
  local insufficient=()
  local gpu free_mb
  for gpu in "${WORKER_GPUS[@]}"; do
    free_mb="$(echo "$mem_lines" | awk -F',' -v g="$gpu" '$1+0==g {gsub(/ /,"",$2); print $2; exit}')"
    [[ -z "$free_mb" ]] && continue
    if (( free_mb < MIN_GPU_FREE_MB )); then
      insufficient+=("gpu${gpu}:${free_mb}MiB")
    fi
  done
  if (( ${#insufficient[@]} > 0 )); then
    die "insufficient free GPU memory (< ${MIN_GPU_FREE_MB} MiB): ${insufficient[*]}"
  fi
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

launch_server() {
  local gpu="$1"
  local port="$2"
  local log_file="$3"
  setsid -w bash -c "
    set -euo pipefail
    cd \"$REPO_ROOT\"
    export CUDA_VISIBLE_DEVICES=\"$gpu\"
    export PYTHONUNBUFFERED=1
    export XLA_PYTHON_CLIENT_PREALLOCATE=\"$XLA_PYTHON_CLIENT_PREALLOCATE\"
    export XLA_PYTHON_CLIENT_MEM_FRACTION=\"$XLA_PYTHON_CLIENT_MEM_FRACTION\"
    export XLA_PYTHON_CLIENT_ALLOCATOR=\"$XLA_PYTHON_CLIENT_ALLOCATOR\"
    local_openpi_src=\"$REPO_ROOT/src\"
    if [[ -n \"$OPENPI_PYTHONPATH\" ]]; then
      export PYTHONPATH=\"\$local_openpi_src:$OPENPI_PYTHONPATH\${PYTHONPATH:+:\$PYTHONPATH}\"
    else
      export PYTHONPATH=\"\$local_openpi_src\${PYTHONPATH:+:\$PYTHONPATH}\"
    fi
    if [[ \"$USE_CONDA_OPENPI\" == \"1\" ]]; then
      source \"$CONDA_BASE/etc/profile.d/conda.sh\"
      conda activate \"$OPENPI_ENV\"
      exec python scripts/serve_b1k.py \
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

launch_segment_worker() {
  local gpu="$1"
  local port="$2"
  local jobs_file="$3"
  local worker_log="$OUT_DIR/segment_worker_gpu${gpu}_p${port}.log"
  setsid -w bash -c "
    set -euo pipefail
    cd \"$BEHAVIOR_DIR\"
    export PYTHONUNBUFFERED=1
    if [[ -n \"$BEHAVIOR_PYTHONPATH\" ]]; then
      export PYTHONPATH=\"$BEHAVIOR_PYTHONPATH\${PYTHONPATH:+:\$PYTHONPATH}\"
    fi
    export NO_PROXY=\"localhost,127.0.0.1,::1\${NO_PROXY:+,\$NO_PROXY}\"
    export no_proxy=\"localhost,127.0.0.1,::1\${no_proxy:+,\$no_proxy}\"
    export OMNIGIBSON_GPU_ID=\"$gpu\"
    export OMNIGIBSON_DATA_PATH=\"\${OMNIGIBSON_DATA_PATH:-$BEHAVIOR_DIR/datasets}\"
    export OMNIGIBSON_DISABLE_EXTENSION_REGISTRY=\"$OMNIGIBSON_DISABLE_EXTENSION_REGISTRY\"
    export OMNIGIBSON_DISABLE_DRIVER_VERSION_CHECK=\"$OMNIGIBSON_DISABLE_DRIVER_VERSION_CHECK\"
    if [[ -n \"$OMNIGIBSON_APPDATA_PATH_BASE\" ]]; then
      if [[ \"$OMNIGIBSON_APPDATA_PATH_MODE\" == \"per_gpu_port\" ]]; then
        export OMNIGIBSON_APPDATA_PATH=\"$OMNIGIBSON_APPDATA_PATH_BASE/$USER/gpu${gpu}_p${port}\"
      else
        export OMNIGIBSON_APPDATA_PATH=\"$OMNIGIBSON_APPDATA_PATH_BASE/$USER/gpu${gpu}\"
      fi
      mkdir -p \"\$OMNIGIBSON_APPDATA_PATH\"
    fi
    export MPLBACKEND=\"\${MPLBACKEND:-Agg}\"
    export TORCHDYNAMO_DISABLE=\"\${TORCHDYNAMO_DISABLE:-1}\"
    export TORCHINDUCTOR_DISABLE=\"\${TORCHINDUCTOR_DISABLE:-1}\"
    export OMNIGIBSON_HEADLESS=\"$HEADLESS\"
    export OG_DEBUG_ASSISTED_RESTORE=\"$OG_DEBUG_ASSISTED_RESTORE\"
    if [[ -n \"$SIM_DISPLAY\" ]]; then
      export DISPLAY=\"$SIM_DISPLAY\"
    else
      unset DISPLAY
    fi
    source \"$CONDA_BASE/etc/profile.d/conda.sh\"
    conda activate \"$BEHAVIOR_ENV\"
    while IFS=\$'\\t' read -r demo_id instance_id segment_idx; do
      [[ -z \"\$demo_id\" ]] && continue
      skill_dir=\$(printf '%s/raw/instance_%s/demo_%s/skill_%03d' \"$OUT_DIR\" \"\$instance_id\" \"\$demo_id\" \"\$segment_idx\")
      mkdir -p \"\$skill_dir\"
      cmd=(python OmniGibson/omnigibson/learning/eval_segment.py
        policy=websocket
        task.name=\"$TASK_NAME\"
        demo_data_path=\"$DEMO_DATA_PATH\"
        segment_level=\"$SEGMENT_LEVEL\"
        segment_idx=\"\$segment_idx\"
        success_mode=\"$SUCCESS_MODE\"
        grounding_topk=\"$GROUNDING_TOPK\"
        dry_run=\"$SEGMENT_DRY_RUN\"
        log_path=\"\$skill_dir\"
        demo_id=\"\$demo_id\"
        headless=$HEADLESS
        write_video=$WRITE_VIDEO
        model.host=\"$MODEL_HOST\"
        model.port=\"$port\"
      )
      if [[ -n \"$RAWDATA_PATH\" ]]; then
        cmd+=(rawdata_path=\"$RAWDATA_PATH\")
      fi
      if [[ -n \"$SEGMENT_MAX_STEPS\" ]]; then
        cmd+=(segment_max_steps=\"$SEGMENT_MAX_STEPS\")
      fi
      if [[ -n \"$ENV_WRAPPER_TARGET\" ]]; then
        cmd+=(env_wrapper._target_=\"$ENV_WRAPPER_TARGET\")
      fi
      if [[ -n \"$SEGMENT_EXTRA_OVERRIDES\" ]]; then
        read -r -a extra_tokens <<< \"$SEGMENT_EXTRA_OVERRIDES\"
        cmd+=(\"\${extra_tokens[@]}\")
      fi
      \"\${cmd[@]}\" >\"\$skill_dir/segment_eval.log\" 2>&1
    done < \"$jobs_file\"
  " >"$worker_log" 2>&1 &
  WORKER_PIDS+=("$!")
  log "Started segment worker: gpu=$gpu port=$port jobs_file=$jobs_file"
}

cleanup() {
  local pid
  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -- -"$pid" 2>/dev/null || true
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${WORKER_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -- -"$pid" 2>/dev/null || true
      kill "$pid" 2>/dev/null || true
    fi
  done
}

run_single_checkpoint_mode() {
  command -v conda >/dev/null 2>&1 || die "conda not found in PATH"
  CONDA_BASE="$(conda info --base)"
  [[ -d "$CONDA_BASE" ]] || die "conda base not found: $CONDA_BASE"
  [[ -d "$BEHAVIOR_DIR" ]] || die "BEHAVIOR_DIR not found: $BEHAVIOR_DIR"
  [[ -d "$DEMO_DATA_PATH" ]] || die "DEMO_DATA_PATH not found: $DEMO_DATA_PATH"
  [[ -d "$FULL_RUN_DIR" ]] || die "FULL_RUN_DIR not found: $FULL_RUN_DIR"
  [[ "$SEGMENT_LEVEL" == "skill" ]] || die "SEGMENT_LEVEL must be skill for this script"
  [[ -n "$CKPT_DIR" ]] || die "CKPT_DIR not provided"

  CKPT_DIR="$(resolve_checkpoint_dir "$CKPT_DIR")"
  [[ -d "$CKPT_DIR" ]] || die "CKPT_DIR not found: $CKPT_DIR"

  resolve_openpi_runtime

  WORKER_GPUS=()
  resolve_gpu_pool WORKER_GPUS
  validate_gpu_memory

  RUN_TAG="${RUN_TAG:-segment_${TASK_NAME}_$(basename "$CKPT_DIR")_$(date +%Y%m%d_%H%M%S)}"
  OUT_DIR="${OUT_DIR:-$REPO_ROOT/segment_eval_runs/$RUN_TAG}"
  mkdir -p "$OUT_DIR"

  export NO_PROXY="localhost,127.0.0.1,::1${NO_PROXY:+,$NO_PROXY}"
  export no_proxy="localhost,127.0.0.1,::1${no_proxy:+,$no_proxy}"

  local manifest_path="$OUT_DIR/segment_manifest.json"
  local jobs_root="$OUT_DIR/jobs"
  mkdir -p "$jobs_root"

  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$BEHAVIOR_ENV"
  python - <<PY
import json
from pathlib import Path
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES

full_run_dir = Path(r"$FULL_RUN_DIR")
demo_data_path = Path(r"$DEMO_DATA_PATH")
task_name = r"$TASK_NAME"
segment_level = r"$SEGMENT_LEVEL"
limit_raw = r"$SEGMENT_LIMIT_PER_DEMO".strip()
segment_indices_raw = r"$SEGMENT_INDICES".strip()
instance_csv = r"$INSTANCE_IDS".strip()
worker_count = int("${#WORKER_GPUS[@]}")
jobs_root = Path(r"$jobs_root")
manifest_path = Path(r"$manifest_path")

segment_indices = None
if segment_indices_raw:
    segment_indices = sorted({int(x) for x in segment_indices_raw.split(",") if x.strip()})

episode_map = {}
for metric_path in sorted(full_run_dir.glob("eval_gpu*_p*/metrics/*.json")):
    try:
        data = json.loads(metric_path.read_text())
    except Exception:
        continue
    stem = metric_path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        continue
    try:
        instance_id = int(parts[-2])
    except ValueError:
        continue
    q_final = data.get("q_score", {}).get("final", 0.0)
    entry = {
        "instance_id": instance_id,
        "metric_path": str(metric_path),
        "episode_q_score_final": q_final,
        "episode_success": bool(q_final and q_final > 0),
    }
    episode_map.setdefault(instance_id, entry)

if instance_csv:
    target_instances = [int(x) for x in instance_csv.split(",") if x.strip()]
else:
    target_instances = sorted(episode_map)

task_idx = TASK_NAMES_TO_INDICES[task_name]
annotations_dir = demo_data_path / "annotations" / f"task-{task_idx:04d}"
if not annotations_dir.exists():
    raise SystemExit(f"annotations dir not found: {annotations_dir}")

demo_by_instance = {}
for ann_path in sorted(annotations_dir.glob("episode_*.json")):
    demo_id = ann_path.stem.replace("episode_", "")
    try:
        instance_id = int(demo_id) // 10 % 1000
    except ValueError:
        continue
    if instance_id in target_instances and instance_id not in demo_by_instance:
        demo_by_instance[instance_id] = demo_id

limit = int(limit_raw) if limit_raw else None
jobs = []
missing_instances = []
for instance_id in target_instances:
    demo_id = demo_by_instance.get(instance_id)
    if demo_id is None:
      missing_instances.append(instance_id)
      continue
    ann_path = annotations_dir / f"episode_{demo_id}.json"
    ann = json.loads(ann_path.read_text())
    segments = ann.get(f"{segment_level}_annotation", [])
    segments = sorted(segments, key=lambda x: x["frame_duration"][0])
    if limit is not None:
        segments = segments[:limit]
    for segment_idx, seg in enumerate(segments):
        if segment_indices is not None and segment_idx not in segment_indices:
            continue
        desc_list = seg.get(f"{segment_level}_description", [])
        segment_desc = desc_list[0] if desc_list else "unknown"
        jobs.append({
            "demo_id": demo_id,
            "instance_id": instance_id,
            "segment_idx": segment_idx,
            "segment_desc": segment_desc,
            "frame_duration": seg.get("frame_duration"),
            "episode_q_score_final": episode_map.get(instance_id, {}).get("episode_q_score_final"),
            "episode_success": episode_map.get(instance_id, {}).get("episode_success"),
        })

for worker_idx in range(worker_count):
    worker_jobs = jobs[worker_idx::worker_count]
    worker_path = jobs_root / f"worker_{worker_idx}.tsv"
    with worker_path.open("w") as f:
        for row in worker_jobs:
            f.write(f"{row['demo_id']}\\t{row['instance_id']}\\t{row['segment_idx']}\\n")

manifest = {
    "task_name": task_name,
    "full_run_dir": str(full_run_dir),
    "demo_data_path": str(demo_data_path),
    "segment_level": segment_level,
    "worker_count": worker_count,
    "target_instances": target_instances,
    "missing_instances": missing_instances,
    "selected_demo_by_instance": demo_by_instance,
    "segment_indices": segment_indices,
    "jobs": jobs,
}
manifest_path.write_text(json.dumps(manifest, indent=2))
print(json.dumps({"jobs": len(jobs), "missing_instances": missing_instances}, indent=2))
PY

  local total_jobs
  total_jobs="$(python - <<PY
import json
from pathlib import Path
manifest = json.loads(Path(r"$manifest_path").read_text())
print(len(manifest["jobs"]))
PY
)"
  (( total_jobs > 0 )) || die "no segment jobs generated"

  log "Writing outputs to: $OUT_DIR"
  log "Task=$TASK_NAME Checkpoint=$CKPT_DIR FullRun=$FULL_RUN_DIR"
  log "Worker GPUs=[${WORKER_GPUS[*]}] Total segment jobs=$total_jobs"

  local i
  for ((i=0; i<${#WORKER_GPUS[@]}; i++)); do
    local worker_port=$((PORT_BASE + i))
    log "worker $i -> gpu=${WORKER_GPUS[i]} port=$worker_port jobs_file=$jobs_root/worker_${i}.tsv"
  done

  if [[ "$DRY_RUN" == "true" ]]; then
    log "Dry-run only: no process started. OUT_DIR=$OUT_DIR"
    return 0
  fi

  SERVER_PIDS=()
  WORKER_PIDS=()
  trap cleanup EXIT
  trap 'cleanup; exit 130' INT
  trap 'cleanup; exit 143' TERM

  log "Launching servers..."
  for ((i=0; i<${#WORKER_GPUS[@]}; i++)); do
    local worker_port=$((PORT_BASE + i))
    local server_log="$OUT_DIR/server_gpu${WORKER_GPUS[i]}_p${worker_port}.log"
    launch_server "${WORKER_GPUS[i]}" "$worker_port" "$server_log"
  done

  log "Waiting for server ready (timeout=${SERVER_READY_TIMEOUT}s)..."
  for ((i=0; i<${#WORKER_GPUS[@]}; i++)); do
    local worker_port=$((PORT_BASE + i))
    local server_log="$OUT_DIR/server_gpu${WORKER_GPUS[i]}_p${worker_port}.log"
    wait_for_server_ready "$worker_port" "$server_log" "$SERVER_READY_TIMEOUT" || die "server failed to become ready on gpu=${WORKER_GPUS[i]} port=$worker_port (log=$server_log)"
  done

  log "Launching segment workers..."
  for ((i=0; i<${#WORKER_GPUS[@]}; i++)); do
    local worker_port=$((PORT_BASE + i))
    local jobs_file="$jobs_root/worker_${i}.tsv"
    if [[ ! -s "$jobs_file" ]]; then
      log "No jobs for worker $i (gpu=${WORKER_GPUS[i]})"
      continue
    fi
    launch_segment_worker "${WORKER_GPUS[i]}" "$worker_port" "$jobs_file"
  done

  local batch_failed=0
  local pid
  for pid in "${WORKER_PIDS[@]}"; do
    if ! wait "$pid"; then
      batch_failed=1
    fi
  done
  (( batch_failed == 0 )) || die "one or more segment workers failed"

  local result_count
  result_count="$(find "$OUT_DIR/raw" -type f -path '*/metrics/*.json' 2>/dev/null | wc -l | tr -d ' ')"
  log "Segment eval completed. metrics_json=$result_count manifest=$manifest_path"
  (( result_count > 0 )) || die "no segment metrics were generated under $OUT_DIR/raw"
}

[[ -n "$FULL_RUN_DIR" ]] || die "full run dir not provided"
[[ -n "$CKPT_DIR" ]] || die "checkpoint dir not provided"
run_single_checkpoint_mode
