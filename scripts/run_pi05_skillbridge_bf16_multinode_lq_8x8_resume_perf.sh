#!/usr/bin/env bash
# Resume the current 8x8 Skill Bridge run in-place with perf-oriented settings.
# Intended for Merlin Robust hot-update on the existing allocation.
#
# Merlin entrypoint:
#   exec bash scripts/run_pi05_skillbridge_bf16_multinode_lq_8x8_resume_perf.sh
#
# Safety contract:
#   - Forces RESUME=1 and OVERWRITE=0.
#   - Pins EXP_NAME to the currently running experiment so a changed Arnold
#     run key cannot accidentally mint a fresh experiment during hot-update.
#   - Verifies that at least one numeric checkpoint exists before launching.
#   - Keeps topology, data, batch size, schedule, and output root identical to
#     the current run; only finite-consensus diagnostics and DeepSpeed
#     overlap-communication behavior differ.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
cd "${REPO_ROOT}"

readonly CURRENT_EXP_NAME="pi05_ki_joint_query_single_task_radio_bf16_8n8g_20260812_151551"
readonly CURRENT_OUTPUT_ROOT="/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/outputs/pi05_skillbridge_a100_lq_bf16_8x8"
readonly CURRENT_CHECKPOINT_DIR="${CURRENT_OUTPUT_ROOT}/checkpoints/${CURRENT_EXP_NAME}"
readonly MIN_RESUME_STEP=80000
readonly TRAINING_PYTHON="/mnt/bn/saiwenresearch/mlx/users/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python"

# Force the exact current run identity and training contract. Do not allow an
# inherited hot-update environment to turn resume off or drift the schedule.
export EXP_NAME="${CURRENT_EXP_NAME}"
export PERSISTENT_OUTPUT_ROOT="${CURRENT_OUTPUT_ROOT}"
export ASSETS_BASE_DIR="${CURRENT_OUTPUT_ROOT}/assets"
export CHECKPOINT_BASE_DIR="${CURRENT_OUTPUT_ROOT}/checkpoints"
export LOG_BASE_DIR="${CURRENT_OUTPUT_ROOT}/logs"
export RESUME=1
export OVERWRITE=0
export CONFIG_NAME="pi05_ki_joint_query_b1k-full_task-ki_on_skillbridge_bf16"
export PYTORCH_TRAINING_PRECISION="bfloat16"
export NUM_TRAIN_STEPS=0
export NUM_TRAIN_EPOCHS=3
export BATCH_SIZE_PER_GPU=4
export NUM_WORKERS=4
export GRADIENT_ACCUMULATION_STEPS=1
export SAVE_INTERVAL=10000
export VAL_LOG_INTERVAL=2000
export ACCEL_CONFIG="${REPO_ROOT}/configs/accelerate_ds_zero2.yaml"
# train_accelerate.py allows this environment variable to override the JSON;
# pin it explicitly so an inherited hot-update environment cannot re-enable it.
export OPENPI_DS_OVERLAP_COMM=false

if [[ ! -d "${CURRENT_CHECKPOINT_DIR}" ]]; then
  echo "ERROR: current checkpoint directory does not exist: ${CURRENT_CHECKPOINT_DIR}" >&2
  exit 2
fi

LATEST_STEP="$(
  {
    find "${CURRENT_CHECKPOINT_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
      | grep -E '^[0-9]+$' \
      | sort -n \
      | tail -1
  } || true
)"
if [[ -z "${LATEST_STEP}" ]]; then
  echo "ERROR: no numeric checkpoint found under ${CURRENT_CHECKPOINT_DIR}" >&2
  exit 2
fi
if (( LATEST_STEP < MIN_RESUME_STEP )); then
  echo "ERROR: refusing to resume below the verified floor: latest=${LATEST_STEP} minimum=${MIN_RESUME_STEP}" >&2
  exit 2
fi

LATEST_CHECKPOINT_DIR="${CURRENT_CHECKPOINT_DIR}/${LATEST_STEP}"
for required_file in manifest.json metadata.pt model.safetensors optimizer.pt; do
  if [[ ! -s "${LATEST_CHECKPOINT_DIR}/${required_file}" ]]; then
    echo "ERROR: latest checkpoint is incomplete; missing/non-empty file required: ${LATEST_CHECKPOINT_DIR}/${required_file}" >&2
    exit 2
  fi
done

ACCELERATE_STATE_DIR="${LATEST_CHECKPOINT_DIR}/accelerate_state"
DEEPSPEED_STATE_DIR="${ACCELERATE_STATE_DIR}/pytorch_model"
if [[ ! -d "${DEEPSPEED_STATE_DIR}" ]]; then
  echo "ERROR: latest checkpoint is missing DeepSpeed state: ${DEEPSPEED_STATE_DIR}" >&2
  exit 2
fi
if [[ "$(<"${ACCELERATE_STATE_DIR}/latest")" != "pytorch_model" ]]; then
  echo "ERROR: invalid DeepSpeed latest tracker: ${ACCELERATE_STATE_DIR}/latest" >&2
  exit 2
fi
if [[ ! -s "${DEEPSPEED_STATE_DIR}/mp_rank_00_model_states.pt" ]]; then
  echo "ERROR: missing/non-empty DeepSpeed model state: ${DEEPSPEED_STATE_DIR}/mp_rank_00_model_states.pt" >&2
  exit 2
fi
for rank in $(seq 0 63); do
  optim_file="${DEEPSPEED_STATE_DIR}/bf16_zero_pp_rank_${rank}_mp_rank_00_optim_states.pt"
  rng_file="${ACCELERATE_STATE_DIR}/random_states_${rank}.pkl"
  if [[ ! -s "${optim_file}" ]]; then
    echo "ERROR: missing/non-empty optimizer shard for rank ${rank}: ${optim_file}" >&2
    exit 2
  fi
  if [[ ! -s "${rng_file}" ]]; then
    echo "ERROR: missing/non-empty RNG shard for rank ${rank}: ${rng_file}" >&2
    exit 2
  fi
done

"${TRAINING_PYTHON}" - "${LATEST_CHECKPOINT_DIR}" "${LATEST_STEP}" <<'PY'
import json
from pathlib import Path
import sys
import torch

checkpoint_dir = Path(sys.argv[1])
expected_step = int(sys.argv[2])
metadata = torch.load(checkpoint_dir / "metadata.pt", map_location="cpu", weights_only=False)
manifest = json.loads((checkpoint_dir / "manifest.json").read_text())
metadata_step = int(metadata.get("global_step", -1))
manifest_step = int(manifest.get("run_metadata", {}).get("global_step", -1))
if metadata_step != expected_step or manifest_step != expected_step:
    raise SystemExit(
        "ERROR: checkpoint step mismatch: "
        f"directory={expected_step} metadata={metadata_step} manifest={manifest_step}"
    )
PY

printf '%s\n' \
  "[resume-perf] REPO_ROOT=${REPO_ROOT}" \
  "[resume-perf] EXP_NAME=${EXP_NAME}" \
  "[resume-perf] RESUME=${RESUME} OVERWRITE=${OVERWRITE}" \
  "[resume-perf] CHECKPOINT_DIR=${CURRENT_CHECKPOINT_DIR}" \
  "[resume-perf] LATEST_CHECKPOINT_STEP=${LATEST_STEP}" \
  "[resume-perf] STATE_SHARDS=optim:64/64 model:1/1 rng:64/64; metadata/manifest matched" \
  "[resume-perf] CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR}" \
  "[resume-perf] DEEPSPEED_CONFIG=${REPO_ROOT}/configs/deepspeed_zero2.json" \
  "[resume-perf] OPENPI_DS_OVERLAP_COMM=${OPENPI_DS_OVERLAP_COMM}"

exec bash "${REPO_ROOT}/scripts/run_pi05_skillbridge_bf16_multinode_lq_8x8.sh"
