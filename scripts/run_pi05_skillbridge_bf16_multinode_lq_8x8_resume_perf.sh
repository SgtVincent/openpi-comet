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
#   - Verifies checkpoint completeness before launching, with retries so that
#     transient BN-mount staleness does not release the 64-GPU allocation.
#   - Keeps topology, data, batch size, schedule, and output root identical to
#     the current run; only finite-consensus diagnostics and DeepSpeed
#     overlap-communication behavior differ.
#
# Shell-option note: validation runs WITHOUT `set -e` on purpose. On Merlin an
# entrypoint that exits releases the whole allocation, and this repo has a
# documented history of transient BN-mount staleness (the `_exp_name_sync`
# freshness timeouts). Validation therefore retries, and a persistent failure is
# reported loudly and then handed to the keepalive wrapper, which holds the GPUs
# for inspection instead of returning them to the queue.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
cd "${REPO_ROOT}"

readonly CURRENT_EXP_NAME="pi05_ki_joint_query_single_task_radio_bf16_8n8g_20260812_151551"
readonly CURRENT_OUTPUT_ROOT="/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/outputs/pi05_skillbridge_a100_lq_bf16_8x8"
readonly CURRENT_CHECKPOINT_DIR="${CURRENT_OUTPUT_ROOT}/checkpoints/${CURRENT_EXP_NAME}"
readonly MIN_RESUME_STEP=80000
readonly TRAINING_PYTHON="/mnt/bn/saiwenresearch/mlx/users/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python"
readonly EXPECTED_WORLD_SIZE=64
readonly VALIDATION_ATTEMPTS="${RESUME_PERF_VALIDATION_ATTEMPTS:-5}"
readonly VALIDATION_RETRY_SLEEP_S="${RESUME_PERF_VALIDATION_RETRY_SLEEP_S:-20}"

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

validate_resume_target() {
  local checkpoint_dir latest_step deepspeed_dir rank optim_file rng_file

  if [[ ! -d "${CURRENT_CHECKPOINT_DIR}" ]]; then
    echo "validation: checkpoint directory not visible: ${CURRENT_CHECKPOINT_DIR}" >&2
    return 1
  fi

  latest_step="$(
    {
      find "${CURRENT_CHECKPOINT_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
        | grep -E '^[0-9]+$' \
        | sort -n \
        | tail -1
    } 2>/dev/null || true
  )"
  if [[ -z "${latest_step}" ]]; then
    echo "validation: no numeric checkpoint found under ${CURRENT_CHECKPOINT_DIR}" >&2
    return 1
  fi
  if (( latest_step < MIN_RESUME_STEP )); then
    echo "validation: latest checkpoint ${latest_step} is below the verified floor ${MIN_RESUME_STEP}" >&2
    return 1
  fi

  checkpoint_dir="${CURRENT_CHECKPOINT_DIR}/${latest_step}"
  for required_file in manifest.json metadata.pt model.safetensors optimizer.pt; do
    if [[ ! -s "${checkpoint_dir}/${required_file}" ]]; then
      echo "validation: missing or empty ${checkpoint_dir}/${required_file}" >&2
      return 1
    fi
  done

  deepspeed_dir="${checkpoint_dir}/accelerate_state/pytorch_model"
  if [[ ! -d "${deepspeed_dir}" ]]; then
    echo "validation: missing DeepSpeed state directory ${deepspeed_dir}" >&2
    return 1
  fi
  if [[ "$(cat "${checkpoint_dir}/accelerate_state/latest" 2>/dev/null)" != "pytorch_model" ]]; then
    echo "validation: invalid DeepSpeed latest tracker in ${checkpoint_dir}/accelerate_state" >&2
    return 1
  fi
  if [[ ! -s "${deepspeed_dir}/mp_rank_00_model_states.pt" ]]; then
    echo "validation: missing or empty ${deepspeed_dir}/mp_rank_00_model_states.pt" >&2
    return 1
  fi
  for rank in $(seq 0 $((EXPECTED_WORLD_SIZE - 1))); do
    optim_file="${deepspeed_dir}/bf16_zero_pp_rank_${rank}_mp_rank_00_optim_states.pt"
    rng_file="${checkpoint_dir}/accelerate_state/random_states_${rank}.pkl"
    if [[ ! -s "${optim_file}" ]]; then
      echo "validation: missing or empty optimizer shard for rank ${rank}" >&2
      return 1
    fi
    if [[ ! -s "${rng_file}" ]]; then
      echo "validation: missing or empty RNG shard for rank ${rank}" >&2
      return 1
    fi
  done

  if ! "${TRAINING_PYTHON}" - "${checkpoint_dir}" "${latest_step}" <<'PY'
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
        "validation: checkpoint step mismatch: "
        f"directory={expected_step} metadata={metadata_step} manifest={manifest_step}"
    )
PY
  then
    echo "validation: checkpoint step metadata did not agree for step ${latest_step}" >&2
    return 1
  fi

  RESUME_TARGET_STEP="${latest_step}"
  return 0
}

RESUME_TARGET_STEP=""
for attempt in $(seq 1 "${VALIDATION_ATTEMPTS}"); do
  if validate_resume_target; then
    break
  fi
  if (( attempt < VALIDATION_ATTEMPTS )); then
    echo "[resume-perf] checkpoint validation attempt ${attempt}/${VALIDATION_ATTEMPTS} failed; " \
         "retrying in ${VALIDATION_RETRY_SLEEP_S}s (BN mount staleness is transient)" >&2
    sleep "${VALIDATION_RETRY_SLEEP_S}"
  fi
done

if [[ -z "${RESUME_TARGET_STEP}" ]]; then
  # Do NOT exit: exiting the Merlin entrypoint releases all 64 GPUs. Hand over to
  # the keepalive wrapper with training disabled so the allocation is held for
  # inspection. Set RESUME_PERF_EXIT_ON_INVALID=1 to fail fast instead.
  echo "ERROR[resume-perf]: checkpoint validation failed after ${VALIDATION_ATTEMPTS} attempts." >&2
  echo "ERROR[resume-perf]: refusing to start training; the allocation will be held instead." >&2
  if [[ "${RESUME_PERF_EXIT_ON_INVALID:-0}" == "1" ]]; then
    exit 2
  fi
  export TRAIN_COMMAND="bash -c 'echo \"resume-perf: checkpoint validation failed; training intentionally not started\"; exit 1'"
  exec bash "${REPO_ROOT}/scripts/run_pi05_skillbridge_lq_keepalive_on_failure.sh"
fi

printf '%s\n' \
  "[resume-perf] REPO_ROOT=${REPO_ROOT}" \
  "[resume-perf] EXP_NAME=${EXP_NAME}" \
  "[resume-perf] RESUME=${RESUME} OVERWRITE=${OVERWRITE}" \
  "[resume-perf] CHECKPOINT_DIR=${CURRENT_CHECKPOINT_DIR}" \
  "[resume-perf] RESUME_TARGET_STEP=${RESUME_TARGET_STEP}" \
  "[resume-perf] STATE_VERIFIED=optim:${EXPECTED_WORLD_SIZE}/${EXPECTED_WORLD_SIZE} model:1/1 rng:${EXPECTED_WORLD_SIZE}/${EXPECTED_WORLD_SIZE}; metadata/manifest matched" \
  "[resume-perf] CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR}" \
  "[resume-perf] DEEPSPEED_CONFIG=${REPO_ROOT}/configs/deepspeed_zero2.json" \
  "[resume-perf] OPENPI_DS_OVERLAP_COMM=${OPENPI_DS_OVERLAP_COMM}"

exec bash "${REPO_ROOT}/scripts/run_pi05_skillbridge_bf16_multinode_lq_8x8.sh"
