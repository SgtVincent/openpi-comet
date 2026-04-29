#!/usr/bin/env bash
set -euo pipefail

find_first_executable() {
  local candidate
  for candidate in "$@"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      printf "%s\n" "${candidate}"
      return 0
    fi
  done
  return 1
}

RUNTIME_ENV_FILE="${HOME}/.openpi_runtime_env.sh"
if [[ -f "${RUNTIME_ENV_FILE}" ]]; then
  # Reuse paths prepared by setup_bashrc.sh when available.
  source "${RUNTIME_ENV_FILE}"
fi

echo "=========================================================="
echo "Starting Sequential Evaluation for 3 make_pizza models"
echo "=========================================================="

REPO_ROOT="/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet"
cd "${REPO_ROOT}"

export TASK_NAME=make_pizza
export BEHAVIOR_DIR=/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K
export OPENPI_ENV=openpi-comet-nas
export BEHAVIOR_ENV=behavior
export OPENPI_PYTHON="$(find_first_executable \
  "/mnt/bn/behavior-data-hl/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python" \
  "${OPENPI_PYTHON:-}" \
  "${REPO_ROOT}/.venv/bin/python" \
  || true)"
export BEHAVIOR_PYTHON="$(find_first_executable \
  "/mnt/bn/behavior-data-hl/chenjunting/miniconda3/envs/behavior/bin/python" \
  "${BEHAVIOR_PYTHON:-}" \
  || true)"

[[ -x "${OPENPI_PYTHON}" ]] || {
  echo "[Error] No executable OPENPI_PYTHON found." >&2
  exit 1
}
[[ -x "${BEHAVIOR_PYTHON}" ]] || {
  echo "[Error] No executable BEHAVIOR_PYTHON found." >&2
  exit 1
}

echo "================ Runtime Python ================"
echo "OPENPI_PYTHON : ${OPENPI_PYTHON}"
echo "BEHAVIOR_PYTHON: ${BEHAVIOR_PYTHON}"
echo "================================================"

# Eval on 10 instances
export EVAL_INSTANCE_IDS=0,1,2,3,4,5,6,7,8,9
export PORT_BASE=8900
export SERVER_STARTUP_WAIT=600  # 增加超时时间，避免 NAS 读权重或 JAX 编译耗时过长导致 timeout
export OMNIGIBSON_DISABLE_EXTENSION_REGISTRY=0
export OMNIGIBSON_DISABLE_DRIVER_VERSION_CHECK=1
export HEADLESS=true

RUN_DIRS=()

# 1. Baseline Model
# echo ">>> Evaluating Baseline Model..."
# export OPENPI_CONFIG_NAME="pi05_b1k-make_pizza_lr1e-4_5ep_sft"
# CKPT1="/mnt/bn/behavior-data-hl/chenjunting/checkpoints/pi05_b1k-make_pizza_lr1e-4_5ep_sft_baseckpt_1ep_20260316_233615"
# export RUN_TAG="parallel_eval_make_pizza_baseline_$(date +%Y%m%d_%H%M%S)"
# RUN_DIRS+=("eval_logs/$RUN_TAG")
# bash scripts/run_b1k_eval_parallel_single_task_headless.sh "$CKPT1"
# echo ">>> Baseline evaluation finished."

# 2. Hamlet Model
echo ">>> Evaluating Hamlet Model..."
export OPENPI_CONFIG_NAME="pi05_hamlet_b1k-make_pizza_lr1e-4_5ep_sft"
CKPT2="/mnt/bn/behavior-data-hl/chenjunting/checkpoints/pi05_hamlet_b1k-make_pizza_lr1e-4_5ep_sft_baseckpt_5ep_20260418_005748"
export RUN_TAG="parallel_eval_make_pizza_hamlet_$(date +%Y%m%d_%H%M%S)"
RUN_DIRS+=("eval_logs/$RUN_TAG")
bash scripts/run_b1k_eval_parallel_single_task_headless.sh "$CKPT2"
echo ">>> Hamlet evaluation finished."

# 3. MemoryVLA Model
echo ">>> Evaluating MemoryVLA Model..."
export OPENPI_CONFIG_NAME="pi05_memoryvla_b1k-make_pizza_lr1e-4_5ep_sft"
CKPT3="/mnt/bn/behavior-data-hl/chenjunting/checkpoints/pi05_memoryvla_b1k-make_pizza_lr1e-4_5ep_sft_baseckpt_5ep_20260418_162427"
export RUN_TAG="parallel_eval_make_pizza_memoryvla_$(date +%Y%m%d_%H%M%S)"
RUN_DIRS+=("eval_logs/$RUN_TAG")
bash scripts/run_b1k_eval_parallel_single_task_headless.sh "$CKPT3"
echo ">>> MemoryVLA evaluation finished."

# 汇总结果
echo "================ EVALUATION SUMMARY ================"
for run_dir in "${RUN_DIRS[@]}"; do
    echo "Summary for $run_dir :"
    if [ -d "$run_dir" ]; then
        "${OPENPI_PYTHON}" summarize_eval_metrics.py "$run_dir"
    else
        echo "Directory not found: $run_dir"
    fi
    echo "---------------------------------------------------"
done