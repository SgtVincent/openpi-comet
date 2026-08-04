#!/bin/bash
# End-to-end Golden Rule test with video output
# Usage: ./run_golden_rule_e2e_test.sh [task_name] [demo_id]
#
# This script:
# 1. Starts serve_golden_rule.py in the openpi-comet-nas environment
# 2. Waits for the server to be ready
# 3. Runs eval_golden_rule.py in the behavior environment
# 4. Outputs ego video with skill plan overlay

set -euo pipefail

TASK_NAME="${1:-turning_on_radio}"
DEMO_ID="${2:-00000010}"
CKPT_DIR="/mnt/bn/navigation-hl/mlx/users/chenjunting/checkpoints/openpi_comet/pi05_skill_pt12_pretrain_4x8_20260325_065102/30997"
CONFIG_NAME="pi05_b1k_skill-pt12_pretrain_lr1e-4_2ep"
DEMO_DATA_PATH="/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos"
LOG_SUFFIX="${LOG_SUFFIX:-}"
LOG_PATH="/mnt/bn/behavior-data-hl/chenjunting/repo/eval_logs/golden_rule_e2e_${TASK_NAME}_${DEMO_ID}${LOG_SUFFIX}"
PORT="${PORT:-8123}"
SKILL_TIMEOUT_STEPS="${SKILL_TIMEOUT_STEPS:-300}"
SKILL_MAX_STEPS_MULTIPLIER="${SKILL_MAX_STEPS_MULTIPLIER:-2.0}"
PLAN_START_SKILL_IDX="${PLAN_START_SKILL_IDX:-0}"
DIAGNOSTIC_SKILL_IDX="${DIAGNOSTIC_SKILL_IDX:-null}"
PROMPT_OVERRIDE="${PROMPT_OVERRIDE:-}"
SKILL_PROMPT_TEMPLATE="${SKILL_PROMPT_TEMPLATE:-}"
SKILL_PROMPT_OVERRIDE="${SKILL_PROMPT_OVERRIDE:-}"
SKILL_PROMPT_DETAIL_MAP_JSON="${SKILL_PROMPT_DETAIL_MAP_JSON:-}"
RESTORE_AT_EACH_PRIMITIVE_START="${RESTORE_AT_EACH_PRIMITIVE_START:-false}"
MAX_LEN="${MAX_LEN:-32}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"

REPO_ROOT="/mnt/bn/behavior-data-hl/chenjunting/repo"
OPENPI_ROOT="${REPO_ROOT}/openpi-comet"
B1K_ROOT="${REPO_ROOT}/BEHAVIOR-1K"
CONDA_SH="/mnt/bn/behavior-data-hl/chenjunting/miniconda3/etc/profile.d/conda.sh"

mkdir -p "${LOG_PATH}"

echo "========================================"
echo "Golden Rule E2E Test"
echo "  Task: ${TASK_NAME}"
echo "  Demo: ${DEMO_ID}"
echo "  Checkpoint: ${CKPT_DIR}"
echo "  Log: ${LOG_PATH}"
echo "  Skill timeout steps: ${SKILL_TIMEOUT_STEPS}"
echo "  Skill max steps multiplier: ${SKILL_MAX_STEPS_MULTIPLIER}"
echo "  Plan start skill idx: ${PLAN_START_SKILL_IDX}"
echo "  Diagnostic skill idx: ${DIAGNOSTIC_SKILL_IDX}"
echo "  Skill prompt template: ${SKILL_PROMPT_TEMPLATE:-<none>}"
echo "  Skill prompt override: ${SKILL_PROMPT_OVERRIDE:-<none>}"
echo "  Skill prompt detail map: ${SKILL_PROMPT_DETAIL_MAP_JSON:-<none>}"
echo "  Restore at each primitive start: ${RESTORE_AT_EACH_PRIMITIVE_START}"
echo "  Max len: ${MAX_LEN}"
echo "  Action horizon: ${ACTION_HORIZON}"
echo "========================================"

# --- Step 1: Start server in openpi-comet-nas env ---
echo "[1/3] Starting policy server..."
SERVER_LOG="${LOG_PATH}/server.log"
source "${CONDA_SH}"
conda activate openpi-comet-nas

SERVER_EXTRA_ARGS=()
if [ -n "${PROMPT_OVERRIDE}" ]; then
    SERVER_EXTRA_ARGS+=(--prompt-override "${PROMPT_OVERRIDE}")
fi
if [ -n "${SKILL_PROMPT_TEMPLATE}" ]; then
    SERVER_EXTRA_ARGS+=(--skill-prompt-template "${SKILL_PROMPT_TEMPLATE}")
fi
if [ -n "${SKILL_PROMPT_OVERRIDE}" ]; then
    SERVER_EXTRA_ARGS+=(--skill-prompt-override "${SKILL_PROMPT_OVERRIDE}")
fi
if [ -n "${SKILL_PROMPT_DETAIL_MAP_JSON}" ]; then
    SERVER_EXTRA_ARGS+=(--skill-prompt-detail-map-json "${SKILL_PROMPT_DETAIL_MAP_JSON}")
fi

python "${OPENPI_ROOT}/scripts/serve_golden_rule.py" \
    --task-name "${TASK_NAME}" \
    --demo-data-path "${DEMO_DATA_PATH}" \
    --demo-id "${DEMO_ID}" \
    --plan-start-skill-idx "${PLAN_START_SKILL_IDX}" \
    --port "${PORT}" \
    --fine-grained-level 2 \
    --control-mode receeding_horizon \
    --max-len "${MAX_LEN}" \
    --action-horizon "${ACTION_HORIZON}" \
    "${SERVER_EXTRA_ARGS[@]}" \
    policy:checkpoint \
    --policy.config "${CONFIG_NAME}" \
    --policy.dir "${CKPT_DIR}" \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "  Server PID: ${SERVER_PID}"
conda deactivate

# --- Step 2: Wait for server ---
echo "[2/3] Waiting for server to be ready..."
for i in {1..300}; do
    if curl -sf --noproxy "127.0.0.1" "http://127.0.0.1:${PORT}/healthz" > /dev/null 2>&1; then
        echo "  Server is ready!"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "  Server process exited unexpectedly!"
        cat "${SERVER_LOG}"
        exit 1
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "  Still waiting for server... (${i}/300)"
    fi
    sleep 2
done

if ! curl -sf --noproxy "127.0.0.1" "http://127.0.0.1:${PORT}/healthz" > /dev/null 2>&1; then
    echo "  Server failed to start within timeout"
    cat "${SERVER_LOG}"
    exit 1
fi

# --- Step 3: Run evaluator in behavior env (has isaacsim/OmniGibson) ---
# We need to add gello to PYTHONPATH since it's in a different repo
echo "[3/3] Running evaluator..."
conda activate behavior
export PYTHONPATH="/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/behavior-1k-solution/BEHAVIOR-1K/joylo:${PYTHONPATH:-}"

# Bypass NVIDIA driver version check in Isaac Sim
export OMNIGIBSON_DISABLE_DRIVER_VERSION_CHECK=1
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_GPU_ID=0

# Preflight: force-import gt_plan_loader before eval_golden_rule.py runs,
# to ensure the module is registered in sys.modules before any stale cache kicks in.
python -B -c "
import sys
# Force load gt_plan_loader into sys.modules manually
import importlib.util
spec = importlib.util.spec_from_file_location(
    'omnigibson.learning.gt_plan_loader',
    '${B1K_ROOT}/OmniGibson/omnigibson/learning/gt_plan_loader.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.modules['omnigibson.learning.gt_plan_loader'] = module
print('Preloaded gt_plan_loader')

# Also preload eval_golden_rule dependencies
for mod_name in ['omnigibson.learning.eval_golden_rule', 'omnigibson.learning.eval_golden_rule_batch']:
    try:
        spec = importlib.util.spec_from_file_location(
            mod_name,
            '${B1K_ROOT}/OmniGibson/omnigibson/learning/' + mod_name.split('.')[-1] + '.py'
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[mod_name] = module
        print(f'Preloaded {mod_name}')
    except Exception as e:
        print(f'Failed to preload {mod_name}: {e}')
"

python "${B1K_ROOT}/OmniGibson/omnigibson/learning/eval_golden_rule.py" \
    task.name="${TASK_NAME}" \
    demo_data_path="${DEMO_DATA_PATH}" \
    demo_id="${DEMO_ID}" \
    log_path="${LOG_PATH}" \
    write_video=true \
    headless=true \
    policy=websocket \
    model.host=127.0.0.1 \
    model.port="${PORT}" \
    skill_timeout_steps="${SKILL_TIMEOUT_STEPS}" \
    skill_max_steps_multiplier="${SKILL_MAX_STEPS_MULTIPLIER}" \
    restore_at_each_primitive_start="${RESTORE_AT_EACH_PRIMITIVE_START}" \
    diagnostic_skill_idx="${DIAGNOSTIC_SKILL_IDX}" \
    control_mode=receeding_horizon \
    max_len="${MAX_LEN}" \
    fine_grained_level=2 \
    wrap_policy_locally=false \
    2>&1 | tee "${LOG_PATH}/eval.log"

EVAL_EXIT=${PIPESTATUS[0]}

# --- Cleanup ---
echo "Stopping server (PID: ${SERVER_PID})..."
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true

if [ -f "${LOG_PATH}/videos/${TASK_NAME}_golden_rule_demo${DEMO_ID}.mp4" ]; then
    echo ""
    echo "========================================"
    echo "SUCCESS! Video output:"
    echo "  ${LOG_PATH}/videos/${TASK_NAME}_golden_rule_demo${DEMO_ID}.mp4"
    echo "========================================"
else
    echo ""
    echo "Warning: Video file not found at expected path"
fi

exit "${EVAL_EXIT}"
