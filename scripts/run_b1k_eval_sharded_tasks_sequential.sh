#!/usr/bin/env bash
set -euo pipefail

# Run multiple tasks sequentially, but shard each task across multiple GPUs.
#
# Motivation:
# - BEHAVIOR tasks can have very different horizons.
# - Running 1 task per GPU pair can lead to poor utilization (fast tasks finish early; long tasks dominate wall time).
# - This script instead evaluates tasks one-by-one, using ALL GPU pairs to shard test-instance indices for the current task.
#
# Under the hood, this calls `scripts/run_b1k_eval_parallel_single_task.sh` for each task.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Comma-separated list of tasks.
TASKS="${TASKS:-putting_shoes_on_rack,picking_up_trash,setting_mousetraps,turning_on_radio,hiding_Easter_eggs,bringing_in_wood,moving_boxes_to_storage,sorting_vegetables}"

NUM_GPUS="${NUM_GPUS:-8}"
PORT_BASE="${PORT_BASE:-8000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"

# Keep consistent defaults with other launchers.
# In shared-server mode, the server uses one GPU and evaluators can use the rest,
# so colocate is typically the best utilization setting.
PAIR_MODE="${PAIR_MODE:-colocate}"
WRITE_VIDEO="${WRITE_VIDEO:-false}"
HEADLESS="${HEADLESS:-true}"
MAX_STEPS="${MAX_STEPS:-}"

EVAL_ENTRYPOINT="${EVAL_ENTRYPOINT:-eval_custom.py}"  # eval.py | eval_custom.py
ENV_WRAPPER_TARGET="${ENV_WRAPPER_TARGET:-omnigibson.learning.wrappers.RGBWrapper}"

# Prefer a shared server per task by default: simulation is usually the bottleneck.
SERVER_MODE="${SERVER_MODE:-shared}"  # per_eval | shared

# Server-side JAX memory controls
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.40}"
XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-10}"

RUN_TAG="${RUN_TAG:-sharded_tasks_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/eval_logs/pi05-b1kpt50-cs32/$RUN_TAG}"
mkdir -p "$OUT_ROOT"

IFS=',' read -r -a TASK_ARRAY <<< "$TASKS"

echo "Writing outputs to: $OUT_ROOT"
echo "Tasks (sequential): $TASKS"
echo "NUM_GPUS=$NUM_GPUS  PAIR_MODE=$PAIR_MODE  EVAL_EPISODES=$EVAL_EPISODES  PORT_BASE=$PORT_BASE"
echo "Evaluator: $EVAL_ENTRYPOINT  Wrapper: $ENV_WRAPPER_TARGET"

for task in "${TASK_ARRAY[@]}"; do
  task_idx=${task_idx:-0}
  echo
  echo "================================================================================"
  echo "== Evaluating task: $task"
  echo "================================================================================"

  # Use a per-task port base to avoid collisions if a previous server fails to exit cleanly.
  TASK_PORT_BASE=$((PORT_BASE + task_idx * 100))

  # Each task writes into its own subdir.
  TASK_NAME="$task" \
    OUT_DIR="$OUT_ROOT/$task" \
    NUM_GPUS="$NUM_GPUS" \
    PORT_BASE="$TASK_PORT_BASE" \
    EVAL_EPISODES="$EVAL_EPISODES" \
    PAIR_MODE="$PAIR_MODE" \
    WRITE_VIDEO="$WRITE_VIDEO" \
    HEADLESS="$HEADLESS" \
    MAX_STEPS="$MAX_STEPS" \
    SERVER_STARTUP_WAIT="$SERVER_STARTUP_WAIT" \
    EVAL_ENTRYPOINT="$EVAL_ENTRYPOINT" \
    ENV_WRAPPER_TARGET="$ENV_WRAPPER_TARGET" \
    SERVER_MODE="$SERVER_MODE" \
    XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PYTHON_CLIENT_PREALLOCATE" \
    XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_PYTHON_CLIENT_MEM_FRACTION" \
    XLA_PYTHON_CLIENT_ALLOCATOR="$XLA_PYTHON_CLIENT_ALLOCATOR" \
    bash "$REPO_ROOT/scripts/run_b1k_eval_parallel_single_task.sh"

  echo "== Finished task: $task"
  task_idx=$((task_idx + 1))
done

echo
echo "All tasks finished. Results under: $OUT_ROOT"