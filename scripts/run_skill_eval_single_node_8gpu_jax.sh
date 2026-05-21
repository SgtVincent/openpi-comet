#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export POLICY_BACKEND="${POLICY_BACKEND:-jax}"
export CONFIG_NAME="${CONFIG_NAME:-pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k}"

exec bash "${SCRIPT_DIR}/run_skill_eval_single_node_8gpu.sh" "$@"
