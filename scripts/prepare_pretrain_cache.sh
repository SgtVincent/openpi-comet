#!/bin/bash
set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

source /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/miniconda3/etc/profile.d/conda.sh
conda activate openpi-comet-nas
export PYTHONPATH="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python:$PYTHONPATH"
export LD_LIBRARY_PATH="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/miniconda3/envs/openpi-comet-nas/lib:$LD_LIBRARY_PATH"

if [[ -z "${TARGET_NODES:-${1:-}}" ]]; then
  echo "Usage: TARGET_NODES=<num_nodes> [NPROC_PER_NODE=8] [CONFIG_NAME=...] bash scripts/prepare_pretrain_cache.sh"
  echo "Or:    bash scripts/prepare_pretrain_cache.sh <num_nodes>"
  exit 2
fi

TARGET_NODES="${TARGET_NODES:-${1}}"
if [[ "${TARGET_NODES}" -le 0 ]]; then
  echo "TARGET_NODES must be > 0, got ${TARGET_NODES}" >&2
  exit 2
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CONFIG_NAME="${CONFIG_NAME:-vlm2_b1k-pt50_cs32_bs64_lr2.5e-5_step50k}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${REPO_ROOT}/checkpoints}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CHECKPOINTS_ROOT}/hf_datasets_cache}"
CLEAN_NODE_CACHE="${CLEAN_NODE_CACHE:-0}"

# Keep behavior consistent with training scripts.
export OPENPI_OFFLINE="${OPENPI_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Enable per-node cache layout: <HF_DATASETS_CACHE>/node{N}
export OPENPI_HF_DATASETS_CACHE_PER_RANK="${OPENPI_HF_DATASETS_CACHE_PER_RANK:-1}"

# Make sure local source tree imports are available.
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

mkdir -p "${HF_DATASETS_CACHE}"

if [[ "${CLEAN_NODE_CACHE}" == "1" ]]; then
  rm -rf "${HF_DATASETS_CACHE}"/node*
fi

WORLD_SIZE="$((TARGET_NODES * NPROC_PER_NODE))"

echo "Preparing HF cache for pretrain"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "TARGET_NODES=${TARGET_NODES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "WORLD_SIZE=${WORLD_SIZE}"
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"

a=0
RANK=0
NODE_DIR="${HF_DATASETS_CACHE}/node0"
mkdir -p "${NODE_DIR}"

echo "[cache-prep] building node0 (RANK=${RANK}, LOCAL_RANK=0)"
CONFIG_NAME="${CONFIG_NAME}" \
HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
WORLD_SIZE="${WORLD_SIZE}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
LOCAL_WORLD_SIZE="${NPROC_PER_NODE}" \
RANK="${RANK}" \
LOCAL_RANK="0" \
NODE_PREP_ID="0" \
OPENPI_FORCE_LOAD_CACHE="0" \
python - <<'PY'
import os
import dataclasses
from pathlib import Path

import tyro

import openpi.training.behavior_dataset as behavior_dataset
from openpi.training.data_config import DataConfig
import openpi.training.train_config as train_config

config_name = os.environ["CONFIG_NAME"]
config = train_config.get_config(config_name)

if isinstance(config.data, list):
    data_factories = list(config.data)
else:
    data_factories = [config.data]

data_configs = []
for cfg in data_factories:
    base = getattr(cfg, "base_config", None)
    repo_id = getattr(cfg, "repo_id", None)
    if repo_id is tyro.MISSING:
        repo_id = None

    if base is not None and isinstance(base, DataConfig):
        data_configs.append(dataclasses.replace(base, repo_id=repo_id or base.repo_id))
    else:
        data_configs.append(cfg.create(config.assets_dirs, config.model))

if not all(behavior_dataset.is_behavior_dataset(dc) for dc in data_configs):
    raise RuntimeError("prepare_pretrain_cache only supports behavior dataset configs.")

# This call triggers BehaviorLeRobotDataset.load_hf_dataset(), generating Arrow cache.
behavior_dataset.create_multi_behavior_dataset(
    data_configs,
    sample_weights=config.sample_weights,
    action_horizon=config.model.action_horizon,
)

cache_root = Path(os.environ["HF_DATASETS_CACHE"]).expanduser()
node_id = int(os.environ["NODE_PREP_ID"])
node_dir = cache_root / f"node{node_id}"
node_dir.mkdir(parents=True, exist_ok=True)
ready = node_dir / ".hf_cache_ready"
tmp_ready = node_dir / f"{ready.name}.{os.getpid()}.tmp"
tmp_ready.write_text("ready\n")
os.replace(tmp_ready, ready)
print(f"[cache-prep] ready marker written: {ready}")
PY

COPY_FLAGS=(-a)
if cp --help 2>/dev/null | grep -q -- "--reflink"; then
  COPY_FLAGS=(-a --reflink=auto)
fi

a=1
while [[ "${a}" -lt "${TARGET_NODES}" ]]; do
  NODE_DIR="${HF_DATASETS_CACHE}/node${a}"
  rm -rf "${NODE_DIR}"
  mkdir -p "${NODE_DIR}"
  echo "[cache-prep] copying node0 -> node${a}"
  cp "${COPY_FLAGS[@]}" "${HF_DATASETS_CACHE}/node0/." "${NODE_DIR}/"
  a="$((a + 1))"
done

echo "All node caches are prepared."
echo "Use FORCE_LOAD_CACHE=1 in distributed training to require prebuilt cache."
