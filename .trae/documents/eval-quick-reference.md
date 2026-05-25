# BEHAVIOR Eval 常用启动模板速查表

> 本文档放在 `openpi-comet/.trae/documents/`，用于直接被 `openpi-comet` 纳入版本管理。

完整说明请配合阅读：

- `openpi-comet/.trae/documents/behavior_eval_orchestration_guide.md`
- `BEHAVIOR-1K/.trae/documents/behavior_eval_runtime_guide.md`
- `BEHAVIOR-1K/.trae/documents/rlinf-style-eval-optimization.md`

## 1. 默认推荐组合

- 入口：`scripts/run_skill_eval_single_node_8gpu.sh`
- 模式：`EVAL_MODE=persistent`
- 恢复：`RESUME=1`
- 视频：`WRITE_VIDEO=1`
- 排障：`SEGMENT_PREDICATE_DUMP_TRACE=1`
- runtime：保持 `headless=true`、`partial_scene_load=false`、`skip_intermediate_obs_in_chunk=true`

## 2. 最常用三条命令

### 2.1 单机正式跑一批 skill

```bash
cd /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet && \
RUN_TAG=behavior_eval_$(date +%Y%m%d_%H%M%S) \
CKPT_DIR=/abs/path/to/checkpoint_dir \
CONFIG_NAME=pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k \
SKILLS="move to,open door" \
MAX_SAMPLES_PER_SKILL=32 \
MAX_SAMPLES_PER_SKILL_TASK=2 \
MAX_TOTAL_JOBS=0 \
MAX_STEPS=120 \
MAX_DYNAMIC_STEPS_CAP=0 \
RESUME=1 \
WRITE_VIDEO=1 \
SEGMENT_PREDICATE_DUMP_TRACE=1 \
EVAL_MODE=persistent \
bash scripts/run_skill_eval_single_node_8gpu.sh
```

### 2.2 quick validation

```bash
cd /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet && \
RUN_TAG=behavior_quick_validation_$(date +%Y%m%d_%H%M%S) \
CKPT_DIR=/abs/path/to/checkpoint_dir \
CONFIG_NAME=pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k \
SKILLS="move to,open door" \
MAX_SAMPLES_PER_SKILL=8 \
MAX_SAMPLES_PER_SKILL_TASK=2 \
WRITE_VIDEO=1 \
SEGMENT_PREDICATE_DUMP_TRACE=1 \
EVAL_MODE=persistent \
bash scripts/run_quick_validation_dataset_single_node_8gpu.sh run
```

### 2.3 direct segment debug

```bash
cd /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet && \
TASK_NAME="opening_door" \
GPU_IDS=0 \
PORT_BASE=9700 \
HEADLESS=1 \
SEGMENT_MAX_STEPS=120 \
SUCCESS_MODE=predicate_satisfied \
bash scripts/run_b1k_skill_segment_eval.sh \
  /abs/path/to/full_run_dir \
  /abs/path/to/checkpoint_dir
```

## 3. 进阶模板

### 3.1 multinode launch

```bash
cd /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet && \
RUN_TAG=behavior_multinode_eval_$(date +%Y%m%d_%H%M%S) \
NODE_RANK=0 \
NUM_NODES=4 \
GPUS_PER_NODE=8 \
LOCAL_GPU_IDS=0,1,2,3,4,5,6,7 \
CKPT_DIR=/abs/path/to/checkpoint_dir \
CONFIG_NAME=pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k \
RESUME=1 \
EVAL_MODE=persistent \
bash scripts/conda_run_pi05_b1kpt50_multinode_skill_eval.sh launch
```

### 3.2 direct serve + full-task eval

```bash
cd /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet && \
python scripts/serve_b1k.py \
  --task_name="opening_door" \
  --control_mode=receeding_horizon \
  --max_len=32 \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k \
  --policy.dir=/abs/path/to/checkpoint_dir
```

## 4. 最常改字段

| 目标 | 最常改字段 |
|---|---|
| 换模型 | `CKPT_DIR`, `CONFIG_NAME`, `POLICY_BACKEND` |
| 缩小范围 | `SKILLS`, `MAX_SAMPLES_PER_SKILL`, `MAX_SAMPLES_PER_SKILL_TASK`, `MAX_TOTAL_JOBS` |
| 控时长 | `MAX_STEPS`, `MAX_DYNAMIC_STEPS_CAP`, `SEGMENT_MAX_STEPS` |
| 提高吞吐 | `EVAL_MODE=persistent` |
| 增强调试 | `WRITE_VIDEO=1`, `SEGMENT_PREDICATE_DUMP_TRACE=1` |

## 5. 抄命令前先检查

1. `CKPT_DIR` 正确。
2. `RUN_TAG` / `OUT_DIR` 新建。
3. `RESUME=1` 是否符合预期。
4. `LOCAL_GPU_IDS` 个数是否等于 `GPUS_PER_NODE`。
5. 当前 conda env 是否正确。
6. 没有误把 arena evaluator / eval-jobqueue 命令混进来。
