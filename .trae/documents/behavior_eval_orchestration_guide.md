# BEHAVIOR Eval Orchestration Guide

> 本文档放在 `openpi-comet/.trae/documents/`，用于被 `openpi-comet` 仓库直接纳入版本管理。
>
> 配套文档：
>
> - `openpi-comet/.trae/documents/eval-quick-reference.md`
> - `BEHAVIOR-1K/.trae/documents/behavior_eval_runtime_guide.md`
> - `BEHAVIOR-1K/.trae/documents/rlinf-style-eval-optimization.md`

## 1. 范围

本文档只讲 **launcher / orchestration 侧**：

- `run_skill_eval_single_node_8gpu.sh`
- `run_quick_validation_dataset_single_node_8gpu.sh`
- `conda_run_pi05_b1kpt50_multinode_skill_eval.sh`
- `run_skill_metric_multinode_sweep.py`
- `persistent_skill_eval_worker.py`
- `serve_b1k.py`

它不替代 behavior 侧运行时说明；`eval.py` / `eval_segment.py` / websocket config / RLinf-style runtime knobs 请看 `BEHAVIOR-1K/.trae/documents/behavior_eval_runtime_guide.md`。

## 2. 默认推荐路径

默认按下面优先级选：

1. 单机正式 skill eval：`scripts/run_skill_eval_single_node_8gpu.sh`
2. quick validation：`scripts/run_quick_validation_dataset_single_node_8gpu.sh`
3. multinode：`scripts/conda_run_pi05_b1kpt50_multinode_skill_eval.sh`
4. direct segment debug：`scripts/run_b1k_skill_segment_eval.sh`
5. direct serve：`scripts/serve_b1k.py`

推荐依据：

- 单机脚本默认走 `launch+merge`，见 `openpi-comet/scripts/run_skill_eval_single_node_8gpu.sh:26`
- 默认 `EVAL_MODE=persistent`，见 `openpi-comet/scripts/run_skill_eval_single_node_8gpu.sh:65`
- 核心调度器是 `openpi-comet/scripts/run_skill_metric_multinode_sweep.py:2588`

## 3. 启动前必须收集的信息

### 3.1 路径类

- `CKPT_DIR`
- `CONFIG_NAME`
- `BEHAVIOR_DIR`
- `DEMO_DATA_PATH`
- `RAWDATA_PATH`
- `OUT_DIR` / `RUN_TAG`

默认入口见 `openpi-comet/scripts/run_skill_eval_single_node_8gpu.sh:196`。

### 3.2 范围类

- `SKILLS`
- `TASK_NAME`
- `MAX_SAMPLES_PER_SKILL`
- `MAX_SAMPLES_PER_SKILL_TASK`
- `MAX_TOTAL_JOBS`

定义见 `openpi-comet/scripts/run_skill_metric_multinode_sweep.py:2634`。

### 3.3 资源与运行模式

- `NUM_NODES`
- `GPUS_PER_NODE`
- `LOCAL_GPU_IDS`
- `PORT_BASE`
- `EVAL_MODE=persistent|process_per_segment`
- `POLICY_BACKEND=auto|torch|jax`

定义见：

- `openpi-comet/scripts/run_skill_eval_single_node_8gpu.sh:65`
- `openpi-comet/scripts/run_skill_metric_multinode_sweep.py:2601`

## 4. 三条常用命令

### 4.1 单机正式 skill eval

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

### 4.2 quick validation

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

### 4.3 multinode launch

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

## 5. launcher 侧关键 flags

### `EVAL_MODE`

- `persistent`：每张 GPU 一个长寿命 worker，复用 Isaac / env / evaluator / server。
- `process_per_segment`：每个 segment 起一个新进程，作为回滚路径。

见 `openpi-comet/scripts/run_skill_eval_single_node_8gpu.sh:65`。

### `RESUME`

- 如果已有 `metrics/*.json`，launcher 会直接跳过该 job。
- 换 checkpoint 或 flags 但不换 `OUT_DIR` 时，最容易误复用。

实现见 `openpi-comet/scripts/run_skill_metric_multinode_sweep.py:2088`。

### `MAX_DYNAMIC_STEPS_CAP`

- 给按 `frame_duration` 推导出的动态步数加上限。
- 太小会截断长段；太大则会拖长坏样本。

### `LOCAL_GPU_IDS` / `GPUS_PER_NODE`

- 二者数量必须一致，否则直接报错。
- 见 `openpi-comet/scripts/run_skill_metric_multinode_sweep.py:2700`。

### `PERSISTENT_WORKER_*`

- `PERSISTENT_WORKER_MAX_SEGMENTS_BEFORE_RESTART`
- `PERSISTENT_WORKER_HEARTBEAT_S`
- `PERSISTENT_WORKER_TASK_RELOAD_TIMEOUT_S`
- `PERSISTENT_WORKER_SEGMENT_TIMEOUT_S`
- `PERSISTENT_WORKER_SHUTDOWN_TIMEOUT`

这些参数用于保证长寿命 worker 跑稳，见 `openpi-comet/scripts/run_skill_eval_single_node_8gpu.sh:69`。

## 6. 输出与排查路径

优先看：

- `OUT_DIR/manifest.json`
- `OUT_DIR/worker_plan.csv`
- `OUT_DIR/persistent_worker_plan.csv`
- `OUT_DIR/launcher_logs/`
- `OUT_DIR/server_logs/`
- `OUT_DIR/worker_results/`
- `OUT_DIR/worker_status/`
- `OUT_DIR/raw/.../metrics/*.json`

## 7. 常见坑

### 7.1 旧 server 串线

不要只看 healthz；要配合 behavior 侧的 `expected_server_*` 身份校验一起看。

### 7.2 `/tmp` 不适合正式结果

调度器默认拒绝正式输出写到 `/tmp`，见 `openpi-comet/scripts/run_skill_metric_multinode_sweep.py:2712`。

### 7.3 `persistent` 自动重启不一定是 bug

先看 `worker_status/persistent_worker_*.jsonl`，再判断是不是逻辑错误。

### 7.4 arena / eval-jobqueue 不是默认路径

当前仓库默认优先使用本文档中的本地 behavior eval orchestration 路径。

