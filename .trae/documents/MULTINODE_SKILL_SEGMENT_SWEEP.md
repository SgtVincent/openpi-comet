# Multinode Skill Segment Sweep

## Goal

在 `4 nodes x 8 GPUs = 32 GPUs` 上并行运行 BEHAVIOR-1K 的 `segment / skill` 级评测，统计单个 checkpoint 在所有已注册 skill 上的成功率差异，并尽量覆盖更多 `task / demo data`。

本方案新增脚本：

- [run_skill_metric_multinode_sweep.py](file:///mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/scripts/run_skill_metric_multinode_sweep.py)
- [conda_run_pi05_b1kpt50_multinode_skill_eval.sh](file:///mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/scripts/conda_run_pi05_b1kpt50_multinode_skill_eval.sh)

它综合复用了以下已有链路的设计：

- [run_skill_metric_runtime_sweep.py](file:///mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/scripts/run_skill_metric_runtime_sweep.py)
- [run_b1k_skill_segment_eval.sh](file:///mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/scripts/run_b1k_skill_segment_eval.sh)
- [summarize_skill_segment_eval.py](file:///mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/scripts/summarize_skill_segment_eval.py)

## What It Does

### 1. Prepare

- 扫描 `2025-challenge-demos/annotations`
- 只保留 `segment_skill_metric_registry.py` 中已注册的 skill
- 默认按“约 6 小时 / 32 GPU”预算做采样：
  - `max_samples_per_skill = 96`
  - `max_samples_per_skill_task = 2`
  这样优先保证 task 覆盖，同时避免全量运行过慢
- 生成共享 `manifest` 和 `worker job` 分片

### 2. Launch

- 每个节点启动 8 个本地 worker
- 每个 worker 绑定 1 张 GPU 和 1 个本地端口
- worker 内部按 `task_name` 分组，尽量复用同一个 server，减少重复启动成本
- 每个 skill segment 的 `segment_max_steps` 自动设置为 `2 x frame_duration`

### 3. Merge

- 汇总所有 `worker_results`
- 生成：
  - `multinode_skill_results.csv`
  - `multinode_skill_summary.csv`
  - `multinode_skill_task_summary.csv`
  - `multinode_skill_summary.md`

## Default Checkpoint

默认 checkpoint 和 config 已经对齐到本次需求：

- ckpt:
  `/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/checkpoints/openpi_comet/pi05-b1kpt50-cs32`
- config:
  `pi05_b1k_skill-pt50_pretrain_lr1e-4_2ep`

如需覆盖，可用：

- `--ckpt-dir`
- `--config-name`

## Recommended 32-GPU Workflow

集群镜像启动后，正式入口脚本会先 `source`：

- `/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/extra_bashrc.sh`

这样代理、conda hook、缓存路径和其他基础环境会先被配置好，然后再执行 `conda activate openpi-comet-nas`。

### Step 1. Prepare Shared Manifest

在 node 0 上执行一次：

```bash
source /mnt/bn/behavior-data-hl/chenjunting/miniconda3/etc/profile.d/conda.sh
conda activate openpi-comet-nas

cd /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet

RUN_DIR=/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/segment_eval_runs/pi05_b1kpt50_multinode_skill_$(date +%Y%m%d_%H%M%S)

python scripts/run_skill_metric_multinode_sweep.py \
  --mode prepare \
  --out-dir "$RUN_DIR" \
  --num-nodes 4 \
  --gpus-per-node 8 \
  --ckpt-dir /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/checkpoints/openpi_comet/pi05-b1kpt50-cs32 \
  --config-name pi05_b1k_skill-pt50_pretrain_lr1e-4_2ep
```

如果希望最大化代表性并接受更长运行时间，可以手动把以下两个参数设为 `0`：

- `--max-samples-per-skill`
- `--max-samples-per-skill-task`

这会退回到“全量尽量覆盖所有可用 `task / demo`”的模式。

正式评测结果请始终写到 repo 下的持久化目录，例如：

- `/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/segment_eval_runs/...`

不要把正式结果放到 `/tmp`，因为集群任务结束后 `/tmp` 目录会被清空。

### Step 2. Launch On 4 Nodes

在每台机器上都运行同一条命令，只修改 `NODE_RANK`：

默认脚本当前等价于：

```bash
MAX_SAMPLES_PER_SKILL=96
MAX_SAMPLES_PER_SKILL_TASK=2
MAX_TOTAL_JOBS=0
```

这组默认值的设计目标是把总 job 数压到 `~3k` 量级，适合作为 `32 x L20` 的一晚或半晚规模实验。

#### Node 0

```bash
source /mnt/bn/behavior-data-hl/chenjunting/miniconda3/etc/profile.d/conda.sh
conda activate openpi-comet-nas
cd /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet

NODE_RANK=0 NUM_NODES=4 GPUS_PER_NODE=8 LOCAL_GPU_IDS=0,1,2,3,4,5,6,7 \
python scripts/run_skill_metric_multinode_sweep.py \
  --mode launch \
  --out-dir "$RUN_DIR" \
  --num-nodes 4 \
  --gpus-per-node 8 \
  --resume
```

#### Node 1

```bash
NODE_RANK=1 NUM_NODES=4 GPUS_PER_NODE=8 LOCAL_GPU_IDS=0,1,2,3,4,5,6,7 \
python scripts/run_skill_metric_multinode_sweep.py \
  --mode launch \
  --out-dir "$RUN_DIR" \
  --num-nodes 4 \
  --gpus-per-node 8 \
  --resume
```

#### Node 2

```bash
NODE_RANK=2 NUM_NODES=4 GPUS_PER_NODE=8 LOCAL_GPU_IDS=0,1,2,3,4,5,6,7 \
python scripts/run_skill_metric_multinode_sweep.py \
  --mode launch \
  --out-dir "$RUN_DIR" \
  --num-nodes 4 \
  --gpus-per-node 8 \
  --resume
```

#### Node 3

```bash
NODE_RANK=3 NUM_NODES=4 GPUS_PER_NODE=8 LOCAL_GPU_IDS=0,1,2,3,4,5,6,7 \
python scripts/run_skill_metric_multinode_sweep.py \
  --mode launch \
  --out-dir "$RUN_DIR" \
  --num-nodes 4 \
  --gpus-per-node 8 \
  --resume
```

### Step 3. Merge Final Results

所有节点跑完后，在任意一台能访问共享 `RUN_DIR` 的机器上执行：

```bash
source /mnt/bn/behavior-data-hl/chenjunting/miniconda3/etc/profile.d/conda.sh
conda activate openpi-comet-nas
cd /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet

python scripts/run_skill_metric_multinode_sweep.py \
  --mode merge \
  --out-dir "$RUN_DIR"
```

## Key Outputs

### Prepare Stage

- `manifest.json`
- `jobs/worker_*.json`
- `worker_plan.csv`
- `planned_skill_coverage.csv`
- `planned_skill_coverage.md`

### Run Stage

- `worker_results/worker_*.jsonl`
- `worker_status/worker_*.done.json`
- `launcher_console_node*.log`
- `server_logs/server_<task>_gpu<id>_p<port>.log`
- `raw/<task>/demo_<id>/skill_<idx>/metrics/*.json`

如果某个 worker 提前失败，`run_skill_metric_multinode_sweep.py` 现在会把对应 `launcher_logs/node*_worker*.log` 的尾部直接打印回主日志，因此 Merlin / Arnold 平台日志里应能直接看到失败 worker 的最近 traceback，而不是只有裸 `exit 1`。

### Merge Stage

- `multinode_skill_results.csv`
- `multinode_skill_summary.csv`
- `multinode_skill_task_summary.csv`
- `multinode_skill_summary.md`

### Visual Review Stage

当你开启 review 流程后，还会额外生成：

- `review/review_manifest.csv`
- `review/review_manifest.json`
- `review/segments/<task>/demo_<id>/skill_<idx>/final_rgb.png`
- `review/segments/<task>/demo_<id>/skill_<idx>/review_payload.json`
- `review/segments/<task>/demo_<id>/skill_<idx>/source_paths.json`

如果运行时已经开启新的关键帧落盘逻辑，同一个 `skill_<idx>` 目录下还可能直接复用：

- `review/start_restore.png`
- `review/end_restore.png`
- `review/final_rollout.png`

## Useful Flags

- `--skills "move to,open door"`:
  只跑指定 skill 子集
- `--max-samples-per-skill 32`:
  每个 skill 只取前 32 个 segment
- `--max-samples-per-skill-task 4`:
  每个 `(skill, task)` 最多取 4 个 segment，适合平衡覆盖和时长
- `--max-total-jobs 128`:
  全局只取前 128 个 segment，适合 smoke test
- `--dry-run`:
  只验证 restore / predicate 生成链路，不做真实 rollout
- `--segment-predicate-dump-trace`:
  保存逐帧 predicate trace，适合定点排查
- `--write-video`:
  输出视频，适合少量失败样本复盘

## Visual Validation Workflow

目标是验证“segment 最后一帧 RGB observation + metric 输出”是否符合人工常识，并把规则迭代收敛到稳定状态。

### 1. 先复用已有 run dir

优先对已有结果生成 review set，而不是每次都整套重跑。示例：

```bash
source /mnt/bn/behavior-data-hl/chenjunting/miniconda3/etc/profile.d/conda.sh
conda activate openpi-comet-nas
cd /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet

python scripts/build_skill_metric_review_set.py \
  --run-dir /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/segment_eval_runs/skill_trace_eval_20260423_202747 \
  --skills "move to,open door" \
  --samples-per-skill 8 \
  --holdout-per-skill 2
```

该脚本会：

- 复用已有 `metrics/*.json`
- 优先复用运行时保存的 `review/*.png`
- 若没有关键帧 png，则从 `videos/*.mp4` 抽最后一帧生成 `final_rgb.png`
- 生成可人工填写的 `review_manifest.csv`

### 2. 填写人工复核结果

在 `review_manifest.csv` 中人工填写以下列：

- `human_judgement`: `pass` / `fail` / `uncertain`
- `human_reason`: 误判原因说明
- `issue_bucket`: 例如 `threshold_too_strict`、`wrong_target`、`rollout_true_failure`

人工判读时，优先看：

- `final_rgb.png` 或 `review/final_rollout.png`
- `review/end_restore.png`
- `review_payload.json` 里的：
  - `template_trace_end`
  - `rollout_final_trace`
  - geometry diagnostics

### 3. 汇总一致率

```bash
python scripts/summarize_skill_metric_review.py \
  --manifest /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/segment_eval_runs/<RUN_DIR>/review/review_manifest.csv
```

会输出：

- `review_summary.csv`
- `review_by_skill.csv`
- `review_by_issue_bucket.csv`
- `review_detailed.csv`
- `review_report.md`

### 4. 规则迭代与回归

推荐顺序：

1. 先修一类 issue bucket
2. 小规模重跑目标 skill
3. 重新生成 review set
4. 重新汇总 review
5. 先看 discovery 集，再看 holdout 集

验收标准建议固定为：

- 人工一致率 `>= 90%`
- holdout 集不回退
- 连续两轮 review 结论稳定

### 5. 单机 8 卡脚本直接联动 review

`scripts/run_skill_eval_single_node_8gpu.sh` 现在支持在 merge 后直接生成 review set：

```bash
POST_MERGE_BUILD_REVIEW_SET=1 \
REVIEW_TARGET_SKILLS="move to,open door" \
REVIEW_SAMPLES_PER_SKILL=8 \
REVIEW_HOLDOUT_PER_SKILL=2 \
bash scripts/run_skill_eval_single_node_8gpu.sh
```

## Smoke Test

已验证下面这条 prepare 命令能正常产出 manifest、coverage 和 worker 切分：

```bash
source /mnt/bn/behavior-data-hl/chenjunting/miniconda3/etc/profile.d/conda.sh
conda activate openpi-comet-nas
cd /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet

python scripts/run_skill_metric_multinode_sweep.py \
  --mode prepare \
  --out-dir /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/segment_eval_runs/skill_multinode_smoke_20260424 \
  --num-nodes 1 \
  --gpus-per-node 2 \
  --skills "move to,open door,sweep surface" \
  --max-samples-per-skill 2 \
  --max-samples-per-skill-task 2
```

该 smoke test 生成了：

- `6` 个 segment job
- `3` 个 unique skills
- `3` 个 unique tasks
- `5` 个 unique demos

并且 `merge` 模式也已验证通过，可以在结果尚未齐全时输出缺失 job 列表。

如果只是临时本地调试，确实需要把输出写到 `/tmp`，可以显式添加：

```bash
--allow-tmp-out-dir
```

默认情况下脚本会拒绝把结果写到 `/tmp`，以避免正式评测结果丢失。
