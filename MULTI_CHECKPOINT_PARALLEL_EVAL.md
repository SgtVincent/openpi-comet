# 一键启动多个训练完成 Checkpoint 并行 Eval（Bash）

本文说明如何使用现有脚本，一键对任意数量的已训练 checkpoint 执行并行评估。

适用仓库：openpi-comet

## 1. 脚本关系

评估入口现在统一为 1 个脚本：

- `scripts/run_b1k_eval_parallel_single_task.sh`
  - 单脚本同时支持：
    - 单 checkpoint 评估
    - 多 checkpoint 列表/多参数并行评估（动态分卡）
    - dry-run 计划预览（不真正启动进程）
- `scripts/monitor_eval_progress.sh`
  - 监控某个 eval run 的日志、metrics 数量、失败信号。

## 2. 前置条件

请先满足：

- openpi-comet 仓库可用，且 `.venv` 存在。
- BEHAVIOR-1K 仓库路径存在：`/home/ubuntu/repo/BEHAVIOR-1K`。
- conda 环境 `behavior` 可用（用于 evaluator）。
- checkpoint 路径真实存在。

建议先检查：
    
```bash
cd /home/ubuntu/repo/openpi-comet
ls .venv/bin/python
conda env list | grep behavior
```

## 3. 一键运行（输入 checkpoint 列表）

直接执行：

```bash
cd /home/ubuntu/repo/openpi-comet
bash scripts/run_b1k_eval_parallel_single_task.sh /path/to/checkpoints.txt
```

其中 `checkpoints.txt` 格式示例（每行一个目录）：

```text
# lines starting with # are ignored
/path/to/ckpt_a
/path/to/ckpt_b
/path/to/ckpt_c
```

也可以直接传多个目录：

```bash
cd /home/ubuntu/repo/openpi-comet
bash scripts/run_b1k_eval_parallel_single_task.sh /path/to/ckpt_a /path/to/ckpt_b /path/to/ckpt_c
```

默认行为：

- `TASK_NAME=turning_on_radio`
- 自动检测可用 GPU（或使用你设置的 `GPU_IDS`）
- 每个 worker GPU 启动 1 个 server + 1 个 evaluator（端口从 `PORT_BASE` 递增）
- `EVAL_INSTANCE_IDS=0,1,2,3,4,5,6,7,8,9`（每模型 10 episodes）
- 每批并行数 = `min(剩余模型数, GPU 总数)`
- 每批内按比例切分 GPU 给每个模型（尽量均分，余数给前面的模型）

## 4. 常用参数覆盖（不改脚本）

可通过环境变量覆盖默认值：

```bash
cd /home/ubuntu/repo/openpi-comet

TASK_NAME=turning_on_radio \
GPU_IDS=0,1,2,3,4,5,6,7 \
EVAL_INSTANCE_IDS=0,1,2,3,4,5,6,7,8,9 \
MAX_STEPS=20 \
PORT_BASE_START=9700 \
PORT_STRIDE=20 \
bash scripts/run_b1k_eval_parallel_single_task.sh /path/to/checkpoints.txt
```

dry-run 示例（只打印调度与命令，不执行）：

```bash
cd /home/ubuntu/repo/openpi-comet
GPU_IDS=0,1,2,3 \
bash scripts/run_b1k_eval_parallel_single_task.sh --dry-run /path/to/checkpoints.txt
```

说明：

- `MAX_STEPS` 不设置时，episode 可能非常长。
- `PORT_BASE_START` + `PORT_STRIDE` 用于给每个模型分配不同端口段，避免冲突。
- `EVAL_INSTANCE_IDS` 控制评估 episodes 数量。
- `GPU_IDS` 控制参与调度的物理卡。
- 若设置 `NUM_GPUS`，只会使用 `GPU_IDS`（或自动检测列表）中的前 `NUM_GPUS` 张卡。

## 5. checkpoint 输入要求

支持两种输入：

- 列表文件（推荐）
- 命令行多个目录

注意：

- 每个目录必须存在。
- 每个目录可以是训练根目录（单 checkpoint 脚本会自动解析最新数字步）或具体步目录。

## 6. 监控运行进度

### 6.1 自动监控最新 run

```bash
cd /home/ubuntu/repo/openpi-comet
bash scripts/monitor_eval_progress.sh
```

### 6.2 监控指定 run 目录

```bash
cd /home/ubuntu/repo/openpi-comet
bash scripts/monitor_eval_progress.sh /home/ubuntu/repo/openpi-comet/eval_logs/parallel_turning_on_radio_pi05-b1kpt50-cs32_20260320_030223
```

### 6.3 只看一次快照

```bash
cd /home/ubuntu/repo/openpi-comet
ONCE=1 bash scripts/monitor_eval_progress.sh /home/ubuntu/repo/openpi-comet/eval_logs/parallel_turning_on_radio_pi05-b1kpt50-cs32_20260320_030223
```

## 7. 结果目录结构

每个 checkpoint 会生成一个 run 目录，例如：

- `eval_logs/parallel_turning_on_radio_pi05-b1kpt50-cs32_20260320_030223`

目录内常见文件：

- `server_gpu*_p*.log`
- `eval_gpu*_p*.log`
- `eval_gpu*_p*/metrics/*.json`
- `eval_gpu*_p*/videos/*.mp4`（若开启视频）

## 8. 快速汇总指标

可用现有汇总脚本查看 `q_score.final` 汇总：

```bash
cd /home/ubuntu/repo/openpi-comet
python summarize_eval_metrics.py eval_logs/parallel_turning_on_radio_pi05-b1kpt50-cs32_20260320_030223
```

## 9. 常见问题与排查

### 9.1 提示 OPENPI_PYTHON 不可执行

解决：

```bash
cd /home/ubuntu/repo/openpi-comet
ls -l .venv/bin/python
```

如路径不同，运行时传：

```bash
OPENPI_PYTHON=/abs/path/to/python bash scripts/run_b1k_eval_parallel_single_task.sh /path/to/ckpt_or_checkpoints.txt
```

### 9.2 提示 conda 环境 behavior 不存在

解决：

```bash
conda env list | grep behavior
```

若环境名不同，运行时传：

```bash
BEHAVIOR_ENV=your_env_name bash scripts/run_b1k_eval_parallel_single_task.sh /path/to/ckpt_or_checkpoints.txt
```

### 9.3 只跑了 3 个 trial，不是 10 个 episode

检查 `EVAL_INSTANCE_IDS` 是否是 10 个 ID：

```bash
EVAL_INSTANCE_IDS=0,1,2,3,4,5,6,7,8,9 bash scripts/run_b1k_eval_parallel_single_task.sh /path/to/ckpt_a
```

如果使用列表文件，命令应为：

```bash
EVAL_INSTANCE_IDS=0,1,2,3,4,5,6,7,8,9 bash scripts/run_b1k_eval_parallel_single_task.sh /path/to/checkpoints.txt
```

### 9.4 运行太慢

可临时限制步数：

```bash
MAX_STEPS=20 bash scripts/run_b1k_eval_parallel_single_task.sh /path/to/ckpt_a
```

## 10. 推荐执行顺序

1. 准备 checkpoint 列表文件（每行一个目录）。
2. 用 `MAX_STEPS=20` 跑一轮 smoke，确认流程通。
3. 去掉 `MAX_STEPS` 或调大后跑正式评估。
4. 用 `monitor_eval_progress.sh` 持续观察状态。
5. 用 `summarize_eval_metrics.py` 汇总结果。
