# DeepSpeed 训练指南（V100 + ZeRO-2 + FP16）

本指南面向“想在 V100 上稳定训练 PI0.5/VLA”的用户，提供一套可复现的 Accelerate + DeepSpeed ZeRO-2 FP16 训练方案、可观测指标、以及常见问题排查路径。

## 1. 适用范围

推荐训练入口：
- `scripts/train_accelerate.py`：支持 bf16/fp16 + DeepSpeed（V100 必选）

相关配置：
- `configs/accelerate_ds_zero2_v100_fp16.yaml`
- `configs/deepspeed_zero2_v100_fp16.json`

## 2. 快速开始（4 nodes x 8 GPUs）

建议直接使用多机启动脚本（会处理 conda 环境、数据根目录、HF cache 等）：

```bash
conda activate openpi-comet-nas

DEBUG_OVERFLOW=1 \
BATCH_SIZE_PER_GPU=8 \
GRADIENT_ACCUMULATION_STEPS=4 \
NUM_WORKERS=10 \
bash scripts/run_pi05_sft_accelerate_deepspeed_multinode_v100_fp16.sh
```

说明：
- 有效全局 batch：`BATCH_SIZE_PER_GPU x WORLD_SIZE x GRADIENT_ACCUMULATION_STEPS`。
- NAS 环境建议把 `NUM_WORKERS` 拉高（多机每卡 10+），避免 IO 抖动导致单 rank 卡住拖垮 NCCL。

## 3. “训练是否正常”的判断标准

训练正常时应满足：
- `loss` 有持续下降趋势（无需与 A100 bf16 逐点对齐，只看趋势/量级）。
- `loss_scale` 稳定（动态 loss scaling 上升或保持都正常；持续下降到很小并伴随大量 skip 才危险）。
- `overflow_debug`（当 `DEBUG_OVERFLOW=1`）长期满足：
  - `finite_ratio_pre=1.0 finite_ratio_post=1.0`
  - `nonfinite_pre=0 nonfinite_post=0`

示例（已验证稳定的 V100 run，Make Pizza SFT）：
- log：`checkpoints/console_logs/pi05_b1k-make_pizza_lr1e-4_5ep_sft_accel_ds_z2_v100fp16_4n8g_20260422_170114/node0.log`
- W&B summary：`loss≈0.00845`，`loss_scale=131072`，`grad_norm≈0.014`

## 4. V100 FP16 稳定性关键点（必须）

以下是 “V100 FP16 能稳定训练” 的必要条件（面向用户的行为准则）：

1. 不要叠两套混合精度
   - 使用 DeepSpeed `fp16.enabled=true` 时，必须避免同时启用另一套 autocast 管线。
2. 对数值敏感的 loss 计算尽量用 fp32
   - 特别是 Flow Matching 的 MSE，在 V100 fp16 下更容易出现中间值溢出。
3. 保持可观测性
   - 推荐 `DEBUG_OVERFLOW=1`，至少在新配置/新 batch 规模下跑一段确认 `finite_ratio` 一直为 1。

注：上述“为什么需要这些处理、历史上遇到过哪些坑、对应代码改动点”，统一放在 `deepspeed_bug_fix.md`（不作为用户指南的一部分）。

## 5. 常见问题（Top）

### 5.1 DataLoader timeout / 某个 rank 卡住

现象：
- 只有某个 rank 报 `DataLoader timed out after 600 seconds`
- 其他 rank 报 `ProcessGroupNCCL watchdog timeout`

优先处理：
- 增大 `NUM_WORKERS`（NAS 上很关键）
- 排查是否某个视频/样本解码会卡死（日志里需要打印 episode/task/chunk 信息才能精确定位）

### 5.2 多机 checkpoint 保存等待 tmp 目录超时

现象：
- `Timed out waiting for tmp_ckpt_dir: .../tmp_1000`

常见原因：
- 多机不同节点写到了不同的 `EXP_NAME` 目录（目录分裂），导致互相等待永远不会出现的路径。

建议：
- 使用本仓库的多机启动脚本（已对齐 `EXP_NAME` 同步），不要手写启动命令。

### 5.3 W&B 网络超时

现象：
- `wandb: WARNING ... Read timed out`

说明：
- 这通常不影响训练正确性，只影响上传；可通过 `WANDB_DISABLED=1` 或离线模式规避。
