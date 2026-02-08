## 目标
- 重新启动并稳定运行 VLM2-VLA pretrain：`vlm2_b1k-pt50_cs32_bs64_lr2.5e-5_step50k`。
- 训练能持续输出 `step/loss` 日志并在 wandb 可见曲线，连续运行 >10min 无崩溃。

## 关键前提（你刚确认已满足）
- `av/pyav` 已能正常安装与使用（避免此前视频解码引发的 SIGSEGV/构建失败）。

## 执行步骤（会实际运行与监控）
1) 预检：确认训练环境可用
- 在 `openpi-comet/.venv` 中依次验证：
  - `import av` 能打印版本
  - `import omnigibson` 不触发异常
  - `from behavior.learning.datas.dataset import BehaviorLeRobotDataset` 能成功导入
  - 从 `/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos/` 读取一个样本（只取 rgb）不报错

2) 单卡 smoke test（先跑通端到端）
- 使用更稳的设置：`--num_workers 0` + `B1K_VIDEO_BACKEND=pyav`（既然 pyav 已可用，优先用它）
- 开启崩溃可观测性：`PYTHONFAULTHANDLER=1`、`TORCH_SHOW_CPP_STACKTRACES=1`
- 启动：
  - `python scripts/train_pytorch.py vlm2_b1k-pt50_cs32_bs64_lr2.5e-5_step50k --exp_name vlm2_vla_pretrain_smoke --num_train_steps 200 --log_interval 5 --save_interval 100000 --batch_size 1 --num_workers 0`
- 通过标准：持续出现 `step=... loss=...`，wandb run 建立且曲线更新。

3) 2 卡 DDP 验证（排除多卡/多进程交互问题）
- `torchrun --standalone --nnodes=1 --nproc_per_node=2 ... --num_workers 0` 起步
- 稳定后把 `--num_workers` 逐步加到 2/4（每次至少观察几百 step 的日志）。

4) 8 卡正式 pretrain 启动并监控 >10min
- 先用 `--num_workers 0` 启动 8 卡，确认稳定后再逐步加 `--num_workers`。
- 命令（核心保持和指南一致）：
  - `torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_pytorch.py vlm2_b1k-pt50_cs32_bs64_lr2.5e-5_step50k --exp_name vlm2_vla_pretrain --num_train_steps 50000 --log_interval 100 --save_interval 5000`
- 持续轮询训练日志，直到确认 >10min 无崩溃且 loss 持续更新。

## 异常分支（如果仍出现 SIGSEGV）
- 立刻收集最后 200 行日志（含 rank/worker 信息），并自动降级复现：
  - 保持 `--num_workers 0`
  - 切换 `B1K_VIDEO_BACKEND=video_reader`（只训练 rgb 时）与 `pyav` 对照
  - 进一步缩小 batch/steps 定位触发点

## 文档同步（TRAIN_GUIDE_MERLIN.md）
- 将“backend 推荐”更新为：pyav 可用时优先 pyav；若遇到多进程不稳定再切 `video_reader`。
- 补上“pyav 可用时的推荐启动命令”与排障顺序，确保你复制粘贴即可复现。

确认后我会按以上步骤实际启动训练、持续监控，并把最终稳定运行的命令与 wandb run 信息汇总给你。