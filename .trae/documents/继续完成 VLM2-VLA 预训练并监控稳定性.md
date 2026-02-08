## 目标
- 重新启动并稳定跑通 `vlm2_b1k-pt50_cs32_bs64_lr2.5e-5_step50k`（VLM2-VLA pretrain）。
- 训练需要能持续输出 `step/loss` 日志并在 wandb 上可见曲线，且连续运行 >10min 无崩溃。

## 执行前快速自检（执行阶段会跑）
- 在 openpi-comet venv 中确认 `av` 可 import（你已说明 uv 环境已更新，此处只是再验证一次）。
- 确认 BEHAVIOR-1K 可导入 `BehaviorLeRobotDataset`。

## 启动策略（从稳到快）
1) 单卡 smoke test（先证明数据解码 + 前向 + loss + wandb 全链路正常）
- 环境变量：
  - `PYTHONFAULTHANDLER=1`、`TORCH_SHOW_CPP_STACKTRACES=1`
  - `B1K_VIDEO_BACKEND=pyav`（你现在 pyav 可用，先用官方推荐；如仍不稳再切 `video_reader`）
- 命令：
  - `python scripts/train_pytorch.py vlm2_b1k-pt50_cs32_bs64_lr2.5e-5_step50k --exp_name vlm2_vla_pretrain_smoke --num_train_steps 200 --log_interval 5 --save_interval 100000 --batch_size 1 --num_workers 0`
- 通过标准：持续打印 `step=... loss=...`，wandb run 正常建立。

2) 2 卡 DDP（排除多进程/多卡交互问题）
- 命令：
  - `torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py vlm2_b1k-pt50_cs32_bs64_lr2.5e-5_step50k --exp_name vlm2_vla_pretrain_2gpu --num_train_steps 2000 --log_interval 10 --save_interval 100000 --batch_size 2 --num_workers 0`
- 稳定后把 `--num_workers` 逐步加到 2/4。

3) 8 卡正式 pretrain（目标运行 >10min）
- 命令（基于 TRAIN_GUIDE_MERLIN.md 的多卡示例）：
  - `torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_pytorch.py vlm2_b1k-pt50_cs32_bs64_lr2.5e-5_step50k --exp_name vlm2_vla_pretrain --num_train_steps 50000 --log_interval 100 --save_interval 5000`
- 初次 8 卡建议仍用 `--num_workers 0` 起步；确认稳定后再加回 `--num_workers 2/4/8`。

## 监控与判定（执行阶段会做）
- 在对话中持续轮询训练输出，直到满足以下任一条件：
  - 连续 >10min 有规律的 `step/loss` 输出且无进程退出；或
  - 发生崩溃（例如 SIGSEGV），收集最后 200 行日志与环境信息，立即切换到“降级复现”命令定位根因。
- wandb：确认 run name 与 exp_name 对应，loss/learning_rate 曲线在更新。

## 崩溃时的降级复现顺序（执行阶段会自动切换）
- `--num_workers 0`（最关键）
- `B1K_VIDEO_BACKEND=video_reader`（只训练 rgb 时更稳）
- 缩小 batch / 缩短 num_train_steps / 提高 log_interval 以便更快看到异常位置

确认后我会按上述顺序实际启动训练、持续监控并把稳定运行的命令与观察到的 wandb run 信息回填到执行结果中。