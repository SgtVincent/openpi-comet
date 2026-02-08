## 目标
- 让 `vlm2_b1k-pt50_cs32_bs64_lr2.5e-5_step50k` 能稳定启动并持续跑起来（至少出现持续的 step/loss 日志与 wandb 曲线）。
- 把可复现/可排障/可恢复的流程补全到 [TRAIN_GUIDE_MERLIN.md](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/TRAIN_GUIDE_MERLIN.md)。

## 现状检查（已确认）
- 训练入口是 [train_pytorch.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/train_pytorch.py)。
- VLM2 预训练配置已存在于 [config.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/training/config.py#L760-L798)：`vlm2_b1k-pt50_cs32_bs64_lr2.5e-5_step50k`。
- 数据加载最终走 `BehaviorLeRobotDataset`（BEHAVIOR-1K）的视频解码路径，默认 backend 倾向 `pyav`，且底层依赖 `av` / `torchvision.io.VideoReader`（见 [lerobot_utils.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/utils/lerobot_utils.py#L55-L152)）。这类解码在多进程/多卡场景下是 SIGSEGV 的高概率来源。

## 计划改动（代码）
1) 让训练可选择更稳的视频解码 backend（避免 PyAV 多进程崩溃）
- 在 [data_loader.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/training/data_loader.py#L128-L152) 的 `create_behavior_dataset()` 初始化 `BehaviorLeRobotDataset(...)` 时新增一个可配置参数：
  - 优先方案：从环境变量读取（例如 `B1K_VIDEO_BACKEND=video_reader|pyav`），并传给 `BehaviorLeRobotDataset(video_backend=...)`。
  - 保持默认行为不变（不设置 env 时沿用当前逻辑），但在文档里推荐 VLM2 pretrain 用 `video_reader` 作为稳定模式。

2) 增加“最小稳定启动”训练开关（仅用于排障，不改变默认）
- 在 [train_pytorch.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/train_pytorch.py) 支持通过环境变量开启更强的崩溃可观测性：
  - `PYTHONFAULTHANDLER=1` / `TORCH_SHOW_CPP_STACKTRACES=1`（仅文档建议即可；如需要也可在脚本入口自动启用）。
  - 在文档里明确：SIGSEGV 优先把 `--num_workers 0` 跑通，再逐步加回。

## 计划改动（文档：TRAIN_GUIDE_MERLIN.md）
1) 修正路径与可复制性
- 把 BEHAVIOR-1K 安装路径从 `/home/ubuntu/repo/BEHAVIOR-1K` 改成当前仓库真实路径 `/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K`。

2) 补全“Pretrain 稳定启动流程”
- 增加一个从小到大的启动阶梯（每一步都能看到 wandb 曲线/step loss）：
  - 单卡 smoke test：`--num_train_steps 50 --log_interval 1 --num_workers 0`。
  - 单机多卡：先 `--nproc_per_node=2`，再扩到 8。
  - 每一步都给出需要看的关键日志标志（例如出现 `step=... loss=...`）。

3) 增加 SIGSEGV 排障章节（针对视频解码/多进程）
- 给出三类“强约束”排障开关（可直接复制）：
  - `--num_workers 0`（最关键）。
  - `B1K_VIDEO_BACKEND=video_reader`（若 torchvision 支持）/ 回退 `pyav`。
  - 关闭/降低额外负载（例如减少 batch、减少 frames、降低 log/save 频率）。

4) 增加 WandB 监控说明
- 明确 wandb 已默认启用（`wandb_enabled=True`），给出：
  - 如何设置 `WANDB_PROJECT` / `WANDB_ENTITY`（不写入任何 key）。
  - 如何从断点恢复：`--resume`；如何避免覆盖：`--overwrite` 的使用注意。

## 验证方式（执行阶段会做）
- 先跑单卡 smoke test，确保能持续打印 step/loss 且 wandb 正常出曲线。
- 再用 torchrun 2 卡与 8 卡逐步验证；持续观察 >10min，确认无 SIGSEGV 并且 loss 在更新。

确认后我将按以上计划进行代码与文档更新，并在本地跑通 smoke test + 多卡启动验证。