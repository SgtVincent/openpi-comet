# VLM2 训练与离线验证说明

本文件说明如何在本仓库中启动 VLM2 的 PyTorch 训练，并在不启动 server / simulator 的情况下，直接在离线数据上做 validation。

## 1. 环境与依赖

### 1.1 安装 OpenPi Comet

请务必每次开始工作前先激活环境，并设置缓存路径：

```bash
cd /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet
source .venv/bin/activate
# 设置本地缓存路径，避免无外网权限导致 tokenizer 下载失败
export OPENPI_DATA_HOME=$(pwd)/.cache/openpi
```

首次安装：
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

WandB 监控：
- 确保当前环境已完成 wandb 登录（例如提前配置好 `WANDB_API_KEY` 环境变量，或在可交互终端执行 `wandb login`）
- 如需自定义项目名，可通过 CLI 覆盖：`--project_name B1K`

### 1.2 安装 BEHAVIOR-1K 依赖（用于数据集读取）

离线 validation 仍会读取 BEHAVIOR-1K 的数据集类（BehaviorLeRobotDataset），因此需要把 BEHAVIOR-1K 作为可编辑依赖安装到同一个 Python 环境中：

```bash
# 需在激活的 .venv 下执行
cd /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/bddl3
uv pip install -e .
cd /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson
uv pip install -e ".[eval]"
```

## 2. 数据准备与配置

### 2.1 数据路径

VLM2 的训练配置在 [src/openpi/training/config.py](src/openpi/training/config.py) 中。
该配置里有 `behavior_dataset_root` 字段，指向 B1K 数据根目录。请根据本机路径修改：

```python
behavior_dataset_root="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos"
```

### 2.2 数据集说明（2025-challenge-demos）

数据集位置：`/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos/`

目录结构：
- `annotations`：语言标注
- `data`：低维状态、动作、任务信息等
- `meta`：episode 级元信息
- `videos`：视觉观测（rgb、depth、seg_instance_id）

数据集统计（官方文档）：
- 10,000 条演示轨迹
- 50 个任务
- 视觉模态：RGB、Depth Linear、Instance Segmentation ID

注意事项：
- 关节力矩（joint efforts）字段在当前版本不可靠，请勿用于训练

### 2.3 归一化统计（推荐）

已复用 `checkpoints/openpi_comet/pi05-b1kpt50-cs32/assets` 下的 norm stats，用于 VLM2 预训练与 SFT。

如需重新计算，可使用训练配置名生成：

```bash
source .venv/bin/activate
uv run scripts/compute_norm_stats.py --config-name vlm2_b1k-pt50_cs32_bs64_lr2.5e-5_step50k
```

## 3. 启动 VLM2 预训练（VLA）

### 3.1 预训练配置

已在 [src/openpi/training/config.py](src/openpi/training/config.py) 中新增：
- `vlm2_b1k-pt50_cs32_bs64_lr2.5e-5_step50k`

该配置使用：
- 数据集：`behavior-1k/2025-challenge-demos`
- 数据路径：`/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos/`
- 基础权重：`checkpoints/openpi_comet/pi05-b1kpt50-cs32`
- **输出目录**：`checkpoints/<exp_name>` (已在 config 中默认指定)

### 3.2 多卡预训练（推荐）

多卡训练时，每个 GPU 分配 10 个 worker（总计 80 workers），以充分利用 IO 带宽。

```bash
source .venv/bin/activate
export OPENPI_DATA_HOME=$(pwd)/.cache/openpi
export B1K_VIDEO_BACKEND=video_reader
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_pytorch.py \
  vlm2_b1k-pt50_cs32_bs64_lr2.5e-5_step50k \
  --exp_name vlm2_vla_pretrain \
  --num_train_steps 50000 \
  --log_interval 100 \
  --save_interval 5000 \
  --num_workers 10 \
  --pytorch-training-precision bfloat16 \
  --overwrite
```

如果出现 OOM 或不稳定，可适当回调 `num_workers`（如降至 8）。

## 4. 启动 VLM2 微调训练（SFT）

### 4.1 单卡训练

```bash
source .venv/bin/activate

python scripts/train_pytorch.py \
  vlm2_b1k-turning_on_radio_lr2.5e-6_step20k_sft \
  --exp_name vlm2_sft_single \
  --num_train_steps 1000 \
  --log_interval 50 \
  --save_interval 200 \
  --batch_size 8 \
  --num_workers 16 \
  --pytorch-training-precision bfloat16
```

### 4.2 多卡训练（推荐）

```bash
source .venv/bin/activate
export OPENPI_DATA_HOME=$(pwd)/.cache/openpi
export B1K_VIDEO_BACKEND=video_reader
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_pytorch.py \
  vlm2_b1k-turning_on_radio_lr2.5e-6_step20k_sft \
  --exp_name vlm2_sft_run \
  --num_train_steps 20000 \
  --log_interval 100 \
  --save_interval 500 \
  --batch_size 64 \
  --num_workers 10 \
  --pytorch-training-precision bfloat16
```

### 4.3 训练产物位置

已修改配置默认输出到 `checkpoints` 目录，产物结构如下：

```
./checkpoints/<exp_name>/<step>/
  ├── model.safetensors
  ├── optimizer.pt
  ├── metadata.pt
  └── assets/
```

## 5. 离线 validation（不启动 server / simulator）

离线 validation 的目标是：在数据集样本上直接跑模型前向，计算预测动作与 GT 动作的差异，避免启动 websocket server 或 OmniGibson。

### 5.1 使用现有脚本（推荐）

仓库里已有脚本 [scripts/test_b1k_openpi.py](scripts/test_b1k_openpi.py) 可以直接做离线验证，只需要改动以下几处：

1) 指定 VLM2 配置与 checkpoint 路径

```python
from openpi.training import config
from openpi.policies import policy_config

checkpoint_dir = "checkpoints/your_exp_name/500"  # 替换为实际目录
policy = policy_config.create_trained_policy(
    config.get_config("vlm2_b1k-turning_on_radio_lr2.5e-6_step20k_sft"),
    checkpoint_dir,
)
```

2) 指定数据集路径与任务名

```python
from omnigibson.learning.datas import BehaviorLeRobotDataset

ds = BehaviorLeRobotDataset(
    repo_id="behavior-1k/2025-challenge-demos",
    root="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos",
    tasks=["turning_on_radio"],
    modalities=["rgb"],
    local_only=True,
    shuffle=False,
)
```

3) 运行脚本

```bash
source .venv/bin/activate
python scripts/test_b1k_openpi.py
```

脚本会逐帧打印预测动作和 GT 动作的 shape，并保存动作对比图到 `actions.png`。

### 5.2 自定义离线验证（示例）

如果你需要统计数值指标，可以用下面的最小示例，计算平均 L2 误差（建议保存为 `scripts/eval_l2.py`）：

```python
import numpy as np
from tqdm import tqdm
from omnigibson.learning.datas import BehaviorLeRobotDataset
from openpi.training import config
from openpi.policies import policy_config
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper

ckpt_dir = "checkpoints/your_exp_name/500"
policy = policy_config.create_trained_policy(
    config.get_config("vlm2_b1k-turning_on_radio_lr2.5e-6_step20k_sft"),
    ckpt_dir,
)
openpi_policy = B1KPolicyWrapper(policy, control_mode="receeding_horizon")

# dataset
root = "/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos"
ds = BehaviorLeRobotDataset(
    repo_id="behavior-1k/2025-challenge-demos",
    root=root,
    tasks=["turning_on_radio"],
    modalities=["rgb"],
    local_only=True,
    shuffle=False,
)

errors = []
for idx in tqdm(range(min(len(ds), 100))):
    data = ds[idx]
    example = {
        "robot_r1::robot_r1:zed_link:Camera:0::rgb": data["observation.images.rgb.head"].permute(1, 2, 0).numpy(),
        "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": data["observation.images.rgb.left_wrist"].permute(1, 2, 0).numpy(),
        "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": data["observation.images.rgb.right_wrist"].permute(1, 2, 0).numpy(),
        "robot_r1::proprio": data["observation.state"],
    }
    pred_action = openpi_policy.act(example)
    gt_action = data["action"]
    errors.append(np.linalg.norm(pred_action - gt_action))

print("mean L2 error:", float(np.mean(errors)))
```

## 6. 常见问题

- 如果训练报 `Normalization stats not found`，请先运行 `scripts/compute_norm_stats.py`。
- 如果多卡训练报显存 OOM，可减少 `--batch_size` 或调低 `--num_workers`。
- 如果出现 `SIGSEGV (exitcode: -11)`，优先按下面顺序排查：
  1) `--num_workers 0`（先把多进程数据加载去掉）
  2) 设置 `B1K_VIDEO_BACKEND=video_reader`（只训练 rgb 时推荐）
  3) 确认 `CUDA_VISIBLE_DEVICES` 与 `--nproc_per_node` 匹配，避免 `invalid device ordinal`
  4) 打开 `PYTHONFAULTHANDLER=1` 与 `TORCH_SHOW_CPP_STACKTRACES=1` 获取更多崩溃信息
- 需要从断点恢复训练：在同一个 `--exp_name` 下追加 `--resume`；如需覆盖旧目录，用 `--overwrite`（不要和 `--resume` 同时用）。
- 离线 validation 只依赖数据集与模型，不启动 server 和 simulator。
