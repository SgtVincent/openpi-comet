# VLM2 训练与离线验证说明

本文件说明如何在本仓库中启动 VLM2 的 PyTorch 训练，并在不启动 server / simulator 的情况下，直接在离线数据上做 validation。

## 1. 环境与依赖

### 1.1 安装 OpenPi Comet

```bash
cd /home/ubuntu/repo/openpi-comet
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
source .venv/bin/activate
```

### 1.2 安装 BEHAVIOR-1K 依赖（用于数据集读取）

离线 validation 仍会读取 BEHAVIOR-1K 的数据集类（BehaviorLeRobotDataset），因此需要把 BEHAVIOR-1K 作为可编辑依赖安装到同一个 Python 环境中：

```bash
cd /home/ubuntu/repo/BEHAVIOR-1K
uv pip install -e bddl
uv pip install -e OmniGibson[eval]
```

## 2. 数据准备与配置

### 2.1 数据路径

VLM2 的训练配置在 [src/openpi/training/config.py](src/openpi/training/config.py) 中，示例配置为：

- `vlm2_b1k-turning_on_radio_lr2.5e-6_step20k_sft`

该配置里有 `behavior_dataset_root` 字段，指向 B1K 数据根目录。请根据本机路径修改：

```python
behavior_dataset_root="/path/to/2025-challenge-demos"
```

### 2.2 计算归一化统计（推荐）

使用训练配置名生成 norm stats：

```bash
uv run scripts/compute_norm_stats.py --config-name vlm2_b1k-turning_on_radio_lr2.5e-6_step20k_sft
```

## 3. 启动 VLM2 训练

### 3.1 单卡训练

```bash
python scripts/train_pytorch.py \
  vlm2_b1k-turning_on_radio_lr2.5e-6_step20k_sft \
  --exp_name vlm2_sft_single \
  --num_train_steps 1000 \
  --log_interval 50 \
  --save_interval 200 \
  --no-wandb-enabled \
  --batch_size 8 \
  --num_workers 2 \
  --pytorch-training-precision bfloat16
```

### 3.2 多卡训练（推荐）

```bash
/home/ubuntu/repo/openpi-comet/.venv/bin/torchrun \
  --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_pytorch.py \
  vlm2_b1k-turning_on_radio_lr2.5e-6_step20k_sft \
  --exp_name vlm2_sft_run \
  --num_train_steps 20000 \
  --log_interval 100 \
  --save_interval 500 \
  --no-wandb-enabled \
  --batch_size 8 \
  --num_workers 2 \
  --pytorch-training-precision bfloat16
```

### 3.3 训练产物位置

`TrainConfig.checkpoint_base_dir` 默认为 `.`，因此训练产物在：

```
./<exp_name>/<step>/
  ├── model.safetensors
  ├── optimizer.pt
  ├── metadata.pt
  └── assets/
```

## 4. 离线 validation（不启动 server / simulator）

离线 validation 的目标是：在数据集样本上直接跑模型前向，计算预测动作与 GT 动作的差异，避免启动 websocket server 或 OmniGibson。

### 4.1 使用现有脚本（推荐）

仓库里已有脚本 [scripts/test_b1k_openpi.py](scripts/test_b1k_openpi.py) 可以直接做离线验证，只需要改动以下几处：

1) 指定 VLM2 配置与 checkpoint 路径

```python
from openpi.training import config
from openpi.policies import policy_config

checkpoint_dir = "/path/to/your/exp_name/500"  # 替换为实际目录
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
    root="/path/to/2025-challenge-demos",
    tasks=["turning_on_radio"],
    modalities=["rgb"],
    local_only=True,
    shuffle=False,
)
```

3) 运行脚本

```bash
python scripts/test_b1k_openpi.py
```

脚本会逐帧打印预测动作和 GT 动作的 shape，并保存动作对比图到 `actions.png`。

### 4.2 自定义离线验证（示例）

如果你需要统计数值指标，可以用下面的最小示例，计算平均 L2 误差：

```bash
python - <<'PY'
import numpy as np
from tqdm import tqdm
from omnigibson.learning.datas import BehaviorLeRobotDataset
from openpi.training import config
from openpi.policies import policy_config
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper

ckpt_dir = "/path/to/your/exp_name/500"
policy = policy_config.create_trained_policy(
    config.get_config("vlm2_b1k-turning_on_radio_lr2.5e-6_step20k_sft"),
    ckpt_dir,
)
openpi_policy = B1KPolicyWrapper(policy, control_mode="receeding_horizon")

# dataset
root = "/path/to/2025-challenge-demos"
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
PY
```

## 5. 常见问题

- 如果训练报 `Normalization stats not found`，请先运行 `scripts/compute_norm_stats.py`。
- 如果多卡训练报显存 OOM，可减少 `--batch_size` 或调低 `--num_workers`。
- 离线 validation 只依赖数据集与模型，不启动 server 和 simulator。
