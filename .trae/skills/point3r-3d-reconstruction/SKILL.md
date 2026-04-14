---
name: "point3r-3d-reconstruction"
description: "Point3R端到端3D重建 from B1K videos + viser可视化（支持流式增量重建）。Invoke when user asks to reconstruct 3D using Point3R or compare with VGGT."
---

# Point3R 3D Reconstruction Skill

## 环境要求

需要激活 `point3r` conda 环境（独立于 openpi-comet-nas）。

**环境搭建（一次性）**：
```bash
conda create -n point3r python=3.11 cmake=3.14.0 -y
conda activate point3r
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install 'llvm-openmp<16' -y
pip install -r src/openpi/third_party/Point3R/requirements.txt
pip install opencv-python-headless scipy einops timm ftfy accelerate evo gradio gsplat h5py hydra-core lpips matplotlib scikit-learn tensorboard tqdm roma trimesh viser pyglet gdown
```

**Point3R 权重下载**（3.1GB，需要代理）：
```bash
proxy_on
mkdir -p src/openpi/third_party/Point3R/src/checkpoints
python -c "
from googledrivedownloader import download_file_from_google_drive
download_file_from_google_drive('1S0Tcx_F2UKtpwbaZ2sQdxWL_YZ9wPIc4',
    'src/openpi/third_party/Point3R/src/checkpoints/point3r_512.pth', unzip=False)
"
```

## 快速开始

### Step 1: Point3R 推理

```bash
conda activate point3r
python scripts/point3r_reconstruct_b1k.py \
    --data_root /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos \
    --episode_id 10 \
    --camera head \
    --num_frames 16 \
    --stride 1 \
    --sampling_mode stride \
    --checkpoint src/openpi/third_party/Point3R/src/checkpoints/point3r_512.pth \
    --output_dir ./outputs/point3r_exp/radio_head \
    --size 512
```

### Step 2: 可视化

```bash
conda activate point3r
python scripts/visualize_point3r.py \
    --npz_dir ./outputs/point3r_exp/radio_head \
    --port 8081 \
    --downsample_factor 4 \
    --max_points 500000
```

## 核心参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_root` | Behavior1K 数据根目录 | 必须指定 |
| `--episode_id` | Episode 编号（task-0000 = turning_on_radio） | 10 |
| `--camera` | 相机：`head` / `left_wrist` / `right_wrist` | head |
| `--num_frames` | 采样帧数（None=所有帧） | 16 |
| `--stride` | 帧间隔（sampling_mode=stride 时生效） | 1 |
| `--sampling_mode` | `stride`：连续采样；`uniform`：均匀采样 | stride |
| `--checkpoint` | Point3R .pth 权重路径 | 必须指定 |
| `--output_dir` | 输出目录 | ./outputs/point3r_exp |
| `--size` | 图像分辨率（Point3R 固定 512） | 512 |
| `--port` | viser 可视化端口 | 8081 |

## 输出结构

```
{output_dir}/
├── point3r_reconstruction.ply  # 合并所有帧的 RGB PLY 点云
├── frame_0000.npz              # 每帧：img, pts3d, camera_pose, conf
├── frame_0001.npz
├── ...
└── metadata.json
```

## Point3R 特点

- **端到端**：直接从多视图图像回归 3D 点，无需深度反投影
- **流式增量**：`--num_frames` 和 `--stride` 控制输入帧的滑动窗口，每次推理融合历史信息
- **相对坐标**：输出的 3D 坐标是相对值，已做中心化处理
- **帧级 pts3d**：每帧 `frame_*.npz` 中的 `pts3d` 是该帧预测的 3D 点位置（HxWx3）

## 流式增量重建用法

对于长 episode（如 stride=15 的 task-0000），建议：
```bash
python scripts/point3r_reconstruct_b1k.py \
    --episode_id 10 \
    --camera head \
    --num_frames 16 \
    --stride 15 \
    --sampling_mode stride \
    --checkpoint src/openpi/third_party/Point3R/src/checkpoints/point3r_512.pth \
    --output_dir ./outputs/point3r_exp/task0000_stride15 \
    --size 512
```

## 依赖

- `point3r` conda 环境（独立环境）
- `point3r_512.pth` 权重（3.1GB，从 Google Drive 下载）
- `openpi-comet-nas` **不需要**（Point3R 独立使用）
