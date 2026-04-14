---
name: "vggt-3d-reconstruction"
description: "VGGT depth-based 3D reconstruction from B1K videos + viser可视化。Invoke when user asks to reconstruct 3D from video using VGGT or compare VGGT vs Point3R."
---

# VGGT 3D Reconstruction Skill

## 环境要求

需要激活 `openpi-comet-nas` conda 环境（包含 VGGT 依赖）。

## 快速开始

### Step 1: 视频帧提取 + VGGT 推理

```bash
conda activate openpi-comet-nas
python scripts/reconstruct_3d_b1k.py \
    --data_root /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos \
    --episode_id 10 \
    --camera head \
    --num_frames 16 \
    --stride 1 \
    --sampling_mode stride \
    --output_dir ./outputs/vggt_exp/radio_head \
    --size 518
```

### Step 2: 可视化

```bash
python scripts/visualize_reconstruction.py \
    --npz_dir ./outputs/vggt_exp/radio_head \
    --port 8080 \
    --downsample_factor 4 \
    --max_points 500000
```

## 核心参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_root` | Behavior1K 数据根目录 | 必须指定 |
| `--episode_id` | Episode 编号（对应 task-0000 的 turning_on_radio） | 10 |
| `--camera` | 相机：`head` / `left_wrist` / `right_wrist` | head |
| `--num_frames` | 采样帧数（None=所有帧） | 16 |
| `--stride` | 帧间隔（sampling_mode=stride 时生效） | 1 |
| `--sampling_mode` | `stride`：均匀间隔采样；`uniform`：等距采样 | stride |
| `--output_dir` | 输出目录 | ./outputs/vggt_exp |
| `--size` | 图像分辨率 | 518 |
| `--port` | viser 可视化端口 | 8080 |

## 输出结构

```
{output_dir}/
├── frame_0000.npz   # 每帧结果：img, depth, camera_pose, conf
├── frame_0001.npz
├── ...
└── metadata.json    # 元信息
```

## 重建原理

VGGT 输出每帧的 RGB 图像 + 深度图 + 相机内参/外参，通过以下公式反投影到 3D：
- `X = (u - cx) * d / fx * scale`
- `Y = (v - cy) * d / fy * scale`
- `Z = d * scale`

然后通过相机外参将每帧点云对齐到世界坐标系。

## 可视化说明

viser 会在 `http://localhost:{port}` 提供交互式 3D 点云查看：
- RGB 颜色表示 3D 坐标中的颜色点
- 每帧相机位姿用坐标系（红绿蓝轴）标注
- `--downsample_factor 4` 表示每 4 个点取 1 个（减少 GPU 负载）

## 依赖

- `openpi-comet-nas` conda 环境
- VGGT checkpoint（通常在 pretrained/ 下或自动下载）
- `viser`, `open3d`, `numpy`, `opencv-python`
