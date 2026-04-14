# Pi3 / Pi3X 3D Reconstruction 使用指南

## 1. 概述

本项目以 **Pi3X** 为主（推荐），在 Behavior1K (B1K) 的 episode mp4 上进行单目视频几何重建，并输出彩色点云：

- 输出：`pi3x_reconstruction.ply`（带 RGB 的点云）
- 可视化：`scripts/3d_reconstruction/visualize_ply_viser.py`（viser Web viewer）

Pi3X 的官方推理入口为 `src/openpi/third_party/Pi3/example_mm.py`。本仓库提供一层 B1K wrapper：

- 重建脚本：`scripts/3d_reconstruction/pi3x_reconstruct_b1k.py`
- 可视化脚本：`scripts/3d_reconstruction/visualize_ply_viser.py`

> 注意：Pi3/Pi3X 权重许可证为 CC BY-NC 4.0（非商用），详见 Pi3 官方 README。

---

## 2. 环境准备（新 conda 环境）

本仓库已创建并验证的新环境名：

```bash
conda activate openpi-comet-nas
conda activate pi3x-nas
```

依赖安装方式（与 Pi3 README 一致）：

```bash
cd /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/third_party/Pi3
pip install -r requirements.txt
pip install viser
```

> 如果 HuggingFace 下载慢：根据项目规则先运行 `proxy_on`，或手动下载 `model.safetensors` 并在脚本中使用 `--ckpt /path/to/model.safetensors`。

---

## 3. 数据来源（Behavior1K）

B1K 视频数据默认在：

```
/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos/
```

视频路径模式：

```
videos/task-XXXX/observation.images.rgb.<camera>/episode_XXXXXXXX.mp4
```

camera 取值：

- `head` -> `observation.images.rgb.head`
- `left_wrist` -> `observation.images.rgb.left_wrist`
- `right_wrist` -> `observation.images.rgb.right_wrist`

---

## 4. 快速开始

### 4.1 跑 Pi3X 重建（B1K wrapper）

```bash
conda activate openpi-comet-nas
conda activate pi3x-nas
cd /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet

python scripts/3d_reconstruction/pi3x_reconstruct_b1k.py \
  --episode_id 130010 \
  --camera head \
  --interval 30
```

输出目录默认：

```
./outputs/pi3x_exp/task0013_ep0010_head_pi3x_i30/
  pi3x_reconstruction.ply
  metadata.json
```

### 4.2 可视化 PLY（viser）

```bash
conda activate openpi-comet-nas
conda activate pi3x-nas
cd /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet

python scripts/3d_reconstruction/visualize_ply_viser.py \
  --ply_path outputs/pi3x_exp/task0013_ep0010_head_pi3x_i30/pi3x_reconstruction.ply \
  --port 8099 \
  --max_points 500000
```

浏览器打开：

`http://localhost:8099/`

---

## 5. 常见问题

### Q1: 权重下载很慢 / 卡住

- 先运行 `proxy_on` 再执行重建；或手动下载 `.safetensors` 并用 `--ckpt` 指定本地路径。

### Q2: 显存不够 / 太慢

- 调大 `--interval`（例如 30/45/60），减少输入帧数量。
- 或者切换 `--device cpu`（更慢，但可用于快速验证流程）。
