# InfiniteVGGT（StreamVGGT）在 Behavior1K 上的实验指南（含 Point3R 对比）

## 1. 目标

- 在 Behavior1K（B1K）视频上跑通 InfiniteVGGT（StreamVGGT）的**流式/滑窗**重建。
- 与 Point3R 在同一 episode（优先 `task-0013/head`）上做可复现对比，并提供 viser 对比可视化。
- 第一阶段对比口径：质化（点云形态/噪声/稳定性）+ 资源统计（耗时/峰值显存/成功率）。

## 2. 数据与 episode id

- 数据根目录：
  - `/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos`
- `task-0013` head 视频示例：
  - `videos/task-0013/observation.images.rgb.head/episode_00130010.mp4`
- episode id 约定：
  - `episode_id = task_id * 10000 + episode_number`
  - 例如 `task_id=13, episode_number=10` → `episode_id=130010`

## 3. 代码入口（本仓库）

- InfiniteVGGT（B1K 流式/滑窗重建）：[infinitevggt_reconstruct_b1k.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/3d_reconstruction/infinitevggt_reconstruct_b1k.py)
- InfiniteVGGT（全 episode 滑窗重建，输出为 Point3R-style frame_*.npz）：[infinitevggt_reconstruct_full_episode.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/3d_reconstruction/infinitevggt_reconstruct_full_episode.py)
- Point3R（B1K 抽帧重建）：[point3r_reconstruct_b1k.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/3d_reconstruction/point3r_reconstruct_b1k.py)
- Point3R（全 episode 滑窗重建）：[point3r_reconstruct_full_episode.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/3d_reconstruction/point3r_reconstruct_full_episode.py)
- 批量对比编排（多 episode）：TODO（该批量编排脚本尚未纳入仓库；建议先手动跑若干 episode，再用 viser 脚本对比）
- viser 对比可视化（Point3R vs InfiniteVGGT）：[visualize_compare_point3r_infinitevggt.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/3d_reconstruction/visualize_compare_point3r_infinitevggt.py)
- viser 对比可视化（两个 Point3R-style 目录对比）：[visualize_compare_point3r_dirs.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/3d_reconstruction/visualize_compare_point3r_dirs.py)

## 4. 环境建议（务实版）

本仓库里涉及两类脚本，建议按用途拆环境：

- **推理/重建脚本**（Point3R / InfiniteVGGT）：建议使用 `point3r` 环境（torch/cuda 与依赖齐全）。
- **可视化脚本**（viser）：建议使用 `openpi-comet-nas` 环境（项目默认环境，一般自带 viser/cv2）。

说明：
- 如果把 Point3R 推理放到 `openpi-comet-nas`，需要补齐 `accelerate` 等依赖，否则会报 `ModuleNotFoundError: accelerate`。

## 5. Checkpoints

### 5.1 Point3R

- 期望路径：
  - `src/openpi/third_party/Point3R/src/checkpoints/point3r_512.pth`

### 5.2 InfiniteVGGT / StreamVGGT

`--checkpoint_path` 支持两种形式：
- 本地 `.pth` 文件路径（严格按 state_dict 加载）
- HuggingFace repo id（例如 `lch01/StreamVGGT`，脚本内部会走 `StreamVGGT.from_pretrained`）

如果使用 HF repo id，建议提前设置 `HF_TOKEN` 以避免限速。

## 6. 输出结构与字段

### 6.1 InfiniteVGGT 输出（滑窗）

```
output_dir/
  metadata.json
  window_0000/
    metadata.json
    images/0000.png ...
    frame_0000.npz ...
  window_0001/...
  _scratch/...
```

`window_*/frame_*.npz` 至少包含：
- `point_map`：(H, W, 3)
- `point_conf`：(H, W)
- `source_frame_index` / `local_frame_index` / `window_id`

### 6.2 Point3R 输出（抽帧）

```
output_dir/
  point3r_reconstruction.ply
  frame_0000.npz
  ...
  metadata.json
```

`frame_*.npz` 常见字段：
- `img`：(H, W, 3)
- `pts3d`：(H, W, 3)（或带 batch 维）
- `conf`：（可选）

## 7. Quick Start（task-0013 / head / episode 10）

推荐先跑 8 帧进行 smoke test，确认链路无误后再扩展为更长序列与多窗口。

### 7.1 Point3R（8 帧 quick run）

```bash
conda run -n point3r python scripts/3d_reconstruction/point3r_reconstruct_b1k.py \
  --data_root /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos \
  --episode_id 130010 --camera head \
  --num_frames 8 --stride 15 --sampling_mode stride \
  --checkpoint src/openpi/third_party/Point3R/src/checkpoints/point3r_512.pth \
  --output_dir ./outputs/point3r_exp/task0013_ep0010_head_n8_s15
```

### 7.2 InfiniteVGGT（8 帧 quick run）

```bash
conda run -n point3r python scripts/3d_reconstruction/infinitevggt_reconstruct_b1k.py \
  --data_root /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos \
  --episode_id 130010 --camera head \
  --sampling_stride 15 --max_sampled_frames 8 \
  --window_size 8 --window_step 8 \
  --checkpoint_path lch01/StreamVGGT \
  --output_dir ./outputs/infinitevggt_exp/task0013_ep0010_head_n8_s15 \
  --overwrite
```

## 8. viser 对比可视化（同一页面切换/并排）

```bash
conda run -n openpi-comet-nas python scripts/3d_reconstruction/visualize_compare_point3r_infinitevggt.py \
  --point3r_dir ./outputs/point3r_exp/task0013_ep0010_head_n8_s15 \
  --infinitevggt_dir ./outputs/infinitevggt_exp/task0013_ep0010_head_n8_s15 \
  --port 8090 \
  --max_points 500000 \
  --point3r_downsample 6 \
  --infinitevggt_downsample 6 \
  --infinitevggt_conf_percentile 25
```

浏览器访问：`http://localhost:8090/`

控件说明：
- `Point3R Frame`：Point3R 的 `frame_XXXX.npz` 序号
- `InfiniteVGGT Window`：选择 `window_XXXX`
- `InfiniteVGGT Local Frame`：窗口内帧序号
- `Point3R 改变时尝试匹配 source_frame_index`：勾选后尝试用 `source_frame_index` 自动把 InfiniteVGGT 定位到对应帧

显示策略说明：
- 两种点云沿 X 轴左右分开（`split_offset`），便于肉眼对照。
- 两种点云各自做去中心化（减均值），因此更适合对比“形态/噪声/稳定性”，不用于评估绝对坐标系对齐。

## 9. 多 episode 批量对比（任务内）

当前仓库未包含 “多 episode 一键对比编排” 脚本。建议做法：

1. 按第 7/11 节分别跑 Point3R 与 InfiniteVGGT（每个 episode 一个输出目录）。
2. 用第 8 节（同一 episode）或第 11.3 节（两个目录）做可视化对比与质检。

## 10. 常见问题

- HuggingFace 下载提示 unauthenticated：设置 `HF_TOKEN` 或提前下载 `.pth` 本地加载。
- InfiniteVGGT 报 CUDA 不可用：推理需要 GPU。
- Point3R 在 `openpi-comet-nas` 报 `ModuleNotFoundError: accelerate`：请用 `point3r` 环境跑推理，或在该环境补依赖。

## 11. Full Episode 对齐对比（参数对齐 Point3R full episode）

目的：完全模仿 [point3r_reconstruct_full_episode.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/3d_reconstruction/point3r_reconstruct_full_episode.py) 的滑窗与参数，在 InfiniteVGGT 上做“同 stride、同 window_size、同 window_step”的全 episode 实验，并用 viser 对比结果。

### 11.1 Point3R full episode

```bash
conda run -n point3r python scripts/3d_reconstruction/point3r_reconstruct_full_episode.py \
  --data_root /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos \
  --episode_id 130010 \
  --camera head \
  --num_frames_per_window 16 \
  --stride 15 \
  --window_step 8 \
  --checkpoint src/openpi/third_party/Point3R/src/checkpoints/point3r_512.pth \
  --output_dir ./outputs/point3r_exp/task0013_ep0010_head_full_w16_s15_ws8 \
  --max_points 500000
```

### 11.2 InfiniteVGGT full episode（输出为 Point3R-style frame_*.npz）

```bash
conda run -n point3r python scripts/3d_reconstruction/infinitevggt_reconstruct_full_episode.py \
  --data_root /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos \
  --episode_id 130010 \
  --camera head \
  --num_frames_per_window 16 \
  --stride 15 \
  --window_step 8 \
  --checkpoint_path lch01/StreamVGGT \
  --total_budget 1200000 \
  --output_dir ./outputs/infinitevggt_exp/task0013_ep0010_head_full_w16_s15_ws8 \
  --max_points 500000 \
  --overwrite
```

### 11.3 viser 对比（两个目录均为 frame_*.npz）

```bash
conda run -n openpi-comet-nas python scripts/3d_reconstruction/visualize_compare_point3r_dirs.py \
  --left_dir ./outputs/point3r_exp/task0013_ep0010_head_full_w16_s15_ws8 \
  --right_dir ./outputs/infinitevggt_exp/task0013_ep0010_head_full_w16_s15_ws8 \
  --port 8093 \
  --left_downsample 8 \
  --right_downsample 8 \
  --max_points 400000 \
  --split_offset 1.6
```
