# Easi3R 在 Behavior1K 上的实验指南（含 task-0013 动态片段与三方对比）

## 1. 目标

- 在 `openpi-comet` 中补齐 Easi3R 的 B1K 运行入口，复用 Behavior1K episode / camera / frame range 工作流。
- 先验证 `point3r` 环境能否直接运行 Easi3R；如果不能，再创建独立 `easi3r` conda 环境。
- 在 `task-0013/head` 中自动选一个“机械臂动态明显”的片段，对比 Easi3R、InfiniteVGGT、Point3R。
- 第一阶段重点看：
  - Easi3R 的动态 mask 是否能把运动区域分离出来
  - 动态区域是否比 Point3R / InfiniteVGGT 更少拖影或噪声
  - 耗时、峰值显存、成功率

## 2. 代码入口（本仓库）

- 动态片段发现：[find\_b1k\_dynamic\_robot\_segment.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/3d_reconstruction/find_b1k_dynamic_robot_segment.py)
- Easi3R（B1K 片段级包装）：[easi3r\_reconstruct\_b1k.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/3d_reconstruction/easi3r_reconstruct_b1k.py)
- Point3R（支持 `start_frame/end_frame`）：[point3r\_reconstruct\_b1k.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/3d_reconstruction/point3r_reconstruct_b1k.py)
- InfiniteVGGT（支持 `start_frame/end_frame`）：[infinitevggt\_reconstruct\_b1k.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/3d_reconstruction/infinitevggt_reconstruct_b1k.py)
- 三方对比编排：TODO（该编排脚本尚未纳入仓库；当前建议按第 10/10.2/11 章节分别跑三方并用 viser 脚本对比）
- Easi3R 原生 4D 可视化：[visualizer.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/third_party/Easi3R/viser/visualizer.py)
- Easi3R（openpi-comet 轻量 viser，可独立端口对比）：[visualize\_easi3r.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/3d_reconstruction/visualize_easi3r.py)
- Point3R / InfiniteVGGT 对比可视化：[visualize\_compare\_point3r\_infinitevggt.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/3d_reconstruction/visualize_compare_point3r_infinitevggt.py) / [visualize\_compare\_point3r\_dirs.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/3d_reconstruction/visualize_compare_point3r_dirs.py)

## 3. 数据与 episode

- 数据根目录：
  - `/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos`
- `task-0013` 示例 episode：
  - `episode_id=130010`
- `head` 相机视频：
  - `videos/task-0013/observation.images.rgb.head/episode_00130010.mp4`
- `head` 分割视频：
  - `videos/task-0013/observation.images.seg_instance_id.head/episode_00130010.mp4`

## 4. 环境建议

- 推理优先尝试 `point3r` 环境：
  - Point3R / InfiniteVGGT 本来就推荐在 `point3r` 环境跑
  - 如果 Easi3R 在这个环境也能工作，就避免额外维护环境
- 可视化优先用 `openpi-comet-nas`：
  - 适合跑 viser 脚本
- 如果 `point3r` 环境跑不通 Easi3R，再新建 `easi3r` 环境

## 5. 先测 point3r 环境

### 5.1 关键导入检查

```bash
conda run -n point3r python -c "
import sys
from pathlib import Path
repo = Path('/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet')
easi3r = repo / 'src/openpi/third_party/Easi3R'
sys.path.insert(0, str(easi3r))
import torch, gradio
from dust3r.model import AsymmetricCroCo3DStereo
print('torch', torch.__version__)
print('gradio ok')
print('dust3r ok')
"
```

### 5.2 如果 point3r 环境失败，再创建 easi3r 环境

```bash
conda create -n easi3r python=3.10 cmake=3.31 -y
conda run -n easi3r pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
conda run -n easi3r pip install -r /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/third_party/Easi3R/requirements.txt
conda run -n easi3r pip install -e /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/third_party/Easi3R/viser
conda run -n easi3r pip install -e /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/third_party/Easi3R/third_party/sam2 --verbose
conda run -n easi3r bash -lc 'cd /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/third_party/Easi3R/croco/models/curope && python setup.py build_ext --inplace'
```

## 6. Checkpoints

### 6.1 Point3R

- 已存在：
  - `src/openpi/third_party/Point3R/src/checkpoints/point3r_512.pth`

### 6.2 InfiniteVGGT

- `--checkpoint_path` 支持本地 `.pth` 或 HuggingFace repo id
- 示例：
  - `lch01/StreamVGGT`

### 6.3 Easi3R / DUSt3R

- Easi3R 当前仓库未内置 DUSt3R / MonST3R 权重
- 可运行上游下载脚本：

```bash
cd /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/third_party/Easi3R/data
bash download_ckpt.sh
```

- 默认 DUSt3R 权重路径：
  - `src/openpi/third_party/Easi3R/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`

## 7. 先找 task-0013 的动态机械臂片段

```bash
conda run -n point3r python scripts/3d_reconstruction/find_b1k_dynamic_robot_segment.py \
  --data_root /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos \
  --task_id 13 \
  --episode_id 130010 \
  --camera head \
  --clip_length 240 \
  --clip_step 120 \
  --sample_stride 15 \
  --output_path ./outputs/dynamic_segments/task0013_ep130010_head.json
```

输出里关注：

- `recommended_clip.clip_start_frame`
- `recommended_clip.clip_end_frame`
- `recommended_clip.label`
- `recommended_clip.score`

## 8. Easi3R smoke test

先用一个较短片段确认链路可跑通：

```bash
conda run -n point3r python scripts/3d_reconstruction/easi3r_reconstruct_b1k.py \
  --data_root /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos \
  --episode_id 130010 \
  --camera head \
  --start_frame 4262 \
  --end_frame 4502 \
  --weights src/openpi/third_party/Easi3R/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --fps 5 \
  --num_frames 65 \
  --sam2_mask_refine \
  --output_dir ./outputs/easi3r_exp/task0013_ep130010_head_smoke \
  --overwrite
```

期望产物：

- `run_metadata.json`
- `{output_dir}/{seq_name}/visualization.mp4`
- `{output_dir}/{seq_name}/attention_video.mp4`
- `{output_dir}/{seq_name}/cluster_video.mp4`
- `{output_dir}/{seq_name}/pred_traj.txt`
- 动态 mask 相关目录

## 9. 三方动态片段对比

当前仓库未包含“一键三方编排”脚本。建议流程：

1. 先用第 7 节找动态片段（得到 start/end frame）。
2. 分别跑：
   - 第 8 节（Easi3R smoke test）
   - 第 10.1 节（Point3R）
   - 第 10.2 节（InfiniteVGGT）
3. 用第 11 节的 viser 脚本做结果对比与质检。

## 10. 单独运行 Point3R / InfiniteVGGT 动态片段

### 10.1 Point3R

```bash
conda run -n point3r python scripts/3d_reconstruction/point3r_reconstruct_b1k.py \
  --data_root /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos \
  --episode_id 130010 \
  --camera head \
  --start_frame 4262 \
  --end_frame 4502 \
  --num_frames 16 \
  --stride 15 \
  --sampling_mode stride \
  --checkpoint src/openpi/third_party/Point3R/src/checkpoints/point3r_512.pth \
  --output_dir ./outputs/point3r_exp/task0013_dynamic_clip \
  --max_points 500000
```

### 10.2 InfiniteVGGT

```bash
conda run -n point3r python scripts/3d_reconstruction/infinitevggt_reconstruct_b1k.py \
  --data_root /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos \
  --episode_id 130010 \
  --camera head \
  --start_frame 4262 \
  --end_frame 4502 \
  --sampling_stride 15 \
  --max_sampled_frames 0 \
  --window_size 16 \
  --window_step 8 \
  --checkpoint_path lch01/StreamVGGT \
  --output_dir ./outputs/infinitevggt_exp/task0013_dynamic_clip \
  --overwrite
```

## 11. 可视化

### 11.1 Easi3R 原生 4D 可视化

```bash
conda run -n openpi-comet-nas python src/openpi/third_party/Easi3R/viser/visualizer.py \
  --data ./outputs/easi3r_exp/task0013_ep130010_head_smoke/task0013_ep130010_head_f04262_04502 \
  --port 9081
```

### 11.2 Easi3R（openpi-comet 轻量 viser）

```bash
conda run -n openpi-comet-nas python scripts/3d_reconstruction/visualize_easi3r.py \
  --easi3r_dir ./outputs/compare_easi3r_point3r_infinitevggt/task0013_head_dynamic_tiny/easi3r \
  --port 9082 \
  --downsample_factor 6 \
  --bg_downsample_factor 8 \
  --max_points 300000
```

### 11.2 Point3R vs InfiniteVGGT

```bash
conda run -n openpi-comet-nas python scripts/3d_reconstruction/visualize_compare_point3r_infinitevggt.py \
  --point3r_dir ./outputs/point3r_exp/task0013_dynamic_clip \
  --infinitevggt_dir ./outputs/infinitevggt_exp/task0013_dynamic_clip \
  --port 8090 \
  --max_points 500000 \
  --point3r_downsample 6 \
  --infinitevggt_downsample 6 \
  --infinitevggt_conf_percentile 25
```

## 12. 输出结构

### 12.1 动态片段发现

```text
outputs/dynamic_segments/
  task0013_ep130010_head.json
```

### 12.2 Easi3R

```text
output_dir/
  run_metadata.json
  _scratch/{seq_name}.mp4
  {seq_name}/
    visualization.mp4
    attention_video.mp4
    cluster_video.mp4
    pred_traj.txt
    pred_intrinsics.txt
    ...
```

### 12.3 三方对比

```text
output_root/
  summary.json
  point3r/
  infinitevggt/
  easi3r/
```

## 13. 常见问题

- `point3r` 环境里 Easi3R 缺依赖：
  - 先看导入检查失败在哪个包，再决定是否切到独立 `easi3r` 环境
- Easi3R 权重不存在：
  - 先运行 `download_ckpt.sh` 或手动下载 DUSt3R 权重
- `demo.py --input_dir` 原生命令跑崩：
  - 直接用 [easi3r\_reconstruct\_b1k.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/3d_reconstruction/easi3r_reconstruct_b1k.py)，不要依赖上游的非交互 CLI 分支
- 动态片段发现为空：
  - 调大 `clip_length`
  - 减小 `sample_stride`
  - 检查 `head` 相机分割视频与 episode id 是否匹配
- Easi3R 更关注动态 mask：
  - 这次优先看 `visualization.mp4`、`attention_video.mp4`、动态 mask 目录，而不是强行与 Point3R 统一成同一种输出格式
