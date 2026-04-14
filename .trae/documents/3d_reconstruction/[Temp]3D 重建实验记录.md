# \[Temp] 3D 重建实验记录

本文档用于记录 `task-0013 / episode_00130010 / head` 的 3D 重建实验、可视化入口与当前状态，便于后续交互与编辑。

## 基本信息

- 仓库：`/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet`
- 数据根目录：`/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos`
- Episode：`episode_id = 130010`（task-0013）
- Camera：`head`
- full-episode 视频：
  - `/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos/videos/task-0013/observation.images.rgb.head/episode_00130010.mp4`
- full-episode 采样设定（Point3R / InfiniteVGGT）：
  - `stride = 15`
  - 采样帧：`0, 15, 30, ..., 13710`（共 `915` 帧）
  - window：`num_frames_per_window = 16`
  - window\_step：`8`

## 当前已启动页面（running）

| 页面                                    | 地址                       | 说明                                                   |
| ------------------------------------- | ------------------------ | ---------------------------------------------------- |
| Point3R vs InfiniteVGGT（full-episode） | <http://127.0.0.1:8096/> | 双源同步 `frame_index` 的 4D 风格 viewer（带相机视锥/轨迹选项）        |
| Easi3R 原生 4D（动态片段）                    | <http://127.0.0.1:9082/> | 当前是动态片段结果（不是 full-episode）                           |
| Easi3R 原生 4D（full-episode, s15/915）   | <http://127.0.0.1:9084/> | stride=15/915 帧采样对齐 full-episode（img=224, winsize=3） |
| Easi3R 原生 4D（full-episode, s100/138）  | <http://127.0.0.1:9085/> | stride=100/138 帧采样 full-episode（img=224, winsize=3） |

## 实验与可视化汇总

| 方法                      | 实验名称                                                  | 目录                                                                                                                  | 状态                                                              | 可视化命令                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ----------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Point3R                 | full-episode (w16, s15, ws8)                          | `outputs/point3r_exp/task0013_ep0010_head_full_w16_s15_ws8/`                                                        | ✅ 完成（含 `metadata.json`/PLY）                                     | `source ~/.bashrc && conda activate openpi-comet-nas && python scripts/3d_reconstruction/visualize_point3r.py --npz_dir outputs/point3r_exp/task0013_ep0010_head_full_w16_s15_ws8 --port 8091 --downsample_factor 10 --max_points 500000 --keep_global_coords`                                                                                                                                                                                                               |
| InfiniteVGGT            | full-episode (w16, s15, ws8)                          | `outputs/infinitevggt_exp/task0013_ep0010_head_full_w16_s15_ws8/`                                                   | ✅ 完成（含 `metadata.json`/PLY）                                     | `source ~/.bashrc && conda activate openpi-comet-nas && python scripts/3d_reconstruction/visualize_point3r.py --npz_dir outputs/infinitevggt_exp/task0013_ep0010_head_full_w16_s15_ws8 --port 8092 --downsample_factor 10 --max_points 500000 --keep_global_coords`                                                                                                                                                                                                          |
| Point3R vs InfiniteVGGT | full-episode 4D 对比页                                   | （同上两目录）                                                                                                             | ✅ 运行中（8096）                                                     | `source ~/.bashrc && conda activate openpi-comet-nas && python scripts/3d_reconstruction/visualize_dynamic_npz_dirs_4d.py --left_dir outputs/point3r_exp/task0013_ep0010_head_full_w16_s15_ws8 --right_dir outputs/infinitevggt_exp/task0013_ep0010_head_full_w16_s15_ws8 --left_label point3r --right_label infinitevggt --port 8096 --keep_global_coords --left_downsample 10 --right_downsample 10 --max_points 500000 --camera_frustum_scale 0.08 --trajectory_stride 6` |
| Easi3R                  | full-episode preview（80帧，仅覆盖开头）                       | `outputs/easi3r_exp/task0013_ep0010_head_full_preview_nf80_it120/.../task0013_ep0010_head_full_preview_nf80_it120/` | ✅ 完成（但采样不等价）                                                    | `source ~/.bashrc && conda activate openpi-comet-nas && python scripts/3d_reconstruction/visualize_easi3r_native_4d.py --data-path outputs/easi3r_exp/task0013_ep0010_head_full_preview_nf80_it120/task0013_ep0010_head_full_preview_nf80_it120 --port 9083 --downsample-factor 3 --bg-downsample-factor 4 --conf-threshold 1.0 --foreground-conf-threshold 0.1 --max-frames 80`                                                                                             |
| Easi3R                  | full-episode stride=15 对齐采样（it10）                     | `outputs/easi3r_exp/task0013_ep0010_head_full_s15_nf915_it10/`                                                      | ⏳ 运行中（log: `.../_scratch/console.log`；采样帧已齐：915 张，0..13710）     | （完成后）`source ~/.bashrc && conda activate openpi-comet-nas && python scripts/3d_reconstruction/visualize_easi3r_native_4d.py --data-path outputs/easi3r_exp/task0013_ep0010_head_full_s15_nf915_it10/task0013_ep0010_head_full_s15_nf915_it10 --port 9084 --downsample-factor 3 --bg-downsample-factor 4 --conf-threshold 1.0 --foreground-conf-threshold 0.1 --max-frames 915`                                                                                               |
| Easi3R                  | full-episode stride=15 对齐采样（it80）                     | `outputs/easi3r_exp/task0013_ep0010_head_full_s15_nf915_it80/`                                                      | ⚠️ 未完成（仅 `_scratch/stride_frames`，无 `run_metadata.json`）        | （完成后）同上，把目录换为 `...it80/...it80`                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Easi3R                  | full-episode stride=15 对齐采样（it10 rerun, GPU7）         | `outputs/easi3r_exp/task0013_ep0010_head_full_s15_nf915_it10_rerun_gpu7/`                                           | ❌ OOM（global\_aligner 阶段；需额外分配 20GiB，80GB 显存不足）                 | （完成后）`source ~/.bashrc && conda activate openpi-comet-nas && python scripts/3d_reconstruction/visualize_easi3r_native_4d.py --data-path outputs/easi3r_exp/task0013_ep0010_head_full_s15_nf915_it10_rerun_gpu7/task0013_ep0010_head_full_s15_nf915_it10_rerun_gpu7 --port 9084 --downsample-factor 3 --bg-downsample-factor 4 --conf-threshold 1.0 --foreground-conf-threshold 0.1 --max-frames 915`                                                                         |
| Easi3R                  | full-episode stride=15 对齐采样（it10, img=224, winsize=3） | `outputs/easi3r_exp/task0013_ep0010_head_full_s15_nf915_it10_img224_w3/`                                            | ✅ 完成（`run_metadata.json` / `scene.glb` / mp4 已生成；peak \~48.5GB） | `source ~/.bashrc && conda activate openpi-comet-nas && python scripts/3d_reconstruction/visualize_easi3r_native_4d.py --data-path outputs/easi3r_exp/task0013_ep0010_head_full_s15_nf915_it10_img224_w3/task0013_ep0010_head_full_s15_nf915_it10_img224_w3 --port 9084 --downsample-factor 3 --bg-downsample-factor 4 --conf-threshold 1.0 --foreground-conf-threshold 0.1 --max-frames 915`                                                                                |
| Easi3R                  | full-episode stride=100（it10, img=224, winsize=3）         | `outputs/easi3r_exp/task0013_ep0010_head_full_s100_nf138_it10_img224_w3/`                                           | ✅ 完成（`run_metadata.json` / `scene.glb` / mp4 已生成；peak \~9.0GB）  | `source ~/.bashrc && conda activate openpi-comet-nas && python scripts/3d_reconstruction/visualize_easi3r_native_4d.py --data-path outputs/easi3r_exp/task0013_ep0010_head_full_s100_nf138_it10_img224_w3/task0013_ep0010_head_full_s100_nf138_it10_img224_w3 --port 9085 --downsample-factor 3 --bg-downsample-factor 4 --conf-threshold 1.0 --foreground-conf-threshold 0.1 --max-frames 138`                                                                                |

## 实验复现命令（重建）

### Point3R full-episode

```bash
source ~/.bashrc
conda activate openpi-comet-nas
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

### InfiniteVGGT full-episode

```bash
source ~/.bashrc
conda activate openpi-comet-nas
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

### Easi3R full-episode（采样对齐：stride=15 / 915帧）

```bash
source ~/.bashrc
conda activate openpi-comet-nas
python scripts/3d_reconstruction/easi3r_reconstruct_b1k.py \
  --data_root /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos \
  --episode_id 130010 \
  --camera head \
  --start_frame 0 \
  --end_frame 13720 \
  --use_source_video \
  --sampling_mode stride \
  --sample_stride 15 \
  --num_frames 915 \
  --fps 0 \
  --scenegraph_type swinstride \
  --winsize 5 \
  --flow_loss_weight 0.0 \
  --min_conf_thr 1.1 \
  --niter 80 \
  --output_dir ./outputs/easi3r_exp/task0013_ep0010_head_full_s15_nf915_it80 \
  --seq_name task0013_ep0010_head_full_s15_nf915_it80 \
  --overwrite
```

## 已知问题 / 待办

- Easi3R stride=15 对齐采样（it10）已确认失败原因：GPU OOM（发生在 `global_aligner -> PointCloudOptimizer(...).to(device)`），见 `outputs/easi3r_exp/task0013_ep0010_head_full_s15_nf915_it10/_scratch/console.log` 末尾 traceback：
  - `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.67 GiB`
  - 报错时 GPU0 仅剩 `~4.20 GiB` 空闲；另有进程占用 `~74.68 GiB` 显存（导致不足以完成全局对齐阶段）
- 解决方向（优先级从高到低）：
  - 换到空闲 GPU（保证有足够显存余量；full-episode 对齐采样约需要 70GB+ 量级显存）
  - 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 降低碎片化风险
  - 仍 OOM 时：降低 `image_size`（512→224）或减小 `winsize` / 改小 scene graph 密度（会影响与原设置可比性，但能换取可跑通）
- 当前已按上述方向启动 rerun：`task0013_ep0010_head_full_s15_nf915_it10_rerun_gpu7`（`CUDA_VISIBLE_DEVICES=7` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`）。
  - 该 rerun 已跑完 `9100` pairs 的 inference，但在 `global_aligner` 进入全局优化时 OOM（`Tried to allocate 20.00 GiB`），说明在 `image_size=512` + `winsize=5` + `915` 帧设置下，global\_aligner 内存峰值会超过 80GB 显存。
- 当前正在跑降配版（保持 stride=15/915 帧采样一致）：`task0013_ep0010_head_full_s15_nf915_it10_img224_w3`（`image_size=224`, `winsize=3`），inference pairs 下降到 `5472`，目标是降低 global\_aligner 显存峰值以跑通。
- Easi3R 原生 viewer 支持 `--max-frames`，即使重建帧数更多，也会被该参数截断显示；full-episode 对齐采样时建议设置为 `915`。

## 相关脚本索引

- Point3R full-episode 重建：[point3r\_reconstruct\_full\_episode.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/3d_reconstruction/point3r_reconstruct_full_episode.py)
- InfiniteVGGT full-episode 重建：[infinitevggt\_reconstruct\_full\_episode.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/3d_reconstruction/infinitevggt_reconstruct_full_episode.py)
- Easi3R（B1K 包装 + 采样模式）：[easi3r\_reconstruct\_b1k.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/3d_reconstruction/easi3r_reconstruct_b1k.py)
- Point3R-style 4D viewer（双目录）：[visualize\_dynamic\_npz\_dirs\_4d.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/3d_reconstruction/visualize_dynamic_npz_dirs_4d.py)
- Easi3R 原生 4D viewer（openpi-comet 入口）：[visualize\_easi3r\_native\_4d.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/3d_reconstruction/visualize_easi3r_native_4d.py)
