# 3D Reconstruction Pipeline (VGGT/CUT3R) 使用指南

## 1. 概述

本项目集成了 **VGGT (Visual Generative 3D Transformer)** 和 **CUT3R** 两个 3D 重建 foundational model，用于对 Behavior1K 机器人操作视频数据进行深度估计和 3D 点云重建。Pipeline 支持单相机 / 多相机、密集 / 稀疏帧采样，并可选择启用 Bundle Adjustment (BA) 全局对齐来提升多视角重建的一致性。

**核心入口脚本**：
- 重建脚本：`scripts/3d_reconstruction/reconstruct_3d_b1k.py`
- 可视化脚本：`scripts/3d_reconstruction/visualize_reconstruction.py`
- VGGT 官方 COLMAP 后处理（BA）：`src/openpi/third_party/vggt/demo_colmap.py`

**本地 VGGT 权重**：
```
checkpoints/openpi_comet/vggt/model.pt
```
> ⚠️ 权重文件本地优先，仅在文件不存在时才会尝试从 HuggingFace 下载。

---

## 2. 环境准备

```bash
conda activate openpi-comet-nas
```

依赖库（已在上述 conda 环境中）：
- `torch`, `numpy`, `Pillow`, `opencv-python`
- `viser`（Web 可视化服务端）
- `pycolmap`（Bundle Adjustment 后处理，如需使用 `--use_ba`）

如需安装 pycolmap：
```bash
pip install pycolmap
```

---

## 3. 数据来源

Behavior1K 视频数据存放在：
```
/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos/
```

目录结构：
```
videos/
  task-0000/
    observation.images.rgb.head/
      episode_00000010.mp4
    observation.images.rgb.left_wrist/
      episode_00000010.mp4
    observation.images.rgb.right_wrist/
      episode_00000010.mp4
  task-0001/
    ...
```

支持的相机 Key：
| camera 参数 | 数据路径中的 Key |
|---|---|
| `head` | `observation.images.rgb.head` |
| `left_wrist` | `observation.images.rgb.left_wrist` |
| `right_wrist` | `observation.images.rgb.right_wrist` |

---

## 4. 快速开始

### 4.1 运行 3D 重建

```bash
# 单相机密集采样（head，16帧，步长1）
python scripts/3d_reconstruction/reconstruct_3d_b1k.py \
    --episode_id 10 \
    --cameras "head" \
    --num_frames 16 \
    --sampling_mode "stride" \
    --sampling_stride 1 \
    --output_dir ./outputs/3d_recon_exp_v2

# 多相机稀疏采样（head + left_wrist + right_wrist，各6帧，步长8）
python scripts/3d_reconstruction/reconstruct_3d_b1k.py \
    --episode_id 10 \
    --cameras "head,left_wrist,right_wrist" \
    --num_frames 6 \
    --sampling_mode "stride" \
    --sampling_stride 8 \
    --output_dir ./outputs/3d_recon_exp_v2
```

**参数说明**：

| 参数 | 说明 |
|---|---|
| `--episode_id` | Episode 编号（如 10 对应 `episode_00000010.mp4`） |
| `--cameras` | 逗号分隔的相机列表，可选 `head`, `left_wrist`, `right_wrist` |
| `--num_frames` | 每个相机采样的帧数 |
| `--sampling_mode` | `stride`（等间隔采样）或 `uniform`（均分视频长度） |
| `--sampling_stride` | 步长（stride 模式下表示每隔 N 帧取一帧） |
| `--output_dir` | 输出根目录 |

**输出文件结构**：
```
<output_dir>/<exp_name>/
  frame_0000.npz   # 包含 extrinsic, intrinsic, depth, conf
  frame_0001.npz
  ...
  metadata.json    # 实验配置元信息
  images/          # 抽帧的原始 PNG 图像（供 BA 使用）
    0000.png
    ...
```

### 4.2 启动 3D 点云可视化

```bash
python scripts/3d_reconstruction/visualize_reconstruction.py \
    --npz_dir ./outputs/3d_recon_exp_v2/ep10_vggt_head_n16_s8_stride \
    --port 8080 \
    --conf_threshold 25.0
```

然后浏览器访问：`http://localhost:8080/`

**参数说明**：

| 参数 | 说明 |
|---|---|
| `--npz_dir` | 包含 `.npz` 文件的目录路径 |
| `--port` | Viser Web 服务端口，默认 8080 |
| `--conf_threshold` | 深度置信度阈值（过滤低质量深度点），默认 25.0 |

### 4.3 运行 Bundle Adjustment（多相机全局对齐）

BA 需要先安装 `pycolmap`，并将抽帧图像保存到 `images/` 子目录（`reconstruct_3d_b1k.py` 会自动保存）：

```bash
python src/openpi/third_party/vggt/demo_colmap.py \
    --scene_dir ./outputs/3d_recon_exp_v2/ep10_vggt_head-left_wrist-right_wrist_n6_s8_stride \
    --use_ba
```

BA 输出：
```
<sparse_dir>/
  images/
    0000.png, 0001.png, ...
  points.ply         # 全局对齐后的彩色点云
  cameras.txt
  images.txt
  points3D.txt
```

---

## 5. 实验设置与结论对照表

| 实验 | 设置 | 资产路径 | 可验证结论 |
|---|---|---|---|
| **Exp1** 单视角稀疏基线 | head, 16帧, stride=8 | `ep10_vggt_head_n16_s8_stride/` | 纯净基线：稳定单一视角+大视差，验证模型在理想条件下的输出质量 |
| **Exp2** 单视角密集 | head, 16帧, stride=1 | `ep10_vggt_head_n16_s1_stride/` | 验证极小视差（连续帧）是否导致深度估计退化 |
| **Exp3** 腕部密集运动 | left_wrist, 16帧, stride=1 | `ep10_vggt_left_wrist_n16_s1_stride/` | 验证腕部相机剧烈运动是否导致位姿/深度崩溃 |
| **Exp4** 多视角稀疏（复现） | head+left_wrist+right_wrist, 各6帧, stride=8 | `ep10_vggt_head-left_wrist-right_wrist_n6_s8_stride/` | 复现原始碎片化问题，验证多视角巨大差异是否是根因 |
| **Exp5** 多视角+BA | 在 Exp4 基础上运行 `demo_colmap.py --use_ba` | `ep10_vggt_head-left_wrist-right_wrist_n6_s8_stride/sparse/` | 关键验证：BA 全局对齐是否能修复多视角坐标系撕裂 |

---

## 6. 输出 npz 文件格式

每个 `frame_XXXX.npz` 文件包含以下字段：

| 字段 | 形状 | 说明 |
|---|---|---|
| `extrinsic` | `(4, 4)` | 相机外参矩阵（camera-from-world，OpenCV 坐标系） |
| `intrinsic` | `(3, 3)` | 相机内参矩阵 |
| `depth` | `(H, W)` | 深度图（单通道），与 input image 同分辨率 |
| `conf` | `(H, W)` | 深度置信度图，值越高表示越可靠 |

> **注意**：VGGT 输出的 `depth` 分辨率与输入图像分辨率相同（默认 518x518），而非原始视频分辨率。

---

## 7. 3D 点云渲染原理（代码参考）

深度图反投影到 3D 点云使用的是 VGGT 官方几何工具函数，**请勿自行手写矩阵乘法**（容易因坐标系 convention 不同而出错）：

```python
from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3

# depth_batch:    (B, H, W)
# extrinsic:      (B, 4, 4)  camera-from-world
# intrinsic:      (B, 3, 3)
# 返回 world_points: (B, H, W, 3)
world_points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)

# 如果需要将点云从 camera 坐标转到 world 坐标：
cam_to_world = closed_form_inverse_se3(extrinsic)  # (B, 4, 4)
```

---

## 8. 常见问题与排查

### Q1: 点云碎片化、撕裂严重
**可能原因**：
1. 多相机（head vs wrists）视角差异过大，VGGT 无法正确融合
2. 腕部相机运动剧烈，特征匹配失败
3. 场景中存在大量动态物体（机械臂）破坏静态场景假设

**建议**：先运行 Exp1（单视角稀疏）作为基线。如果基线质量良好，则问题确认是多相机融合导致的。进一步可尝试 Exp5（+BA）看全局对齐能否修复。

### Q2: 深度图几乎全黑或全白
**可能原因**：深度值范围不在模型预期区间，可能是因为模型权重加载失败或精度不匹配。
**排查**：检查是否正确使用了 `torch.bfloat16` 或 `torch.float16`，以及 local checkpoint 是否完整加载。

### Q3: Bundle Adjustment 失败
**可能原因**：
1. `pycolmap` 未安装
2. 图像数量不足（至少需要 2 张以上有足够视差的图像）
3. 相机内参/外参初始化太离谱，导致 BA 优化发散

### Q4: 终端显示 "ModuleNotFoundError: No module named 'torch'"
**原因**：未激活正确的 conda 环境。请在命令前加 `conda activate openpi-comet-nas &&`。

---

## 9. 关键文件索引

```
openpi-comet/
├── scripts/3d_reconstruction/
│   ├── reconstruct_3d_b1k.py       # 3D 重建主脚本
│   └── visualize_reconstruction.py # Web 可视化脚本
├── src/openpi/third_party/
│   ├── vggt/                       # VGGT 模型库
│   │   ├── demo_colmap.py          # BA 后处理入口
│   │   └── vggt/utils/geometry.py  # 深度反投影工具函数
│   └── cut3r/                      # CUT3R 模型库（备用）
├── checkpoints/openpi_comet/vggt/model.pt  # 本地 VGGT 权重
└── .trae/documents/
    └── 3d_reconstruction/3D_RECONSTRUCTION_GUIDE.md  # 本文档
```

---

## 10. 参考资料

- **VGGT 论文**：Visual Generative 3D Transformer（Facebook Research）
- **VGGT 官方仓库**：https://github.com/facebookresearch/gsvit
- **COLMAP**：Structure-from-Motion 和 Multi-View Stereo 工具
- **pycolmap**：Python binding for COLMAP
- **viser**：轻量级 Web 3D 可视化服务端
