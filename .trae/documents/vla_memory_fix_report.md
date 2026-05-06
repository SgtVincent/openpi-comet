## VLA Memory Fix Report

### Background

VLA 在 B1K 的 `receeding_horizon` 控制方式下，每次 `policy.infer()` 生成一段 action chunk（默认 32 步），环境执行完 chunk 才返回下一次视觉观测。原实现中 VLM2 的 dual-memory 在每次推理都会被重置，导致 **memory 无法跨 replanning 次持久化**；同时把多相机 view “伪装成 video” 导致 VGGT 重复计算。

参考背景分析文档：[vla_memory_fix_plan.md](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/.trae/documents/vla_memory_fix_plan.md)

### Completed Fixes

#### 1) DualMemory 支持 runtime state 导出/恢复（为 session 化 streaming memory 打基础）

- 新增：
  - `DualMemoryModule.clear_runtime_state()`
  - `DualMemoryModule.get_runtime_state()`
  - `DualMemoryModule.set_runtime_state(...)`
- 目的：
  - 不进入 checkpoint（都是 `persistent=False` 的 runtime buffers），仅用于推理时 “跨调用持久化” 与 “多连接隔离”。
- 代码：
  - [dual_memory.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/vlm2/dual_memory.py)

#### 2) VLM2 推理改为 streaming memory（跨 `policy.infer()` 保留），训练路径保持按 batch reset

- 变更：
  - `VLM2WithPi05.process_video_with_memory(..., reset_before_process=False)`：默认不 reset
  - 训练 `forward()` / `VLM2SubtaskWithPi05.forward()`：显式 `reset_before_process=True`，保持原训练假设（每个样本独立）
- 目的：
  - 在 B1K receding-horizon 下，让 memory 的 “时间维” 对齐到 replanning 次（跨 `infer`）而不是 view 序列。
- 代码：
  - [process_video_with_memory](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/vlm2/vlm2_model.py#L373-L416)
  - [VLM2WithPi05.forward](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/vlm2/vlm2_model.py#L485-L551)
  - [VLM2SubtaskWithPi05.forward](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/vlm2/vlm2_model.py#L835-L980)

#### 3) 3D encoder (VGGT) 默认只用 head camera（base_0_rgb）

- 变更：
  - 推理 `sample_actions()` 使用 `preprocess_observation_pytorch(..., image_keys=("base_0_rgb",))`
  - 仅构造单帧 `video_frames = base_image.unsqueeze(1)`（shape `(B, 1, C, H, W)`）
- 目的：
  - 避免 “multi-timestep + multi-view” 尚未验证的风险；
  - 避免旧实现中将多 view + padding 重复帧拼成 “伪视频”，导致 VGGT 重复计算/语义混乱。
- 代码：
  - [VLM2WithPi05.sample_actions](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/vlm2/vlm2_model.py#L659-L747)

#### 4) Streaming memory 的 session 隔离（支持 websocket 多连接共享同一模型）

- 模型侧新增：
  - `VLM2WithPi05.set_active_session(session_id)`
  - `VLM2WithPi05.reset_streaming_state(session_id)`
  - `VLM2WithPi05.clear_session(session_id)`
  - 内部用 `DualMemoryModule.get/set_runtime_state()` 保存/恢复 per-session buffers
- Wrapper 接入：
  - `B1KPolicyWrapper` 增加 `_session_id`
  - `spawn_session()` 分配新 `_session_id`
  - 每次 `policy.infer()` 前 best-effort 调用 `model.set_active_session(...)`
  - `reset()` 时 best-effort `reset_streaming_state(...)` 清空该连接的 memory
- 代码：
  - [VLM2WithPi05 session APIs](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/vlm2/vlm2_model.py#L306-L368)
  - [B1KPolicyWrapper session wiring](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/shared/eval_b1k_wrapper.py#L28-L140)

#### 5) 修复离线脚本的明显问题

- `scripts/test_b1k_openpi.py` 修复错误导入：`from numpy import np` -> `import numpy as np`
- 代码：
  - [test_b1k_openpi.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/test_b1k_openpi.py)

### Tests Run (Completed)

#### Pytest

新增测试：
- DualMemory runtime state roundtrip：[dual_memory_test.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/vlm2/dual_memory_test.py)
- VLM2 session/streaming memory 行为（轻量，不依赖完整模型 init）：[vlm2_model_streaming_test.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/vlm2/vlm2_model_streaming_test.py)
- B1K wrapper session/reset wiring：[eval_b1k_wrapper_test.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/shared/eval_b1k_wrapper_test.py)

执行命令（注意本环境需要显式 source conda.sh）：

```bash
source /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/miniconda3/etc/profile.d/conda.sh
conda activate openpi-comet-nas
pytest -q src/openpi/models_pytorch/vlm2/dual_memory_test.py \
  src/openpi/models_pytorch/vlm2/vlm2_model_streaming_test.py \
  src/openpi/shared/eval_b1k_wrapper_test.py
```

结果：`6 passed`

#### Diagnostics

- `dual_memory.py` / `eval_b1k_wrapper.py` / 新增测试文件均无 IDE 诊断错误。
- `vlm2_model.py` 存在既有的 Pylance 类型提示诊断（主要是第三方模型接口类型不精确），不影响本次功能修复与 pytest 执行。

### Notes / Known Limitations

- 当前修复聚焦于 **推理侧**：让 memory 在 receding-horizon 场景按 replanning 次持久化，并默认 head-only 3D。
- 训练侧若要让 memory 真正 “学会用”，仍需要 episode-level / TBPTT 或 detach memory 的训练改造（本轮未做，避免引入训练范式大改动）。
- action-history token（将上一段 action chunk 压缩为 transition token）属于增益项，本轮未引入，以减少变量。

### Follow-ups (Next Experiments)

- 离线 replay 对比（streaming on/off，head-only vs 旧版伪视频）：
  - 主要指标：`infer_ms`、稳定性（NaN/崩溃）、memory_count 行为是否符合预期。
- 在线 B1K eval：
  - 至少一个短任务 + 一个长任务，重点验证：episodic memory capacity 达到后 merge/LRU 稳定、session 不串、推理时延下降。

### Related Plan Doc

开发与测试计划（可执行步骤清单）：[vla_memory_fix_dev_test_plan.md](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/.trae/documents/vla_memory_fix_dev_test_plan.md)
