## MemoryVLA Baseline Implementation

### Goal

实现一个 `PI0Pytorch` / `pi0.5` 兼容的 `MemoryVLA-inspired` 外挂显式记忆 baseline，并明确区分：

- Phase A：单流 MVP（本轮已实现）
- Phase B：忠实双流（perceptual + cognitive，后续升级）

### Implemented Files

- 配置：
  - [memoryvla_config.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models/memoryvla_config.py)
- Memory 模块：
  - [memoryvla_memory.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/memory_baselines/memoryvla_memory.py)
- 整模型：
  - [pi0_memoryvla.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/pi0_memoryvla.py)
- 训练入口：
  - [train_config.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/training/train_config.py)
  - [pretrain_config.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/training/pretrain_config.py)
  - [test_config.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/training/test_config.py)
  - [train_pytorch.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/train_pytorch.py)
- 测试：
  - [memoryvla_memory_test.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/memory_baselines/memoryvla_memory_test.py)
  - [pi0_memoryvla_test.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/pi0_memoryvla_test.py)
  - [test_pi05_memoryvla.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/test_pi05_memoryvla.py)

### Phase A Design

- backbone：继承 `PI0Pytorch`
- memory 注入点：覆写 `embed_prefix(...)`
- 当前 query token：
  - 先跑一次 prefix-only 编码
  - 对 prefix hidden 做 mean pooling，得到单个 summary token
- external memory：
  - `SingleStreamMemoryBank`
  - 固定容量 bank
  - similarity-aware replacement / append
- retrieval：
  - 当前 summary 作为 query
  - 对 bank 做 softmax attention retrieval
- fusion：
  - `GatedMemoryFusion`
  - 对当前 token 和 retrieved token 做逐维 sigmoid gate 融合
- prefix 注入：
  - 将融合后的 memory token 追加到原 prefix embeddings 后，再交给 action expert 走原路径

### Current Limitation

- 本轮只实现单流 perceptual memory，没有实现 cognitive stream。
- `memoryvla_use_cognitive_stream` / `memoryvla_cognitive_pool` 只是在 config 层预留接口。
- 训练阶段同样默认 `update_memory=False`，避免把随机 batch 错误当成真实时序 history。

### Phase B Upgrade Plan

后续升级为更忠实的双流版本时，建议保持以下接口不变：

- `MemoryVLAModule.encode_memory_item(...)`
- `MemoryVLAModule.retrieve(...)`
- `MemoryVLAModule.forward(...)`

升级点：

- 新增 `cognitive_bank`
- 引入 `lang_hidden` / EOS pooled summary
- 引入 `merge_streams(...)` 做双流融合

### Verification

```bash
source /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/miniconda3/etc/profile.d/conda.sh
conda activate openpi-comet-nas
pytest -q src/openpi/models_pytorch/memory_baselines/memoryvla_memory_test.py src/openpi/models_pytorch/pi0_memoryvla_test.py
python scripts/test_pi05_memoryvla.py
```

结果：

- `memoryvla_memory_test.py` + `pi0_memoryvla_test.py`: `6 passed`
- `test_pi05_memoryvla.py`:
  - `prefix_embs (2, 6, 8)`
  - `loss (2, 32, 32)`
  - `sampled_actions (2, 32, 32)`
