## HAMLET Baseline Implementation

### Goal

在不引入 3D encoder、不修改现有 `pi0.5` 数据 schema 的前提下，实现一个 `PI0Pytorch` 兼容的显式 memory baseline，用于和原始 `pi0.5` baseline 做单变量对比。

### Implemented Files

- 配置：
  - [hamlet_config.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models/hamlet_config.py)
- Memory 模块：
  - [hamlet_memory.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/memory_baselines/hamlet_memory.py)
- 整模型：
  - [pi0_hamlet.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/pi0_hamlet.py)
- 训练入口：
  - [train_config.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/training/train_config.py)
  - [pretrain_config.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/training/pretrain_config.py)
  - [test_config.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/training/test_config.py)
  - [train_pytorch.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/train_pytorch.py)
- 测试：
  - [hamlet_memory_test.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/memory_baselines/hamlet_memory_test.py)
  - [pi0_hamlet_test.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/pi0_hamlet_test.py)
  - [test_pi05_hamlet.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/test_pi05_hamlet.py)

### Design

- backbone：直接继承 `PI0Pytorch`
- memory 注入点：覆写 `embed_prefix(...)`
- 当前 step moment summary：
  - 先将 learnable moment tokens 追加到 prefix embeddings
  - 单独跑一次 prefix-only 编码
  - 取最后 `hamlet_num_moment_tokens` 个 hidden states 作为当前时刻 summary
- history memory：
  - 使用 `HamletMemoryAdapter`
  - 内部维护固定长度 history buffer
  - 通过 2 层 causal Transformer 做 history contextualization
  - 输出使用 `current + tanh(gate) * contextualized`
- 推理 session：
  - `set_active_session(...)`
  - `reset_streaming_state(...)`
  - `clear_session(...)`

### Current Limitation

- 训练阶段默认 `update_memory=False`，即不在 batch 之间持久化 history。
- 这是为了避免把 i.i.d. batch 错误当作真实时序。
- 因此本轮 HAMLET 更准确地说是：
  - 推理侧具备 streaming history 接口
  - 训练侧仍是单步兼容实现

### Verification

```bash
source /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/miniconda3/etc/profile.d/conda.sh
conda activate openpi-comet-nas
pytest -q src/openpi/models_pytorch/memory_baselines/hamlet_memory_test.py src/openpi/models_pytorch/pi0_hamlet_test.py
python scripts/test_pi05_hamlet.py
```

结果：

- `hamlet_memory_test.py` + `pi0_hamlet_test.py`: `6 passed`
- `test_pi05_hamlet.py`:
  - `prefix_embs (2, 13, 8)`
  - `loss (2, 32, 32)`
  - `sampled_actions (2, 32, 32)`
