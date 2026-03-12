# OpenPI-Comet：可插拔 Action Expert（PyTorch）

本文档说明 OpenPI-Comet 中新增的 Action Expert 抽象层与 il_lib 适配层，用于在不影响 `il_lib` 独立训练/运行的前提下，在 OpenPI 的 PI0(PyTorch) 链路里快速替换不同 action expert 结构。

## 代码入口

- 抽象/registry：
  - [base.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/action_experts/base.py)
  - [registry.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/action_experts/registry.py)
- 内置 expert：
  - Gemma token expert（保持原始 PI0 逻辑）：[gemma_token_expert.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/action_experts/gemma_token_expert.py)
  - il_lib MoE velocity field（新增适配）：[il_moe_velocity_expert.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/action_experts/il_moe_velocity_expert.py)
- 注入点：
  - ModelConfig 新增字段：
    - [pi0_config.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models/pi0_config.py)
    - [vlm2_vla_config.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models/vlm2_vla_config.py)
    - [action_expert_config.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models/action_expert_config.py)
  - PyTorch 模型加载注入：[model.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models/model.py#L252-L290)
  - PI0Pytorch 使用可插拔 expert：[pi0_pytorch.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/pi0_pytorch.py)

## 已支持的 expert

### gemma_token（默认）

- 使用 PaliGemma prefix + Gemma token expert，行为与原实现一致。

### il_moe_velocity（新增）

- 需要运行环境可 import `il_lib`。
- 用 OpenPI prefix encoder 得到 prefix hidden states 并池化成 `prefix_cond`，将 `state` 投影为 `state_cond`，两者相加得到 cond。
- 调用 `il_lib.nn.flow_matching.moe_velocity_field.ActionBlockMoEVelocityField` 预测 flow-matching velocity `v_t`，采样循环保持不变。

## 配置与用法

ModelConfig 新增字段：

- `model.pytorch_action_expert.name`: `gemma_token | il_moe_velocity`
- `model.pytorch_action_expert.il_moe_velocity.*`: MoE expert 的超参（cond_dim / num_experts / top_k / hidden_dim 等）

示例（在 openpi conda 环境中）：

```bash
python openpi-comet/scripts/train.py pi05_b1k-base --model.pytorch-action-expert.name gemma_token
python openpi-comet/scripts/train.py pi05_b1k-base --model.pytorch-action-expert.name il_moe_velocity --model.pytorch-action-expert.il-moe-velocity.cond-dim 512 --model.pytorch-action-expert.il-moe-velocity.num-experts 4 --model.pytorch-action-expert.il-moe-velocity.top-k 2
```

## VLM2WithPi05 现状

`VLM2WithPi05` 目前仅支持 `action_expert_name="gemma_token"`；若配置为其他 expert 会直接报错。

## 最小 smoke test

脚本：

- [test_action_expert_plugin.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/test_action_expert_plugin.py)

运行：

```bash
python openpi-comet/scripts/test_action_expert_plugin.py --expert gemma_token
python openpi-comet/scripts/test_action_expert_plugin.py --expert il_moe_velocity
```
