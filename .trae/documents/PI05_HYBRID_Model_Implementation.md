# PI05_HYBRID Model Implementation（最终改动 / Concern / 使用方式）

本文档基于当前仓库代码、`PI05_FIX.md`、以及已有训练日志，整理 PI05_HYBRID（π0.5 Hybrid）特性的最终实现状态、关键改动、验证结果与仍需注意的限制。

## 目标与实现概览

PI05_HYBRID 的目标是在同一个 VLM backbone 前向中，同时学习：

- 连续动作的 flow matching（与 PI05 一致的动作专家路径）
- 子任务文本（subtask text）的自回归 token 预测（cross-entropy 监督）

组合损失对应论文 Eq.(1)：

- `L = CE(subtask_tokens) + alpha * flow_matching_loss`
- 当前默认 `alpha=10.0`

训练时的 token 组织为：

- Prefix：`[image_tokens, prompt+state_tokens, subtask_tokens(causal)]`
- Suffix：`[action_tokens]`
- 单次 forward 同时产出 `text_logits -> CE` 与 `v_t -> MSE(flow matching)`

代码入口：

- 训练侧 PyTorch 模型：[pi05_hybrid.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/pi05_hybrid.py)
- Action expert（同时产出 v_t 与 text logits）：[hybrid_expert.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/action_experts/hybrid_expert.py)
- 配置入口：[pi05_hybrid_config.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models/pi05_hybrid_config.py)

## 代码改动清单

### 新增核心文件

- [pi05_hybrid_config.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models/pi05_hybrid_config.py)
  - `Pi05HybridConfig`
  - 暴露 `alpha`、`subtask_max_len`、`vocab_size` 等配置
  - 明确 JAX `create()` 未实现，仅支持 PyTorch 训练/推理路径
- [pi05_hybrid.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/pi05_hybrid.py)
  - `PI05HybridPytorch`
  - `forward()` 返回 `{loss, flow_loss, ce_loss}`
  - 提供 `predict_subtask_tokens()`、`predict_subtask()`、`sample_actions_hierarchical()`
- [hybrid_expert.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/action_experts/hybrid_expert.py)
  - `HybridActionExpert`
  - `compute_hybrid_loss_train()` 单次 forward 同时计算文本 CE 与动作 velocity

### 关键修改文件

- [model.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models/model.py)
  - `ModelType` 新增 `PI05_HYBRID`
  - `Observation` 新增 `subtask_tokens/subtask_mask/subtask_loss_mask/subtask_ar_mask`
  - `preprocess_observation()` 与 `load_pytorch()` 路径支持 hybrid
- [tokenizer.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models/tokenizer.py)
  - 新增 `HybridTokenizer`
  - `tokenize_subtask()` 现在显式 prepends BOS，并且不再丢掉第一个真实 subtask token 的 CE 监督
- [preprocessing_pytorch.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/preprocessing_pytorch.py)
  - PyTorch preprocess 路径透传全部 hybrid subtask 字段，避免训练/推理中静默丢字段
- [transforms.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/transforms.py)
  - `PromptFromLeRobotItem` 透传 `subtask_text`
  - 新增 `TokenizeHybridInputs`，把 `prompt/state/subtask_text` 转成 hybrid observation 字段
- [data_config.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/training/data_config.py)
  - `ModelTransformFactory` 新增 `PI05_HYBRID` 分支
  - B1K repack transform 增加 `subtask_text`
- [train_config.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/training/train_config.py)
  - `pytorch_model_name` 新增 `"hybrid"`
  - 默认 checkpoint/log 输出路径统一到 `./outputs/checkpoints` 与 `./outputs/logs`
- [pretrain_config.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/training/pretrain_config.py)
  - 新增预训练配置 `pi05_hybrid_b1k-pt50_cs32_bs64_lr2.5e-5_5ep`
- [sft_make_pizza_config.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/training/sft_make_pizza_config.py)
  - 新增单任务配置 `pi05_hybrid_b1k-make_pizza_lr2.5e-6_5ep_sft`
- [registry.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/action_experts/registry.py)
  - 注册 `"hybrid"` action expert
- [train_pytorch.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/train_pytorch.py)
  - 新增 hybrid 模型构建路径
  - 兼容 forward 返回 dict 的 loss 处理
  - wandb 记录 `hybrid/flow_loss` 与 `hybrid/ce_loss`
  - 非 VLM2 训练不再承担 `vlm2_model` 启动成本
  - `wandb` 改为惰性导入
- [dataset.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/behavior/learning/datas/dataset.py)
  - streaming / 非 streaming `__getitem__` 均注入 `item["subtask_text"]`
  - 当前固定使用 orchestrator level=1 task 作为 CE 监督
- [policy.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/policies/policy.py)
  - `Policy.infer()` 现在支持两阶段 hybrid 推理
  - 以 prompt 为 key 缓存生成的 subtask tokens
  - 输出里暴露 `generated_subtask`
- [b1k_policy.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/policies/b1k_policy.py)
  - 将最小 `R1Pro` proprio 索引常量本地化，避免训练配置导入时无意义拉起 OmniGibson

### 训练 / 验证辅助文件

- [run_pi05_hybrid_train.sh](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/run_pi05_hybrid_train.sh)
  - 通用 hybrid 训练启动脚本
- [run_pi05_hybrid_make_pizza_1k.sh](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/run_pi05_hybrid_make_pizza_1k.sh)
  - 单任务 `make_pizza` 1000 step 验证脚本
  - 默认保守参数：`batch_size_per_gpu=1`、`num_workers=0`
- [run_pi05_hybrid_validation.sh](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/run_pi05_hybrid_validation.sh)
  - 运行 hybrid 功能验证入口
- [test_pi05_hybrid_feature.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/test_pi05_hybrid_feature.py)
  - 覆盖训练侧 sample wiring、层级推理和可选 train smoke

## 数据与监督信号（subtask_text）

### subtask_text 的来源

当前实现将 Behavior-1K 数据中的 orchestrator level=1 作为子任务文本监督：

- `item["task"]`：仍使用 `fine_grained_level` 对应的 task
- `item["subtask_text"]`：固定取 `level=1` 的 `task` 字段，用于 hybrid 的 CE 监督

对应实现：

- [dataset.py:__getitem__](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/behavior/learning/datas/dataset.py#L440-L448)
- [dataset.py:_get_task_at_level](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/behavior/learning/datas/dataset.py#L602-L614)

### 没有 subtask_text 时的退化行为

如果样本没有 `subtask_text`：

- [TokenizeHybridInputs](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/transforms.py#L398-L445) 会生成全 0 的 subtask tokens/masks
- [PI05HybridPytorch.forward](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/pi05_hybrid.py#L95-L167) 会自动退化为仅 flow matching（`ce_loss=0`）

## 已修复的关键问题

这次最终版实现里，之前在 `PI05_对比分析.md` / `PI05_FIX.md` 中暴露的关键问题已经补齐：

- 第一个 subtask token 的 CE 监督现在通过 BOS 对齐，不再被跳过
- hybrid prefix/subtask mask 拼接现在有统一 helper，不再依赖脆弱的局部拼接逻辑
- preprocess 路径已透传全部 subtask 字段，避免 hybrid 字段在中间层静默丢失
- `predict_subtask_tokens()` 现在用真实 prefix 末位 hidden state，不再使用 padding 后的错误位置
- hierarchical inference 已正式落地，不再只是文档或注释引用
- EOS 不再写死为 `1`，而是读取 sentencepiece `eos_id()`
- `Policy.infer()` 已支持自动两阶段推理与 prompt 级 subtask cache
- `BaseModelConfig.load_pytorch()` 已支持加载 hybrid checkpoint

## 如何训练

### 1) 环境

建议在训练/验证前激活：

```bash
conda activate openpi-comet-nas
```

额外依赖重点关注：

- `sentencepiece`
- `gs://big_vision/paligemma_tokenizer.model` 的访问能力，或本地缓存好的 tokenizer 文件

### 2) 预训练配置

预置 hybrid pretrain config：

- `pi05_hybrid_b1k-pt50_cs32_bs64_lr2.5e-5_5ep`

启动方式：

```bash
bash scripts/run_pi05_hybrid_train.sh
```

或直接：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_pytorch.py pi05_hybrid_b1k-pt50_cs32_bs64_lr2.5e-5_5ep \
  --exp_name pi05_hybrid_train
```

### 3) 单任务 make_pizza 验证配置

单任务 SFT config：

- `pi05_hybrid_b1k-make_pizza_lr2.5e-6_5ep_sft`

验证脚本：

```bash
bash scripts/run_pi05_hybrid_make_pizza_1k.sh
```

该脚本默认：

- `CONFIG_NAME=pi05_hybrid_b1k-make_pizza_lr2.5e-6_5ep_sft`
- `NUM_TRAIN_STEPS=1000`
- `BATCH_SIZE_PER_GPU=1`
- `NUM_WORKERS=0`

### 4) 训练日志与指标

训练 loop 对 hybrid 会额外记录：

- `hybrid/flow_loss`
- `hybrid/ce_loss`

同时训练产物默认落在：

- `outputs/checkpoints/<exp_name>/...`
- `outputs/logs/<exp_name>/rank0.log`

## 如何使用（推理 / 调用）

### 推理动作

`PI05HybridPytorch.sample_actions()` 现在有两种路径：

- 如果 observation 已带 ground-truth `subtask_tokens`，直接按条件化 subtask 采样动作
- 如果没有 subtask，则自动走 hierarchical path：先生成 subtask，再生成动作

对应实现：

- [sample_actions](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/pi05_hybrid.py#L205-L223)
- [sample_actions_hierarchical](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/pi05_hybrid.py#L225-L244)

### 预测 subtask 文本

`predict_subtask_tokens()` / `predict_subtask()` 提供从 prefix 自回归生成 subtask 的能力：

- [predict_subtask_tokens](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/pi05_hybrid.py#L246-L321)
- [predict_subtask](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/pi05_hybrid.py#L323-L332)

`Policy.infer()` 在没有 ground-truth subtask 时，会：

- 先调用 `predict_subtask_tokens()`
- 将生成结果缓存到 prompt 级 cache
- 重建 hierarchical observation
- 输出动作，并额外返回 `generated_subtask`

## 验证结果

### 1) 层级推理验证

真实 checkpoint `checkpoints/openpi_comet/pi05-b1kpt50-cs32` 上已验证：

- auto-generated-subtask 路径可运行
- ground-truth-subtask 路径可运行

产物：

- `checkpoints/console_logs/pi05_hybrid_validation_light13.trace`
- `checkpoints/console_logs/pi05_hybrid_validation_light13.log`

### 2) one-step hybrid train smoke

真实环境 `openpi-comet-nas` 中，hybrid forward/backward/optimizer step 已跑通。

产物：

- `checkpoints/console_logs/pi05_hybrid_train_smoke_validation4.trace`

关键 marker：

- `train_smoke:model_load_done`
- `train_smoke:forward_done`
- `train_smoke:backward_done`
- `train_smoke:step_done`
- `train_smoke_ok`

### 3) 正式 train_pytorch 路径的一步真实数据验证

正式 `scripts/train_pytorch.py` 已在真实 Behavior-1K 数据上完成一步训练并落盘 checkpoint。

本次保守覆盖参数：

- `--batch-size-per-gpu 1`
- `--num-workers 0`

产物：

- `checkpoints/console_logs/pi05_hybrid_real_train_validation4.log`
- `checkpoints/console_logs/pi05_hybrid_real_train_validation4.status`
- `outputs/checkpoints/pi05_hybrid_real_train_validation4/1/model.safetensors`
- `outputs/checkpoints/pi05_hybrid_real_train_validation4/1/optimizer.pt`

### 4) 单任务 make_pizza 1000 step 验证

单任务实验 `pi05_hybrid_make_pizza_step1000_validation_nw0` 已从日志确认完成 1000 step。

主日志：

- `outputs/logs/pi05_hybrid_make_pizza_step1000_validation_nw0/rank0.log`

关键 checkpoint 保存点：

- step 250
- step 500
- step 750
- step 999
- step 1000

日志中可见的后半段 loss 走势：

- step 800: `loss=8.8052`
- step 840: `loss=8.0275`
- step 900: `loss=8.0625`
- step 920: `loss=7.8276`
- step 940: `loss=7.7145`
- step 960: `loss=7.6479`
- step 980: `loss=8.1653`

这说明：

- 单任务 hybrid 训练已经不是只停留在 one-step smoke
- 在 `batch_size_per_gpu=1`、`num_workers=0` 的保守设置下，真实数据 1000 step 能完整跑通并持续落盘

## 已知限制 / Concern

### 1) JAX 侧未实现

`Pi05HybridConfig.create()` 仍明确抛出 `NotImplementedError`，当前 hybrid 仅支持 PyTorch 路径。

### 2) tokenizer 依赖外部下载

`HybridTokenizer` 与 `predict_subtask()` 都依赖 sentencepiece tokenizer；在无外网 / 无 GCS 权限环境下，需要提前准备缓存文件或替换下载逻辑。

### 3) checkpoint 仍使用宽松加载策略初始化 hybrid

训练脚本在 `pytorch_model_name == "hybrid"` 时使用 `strict=False` 加载已有权重，以兼容从 PI05 checkpoint 初始化 hybrid 的场景。这意味着：

- 初始化路径已可用
- 但实际训练前后仍建议检查 `flow_loss/ce_loss` 的数值合理性

### 4) `subtask_text` 语义仍是工程近似

当前并不是直接读取 `skill_description/primitive_description` 原字段，而是使用 orchestrator level=1 task 作为 subtask supervision。如果要更严格贴论文定义，还需要进一步确认：

- level=1 task 是否与目标 subtask 语义等价
- 数据侧是否应切换到更贴近 `skill_description/primitive_description` 的原始字段

### 5) 默认大配置 runtime shape 还没有完整长跑验证

当前最稳的已验证组合仍是：

- `batch_size_per_gpu=1`
- `num_workers=0`

默认较大配置（例如 `batch_size_per_gpu=32`、`num_workers=16`）在当前节点上的完整长跑稳定性，还没有在本轮工作中被同等强度验证。

## 当前结论

截至当前版本，仓库已经具备：

- hybrid 训练主路径
- hybrid-aware checkpoint 加载路径
- hierarchical inference 与 prompt cache
- 真实 checkpoint 的层级推理验证
- 真实数据的一步 formal 训练验证
- 单任务 `make_pizza` 的 1000 step 完整训练验证

换句话说，PI05_HYBRID 在 openpi-comet 中已经从“搭骨架”推进到“可在真实数据上跑完整体验证”的状态，当前剩余问题主要集中在更大规模 runtime 稳定性与 subtask 语义选择，而不再是主功能缺失。
