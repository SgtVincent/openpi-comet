# PI0 × BEHAVIOR-1K Skill Pretrain 最终方案

## 目标与约束

- 新增可复现实验配置：`pi0_b1k_skill-pt50_pretrain_lr1e-4_1ep`
  - 初始参数来自 `pi05_base_pytorch`
  - 训练仅使用 `PI0Pytorch` 的 flow-matching MSE loss（不包含 subtask CE / subtask loss）
  - 指令输入（prompt）来自 BEHAVIOR-1K annotations 的 `skill_description`
- 以最小侵入方式新增 dataset 封装：`BehaviorLeRobotSkillDataset`
  - 作为 `BehaviorLeRobotDataset` 的轻量子类/封装
  - 只改变 “prompt 来源”，其余字段保持完全兼容现有 training pipeline
- Norm stats 与训练/部署一致性是硬约束
  - 训练阶段从 config assets 目录加载 norm stats（缺失会导致后续 Normalize 失败或训练/部署不一致）
  - 部署阶段强制从 checkpoint 的 `assets/` 目录加载 norm stats，确保与训练时一致（见 [create_trained_policy](file:///mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet/src/openpi/policies/policy_config.py#L60-L70)）

## 两次讨论汇总结论（Skill dataset / 训练 / norm stats）

- Prompt 注入不在模型侧做改动：沿用现有 `PromptFromLeRobotItem` 的 `task -> prompt` 机制（见 [transforms.py](file:///mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet/src/openpi/transforms.py#L386-L398)），在数据侧把 `task` 替换成 `skill_description`。
- `skill_description` 的来源与对齐方式：
  - annotations 的 `skill_annotation[*].skill_description`（见 [dataset_utils.py](file:///mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet/src/behavior/learning/datas/dataset_utils.py#L142-L155)）
  - 复用 `BehaviorLeRobotDataset._get_resample_key_from_skill_ann()` 做 “帧 -> skill segment” 映射（见 [dataset.py](file:///mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet/src/behavior/learning/datas/dataset.py#L775-L835)）
- 训练 recipe 明确以 “PI0 的 flow-matching MSE” 为唯一优化目标：
  - PyTorch 侧对应 `PI0Pytorch.forward()` 的 `F.mse_loss(u_t, v_t)`（见 [pi0_pytorch.py](file:///mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet/src/openpi/models_pytorch/pi0_pytorch.py#L337-L362)）
- Norm stats 必须先算再训：
  - stats 写入 `<assets_dir>/<asset_id>/norm_stats.json`（见 [compute_norm_stats.py](file:///mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet/scripts/compute_norm_stats.py#L89-L154) 与 [normalize.save](file:///mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet/src/openpi/shared/normalize.py#L134-L138)）
  - 训练保存 checkpoint 时，会把 `data_config.norm_stats` 保存进 `checkpoint/assets/<asset_id>/norm_stats.json`（见 [save_state](file:///mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet/src/openpi/training/checkpoints.py#L69-L90)），供部署时加载（见 [create_trained_policy](file:///mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet/src/openpi/policies/policy_config.py#L64-L70)）

## 数据集方案：BehaviorLeRobotSkillDataset（最小改动、显式开关）

### 1) 新增 dataset 封装

新增文件：
- `src/openpi/training/behavior_skill_dataset.py`

实现要点：
- 继承 `behavior.learning.datas.dataset.BehaviorLeRobotDataset`
- 覆盖 `__getitem__`：
  - `item = super().__getitem__(idx)`（保留现有字段，如 `episode_index / timestamp / task` 等）
  - `(_, skill_desc) = self._get_resample_key_from_skill_ann(item)`
  - 若 `skill_desc` 非空：写回 `item["task"] = skill_desc`
  - 否则：不改动 `task`，避免因 annotations 缺失而训练崩溃

### 2) 在 training 的 dataset factory 里接入（opt-in）

涉及入口：
- 现有 B1K dataset 创建逻辑：`create_behavior_dataset()`（见 [behavior_dataset.py](file:///mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet/src/openpi/training/behavior_dataset.py)）
- `DataConfigFactory` 会优先从 assets_dir 加载 norm stats（见 [data_config.py](file:///mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet/src/openpi/training/data_config.py#L272-L293)）

接入策略：
- 增加布尔开关：`prompt_from_skill_description: bool = False`
- 在 `create_behavior_dataset()` 内：
  - 若 `prompt_from_skill_description=True`：实例化 `BehaviorLeRobotSkillDataset`
  - 否则：保持现有 `BehaviorLeRobotDataset`

## 训练配置：pi0_b1k_skill-pt50_pretrain_lr1e-4_1ep

参考与复用：
- 现有对照实验脚本：`scripts/run_pi05_b1k_skill-pt50_pretrain_lr1e-4_2ep.sh`
- 现有对照训练配置：`pretrain_config.py` 中 `pi05_b1k_skill-pt50_pretrain_lr1e-4_2ep`

核心设定（以 2ep 配置为模板，改为 PI0 + 1ep + skill prompt）：
- `name="pi0_b1k_skill-pt50_pretrain_lr1e-4_1ep"`
- `pytorch_model_name="pi0"`（确保走 PI0 训练路径）
- `model=pi0_config.Pi0Config(pi05=True, action_horizon=32)`（与 `pi05_base_pytorch` 权重结构对齐）
- `data=LeRobotB1KDataConfig(..., base_config=DataConfig(..., prompt_from_skill_description=True, fine_grained_level=0, subtask_source="orchestrator"))`
  - `prompt_from_skill_description=True` 触发 SkillDataset 输出 `task=skill_description`
  - `fine_grained_level=0` 可保留（最终由 SkillDataset 覆盖 `task`，避免依赖 orchestrator 的层级结构）
  - subtask 相关字段不设置/保持默认（该实验不需要 subtask loss）
- 权重：
  - 本地：`pytorch_weight_path="checkpoints/pi05_base_pytorch"`
  - 远端：`weight_loader=CheckpointWeightLoader("sunshk/pi05_base_pytorch")`
- 训练步数：
  - `num_train_epochs=1`（`num_train_steps=0` 按现有配置约定）

## Norm stats：必须先算再训（并保证训练/部署一致）

### 1) 计算 norm stats

入口脚本： [compute_norm_stats.py](file:///mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet/scripts/compute_norm_stats.py)

推荐命令（本环境优先 conda）：

```bash
conda activate openpi-comet-nas
python scripts/compute_norm_stats.py --config-name pi0_b1k_skill-pt50_pretrain_lr1e-4_1ep
```

快速自检（只跑少量 frames）：

```bash
python scripts/compute_norm_stats.py --config-name pi0_b1k_skill-pt50_pretrain_lr1e-4_1ep --max-frames 4096
```

预期产物：
- `<assets_dir>/<asset_id>/norm_stats.json`

### 2) 为什么部署一定要用 checkpoint/assets 的 stats

- 训练 config 在构建 `DataConfig` 时会从 assets_dir 加载 norm stats（见 [data_config.py](file:///mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet/src/openpi/training/data_config.py#L272-L293)），并在保存 checkpoint 时把同一份 stats copy 到 `checkpoint/assets/`（见 [save_state](file:///mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet/src/openpi/training/checkpoints.py#L69-L90)）
- 部署端默认从 `checkpoint/assets/<asset_id>/norm_stats.json` 加载（见 [create_trained_policy](file:///mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet/src/openpi/policies/policy_config.py#L64-L70)），避免 “训练用了 A stats、部署用了 B stats” 的隐蔽不一致

## 运行与验证清单（最小闭环）

1) 配置能被解析：

```bash
conda activate openpi-comet-nas
python -c "from openpi.training.config import get_config; print(get_config('pi0_b1k_skill-pt50_pretrain_lr1e-4_1ep'))"
```

2) Dataset smoke（只跑少量 step）：
- 用 `python scripts/train_pytorch.py pi0_b1k_skill-pt50_pretrain_lr1e-4_1ep --exp_name <run_name>` 启动 1–2 个 step（见 [train_pytorch.py](file:///mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet/scripts/train_pytorch.py#L7-L24)）
- 重点确认 batch 中最终 `prompt` 来自 `skill_description`（而不是原始 primitive/task）

3) 权重加载验证：
- `checkpoints/pi05_base_pytorch/model.safetensors` 存在时 strict load 通过

4) 训练 loss 形态验证：
- 日志应只出现主 loss（flow-matching MSE），不应出现 `subtask/ce_loss`

## 风险与回滚

- `skill_description` 在部分帧可能为空（annotations 缺失或映射不到 segment）
  - 处理：为空时不覆盖 `task`，回退到原 `task`，保证训练不崩溃
- norm stats 不存在或 asset_id 不匹配
  - 处理：先跑 `scripts/compute_norm_stats.py --config-name ...`，并确认生成在 `<assets_dir>/<asset_id>/norm_stats.json`
- 回滚路径
  - 关闭 `prompt_from_skill_description` 即可回到原行为（不影响已有实验与 pipeline）
