---
name: "skill-segment-eval"
description: "运行或排查 BEHAVIOR-1K / openpi-comet 的 skill/segment eval。优先走当前 canonical per-skill eval 路径（segment_predicates），仅在需要对比旧语义时回到 legacy BDDL predicate 模式。"
---

# Skill Segment Eval

## 何时使用

当用户提到下面任一类需求时使用本 skill：

- 跑 `eval_segment.py`
- 跑 per-skill eval / segment eval / skill metric eval
- 检查 `segment_predicates` 结果、视频、trace、metrics
- 排查 `predicate_subgoal` / `segment_predicates` 差异
- 运行 openpi-comet 的 skill eval launcher

## 先给用户的默认判断

先明确区分两条路径：

1. 当前正式 canonical 路径：

```text
segment_level=skill
success_mode=segment_predicates
```

2. legacy 调试路径：

```text
success_mode=predicate_subgoal|predicate_progress|state_match
```

除非用户明确说“看 BDDL delta / q_score / predicate_subgoal”，否则默认走第 1 条。

## 代码与文档入口

### BEHAVIOR-1K runtime

- `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_segment.py`
- `BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs/eval_segment_config.yaml`
- `BEHAVIOR-1K/.trae/documents/skill_eval_user_guide.md`
- `BEHAVIOR-1K/.trae/documents/behavior_eval_runtime_guide.md`

### openpi-comet orchestration

- `openpi-comet/scripts/run_skill_metric_multinode_sweep.py`
- `openpi-comet/scripts/run_skill_eval_single_node_8gpu.sh`
- `openpi-comet/.trae/documents/behavior_eval_orchestration_guide.md`

## 默认执行顺序

### 1. 先核对环境与路径

- repo root 是否正确：
  - `BEHAVIOR-1K` runtime 代码在 `/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K`
  - orchestration 代码在 `/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet`
- Python 不要信任当前 shell 的 `which python`；优先显式使用：
  - `/mnt/bn/navigation-hl/mlx/users/chenjunting/miniconda3/envs/behavior/bin/python`

### 2. 先决定是单 segment debug 还是批量 launcher

- 单条样例 / 单段 debug：直接跑 `eval_segment.py`
- 正式批量、多 GPU、长跑：优先 `run_skill_metric_multinode_sweep.py`

### 3. 若是 websocket policy，必须做身份校验

显式检查：

- `model.expected_task_name`
- `model.expected_task_prompt_sha256`
- `model.expected_server_run_id`
- `model.expected_server_token`

不要只看 `healthz`。

## 常用命令模板

### 单 segment canonical 调试

```bash
cd /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K

PYTHONPATH=/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson \
/mnt/bn/navigation-hl/mlx/users/chenjunting/miniconda3/envs/behavior/bin/python \
  OmniGibson/omnigibson/learning/eval_segment.py \
  policy=websocket \
  task.name=<TASK_NAME> \
  demo_data_path=/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos \
  rawdata_path=/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-rawdata \
  demo_id=<DEMO_ID> \
  segment_level=skill \
  segment_idx=<SEGMENT_IDX> \
  success_mode=segment_predicates \
  log_path=<LOG_PATH> \
  headless=true \
  write_video=true \
  segment_predicate_dump_trace=true
```

### 批量 launcher

```bash
cd /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet

python -u scripts/run_skill_metric_multinode_sweep.py \
  --mode launch \
  --out-dir <OUT_DIR> \
  --node-rank 0 \
  --num-nodes 1 \
  --gpus-per-node 8 \
  --config-name <CONFIG_NAME> \
  --policy-backend torch \
  --ckpt-dir <CKPT_DIR> \
  --behavior-dir /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K \
  --demo-data-path /mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos \
  --rawdata-path /mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-rawdata \
  --max-samples-per-skill 1 \
  --write-video \
  --segment-predicate-dump-trace
```

## legacy 使用边界

只有在这些场景才切回 legacy：

- 用户明确要求 `predicate_subgoal`
- 要解释 BDDL grounding / `q_score`
- 要和老日志、老文档对齐

此时可以引用：

- `BEHAVIOR-1K/.trae/documents/bddl_predicate_segment_eval_interface.md`
- `BEHAVIOR-1K/.trae/documents/bddl_predicate_segment_eval_usage.md`
- `BEHAVIOR-1K/.trae/skills/bddl_predicate_segment_eval.md`

## 验证标准

至少给出下面一种证据：

- 直接运行出的 `metrics/*.json`
- `segment_eval.log`
- `videos/*.mp4` / `review/*.png`
- 或者当前代码上的核心逻辑验证结果

当前已知可复用的轻量验证结论：

- `BEHAVIOR-1K/main` 与 `origin/main` 在 segment eval 相关文件上已对齐
- `OmniGibson/tests/test_segment_predicate_eval.py` 的 17 个核心测试函数在当前 merged 代码上已跑通

## 注意事项

- master 节点不一定有 GPU；不要在没有合适 runtime 的地方声称“已完成完整 Isaac smoke”。
- 多个独立 GPU 实验要显式设 `CUDA_VISIBLE_DEVICES`，避免串卡。
- 如果需要清残留评测进程，参考 `openpi-comet/.trae/skills/behavior-eval-cleanup/SKILL.md`，不要直接 broad kill。
