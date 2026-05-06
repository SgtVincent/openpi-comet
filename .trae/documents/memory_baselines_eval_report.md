## Memory Baselines Eval Report

### Scope

本报告记录本轮新增的两个 `pi0.5` 兼容显式 memory baseline：

- HAMLET baseline
- MemoryVLA-inspired baseline（Phase A: single-stream MVP）

### HAMLET

- 设计文档：[hamlet_baseline_impl.md](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/.trae/documents/hamlet_baseline_impl.md)
- 新增训练配置：
  - `pi05_hamlet_b1k-pt50_cs32_bs64_lr2.5e-5_5ep`
  - `pi05_hamlet_test`
- 已完成验证：
  - `pytest -q src/openpi/models_pytorch/memory_baselines/hamlet_memory_test.py src/openpi/models_pytorch/pi0_hamlet_test.py`
  - `python scripts/test_pi05_hamlet.py`
  - `python scripts/train_pytorch.py pi05_hamlet_test --num_train_steps 1 ...`
- 当前结果：
  - 模块/模型级测试 `6 passed`
  - smoke script 输出：
    - `prefix_embs (2, 13, 8)`
    - `loss (2, 32, 32)`
    - `sampled_actions (2, 32, 32)`
  - 1-step train smoke：
    - 初始失败：已通过数据加载、模型初始化、前向、反向传播，但阻塞在 `optim.step()` 的 Adam 状态初始化
    - OOM 直接证据：
      - 模型创建后显存约 `7.90GB`
      - `after_backward` 后显存约 `15.28GB`
      - `optim.step()` 还需再申请 `64 MiB` 给 `exp_avg_sq`
      - 同时 GPU 上存在外部进程 `PID 1892955` 占用约 `78.84 GiB`
    - 修复后重跑成功：
      - 方案：`OPENPI_TRAIN_LORA_ONLY=1` + `batch_size=1` + 选择空闲 GPU
      - 成功结果：
        - trainable params = `104,921,089` (`2.82%`)
        - `after_backward` 显存约 `8.34GB`
        - `step=1` 正常完成，`loss=1.7145`

### MemoryVLA

- 设计文档：[memoryvla_baseline_impl.md](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/.trae/documents/memoryvla_baseline_impl.md)
- 新增训练配置：
  - `pi05_memoryvla_b1k-pt50_cs32_bs64_lr2.5e-5_5ep`
  - `pi05_memoryvla_test`
- 已完成验证：
  - `pytest -q src/openpi/models_pytorch/memory_baselines/memoryvla_memory_test.py src/openpi/models_pytorch/pi0_memoryvla_test.py`
  - `python scripts/test_pi05_memoryvla.py`
  - `python scripts/train_pytorch.py pi05_memoryvla_test --num_train_steps 1 ...`
- 当前结果：
  - 模块/模型级测试 `6 passed`
  - smoke script 输出：
    - `prefix_embs (2, 6, 8)`
    - `loss (2, 32, 32)`
    - `sampled_actions (2, 32, 32)`
  - 1-step train smoke：
    - 初始失败模式与 HAMLET 一致：`optim.step()` 时 Adam 状态初始化触发 `CUDA OOM`
    - 修复后重跑成功：
      - 方案：`OPENPI_TRAIN_LORA_ONLY=1` + `batch_size=1` + 选择空闲 GPU
      - 成功结果：
        - trainable params = `16,785,408` (`0.46%`)
        - `after_backward` 显存约 `7.59GB`
        - `step=1` 正常完成，`loss=1.7254`

### CUDA OOM Analysis

根因不是单一因素，而是两个因素叠加：

1. **外部进程占满目标 GPU**
   - 初始失败日志显示：
     - 目标卡总显存约 `79.33 GiB`
     - 外部进程 `PID 1892955` 占用约 `78.84 GiB`
     - 训练进程几乎没有余量完成 optimizer state 初始化

2. **AdamW 在 `optim.step()` 首次初始化状态时会额外申请显存**
   - 即使前向/反向能过，`exp_avg` / `exp_avg_sq` 仍需要为全部可训练参数分配 optimizer state
   - 对 2B 级 backbone，即便用了 LoRA variant，如果仍放开太多参数训练，也会在 `step()` 时爆显存

### OOM Fix Applied

本轮采用了以下可复现修复方案：

1. **选择空闲 GPU**
   - 重新检查 `nvidia-smi`
   - 选择仅有约 `452 MiB` 占用的卡，例如 `CUDA_VISIBLE_DEVICES=2`

2. **限制为 LoRA-only + memory-module-only 训练**
   - 在 [train_pytorch.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/train_pytorch.py) 新增：
     - `OPENPI_TRAIN_LORA_ONLY=1`
   - 作用：
     - 仅训练 `lora` 参数
     - 外加 baseline 自己的 memory 模块参数：
       - `moment_token_pool`
       - `hamlet_memory`
       - `memory_to_prefix_proj`
       - `memoryvla`

3. **使用最小 smoke 批大小**
   - `--batch_size 1`
   - `--num_workers 0`

### Recommended Training Commands

用于再次复现已通过的 1-step smoke：

```bash
CUDA_VISIBLE_DEVICES=2 OPENPI_TRAIN_LORA_ONLY=1 WANDB_DISABLED=1 \
python scripts/train_pytorch.py pi05_hamlet_test \
  --exp_name hamlet_1step_smoke_lora_only \
  --num_train_steps 1 \
  --batch_size 1 \
  --num_workers 0 \
  --overwrite \
  --no-wandb-enabled

CUDA_VISIBLE_DEVICES=2 OPENPI_TRAIN_LORA_ONLY=1 WANDB_DISABLED=1 \
python scripts/train_pytorch.py pi05_memoryvla_test \
  --exp_name memoryvla_1step_smoke_lora_only \
  --num_train_steps 1 \
  --batch_size 1 \
  --num_workers 0 \
  --overwrite \
  --no-wandb-enabled
```

### Dual-GPU Sanity Check (No Concurrent OOM)

为验证 “是否因为同卡同时跑两个 smoke 导致 OOM”，在两个不同 GPU 上分别运行两个 smoke（并发）：

- HAMLET: `CUDA_VISIBLE_DEVICES=2`
- MemoryVLA: `CUDA_VISIBLE_DEVICES=3`
- 均使用：`OPENPI_TRAIN_LORA_ONLY=1`，`batch_size=1`

结果：两个 smoke 都能完成 `step=1`，未再出现 OOM。

### Formal 1-step Smoke Logs (2026-04-15)

#### HAMLET (GPU2)

Command:

```bash
CUDA_VISIBLE_DEVICES=2 OPENPI_TRAIN_LORA_ONLY=1 WANDB_DISABLED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train_pytorch.py pi05_hamlet_test \
  --exp_name hamlet_1step_formal_20260415_170127_gpu2 \
  --num_train_steps 1 --batch_size 1 --num_workers 0 --overwrite --no-wandb-enabled
```

Key logs:

```text
Step 0 (after_model_creation): GPU memory - allocated: 7.89GB, reserved: 7.90GB
OPENPI_TRAIN_LORA_ONLY enabled: trainable params=104921089 (2.82% of total 3721678609)
Step 0 (after_backward): GPU memory - allocated: 8.33GB, reserved: 13.08GB
step=0 ... loss=1.7145 ... step=1
```

#### MemoryVLA (GPU3)

Command:

```bash
CUDA_VISIBLE_DEVICES=3 OPENPI_TRAIN_LORA_ONLY=1 WANDB_DISABLED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train_pytorch.py pi05_memoryvla_test \
  --exp_name memoryvla_1step_formal_20260415_170127_gpu3 \
  --num_train_steps 1 --batch_size 1 --num_workers 0 --overwrite --no-wandb-enabled
```

Key logs:

```text
Step 0 (after_model_creation): GPU memory - allocated: 7.54GB, reserved: 7.55GB
OPENPI_TRAIN_LORA_ONLY enabled: trainable params=16785408 (0.46% of total 3633542928)
Step 0 (after_backward): GPU memory - allocated: 7.58GB, reserved: 8.08GB
step=0 ... loss=1.7254 ... step=1
```

如果再次遇到 OOM，建议按优先级处理：

1. 先换到空闲 GPU
2. 保持 `OPENPI_TRAIN_LORA_ONLY=1`
3. 保持 `batch_size=1`
4. 如仍冲突，再关闭其他占卡进程或换到完全空闲机器

### Make Pizza SFT Configs

已新增 `make_pizza` 任务上的两个 baseline SFT config：

- `pi05_hamlet_b1k-make_pizza_lr1e-4_5ep_sft`
- `pi05_memoryvla_b1k-make_pizza_lr1e-4_5ep_sft`

位置：

- [sft_make_pizza_config.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/training/sft_make_pizza_config.py)

对应训练脚本：

- [run_pi05_sft_make_pizza_1ep.sh](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/run_pi05_sft_make_pizza_1ep.sh)
- [run_pi05_hamlet_sft_make_pizza.sh](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/run_pi05_hamlet_sft_make_pizza.sh)
- [run_pi05_memoryvla_sft_make_pizza.sh](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/run_pi05_memoryvla_sft_make_pizza.sh)

示例：

```bash
CONFIG_NAME=pi05_hamlet_b1k-make_pizza_lr1e-4_5ep_sft \
  bash scripts/run_pi05_sft_make_pizza_1ep.sh

CONFIG_NAME=pi05_memoryvla_b1k-make_pizza_lr1e-4_5ep_sft \
  bash scripts/run_pi05_sft_make_pizza_1ep.sh
```

或者用更“集群友好”的固定脚本名（默认 5ep，可用 `NUM_TRAIN_EPOCHS` 覆盖）：

```bash
bash scripts/run_pi05_hamlet_sft_make_pizza.sh
bash scripts/run_pi05_memoryvla_sft_make_pizza.sh
```

如果只是做 baseline 是否可训练的快速验证，建议额外加：

```bash
export OPENPI_TRAIN_LORA_ONLY=1
export BATCH_SIZE_PER_GPU=1
export NUM_WORKERS=0
```

### Max Batch Size Sweep (A100/A800 80G, Full Finetune)

本节记录在当前集群 80G 卡上，使用 `make_pizza` SFT config、关闭 LoRA-only（`OPENPI_TRAIN_LORA_ONLY=0`）时，
通过 1-step smoke（`--num_train_steps 1`）测得的最大 `--batch_size_per_gpu`。

前置：使用代理下载并缓存 tokenizer 到项目根目录 `.cache`（避免访问 `storage.googleapis.com` 超时）。

- proxy: `http://sys-proxy-rd-relay.byted.org:8118`
- cached: `openpi-comet/.cache/openpi/big_vision/paligemma_tokenizer.model`

通用运行参数：

- `OPENPI_DATA_HOME=${REPO_ROOT}/.cache/openpi`
- `OPENPI_LOAD_DATASET_NUM_PROC_CAP=8`（避免 HF dataset cache 构建进程风暴）
- `--force-load-cache`（复用 HF datasets cache）

#### Results

- HAMLET (`pi05_hamlet_b1k-make_pizza_lr1e-4_5ep_sft`)
  - `batch_size_per_gpu=11`: PASS（`peak_allocated ~83.66GB`）
  - `batch_size_per_gpu=12`: FAIL（`CUDA OOM` in backward; `Tried to allocate ~484MiB`）
- MemoryVLA (`pi05_memoryvla_b1k-make_pizza_lr1e-4_5ep_sft`)
  - `batch_size_per_gpu=11`: PASS（`peak_allocated ~82.81GB`）
  - `batch_size_per_gpu=12`: FAIL（`CUDA OOM` in backward; `Tried to allocate ~482MiB`）

### Architecture Bug Fixes (2026-04-17, updated 2026-04-29)

基于 [.trae/documents/hamlet_memoryvla_fix_plan.md](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/.trae/documents/hamlet_memoryvla_fix_plan.md)
的诊断，本轮对两个 baseline 做了最小但关键的实现修复。以下内容已按当前工作区代码状态更新：

1. **去掉训练态额外的完整 VLM forward**
   - `Pi05WithHamlet.embed_prefix()` 不再调用额外的 `paligemma_with_expert.forward()` 编码 moment token
   - `Pi05WithMemoryVLA.embed_prefix()` 不再调用额外的 prefix-only `paligemma_with_expert.forward()`
   - 两者都改为基于 `prefix_embs` 的 masked summary + 轻量线性投影生成 memory 输入

2. **去掉训练态 session-state 写回，但当前仍仅在推理态更新 runtime memory**
   - 当前代码里 `Pi05WithHamlet` 和 `Pi05WithMemoryVLA` 仍使用 `update_memory=not self.training`
   - 也就是说，训练态不会跨 batch 持久化 runtime memory；推理态会更新 memory
   - 仅在 `not self.training` 且有 active session 时保存 `_session_memory_state`

3. **将 runtime history / memory bank 改为固定 buffer + 原地写入**
   - HAMLET: `history_buffer` + `history_write_index`
   - MemoryVLA: `memory_bank` + `memory_write_index`
   - 避免每步 `torch.cat` 导致的大量新 tensor 分配

4. **修复 MemoryVLA 的 DDP/static-graph 与 autograd 细节问题**
   - 当 memory 为空时，`GatedMemoryFusion` 仍走同一条参数路径，避免 `OPENPI_DDP_STATIC_GRAPH=1` 下参数使用模式跨 step 变化
   - `retrieve()` 改为基于 detached snapshot，避免同一次 forward 中 `update()` 原地改写 bank 触发 backward version mismatch

5. **修复 HAMLET 注册加载推理时的 dtype mismatch**
   - `Pi05WithHamlet._encode_current_moment_tokens()` 现在会先把 `prefix_summary` cast 到 `prefix_summary_proj.weight.dtype`
   - 再通过 `prefix_summary_proj` 做线性投影，最后 cast 回 `prefix_embs.dtype`
   - 该修复解决了 `test_registered_model_loading_inference` 中出现的 `mat1 and mat2 must have the same dtype` 错误

#### 代码验证

```bash
pytest -q \
  src/openpi/models_pytorch/memory_baselines/hamlet_memory_test.py \
  src/openpi/models_pytorch/pi0_hamlet_test.py \
  src/openpi/models_pytorch/memory_baselines/memoryvla_memory_test.py \
  src/openpi/models_pytorch/pi0_memoryvla_test.py

python scripts/test_pi05_hamlet.py
python scripts/test_pi05_memoryvla.py
```

结果：

- `pytest`: 12 passed
- `test_pi05_hamlet.py`: PASS
- `test_pi05_memoryvla.py`: PASS

#### Registered Checkpoint Load + Inference Smoke (2026-04-29)

使用 `scripts/test_registered_model_loading_inference.py` 对真实 `make_pizza` SFT checkpoint 做注册加载与单步推理验证：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/test_registered_model_loading_inference.py \
  --config pi05_hamlet_b1k-make_pizza_lr1e-4_5ep_sft \
  --ckpt-dir outputs/checkpoints/pi05_hamlet_b1k-make_pizza_lr1e-4_5ep_sft_baseckpt_5ep_20260418_005748 \
  --device cuda:0 \
  --default-prompt "make pizza"

CUDA_VISIBLE_DEVICES=0 python -u scripts/test_registered_model_loading_inference.py \
  --config pi05_memoryvla_b1k-make_pizza_lr1e-4_5ep_sft \
  --ckpt-dir outputs/checkpoints/pi05_memoryvla_b1k-make_pizza_lr1e-4_5ep_sft_baseckpt_5ep_20260418_162427 \
  --device cuda:0 \
  --default-prompt "make pizza"
```

结果：

- HAMLET:
  - 初始状态：checkpoint 可加载，但 `policy.infer` 失败，报错 `RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float`
  - 修复后：PASS
  - 解析到的 checkpoint step: `299795`
  - 推理输出：`actions_shape=[32, 23]`
  - 单步推理耗时：`infer_ms ~= 702.09`
- MemoryVLA:
  - PASS
  - 解析到的 checkpoint step: `149895`
  - 推理输出：`actions_shape=[32, 23]`
  - 单步推理耗时：`infer_ms ~= 892.68`

### 8-GPU / 10-step Full-Finetune Regression

在与先前失败完全一致的设定下重新测试：

- `world_size=8`
- `batch_size_per_gpu=8`
- `num_train_steps=10`
- `OPENPI_DDP_STATIC_GRAPH=1`
- `OPENPI_TRAIN_LORA_ONLY=0`

#### HAMLET 修复后结果

日志：[stderr.log](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/checkpoints/torchrun_logs/hamlet_8gpu_10step_fix_20260417_0033/e3a8453a-8b0c-4e52-8433-6b0cc44959a6_15emam9w/attempt_0/0/stderr.log)

- 模型创建后：`allocated=7.91GB`
- `step 0 after_backward`: `peak_allocated=25.76GB`, `peak_reserved=26.05GB`
- `step 1 after_backward`: `peak_allocated=37.37GB`, `peak_reserved=37.40GB`
- `step=10` 正常完成，未出现 OOM

对比修复前：

- 修复前同设定在 `step 1` 前向时 OOM，`peak_allocated ~72.22GB`
- 修复后 10-step 峰值约 `37.37GB`

#### MemoryVLA 修复后结果

日志：[stderr.log](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/checkpoints/torchrun_logs/memoryvla_8gpu_10step_fix3_20260417_0335/41b46a57-b3f4-4c68-b0e4-0c851101ca1a_p3nakq1r/attempt_0/0/stderr.log)

- 模型创建后：`allocated=7.56GB`
- `step 0 after_backward`: `peak_allocated=24.70GB`, `peak_reserved=25.00GB`
- `step 1 after_backward`: `peak_allocated=35.61GB`, `peak_reserved=35.64GB`
- `step=10` 正常完成，未出现 OOM / static-graph / autograd inplace 错误

对比修复前：

- 修复前同设定先后触发：
  - `step 1 backward` 的 OOM
  - `DDP static_graph` 参数使用模式变化错误
  - bank 原地更新导致的 backward version mismatch
- 修复后 10-step 峰值约 `35.61GB`

### Combined Verification

本轮在 `openpi-comet-nas` 环境下已通过：

```bash
pytest -q \
  src/openpi/models_pytorch/memory_baselines/hamlet_memory_test.py \
  src/openpi/models_pytorch/pi0_hamlet_test.py \
  src/openpi/models_pytorch/memory_baselines/memoryvla_memory_test.py \
  src/openpi/models_pytorch/pi0_memoryvla_test.py

python scripts/test_pi05_hamlet.py
python scripts/test_pi05_memoryvla.py

CUDA_VISIBLE_DEVICES=0 python -u scripts/test_registered_model_loading_inference.py \
  --config pi05_hamlet_b1k-make_pizza_lr1e-4_5ep_sft \
  --ckpt-dir outputs/checkpoints/pi05_hamlet_b1k-make_pizza_lr1e-4_5ep_sft_baseckpt_5ep_20260418_005748 \
  --device cuda:0 \
  --default-prompt "make pizza"

CUDA_VISIBLE_DEVICES=0 python -u scripts/test_registered_model_loading_inference.py \
  --config pi05_memoryvla_b1k-make_pizza_lr1e-4_5ep_sft \
  --ckpt-dir outputs/checkpoints/pi05_memoryvla_b1k-make_pizza_lr1e-4_5ep_sft_baseckpt_5ep_20260418_162427 \
  --device cuda:0 \
  --default-prompt "make pizza"
```

### Known Gaps

- 两个 baseline 当前都没有做 episode-level sequential training 改造。
- 两个 baseline 当前训练态都不会跨 batch 持久化 runtime memory。
- `MemoryVLA` 当前只完成 Phase A 单流实现，尚未实现双流 cognitive branch。

### Next Step

- 使用新增 `make_pizza` SFT config 运行：
  - `CONFIG_NAME=pi05_hamlet_b1k-make_pizza_lr1e-4_5ep_sft bash scripts/run_pi05_sft_make_pizza_1ep.sh`
  - `CONFIG_NAME=pi05_memoryvla_b1k-make_pizza_lr1e-4_5ep_sft bash scripts/run_pi05_sft_make_pizza_1ep.sh`
- 再决定是否进入：
  - HAMLET with sequential training
  - MemoryVLA dual-stream upgrade
