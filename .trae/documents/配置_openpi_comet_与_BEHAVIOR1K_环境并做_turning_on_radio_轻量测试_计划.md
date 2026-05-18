# 计划：配置 openpi-comet / BEHAVIOR-1K Conda 环境并执行 turning_on_radio 轻量测试

## Summary

- 目标是在 `/mnt/bn/behavior-data-hl/chenjunting/miniconda3/` 下重新创建两个 Conda 环境：
  - `openpi-comet-nas`
  - `behavior`
- 配置完成后，使用 checkpoint `/mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet/checkpoints/openpi_comet/pi05-b1kpt12-cs32`，在 `turning_on_radio` 任务上跑一次轻量级端到端测试。
- 用户已确认：
  - 若安装过程中涉及 NVIDIA EULA / Dataset License，可自动接受。
  - “快速测试”采用“轻量评测”级别，而不是完整 10 实例默认评测。

## Current State Analysis

### 1. openpi-comet 环境定义与安装入口

- `openpi-comet` 的环境定义文件是 [environment.yml](file:///mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet/environment.yml#L1-L467)。
- 该文件定义的环境名为 `openpi-comet-nas`，并且 `prefix` 已指向：
  - `/mnt/bn/behavior-data-hl/chenjunting/miniconda3/envs/openpi-comet-nas`
- `openpi-comet` 的安装流程记录在 [installation.md](file:///mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet/installation.md#L1-L49)，核心步骤包括：
  - 通过 `environment.yml` 创建/更新 Conda 环境
  - 在激活环境后执行 `uv pip install -e .`
  - 在同一环境中安装 `bddl3` 与 `OmniGibson[eval]`
  - 将 `src/openpi/models_pytorch/transformers_replace/` 覆盖到目标环境的 `transformers` 包目录

### 2. BEHAVIOR-1K 环境定义与安装入口

- `BEHAVIOR-1K` 的官方安装说明位于 [docs/getting_started/installation.md](file:///mnt/bn/behavior-data-hl/chenjunting/repo/BEHAVIOR-1K/docs/getting_started/installation.md#L16-L109)。
- 官方推荐使用 [setup.sh](file:///mnt/bn/behavior-data-hl/chenjunting/repo/BEHAVIOR-1K/setup.sh#L192-L239) 创建 `behavior` 环境，且明确要求 Python 3.10。
- `setup.sh --new-env` 会直接创建名为 `behavior` 的环境，并安装所选组件。
- 结合评测需求，至少需要 `omnigibson`、`bddl`、`dataset` 和 `eval` 相关组件。

### 3. 当前机器上的环境现状

- 当前 `conda env list` 仅显示：
  - `base`（位于 `/mnt/bn/behavior-data-hl/chenjunting/miniconda3`）
  - 一个不在当前 Conda 根目录下的 `behavior`（位于 `/home/tiger/miniconda3/envs/behavior`）
- 在当前 Conda 根下执行 `conda run -n behavior ...` 会报：
  - `EnvironmentLocationNotFound`
- 这说明当前活跃的 Conda 安装并不能直接使用 `/home/tiger/...` 下的 `behavior` 环境，因此按用户要求在 `/mnt/bn/behavior-data-hl/chenjunting/miniconda3/` 下重建两个环境是必要的。

### 4. 当前机器上的工具与资源现状

- `conda` 可用：`/mnt/bn/behavior-data-hl/chenjunting/miniconda3/bin/conda`
- `uv` 可用：`/home/tiger/.local/bin/uv`
- `mamba` 当前未发现，因此执行时应优先使用 `conda env create/update`
- GPU 可见且当前探测到 4 张卡：
  - GPU 0-3 均为 `NVIDIA L20`

### 5. checkpoint 与评测入口现状

- 目标 checkpoint 目录存在：
  - `/mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet/checkpoints/openpi_comet/pi05-b1kpt12-cs32`
- 目录内已存在 `params/`，符合脚本识别 checkpoint 的条件。
- openpi-comet 已提供一键评测脚本 [run_b1k_eval_parallel_single_task.sh](file:///mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet/scripts/run_b1k_eval_parallel_single_task.sh#L1-L529)。
- 该脚本默认：
  - `TASK_NAME=turning_on_radio`
  - `OPENPI_ENV=openpi-comet-nas`
  - `BEHAVIOR_ENV=behavior`
  - evaluator 调用 `OmniGibson/omnigibson/learning/eval_custom.py`
- 评测思路与 openpi-comet README 的 serve → eval 流程一致，见 [README.md](file:///mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet/README.md#L230-L263)。

## Proposed Changes

### 1. 创建并验证 openpi-comet Conda 环境

**涉及文件 / 入口**

- [environment.yml](file:///mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet/environment.yml#L1-L467)
- [installation.md](file:///mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet/installation.md#L1-L49)
- `src/openpi/models_pytorch/transformers_replace/`

**执行内容**

- 使用当前 Conda 根目录，在 `openpi-comet` 仓库下执行：
  - `conda env create -f environment.yml`
  - 若环境已部分存在，则改为 `conda env update -f environment.yml --prune`
- 激活 `openpi-comet-nas`
- 按安装文档补齐本地 editable 安装：
  - `GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .`
  - `uv pip install -e /mnt/bn/behavior-data-hl/chenjunting/repo/BEHAVIOR-1K/bddl3`
  - `uv pip install -e "/mnt/bn/behavior-data-hl/chenjunting/repo/BEHAVIOR-1K/OmniGibson[eval]"`
- 执行 `transformers` 覆盖补丁：
  - 将 `src/openpi/models_pytorch/transformers_replace/*` 拷贝到该环境对应的 `site-packages/transformers/`

**为什么这样做**

- 这是 `openpi-comet` 文档给出的官方 Conda 安装路径。
- 同时将 `bddl3` / `OmniGibson[eval]` 装进 `openpi-comet-nas`，可以满足本仓库推理服务端与相关脚本运行所需依赖。

### 2. 创建并验证 BEHAVIOR-1K Conda 环境

**涉及文件 / 入口**

- [setup.sh](file:///mnt/bn/behavior-data-hl/chenjunting/repo/BEHAVIOR-1K/setup.sh#L192-L239)
- [docs/getting_started/installation.md](file:///mnt/bn/behavior-data-hl/chenjunting/repo/BEHAVIOR-1K/docs/getting_started/installation.md#L50-L109)
- [project_overview.md](file:///mnt/bn/behavior-data-hl/chenjunting/repo/BEHAVIOR-1K/.trae/documents/project_overview.md#L34-L76)

**执行内容**

- 在 `BEHAVIOR-1K` 仓库下使用官方脚本创建新环境：
  - `./setup.sh --new-env --omnigibson --bddl --dataset --eval --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos`
- 若脚本因已有残留环境而退出，则先删除当前 Conda 根目录下冲突环境后重跑。
- 创建完成后，在 `behavior` 环境中做最小导入校验与基础路径检查。

**为什么这样做**

- 用户明确要求配置 `BEHAVIOR-1K` 自己的 Conda 环境。
- 对于 turning_on_radio 评测，单独准备 `behavior` 环境更符合官方使用方式，也能降低 openpi 推理环境与仿真/评测环境相互污染的风险。
- `--dataset` 与 `--eval` 是为了保证轻量评测可真正落地，而不是只完成代码依赖安装。

### 3. 以轻量模式执行一次 turning_on_radio 端到端测试

**涉及文件 / 入口**

- [run_b1k_eval_parallel_single_task.sh](file:///mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet/scripts/run_b1k_eval_parallel_single_task.sh#L1-L529)
- [MULTI_CHECKPOINT_PARALLEL_EVAL.md](file:///mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet/.trae/documents/MULTI_CHECKPOINT_PARALLEL_EVAL.md#L70-L99)
- checkpoint 目录：
  - `/mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet/checkpoints/openpi_comet/pi05-b1kpt12-cs32`

**执行内容**

- 使用 openpi-comet 自带评测脚本，而不是手工拆成多个终端命令。
- 采用“轻量评测”定义：
  - `TASK_NAME=turning_on_radio`
  - `GPU_IDS=0`
  - `NUM_GPUS=1`
  - `EVAL_INSTANCE_IDS=0,1`
  - `MAX_STEPS=20`
  - `HEADLESS=true`
  - `WRITE_VIDEO=false`
  - `BEHAVIOR_DIR=/mnt/bn/behavior-data-hl/chenjunting/repo/BEHAVIOR-1K`
  - `OPENPI_ENV=openpi-comet-nas`
  - `BEHAVIOR_ENV=behavior`
- 运行：
  - `bash scripts/run_b1k_eval_parallel_single_task.sh /mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet/checkpoints/openpi_comet/pi05-b1kpt12-cs32`

**为什么这样做**

- 该脚本已内建 server + evaluator 联调逻辑，更接近仓库现有最佳实践。
- 2 个实例 + 限制步数，能验证端到端闭环是否可用，同时控制首次评测的资源与时间成本。

### 4. 汇总结果并给出可复现命令

**涉及文件 / 入口**

- `eval_logs/` 输出目录
- [monitor_eval_progress.sh](file:///mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet/scripts/monitor_eval_progress.sh)
- [MULTI_CHECKPOINT_PARALLEL_EVAL.md](file:///mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet/.trae/documents/MULTI_CHECKPOINT_PARALLEL_EVAL.md#L114-L218)

**执行内容**

- 记录最终实际使用的环境创建命令、验证命令与评测命令。
- 汇总：
  - 环境创建位置
  - 关键 import 校验结果
  - eval run 目录
  - server / eval 日志位置
  - metrics 产出情况
- 如评测失败，优先基于日志给出首个可执行修复点。

## Assumptions & Decisions

- 决策：两个环境都安装在 `/mnt/bn/behavior-data-hl/chenjunting/miniconda3/envs/` 下，不复用 `/home/tiger/miniconda3/envs/behavior`。
- 决策：`openpi-comet` 环境按仓库 Conda 文档安装，并额外完成 editable 安装与 `transformers` 覆盖补丁。
- 决策：`BEHAVIOR-1K` 环境按官方 `setup.sh` 安装，使用自动接受条款参数。
- 决策：轻量评测定义为 `2` 个 eval instance、`1` 张 GPU、`MAX_STEPS=20`。
- 假设：本机具备足够磁盘空间与网络条件，以支持 `behavior` 环境及 `dataset` 所需内容安装。
- 假设：`turning_on_radio` 所需评测资产可通过官方安装流程获得，且当前 checkpoint 与 `pi05_b1k-base` 配置兼容。

## Verification Steps

### A. 环境层验证

- `conda env list` 中可见：
  - `/mnt/bn/behavior-data-hl/chenjunting/miniconda3/envs/openpi-comet-nas`
  - `/mnt/bn/behavior-data-hl/chenjunting/miniconda3/envs/behavior`
- `openpi-comet-nas` 中验证：
  - `python -c "import openpi, transformers, omnigibson, bddl; print('ok')"`
- `behavior` 中验证：
  - `python -c "import omnigibson, bddl, torch; print('ok')"`

### B. 补丁层验证

- 在 `openpi-comet-nas` 中验证：
  - `python -c "import transformers; print(transformers.__version__)"`
- 必要时检查补丁后的关键文件是否已落到目标 `site-packages/transformers/`。

### C. 轻量评测验证

- 成功启动至少 1 个 `serve_b1k.py` 进程与 1 个 evaluator 进程。
- `eval_logs/<run_tag>/` 下生成：
  - `server_gpu*_p*.log`
  - `eval_gpu*_p*.log`
  - `eval_gpu*_p*/metrics/*.json`
- 日志中不出现导入失败、环境不存在、checkpoint 解析失败、端口冲突等阻断性错误。

### D. 最终交付验证

- 向用户回报：
  - 两个环境的实际创建结果
  - 安装中遇到的关键问题与处理方式
  - turning_on_radio 轻量测试命令
  - 评测输出目录与核心结果摘要
