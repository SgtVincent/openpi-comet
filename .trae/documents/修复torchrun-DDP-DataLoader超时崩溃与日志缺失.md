## 背景与现象

本仓库在 8 卡 torchrun DDP 预训练（`scripts/train_pytorch.py`）中曾出现两类“看起来像超时/卡死”的问题：

- **日志不全，难以定位根因**
  - `wandb/latest-run/files/` 不一定有 `output.log`，且 DDP 场景下通常只记录 rank0 的 stdout/stderr，其他 rank 的 traceback / NCCL / elastic 子进程错误经常缺失。
- **训练中途出现 `Timeout (0:30:00)!` + 随后崩溃（甚至 SIGSEGV）**
  - 控制台日志里周期性出现 `Timeout (0:30:00)!`，并伴随线程栈打印；更糟糕时 torchrun 汇总报 `ChildFailedError`，某个 rank 以 `exitcode: -11 (SIGSEGV)` 退出。

本文档合并了两份旧文档，并基于**当前仓库已经落地的代码**更新：说明我们是如何修复 torchrun DDP 下 DataLoader 导致的卡死/超时误判/崩溃，并把日志采集链路补齐到“可追溯、可定位”。

## 一句话结论

DDP 训练的“不稳定”并非单点原因，而是多个因素叠加：

- **(A) 超时堆栈打印被误判为真实 Timeout**（faulthandler 定时 dump 默认 30min）
- **(B) 多 rank 并发初始化数据集/缓存导致锁竞争与等待**（HF datasets parquet cache + filelock）
- **(C) DataLoader worker 侧触发 JAX/XLA 初始化（甚至触达 GPU）**，造成 worker hang 或资源竞争
- **(D) DDP barrier 未绑定 device_id**，在 NCCL 场景存在潜在 hang 风险/警告

修复策略是：**日志可观测性先行** + **离线/缓存路径统一** + **DDP barrier 设备绑定** + **DataLoader worker 防 JAX 侵入** + **给 DataLoader 加超时/worker 运行开关**。

---

## 1. 修复：DDP 日志缺失（让每个 rank 的错误都落盘）

### 1.1 为什么 wandb / latest-run 不可靠

- `wandb/latest-run/files/` 目录本身就不保证有 `output.log`。
- 即便有 `output.log`，在 DDP 下通常也只包含 rank0 的输出；其它 rank 的 crash、NCCL error、elastic 子进程错误常常缺失。

### 1.2 现在的“全量可追溯日志”体系（推荐优先级）

1) **torchrun 自带重定向日志（最全，含 stdout + stderr，且按 rank 分开）**

- 启动命令开启：
  - `--log_dir checkpoints/torchrun_logs/<exp_name>`
  - `--redirects 3 --tee 3`（3=stdout+stderr）
- 产物路径类似：
  - `checkpoints/torchrun_logs/<exp_name>/<run_id>/attempt_0/<rank>/{stdout.log,stderr.log}`
- 启动脚本已对齐：见 [run_vlm2_ddp.sh](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/run_vlm2_ddp.sh)

2) **训练脚本 per-rank 文件日志（更适合看阶段/指标/关键 INFO）**

- 产物路径：
  - `checkpoints/<exp_name>/logs/rank{rank}.log`
- 训练入口已在 DDP 初始化后创建并写入，并通过 `sys.excepthook` 记录未捕获异常：见 [train_pytorch.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/train_pytorch.py)

3) **控制台聚合日志（方便快速扫进度，但不适合作为唯一证据）**

- 产物路径：
  - `checkpoints/console_logs/<exp_name>.log`
- 注意：控制台聚合日志会混杂多 rank 输出，且易受外部终端截断/分页影响。

---

## 2. 修复：排除“假 Timeout”（faulthandler 30min 定时 dump）

### 2.1 现象

日志里出现 `Timeout (0:30:00)!` 并打印线程栈，容易被误解为训练本身超时或平台超时。

### 2.2 根因

这是训练脚本主动调用 `faulthandler.dump_traceback_later(1800)` 的效果：它会在指定时间后**无条件**把堆栈 dump 到 stderr，并不表示程序卡死。

### 2.3 修复（已落地）

改为默认关闭定时 dump，仅在需要排障时显式开启：

- 默认：`OPENPI_FAULT_TIMEOUT_S=0`（不启用定时 dump）
- 需要时：
  - `export OPENPI_FAULT_TIMEOUT_S=1800`
  - `export OPENPI_FAULT_REPEAT=1`（是否周期性重复）

见实现：[train_pytorch.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/train_pytorch.py)

---

## 3. 修复：torchrun DDP 下 DataLoader 卡死/超时崩溃（核心）

### 3.1 典型表征

- 某个 rank 不再产生日志推进（step 不动），其他 rank 可能卡在 barrier / allreduce / dataloader prefetch。
- 之后 torch elastic 报 `ChildFailedError`，甚至某个 rank `SIGSEGV`（exitcode -11）。
- 栈中常出现 `torch/utils/data/dataloader.py` / `multiprocessing` / `_feed` / worker loop 等。

### 3.2 根因组合（我们实际遇到的）

1) **DDP 多 rank 并发触发 HF datasets parquet cache 构建/读写**

- `datasets.load_dataset("parquet", data_files=...)` 会创建/复用本地缓存并使用 filelock。
- 多 rank 并发初始化时容易发生锁竞争与长等待（尤其共享路径/NAS）。

2) **DataLoader worker 侧“间接 import 了 JAX”，触发 XLA backend 初始化**

- worker 进程通过 `spawn` 启动，会重新 import 训练栈依赖。
- 如果 worker 侧触发了 JAX/XLA（甚至触达 GPU），在多进程 + 多线程 + 高 I/O 的组合下更容易卡死或异常。

3) **DDP barrier 未指定 device_id 的潜在风险**

- torch.distributed 会提示 barrier 未指定 device_id，映射不明可能导致 hang。

### 3.3 修复点 1：强制离线 + 统一 HF 缓存位置（避免“隐式下载/锁竞争”）

训练入口在启动时设置：

- `OPENPI_OFFLINE=1` 时默认：
  - `HF_HUB_OFFLINE=1`
  - `HF_DATASETS_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`
- 统一 cache：
  - `HF_HOME=checkpoints/hf_home`
  - `HF_DATASETS_CACHE=checkpoints/hf_datasets_cache`
  - `HUGGINGFACE_HUB_CACHE=checkpoints/hf_home/hub`

见实现：[configure_hf_cache](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/train_pytorch.py#L109-L143)

### 3.4 修复点 2：DDP 下串行化数据集初始化（rank0 先建 cache，其它 rank 等）

避免多 rank 同时构建/读写 datasets cache：

- `rank != 0` 先 barrier
- `rank0` 完成 `create_data_loader()` 后再 barrier 放行

见实现：[build_datasets](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/train_pytorch.py#L199-L209)

### 3.5 修复点 3：barrier 绑定 device_id（降低 NCCL 场景 hang 风险）

新增 `ddp_barrier(local_rank=...)`：优先调用 `dist.barrier(device_ids=[local_rank])`，并在不支持时回退到普通 barrier。

见实现：[ddp_barrier](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/train_pytorch.py#L112-L125)

### 3.6 修复点 4：阻断 worker 侧 JAX/XLA 侵入（只影响 PyTorch + num_workers>0）

在 `framework=="pytorch" and num_workers>0` 时，DataLoader 构造前设置环境变量（让 worker 继承）：

- `JAX_PLATFORMS=cpu`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- `XLA_PYTHON_CLIENT_ALLOCATOR=platform`

见实现：[TorchDataLoader.__init__](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/training/data_loader.py#L340-L344)

### 3.7 修复点 5：给 DataLoader 增加“可控稳定性开关”（长期跑更稳）

新增环境变量控制项：

- `OPENPI_PERSISTENT_WORKERS`：默认 1；建议 DDP 长跑遇到不稳时设为 0
- `OPENPI_DATALOADER_TIMEOUT_S`：默认 0（不启用）；建议先设 300~600，避免 worker 卡死时主进程无限等
- `OPENPI_DATALOADER_PREFETCH_FACTOR`：默认 2

见实现：[TorchDataLoader.__init__](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/src/openpi/training/data_loader.py#L366-L392)

### 3.8 额外防护：默认禁用 TorchDynamo（减少编译/子进程干扰）

当未显式开启 `OPENPI_TORCH_COMPILE_SAMPLE_ACTIONS=1` 时，训练入口默认设置：

- `TORCHDYNAMO_DISABLE=1`

目的：减少 torch.compile / inductor compile worker 对多进程训练稳定性的干扰（尤其排障阶段）。

见实现：同 [configure_hf_cache](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/train_pytorch.py#L109-L143)。

---

## 4. 数据读取自测：快速判断“是不是数据/worker 自身问题”

自测脚本：

- [test_local_b1k_dataset.py](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/test_local_b1k_dataset.py)

示例（只测 dataset/dataloader，不引入 DDP/模型）：

```bash
source .venv/bin/activate
python scripts/test_local_b1k_dataset.py \
  --root /mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos/ \
  --repo_id behavior-1k/2025-challenge-demos \
  --episodes 0:4 \
  --num_samples 2 \
  --test_dataloader \
  --batch_size 2 \
  --num_workers 10 \
  --num_batches 2
```

---

## 5. 推荐启动方式（稳定优先）

直接运行：

- [run_vlm2_ddp.sh](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/scripts/run_vlm2_ddp.sh)

该脚本已包含：

- HF 离线与 cache 固定
- torchrun `--log_dir/--redirects/--tee`
- DataLoader 稳定性环境变量（persistent_workers/timeout/prefetch）

---

## 6. 长日志排障：使用 training-log-debugger Skill

当日志超过万行时，优先用脚本提取“致命信号 + 上下文”：

```bash
python .trae/skills/training-log-debugger/scripts/log_inspect.py \
  checkpoints/console_logs/vlm2_vla_pretrain.log \
  --since-line 1 --topk 40 --context 80 \
  --patterns 'SIGSEGV|ChildFailedError|CUDA error|DataLoader worker|NCCL (WARN|ERROR)'
```

Skill 说明见：

- [training-log-debugger/SKILL.md](file:///mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/repo/openpi-comet/.trae/skills/training-log-debugger/SKILL.md)
