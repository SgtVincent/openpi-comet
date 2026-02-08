---
name: "training-log-debugger"
description: "从超长训练日志中提取关键证据并定位根因。遇到训练卡住、超时、SIGSEGV、NCCL/DataLoader 异常时调用。"
---

# Training Log Debugger

## 目标

给定一份超长训练日志（例如 torchrun + tee 的总控台日志，或 per-rank 的 `rank*.log`），快速提取：
- 关键失败点（SIGSEGV、CUDA/NCCL、DataLoader worker died、OOM、deadlock、异常退出）
- 进度与吞吐（step、loss、lr、it/s 波动与中断点）
- 可疑的周期性噪声（例如定时 faulthandler dump）与其影响
- 下一步最小化复现与修复建议（优先给出可落地的代码改动或启动参数）

## 工作方式（必须遵循）

1. **先定位“最后一次明确的信号”**
   - 优先找 `SIGSEGV`/`ChildFailedError`/`exitcode`/`CUDA error`/`NCCL WARN/ERROR`/`DataLoader worker (pid=...) exited`。
   - 记录它出现的“行号范围”和“时间戳”（如有）。

2. **再定位“失败前 1-5 分钟发生了什么”**
   - 取失败点前后各 300~800 行作为上下文，重点看：
     - 正在执行的阶段：dataloader、forward、backward、optimizer.step、checkpoint save、eval 等
     - 是否出现周期性堆栈 dump（`Timeout (0:30:00)!` 一类），避免误判为真实超时

3. **分 rank 交叉验证**
   - 如果有 `checkpoints/<exp>/logs/rank*.log` 或 torchrun `attempt_0/<rank>/stderr.log`：
     - 对比 rank0 与失败 rank 的最后 200 行
     - 若只有某个 rank 报错，倾向于 DataLoader / device mapping / 进程间同步问题

4. **输出必须可操作**
   - 给出“最可能根因（Top 1~3）+ 证据（精确行号/片段）+ 立即可试的修复（命令或代码改动）”。
   - 如果需要进一步证据，明确要检索的关键词/文件/行号范围。

## 工具

### A. 快速摘要脚本（推荐）

运行：

```bash
python .trae/skills/training-log-debugger/scripts/log_inspect.py \
  /path/to/console.log \
  --topk 80
```

常用参数：
- `--since-line N`：从指定行号开始分析（适合增量）
- `--context 60`：每个命中点输出的上下文行数
- `--patterns "SIGSEGV|ChildFailedError|CUDA error|NCCL|DataLoader|Timeout \\(0:30:00\\)"`：自定义关注点

### B. 纯文本检索（当脚本不够用时）

优先做两类检索：
- **失败信号**：`SIGSEGV|ChildFailedError|exitcode|CUDA|NCCL|DataLoader|Killed|OOM|nan|inf`
- **阶段标记**：`Loading weights|Loaded metadata|Training:|save|checkpoint|eval`

## 输出格式（必须）

- **现象**：一句话描述（例如“DDP 训练在 step=XXX 后 rank3 SIGSEGV 导致全局退出”）
- **证据**：列出 3~8 条，包含“文件路径 + 行号范围”
- **根因假设**：Top 1~3（按可能性排序）
- **修复建议**：立即可试（1~3条）+ 长期修复（可选）

