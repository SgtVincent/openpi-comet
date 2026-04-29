---
name: "behavior-eval-cleanup"
description: "清理 BEHAVIOR-1K/openpi-comet 评测（serve/eval/isaacsim/kit/omnigibson）残留进程与 GPU 显存占用。Invoke when GPU 显存被残留评测进程占用或评测退出不干净。"
---

# Behavior Eval Cleanup

用于清理 BEHAVIOR-1K / openpi-comet evaluation pipeline 残留的 `serve/eval/isaacsim/kit/omnigibson` 进程，避免显存被长期占用、下次评测端口冲突、以及 GPU util 异常。

## 何时使用

- `nvidia-smi` 显示大量 GPU memory 被占用，但当前任务看似已经结束
- 你刚刚 Ctrl-C/kill 过评测，但显存仍未释放
- 评测脚本异常退出，怀疑 `isaacsim/kit` 没有被回收

## 安全原则（避免误杀）

- 先确认 PID 属于你自己的评测作业：优先用 `pgrep -af` / `ps -fp` 看命令行与用户
- 如果 `nvidia-smi` 里出现 `process_name=[Not Found]` 且 `ps -p <pid>` 查不到：
  - 常见原因是 PID namespace / 权限问题（例如你在容器里看到了宿主机 PID）
  - 需要在“能看到这些 PID 的那台机器/那个终端环境”里执行下面的确认与清理命令

## 步骤 1：取证（建议保留一份快照）

```bash
date
nvidia-smi
nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader
```

按关键词找疑似残留进程（先 dry-run，不做杀进程）：

```bash
pgrep -af 'serve_b1k.py|eval_custom.py|eval_segment.py|isaacsim|kit|omnigibson'
```

如果你已经拿到 PID 列表，进一步确认：

```bash
ps -o user,pid,ppid,etime,cmd -p <PID1>,<PID2>,<PID3>
```

## 步骤 2：温和清理（推荐）

对确认属于你任务的 PID，先发送 SIGTERM：

```bash
kill <PID>
sleep 5
```

仍未退出再升级为 SIGKILL（最后手段）：

```bash
kill -9 <PID>
```

## 步骤 3：“一锅端”清理（高风险，谨慎）

当你确认机器上没有其他人的 Isaac/kit 作业，才使用：

```bash
pkill -f 'serve_b1k.py|eval_custom.py|eval_segment.py|isaacsim|kit|omnigibson'
```

更安全的做法是先列出将被匹配到的进程：

```bash
pgrep -af 'serve_b1k.py|eval_custom.py|eval_segment.py|isaacsim|kit|omnigibson'
```

## 步骤 4：验证显存是否释放

```bash
nvidia-smi
nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader
```

验证要点（以本次实测为例）：

- 清理前：`nvidia-smi --query-compute-apps ...` 里能看到多个 PID，占用 ~8GiB / ~13GiB 等大块显存（`process_name` 可能显示 `[Not Found]`）
- 清理后：`--query-compute-apps` 输出为空，并且每卡显存回落到几百 MiB 的 baseline（例如 ~`334MiB`），通常只剩驱动 / persistence 的占用
- 建议再补一个交叉验证：`nvidia-smi pmon -c 1` 不应再列出 PID

```bash
nvidia-smi pmon -c 1
```

如果显存仍未释放：

- 可能仍有残留进程（重复步骤 1-3）
- 也可能是同机其他用户/容器在跑（检查 `ps -ef | egrep 'isaacsim|kit|omnigibson'` 以及集群调度器作业信息）

如果显存已经释放，但 `nvidia-smi` 仍显示 GPU-Util 偏高且 Process 列表为空：

- 先等 10-30 秒再看（采样/显示延迟偶尔会出现）
- 用 `nvidia-smi dmon -c 3` / `nvidia-smi pmon -c 3` 复核是否真的没有活跃 compute 进程
