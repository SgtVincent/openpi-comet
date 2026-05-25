# Persistent Skill Eval 框架实现说明

> 适用范围：`openpi-comet/scripts/run_skill_metric_multinode_sweep.py`
>
> - `openpi-comet/scripts/persistent_skill_eval_worker.py`
> - `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_segment.py`
> - `openpi-comet/scripts/run_skill_eval_single_node_8gpu.sh`
>
> OpenSpec 变更：`openspec/changes/persistent-skill-eval-framework/`

## 1. 背景：为什么要做 persistent env

旧的 `run_skill_metric_multinode_sweep.py --mode worker` 路径采用「每个 segment 起一个 Python 子进程」的方式：

- 每段 rollout 都要重新 `import omnigibson` → 拉起 Isaac Sim → 加载场景 → 加载 task instance → 启动 policy server。
- 实测中位时间 **rollout ≈ 106 s**，**setup ≈ 688 s**，setup 占比约 **86%**。
- 对于 34 skill × 32 segment = 1088 段的 sweep，setup 累计要烧掉数十个 GPU·小时。

*观察到 manifest 里的 segment 是按 task 分组的（同一个* *`task_name`* *下有几十段），如果同一个 GPU 上**复用一份已加载的 env / task / policy server**，setup 就只在「换 task」时付一次，大头开销可以摊掉。Persistent eval 框架就是把这件事工程化。*

## 2. 总体架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│  run_skill_eval_single_node_8gpu.sh   (EVAL_MODE=persistent 默认)        │
│        │                                                                  │
│        ▼                                                                  │
│  run_skill_metric_multinode_sweep.py --mode launch-persistent             │
│        │                                                                  │
│        ├─ build_manifest()                                                │
│        ├─ materialize_persistent_jobs()    ← task-affinity 调度           │
│        ├─ subprocess.Popen × 8  →  persistent_skill_eval_worker.py        │
│        │       (每个 GPU 一个长寿命进程, gpu_id / port_base / rank)      │
│        ├─ append({"action":"shutdown"}) × 8  ← 队列结束                   │
│        └─ merge_results()                  ← worker_*.jsonl ∪ persistent_*│
│                                                                          │
│  worker_jobs/persistent_worker_{rank}.jobs.jsonl  (launcher → worker)     │
│  worker_results/persistent_worker_{rank}.jsonl    (worker → launcher)     │
│  worker_status/persistent_worker_{rank}.jsonl     (心跳/事件流)           │
└──────────────────────────────────────────────────────────────────────────┘
```

各角色单一职责：

| 组件                                                                         | 职责                                                                                             |
| -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `run_skill_eval_single_node_8gpu.sh`                                       | 单机 8 卡入口；通过 `EVAL_MODE` 选择 `persistent` / `process_per_segment` 路径，导出 persistent worker 的环境变量。 |
| `run_skill_metric_multinode_sweep.py` (launcher)                           | 构建 manifest、做 task-affinity 调度、拉起 8 个 worker、下发 shutdown、聚合结果。                                 |
| `persistent_skill_eval_worker.py`                                          | 每个 GPU 一个长寿命进程；保持 OmniGibson + policy server 常驻，串行消费 JSONL 队列。                                 |
| `eval_segment.py` 中的 `run_segment_on_env()` / `_reconfigure_for_segment()` | 把单段评测的「reset → rollout → 判定 → 写 metrics/视频」抽成可复用函数，在同一个 evaluator 上重入。                         |

## 3. eval\_segment.py 的可复用化（Task 1）

`BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_segment.py` 在保持 CLI 行为不变的前提下，新增三段：

- `_reconfigure_for_segment(evaluator, sample) -> Dict[str, Any]`
  - 只允许在**同一个 task\_name** 内复用 evaluator；切 task 必须重新构造（强制约束写在前面，越界会抛 `RuntimeError`）。
  - 改写 `cfg.demo_id / cfg.segment_idx / cfg.segment_max_steps / cfg.log_path / cfg.write_video / cfg.success_mode` 等 per-segment 字段。
  - 关闭并清理上一段残留的 `current_rawdata_hdf5` / `current_primitive_state_cache`，避免跨 demo 复用错误的 HDF5 句柄。
  - 创建输出目录并（按需）打开新的 video writer；旧 writer 由 `Evaluator.video_writer` setter 自动 flush。
- `run_segment_on_env(evaluator, sample, *, write_metrics=True) -> Dict[str, Any]`
  - 单段 rollout 主体：调用 `_reconfigure_for_segment`，再走原来 `run_single_segment(...)` 的内层循环，最后做 video tail padding、写 metrics、flush video。
  - `finally` 里**永远**关闭 rawdata 句柄、清空 primitive state 缓存，让下一段从干净状态开始。
- `_build_sample_from_cli_config(config) -> Dict[str, Any]`
  - 把原来 `__main__` 里读 cfg → 拼 sample 的逻辑抽出来，方便 CLI 路径和 worker 路径共用同一个字段约定。

`__main__` 简化为：构建 sample → `with SubTaskEvaluator(config) as evaluator:` → `run_segment_on_env(evaluator, sample, write_metrics=True)`。CLI 行为与改造前一致（同一份 metrics JSON、同一个视频文件路径）。

## 4. Persistent worker 进程（Task 2）

`openpi-comet/scripts/persistent_skill_eval_worker.py` 是每张 GPU 上的长寿命进程，关键设计：

### 4.1 启动顺序：先 gm flag，再 import env

`_apply_gm_flags()` 在**任何** `omnigibson` import 前先设置：

```python
gm.HEADLESS = True
gm.GUI_VIEWPORT_ONLY = True
gm.RENDER_VIEWER_CAMERA = False
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
```

这五个 flag 是 headless eval 的基本盘，写错顺序的话 OmniGibson 会按默认值起渲染窗口，setup 时间和显存都会膨胀。

### 4.2 JSONL 队列 + 游标

- 输入：`worker_jobs/persistent_worker_{rank}.jobs.jsonl`，由 launcher 单写、worker 单读。
- 用 `cursor` 记录已消费的行数，每轮 `_drain_jobs()` 增量拉新行；不依赖文件锁，依赖「launcher 只 append、不重写」的不变量。
- 支持的 action：`assign`（带 `sample` payload）、`shutdown`；其他 action 会被记一条 `unknown_action` 状态事件后丢弃。

### 4.3 Policy server 生命周期与 task 切换

`_ensure_task_loaded(task_name, sample)` 是 hot path：

1. 如果 `self._loaded_task_name == task_name` 且 evaluator 还在，**直接复用**，零 setup。
2. 否则发 `task_switching` 事件、`_unload_evaluator()`（调用 `SubTaskEvaluator.__exit__`）、`_stop_server()`（SIGTERM + SIGKILL deadline + `wait_for_port_free`）。
3. 重新 `_start_server()`：
   - `find_free_port(port_base, stride=gpus_per_node)` 抢一个本卡专属端口；
   - `build_server_identity(...)` 生成 `server_run_id` / `server_token` / `task_prompt_sha256`；
   - `start_server(...)` 拉起 `serve_b1k.py`，日志写到 `server_logs/`；
   - `wait_for_server_proc(...)` 用 metadata-bearing health 检查（不是裸 healthz）。
4. 用 `_build_eval_cfg(...)` 通过 `hydra.initialize_config_dir + compose` 构造 `eval_segment_config.yaml`，注入 `model.expected_*` 四元组（防止打到错的 server）以及当前 launcher 传入的 runtime knobs（默认 `partial_scene_load=false`、`skip_intermediate_obs_in_chunk=true`）。
5. `SubTaskEvaluator(cfg).__enter__()` 加载场景 + task instance；之后这一份 evaluator 会被同 task 的所有 segment 复用。

「同一个 task 内的所有 segment 复用 evaluator」就是节省 setup 的核心机制。

### 4.4 单段执行 `_run_assignment(sample)`

1. **Resume 短路**：如果 `raw/<task>/demo_<id>/skill_<NNN>/metrics/*.json` 已经存在，直接调 `load_metrics_row(...)` 拼一行 `resume_hit=true` 写回 `worker_results/`，不走 rollout。
2. 否则 `_ensure_task_loaded()` 后，调用 `run_segment_on_env(self._evaluator, sample, write_metrics=True)`。
3. `finally` 里聚合 row（`worker_kind="persistent"`、`worker_rank` / `gpu_id` / `segment_runtime_s` / `dynamic_max_steps`），append 到 `worker_results/persistent_worker_{rank}.jsonl`。
4. 所有事件（`segment_start` / `segment_done` / `segment_resume_hit` / `segment_error`）写到 `worker_status/persistent_worker_{rank}.jsonl`，方便 launcher 端做诊断。

### 4.5 Soft restart：用 `os.execv` 清白起步

Isaac Sim 长跑会有显存碎片、stage 残留、PhysX cache 等不易释放的状态。worker 内置两种触发：

- 计数触发：`segments_since_boot >= PERSISTENT_WORKER_MAX_SEGMENTS_BEFORE_RESTART`（默认 64）。
- 异常触发：`_run_assignment` 抛任何未捕获异常 → 立刻发 `segment_error` 事件 → soft restart。

`_soft_restart()` 的清理顺序：发 `restart` 事件 → `_unload_evaluator()` → `_stop_server()` → `os.execv(self._original_executable, ...)`。`os.execv` 用同样的 argv 替换当前进程，新进程会重新读 `worker_jobs/...jobs.jsonl`，从 `cursor=0` 开始扫但用 `_done_keys`（从 `worker_results/` 已有 `job_key` 重建）短路掉已完成的段，所以**重启不会丢段也不会重复算**。

### 4.6 清退路径

- `{"action":"shutdown"}` → 完成在飞 segment（其实拿到 shutdown 时一般不在飞）→ `_unload_evaluator() / _stop_server()` → 写一条 `state: shutdown` 的最终 heartbeat → 退出 0。
- Launcher 死掉：`_launcher_alive()` 通过 `os.kill(launcher_pid, 0)` 探测，pid 失踪即发 `launcher_gone` 然后清退，避免成为孤儿。
- `KeyboardInterrupt` → 同样走清退；`SIGPIPE` 被 `SIG_DFL` 重置防止 launcher 关 stdout 时把自己打挂。

### 4.7 状态可观测性

`worker_status/persistent_worker_{rank}.jsonl` 是诊断的主要入口，事件包括：`started / heartbeat / server_started / server_ready / server_stopped / task_loaded / task_unloaded / task_switching / segment_start / segment_done / segment_error / segment_resume_hit / restart / launcher_gone / shutdown_requested / interrupted`。Heartbeat 默认 60 s 一次（`PERSISTENT_WORKER_HEARTBEAT_S`），带 `loaded_task` / `segments_since_boot` / `state`。

## 5. Launcher：task-affinity 调度与 launch-persistent 模式（Task 3）

`run_skill_metric_multinode_sweep.py` 新增 `--mode launch-persistent` 与 `--mode persistent-worker`，旧的 `--mode launch` / `--mode worker` 路径**保留作为 rollback**。

### 5.1 任务亲和调度 `materialize_persistent_jobs(args, jobs)`

1. 用现有 `build_worker_assignments(...)` 把 manifest 按 `task_name` 分组，单组超过 `2 * mean_group_size` 时拆给多张 GPU；最终保证「同 GPU 上同 task 的段连续」，从而最大化 evaluator 复用率。
2. 对每个 `(task_name, demo_id, skill_idx)` 预计算 `dynamic_max_steps`（沿用 `get_dynamic_max_steps(frame_duration)` 的公式），落到 jobs 行里，worker 不再重复算。
3. `--resume` 开启时，扫 `raw/<task>/demo_<id>/skill_<NNN>/metrics/*.json`，已有的 `job_key` 直接跳过不下发。
4. 把每张卡的队列写到 `worker_jobs/persistent_worker_{rank}.jobs.jsonl`，每行是 `{"action":"assign", "sample": {...}}`，字段与旧 `run_segment_eval(sample=...)` 兼容。

### 5.2 启动 8 个 worker：`launch_node_persistent(args)`

- 每张卡一个 `subprocess.Popen`，`env` 由 `_persistent_worker_env(args, gpu_id, port)` 构建：
  - `CUDA_VISIBLE_DEVICES = str(gpu_id)`；
  - `OMNIGIBSON_APPDATA_PATH` 按 `OMNIGIBSON_APPDATA_SCOPE`（`gpu` / `gpu_port` / `run_gpu_port`）做隔离，避免多进程共享 Omniverse appdata 造成串扰；
  - `PYTHONPATH` 块拼上 `BEHAVIOR-1K/OmniGibson/`、`bddl3/`、`openpi-comet/src/` 等仓内路径；
  - `http_proxy` / `https_proxy` 透传。
- 命令行：`python persistent_skill_eval_worker.py --out-dir ... --worker-rank R --gpu-id G --port-base ... --gpus-per-node 8 --launcher-pid <self> [--resume] [--write-video] [--segment-predicate-dump-trace] [--ckpt-dir ...] [--config-name ...]`。
- 日志重定向到 `launcher_logs/persistent_worker{rank}.log`。
- 所有 segment 下发完后，给每个 GPU 的 jobs 文件 append `{"action":"shutdown"}`，然后 `wait()` 到超时（`PERSISTENT_WORKER_SHUTDOWN_TIMEOUT`，默认 900 s），再依次 SIGTERM → SIGKILL。

### 5.3 结果聚合

`merge_results()` 现在 glob `worker_results/worker_*.jsonl` 与 `worker_results/persistent_worker_*.jsonl` 两类文件，按 `job_key` 去重（persistent 与 legacy 的 schema 一致）。`summary.json` / `stop_snapshots/` / `segment_predicate_audit/` 这些下游脚本不需要任何改动，它们只看聚合后的行集合。

## 6. 单机 8 卡入口脚本（Task 4）

`openpi-comet/scripts/run_skill_eval_single_node_8gpu.sh` 在保留全部旧 knob（`MAX_SAMPLES_PER_SKILL` / `GPUS_PER_NODE` / `LOCAL_GPU_IDS` / `CKPT_DIR` / `CONFIG_NAME` / `WRITE_VIDEO` / `SEGMENT_PREDICATE_DUMP_TRACE` / `OMNIGIBSON_APPDATA_SCOPE`）的前提下，新增：

| 环境变量                                            | 默认值          | 含义                                                                                   |
| ----------------------------------------------- | ------------ | ------------------------------------------------------------------------------------ |
| `EVAL_MODE`                                     | `persistent` | `persistent` 走 `--mode launch-persistent`；`process_per_segment` 退回旧 `--mode launch`。 |
| `PERSISTENT_WORKER_MAX_SEGMENTS_BEFORE_RESTART` | `64`         | 单 worker 累计跑多少段后强制 soft restart。                                                     |
| `PERSISTENT_WORKER_HEARTBEAT_S`                 | `60`         | Heartbeat 写入间隔。                                                                      |
| `PERSISTENT_WORKER_TASK_RELOAD_TIMEOUT_S`       | `1800`       | 单次 task switch（含 server 重起 + 场景重加载）的总超时。                                             |
| `PERSISTENT_WORKER_SHUTDOWN_TIMEOUT`            | `900`        | Launcher 等 worker 优雅退出的最长时间。                                                         |

脚本启动时会把当前 `eval_mode` 与上述 knob 打到日志，方便事后审计。

## 7. 复用率：persistent 究竟省了什么

理论分析：

- **Per-segment 子进程模式**：1088 段 × (688 s setup + 106 s rollout) ≈ **240 GPU·h**（8 卡并行约 30 小时）。
- **Persistent 模式**：8 张卡，每个 task 只付 1 次 setup；按 34 task × 8 GPU 上限估，最多 **34 + 几次 soft-restart + 异常重启 ≈ 40 次 setup**，rollout 总量不变。理论开销 ≈ 40 × 688 + 1088 × 106 ≈ **35 GPU·h**（8 卡并行约 4.5 小时），**3–6× 加速**。
- 实际加速取决于：
  1. task-affinity 调度命中率（同 task 段被均匀切到一张卡，命中率 100%；被切到多卡时一定程度上重复 setup）；
  2. 每张卡的 task 数量上限 vs `MAX_SEGMENTS_BEFORE_RESTART`（卡多 task 时频繁 task switch 也会吃 setup）；
  3. 异常率（每次 segment\_error 触发一次 soft restart）。

## 8. 切换与回滚

- **默认走 persistent**：直接 `bash openpi-comet/scripts/run_skill_eval_single_node_8gpu.sh run ...`，不需要额外 env。
- **退回旧路径**：`EVAL_MODE=process_per_segment bash openpi-comet/scripts/run_skill_eval_single_node_8gpu.sh run ...`，launcher 走 `--mode launch`，行为与 persistent 之前完全一致；旧的 `worker_results/worker_*.jsonl` schema 也保持不变。
- **混用同一个 out-dir**：`merge_results()` 会同时收 `worker_*.jsonl` 与 `persistent_worker_*.jsonl` 并按 `job_key` 去重，所以 persistent 和 legacy 的产出可以并存（例如 persistent 跑大头，individual 段重跑用 legacy 兜底）。

## 9. 烟雾验证清单（Task 5，已完成）

OpenSpec 任务 5 已经完成：

1. **5.1** 单 GPU × 单 task（推荐 `hanging_pictures`）× 4 段，确认 Isaac 只 boot 一次、4 个 metrics JSON 落盘；
2. **5.2** 同 4 段在 `EVAL_MODE=process_per_segment` 下重跑，diff `success` / `result_type` / `final_step` / `first_predicate_satisfied_step`，必须完全一致；
3. **5.3** 2 GPU × 2 task × 32 segment 实测，对比 8 卡折算时长是否 ≥ 3× 加速、`worker_status/persistent_worker_*.jsonl` 无 `error` 事件；
4. **5.4** 用 `stop_snapshots/remaining_segments_*.txt` 作为 manifest filter，恢复被中止的 34×32 sweep；
5. **5.5** 在 `AGENTS.md` 「Eval launch safety tips」末尾追加一行，提及 `EVAL_MODE=process_per_segment` 这个 rollback 开关。

## 10. 文件索引

| 文件                                                                 | 作用                                                                                                    |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_segment.py`       | `run_segment_on_env()` / `_reconfigure_for_segment()` / `_build_sample_from_cli_config()`             |
| `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_subtask_reset.py` | `SubTaskEvaluator`（被 worker 直接 instantiate）                                                           |
| `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_custom.py`        | `Evaluator` 基类（`load_env` / `load_task_instance` / `video_writer` setter）                             |
| `openpi-comet/scripts/persistent_skill_eval_worker.py`             | 长寿命 GPU worker，本框架核心                                                                                  |
| `openpi-comet/scripts/run_skill_metric_multinode_sweep.py`         | Launcher，新增 `--mode launch-persistent` / `--mode persistent-worker` 与 `materialize_persistent_jobs()` |
| `openpi-comet/scripts/run_skill_eval_single_node_8gpu.sh`          | 单机 8 卡入口；`EVAL_MODE` 选 persistent / legacy                                                            |
| `openspec/changes/persistent-skill-eval-framework/`                | 设计文档与任务清单（proposal / design / spec / tasks）                                                           |
