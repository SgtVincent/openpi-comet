# B1K 分布式 Arrow Cache 协议与运维说明

本文说明 full B1K BF16 路径中最终批准的分布式 Hugging Face Arrow cache 实现。内容以当前代码为准，覆盖数据选择、身份模型、节点内构建、跨 rank 失败共识、严格复用、兼容性、运维方式和已完成审计；不描述尚未实现的训练框架重构。

## 1. 背景与根因

### 1.1 Full B1K 的数据范围

正式配置 `pi05_ki_joint_query_b1k-full_task-ki_on_bf16` 不设置 `tasks` 过滤，因此选择全部 B1K challenge tasks；`episodes_index` 按 task 独立应用：

- train：每个 task 的 `[0, 180)`，即索引 `0` 至 `179`；
- validation：每个 task 的 `[180, 200)`，即索引 `180` 至 `199`。

数据集先按 task 汇总并排序 episode，再用 `episodes_index` 逐 task 取值，最后展平并排序。实现只保留满足 `i < len(ep_indices)` 的索引。因此，如果 50 个 task 都至少有 180 个有效 episode，训练选择的常见规模上界约为 `50 × 180 = 9000` 个 episode；**9000 不是代码不变量**，任一 task 的有效 episode 不足时，越界索引会被跳过。

### 1.2 Arrow cache 位于数据分片之前

所选 Parquet 源的 `load_dataset("parquet")` 和 Arrow 准备发生在 `BehaviorLerobotDataset` 构造期间。调用顺序是：

1. `create_torch_dataset(...)` 创建数据集并完成 Arrow 准备；
2. 随后才创建 `DistributedSampler` / `DataLoader`；
3. 更晚才执行 `accelerator.prepare(model, optimizer, loader)`；
4. chunk streaming 的 `rank × worker` 分区发生在运行期 `__getitem__` 中。

因此，PyTorch/Accelerate 的样本分片不能避免启动阶段的 Arrow 准备，也不能替代 Arrow artifact 的构建协调。

### 1.3 原启动问题

full B1K 的全量 Parquet→Arrow 准备可能持续超过 NCCL barrier watchdog 的容忍时间。把整个构建包在长时间的全局 `barrier()` 或 `main_process_first()` 中并不安全：collective 既不了解 artifact 是否完整，也不能把构建失败转换成所有 rank 一致的可重试/终止结果；长时间等待还会暴露 watchdog timeout 风险。

当前实现不在 Arrow I/O 期间持有 NCCL collective，而使用 c10d default Store 作为控制面，并以节点本地原子 marker 协调同节点 rank。

### 1.4 Arrow 中存放什么

Arrow 保存帧级结构化行及相关 metadata，例如 `timestamp`、`task_index`、`episode_index`、state、action，以及用于定位外部视频的引用/帧索引信息。它**不包含已解码的视频像素**。RGB、depth、segmentation 等视频仍位于外部文件中，数据访问阶段通过 `OBS_LOADER_MAP` 或 `decode_video_frames(...)` 按需懒解码。Arrow 完整性与外部视频文件完整性是两个不同边界。

## 2. 文件与职责

| 文件 | 主要职责 | 明确边界 |
|---|---|---|
| `src/behavior/learning/datas/dataset.py` | 选择 tasks；按 task 排序、筛选并展平 episode；基于完整 `self.episodes` 生成 Parquet source request；normal builder/reader 通过 `load_dataset("parquet")`；strict reader 通过 `datasets.Dataset.from_file` 直接打开 manifest 指定的 Arrow；数据访问时懒解码视频 | Arrow 准备在 sampler/Accelerate sharding 之前；strict 路径不调用 build/repair；视频像素不写入 Arrow |
| `src/behavior/learning/datas/hf_cache_sync.py` | 计算 run/selection/request/generation 身份；通过 c10d default Store 协调 attempt；解析 `node<N>` cache；每节点选举一个 builder；提供 setup/load/readiness 的失败共识、ack 与 retry 隔离；原子发布本地 marker；生成并校验 protocol-v3 prepared marker；校验路径 containment、size、mtime 和全文件 SHA-256 | `_MARKER_PROTOCOL_VERSION=3`；marker 内 `prepared_manifest` 的 schema 为 `manifest_version=1`；prepared 身份是 request-scoped，attempt 状态是 generation-scoped |
| `scripts/train_accelerate.py` | 推导并导出 `HF_DATASETS_CACHE`；将 `force_load_cache` 接入数据集；先构建 train，再构建 val；`PREPARE_HF_CACHE_ONLY` 在二者完成后、模型加载前退出；数据 manifest probe 默认 `0` | 配置阶段只导出路径，不预先检查、创建或遍历 HF datasets artifact tree；`probe=0` 只跳过后续 loader sample probe，不跳过 train/val cache 构建 |
| `scripts/run_pi05_ki_joint_query_full_b1k_bf16_multinode_hl.sh` | 绑定 node-local `/tmp` cache；解析并导出同一 job 的 `OPENPI_HF_CACHE_RUN_ID`；校验 manual/strict 使用规则；传递 PREPARE/FORCE；执行本地 cache/TMP preflight；启动 Accelerate | `_LOCAL_CACHE_DIRS` 故意排除 `HF_DATASETS_CACHE`；launcher preflight 不创建或验证 datasets artifact tree，strict 校验延迟到分布式 setup |

补充：`src/openpi/training/data_loader.py` 在数据集创建完成后才建立 `DistributedSampler`；chunk streaming 则在 `dataset.py` 中用 `global_worker_id = rank * num_workers + worker_id` 划分 chunk。

## 3. Identity 模型

| Identity | 组成与稳定性 | 使用范围 |
|---|---|---|
| `OPENPI_HF_CACHE_RUN_ID` / `run_id` | 同一 job/rendezvous 的 rank-consistent 身份。优先取显式 `OPENPI_HF_CACHE_RUN_ID`，其次 `ARNOLD_JOB_ID`、`TORCHELASTIC_RUN_ID`，或完整的 `MASTER_ADDR:MASTER_PORT:WORLD_SIZE` fallback；launcher 也会把 managed task identity 导出为显式值 | 隔离同一协调会话，参与 coordination namespace 和默认 node-local cache root 推导；所有 rank 必须一致 |
| `selection_id` | dataset root、source mode、canonical load options，以及**有序** source paths 的 SHA-256 截断；不读取文件 stat | 表示语义选择；源路径和加载语义不变时跨 retry 稳定 |
| `request_id` | `selection_id` 加上去重后的 Parquet 源 metadata：绝对路径、`size`、`mtime_ns`、`ctime_ns`、`mode` | 表示可持久复用的 prepared artifact request；prepared marker 以它命名 |
| HF dataset `_fingerprint` | `datasets` 生成的 dataset fingerprint | 记录在 `prepared_manifest` 供审计，但不是 strict lookup 的唯一键，也不替代 request identity 和 artifact 校验 |
| `generation_id` | `run_id + selection_id + invocation_index + UUID` 的 SHA-256 截断 | 表示本次 invocation/attempt；新的协议级 retry 或重新调用会分配新 generation，旧 generation 的 ready/failure 不能满足新 attempt |

本地 marker 分为两类：

- 持久 prepared marker：`.openpi_hf_cache_sync/prepared/<request_id>.ready.json`，`generation_id=null`，绑定 `request_id`；
- attempt marker：`.openpi_hf_cache_sync/attempts/<generation_id>/<request_id>.{ready,failed}.json`，绑定当前 generation。

Store 的 setup、ready、failure 和 `failure-observed/<rank>` key 均位于 `request_id/generation_id` namespace。这样可复用的是经过验证的 prepared artifact，而不是上一轮 attempt 的成功/失败状态。

需要注意：source request fingerprint 使用文件 metadata，不是源 Parquet 内容 SHA。prepared Arrow artifact 则逐字节计算完整 SHA-256，见第 8 节限制。

## 4. 端到端协议

### 4.1 正常与严格模式共同前半段

1. Trainer/launcher 推导并导出 `HF_DATASETS_CACHE`，但不对该 datasets tree 做 `mkdir`、存在性检查或 artifact 扫描。
2. 所有 rank 根据同一语义选择计算 `selection_id`。分布式时 global rank 0 对去重后的源 Parquet 做 stat，计算 `request_id`，再把包含 `request_id` 和新 `generation_id` 的 descriptor 写入 c10d Store；其他 rank 读取并校验 descriptor。
3. 每个 rank 在 `coordinate_global_cache_setup(...)` 内调用 `setup_node_cache_paths(...)`：
   - `world_size <= 1` 时使用配置 root 本身；
   - 分布式时按 `node_rank = rank // local_world_size` 解析为 `<configured-root>/node<N>`；
   - normal mode 创建缺失的 node cache directory；
   - strict mode 只接受已存在目录，禁止创建或修复。
4. normal mode 中 `local_rank == 0` 是该节点唯一 builder；同节点其他 rank 是 reader。`OPENPI_HF_DATASETS_CACHE_PER_RANK=1` 是历史命名，实际语义是“一份 cache/节点”，不是“一份 cache/GPU rank”。

### 4.2 Normal builder/reader

5. builder 清理当前 generation 的旧 attempt marker，然后执行 `load_dataset("parquet", data_files=完整 self.episodes 对应路径, ...)`。成功后：
   - 收集 `hf_dataset.cache_files` 中的 Arrow 文件；
   - 对 artifact 做 cache-root containment 校验；
   - 记录 relative path、kind、size、`mtime_ns`、全文件 SHA-256 和 dataset `_fingerprint`；
   - 用临时文件写入、`fsync`、`os.replace` 原子发布 prepared marker，最后发布当前 generation ready marker。
6. 同节点 reader 轮询当前 generation 的 ready/failure marker；ready 后再调用 `load_dataset("parquet")`，由 Hugging Face datasets 打开已经准备好的 node-local cache。reader 不参与 builder election。

### 4.3 失败共识与 retry 隔离

7. 在 descriptor 已分配 `request_id/generation_id` 之后，setup、builder、reader、strict read 或 global readiness 任一失败都会：
   - 由第一个失败 rank 通过 Store `compare_set` 发布 canonical failure；
   - 所有 rank 观察相同 payload；
   - 每个 rank 写入 `failure-observed/<rank>` ack；
   - 全部 ack 完成后抛出相同 `DistributedCacheError`，并保留 `retryable` 分类。
8. 新的协议 attempt 使用新 `generation_id`。Store key 和本地 attempt marker 都按 generation 隔离，因此旧失败不会污染新 attempt。已验证的 request-scoped prepared marker可以被后续 generation 使用。

`FileNotFoundError(filename=None)` 这一已知 filelock contention 形态可分类为 retryable；普通 schema/完整性错误等不被泛化为可重试。`_do_load` 内部的短暂 filelock retry 仍在同一 generation 内；重新进入协议级 invocation 才获得新 generation。

**边界说明：**上述 canonical failure ack 仅适用于 attempt 已获得 `request_id/generation_id` 之后。若 global rank 0 在 descriptor 生成前计算 source request fingerprint 就失败，代码使用 coordination namespace 的 failure notification 通知其他 rank；此时尚无 generation，不能描述成 generation-scoped ack。

### 4.4 Strict reader 与后续数据分片

9. `FORCE_LOAD_CACHE=1` 时没有 builder。每个 rank 都校验本节点 `<request_id>.ready.json` 与全部 artifact：
   - marker protocol/status/request identity；
   - artifact 必须位于 cache root 内；
   - 文件必须存在；
   - size、`mtime_ns`、全文件 SHA-256 必须完全匹配。
10. 校验通过后，每个 rank 用 `datasets.Dataset.from_file(...)` 直接打开 manifest 中按序列出的 Arrow；多个 part 用 `concatenate_datasets` 合并。打开前后还会比较 cache tree snapshot，发现写入即失败。strict 路径不调用 `load_dataset`、不 rebuild、不 repair。
11. 只有 train/val cache 和 dataset object 均已存在后，代码才应用 `DistributedSampler`、DataLoader、Accelerator/DeepSpeed 包装以及 chunk streaming 的 rank×worker 分区。

## 5. 兼容性矩阵

| Scenario | Distribution state | Cache path | Builder count | Reader behavior | Failure/retry | Constraints |
|---|---|---|---|---|---|---|
| 非分布式单进程 | `torch.distributed` 不可用或未初始化，且 `WORLD_SIZE<=1` | 已配置时直接使用 configured root，不追加 `node0`；未配置时 normal 可直接 `load_dataset`，strict 拒绝无法验证的复用 | normal：1；strict：0 | normal 由本进程构建/读取；strict 校验后 `Dataset.from_file` | 无跨 rank Store 共识；本地错误包装为 `DistributedCacheError` | reviewed trainer/launcher 会导出 cache path；若环境声称 `WORLD_SIZE>1` 但 PG 未初始化则 fail closed |
| 已初始化 `world_size=1` | Process group 已初始化，单 rank c10d default Store | configured root 本身，不追加 `node0` | normal：1；strict：0 | normal 单 rank builder；strict 单 rank直接校验/open Arrow | 仍执行 generation-scoped Store setup/readiness，ack 集合只有 rank 0 | `RANK/WORLD_SIZE` 环境必须与 PG 一致 |
| 单节点多 GPU/rank | PG initialized，`world_size>1`，全部 rank 位于一节点 | `<HF_DATASETS_CACHE>/node0`，节点内共享一份 Arrow | normal：每节点 1，即总计 1；strict：0 | normal 同节点 reader 等待原子 marker，再打开缓存；strict 每 rank 独立校验本节点 artifact 并直接 open | Store canonical failure + 全 rank ack；新 invocation/new generation 隔离 retry | 要求正确的 `LOCAL_RANK`、`LOCAL_WORLD_SIZE`，并保持连续 global rank 映射 |
| 多节点多 GPU（审阅拓扑 4×8） | 32 ranks，4 nodes × 8 local ranks | 每个物理节点使用自己的 node-local root，并解析为该节点的 `node0`…`node3`；每节点各有一份 Arrow copy | normal：每节点 1，即 4；strict：0 | normal 每节点 7 个 peer 等本节点 marker；strict 32 个 rank 各自验证所在节点的 prepared artifact | 任一节点 setup/load/read 失败都会经 Store 形成 32-rank 一致结果；retry 换 generation | 假设 global ranks 按节点连续，且 `LOCAL_RANK/LOCAL_WORLD_SIZE` 准确；strict 时每个节点都必须已有匹配的 `node<N>` cache |

兼容性审计结论是：非分布式单进程、initialized PG `world_size=1`、单节点多 rank，以及 4×8 多节点路径均受支持，未发现必须修复的代码问题。证据边界是：4×8 使用 32-rank thread/FakeStore 模拟；真实 Gloo spawned 测试为 2 个进程，**不是**一次完整 4×8 GPU 训练。

## 6. 为什么仍保留自定义 coordinator

### 6.1 原生组件已经负责的部分

- Accelerate：进程启动/rendezvous、rank topology、梯度同步，以及 DataLoader 包装/样本分片；
- PyTorch：ProcessGroup、Store、collective、`DistributedSampler` 等基础构件；
- DeepSpeed：模型/优化器的分布式训练职责。

### 6.2 原生组件没有开箱提供的部分

这些组件没有直接提供以下 B1K cache artifact 协议：

- node-local Arrow builder election；
- request-persistent、generation-isolated 的 prepared-cache lifecycle；
- 原子 local ready/failure marker 与可验证 manifest；
- 跨 rank canonical failure、ack 和 retry/abort 一致决策；
- strict、只读、逐 artifact 校验的复用路径。

当前 coordinator 复用 PyTorch Store 作为控制面，但 artifact lifecycle、节点内选举和完整性策略由项目代码实现。

### 6.3 不使用长 barrier / `main_process_first()` / global-rank-0-only

- **global rank 0 only**：只能写 rank 0 所在节点的 `/tmp`，不能为其他节点生成 node-local Arrow；
- **长 `barrier()`**：只表达“到达同步点”，不表达 artifact 身份、完整性、失败来源或 retry 分类，并可能触发 watchdog timeout；
- **`main_process_first()`**：能串行化先后顺序，但不能完成每节点 builder election，也不提供 generation 隔离、manifest 完整性或失败共识。

未来可以把 dataset preparation/cache coordination 从 DataLoader 和 training-loop 责任中进一步模块化，但该重构**尚未实现**，本文不把它描述为当前行为。

## 7. Operations 与 Preflight

以下示例均围绕 reviewed launcher：

```bash
LAUNCHER=scripts/run_pi05_ki_joint_query_full_b1k_bf16_multinode_hl.sh
```

### 7.1 Managed normal launch

在 Merlin/Arnold 为每个节点注入 `ARNOLD_*` topology/job identity 的环境中，由各节点执行同一命令：

```bash
bash "${LAUNCHER}"
```

launcher 会优先使用显式 `OPENPI_HF_CACHE_RUN_ID`，否则使用 managed job/task identity，并把 bulk cache 绑定到 node-local `/tmp`。

### 7.2 显式 run identity 的 launcher preflight

```bash
OPENPI_LAUNCH_PREFLIGHT_ONLY=1 \
OPENPI_HF_CACHE_RUN_ID='b1k-preflight-20260729-001' \
bash "${LAUNCHER}"
```

该模式验证 run identity、local cache binding、TMP alias、写权限和 `multiprocess.Manager` primitive，然后跳过分布式启动。它**不会**创建或验证 `HF_DATASETS_CACHE` artifact tree；strict artifact validation 只在 process group 建立后、`coordinate_global_cache_setup` 内发生。

### 7.3 Prepare-only：构建 train+val 后退出

manual normal/prepare run 必须使用一个新的、非空、所有节点一致的显式 run ID：

```bash
OPENPI_HF_CACHE_RUN_ID='b1k-full-cache-job-20260729-001' \
PREPARE_HF_CACHE_ONLY=1 \
bash "${LAUNCHER}"
```

Trainer 按 train→val 顺序创建两个 dataset/cache；两者完成后在加载模型和开始训练前退出。`OPENPI_DATA_MANIFEST_PROBE_BATCHES=0` 的默认值不会取消这一步，只会跳过后续基于真实 batch 的 manifest probe。

### 7.4 同一 job/allocation 内 strict 复用

在相同物理节点、相同 node-local cache 尚存在时，使用与 prepare 阶段相同的 run ID：

```bash
OPENPI_HF_CACHE_RUN_ID='b1k-full-cache-job-20260729-001' \
FORCE_LOAD_CACHE=1 \
bash "${LAUNCHER}"
```

strict 模式要求每个节点的 `node<N>` 都有匹配 `request_id` 的 protocol-v3 prepared marker 和完整 artifact；缺失、损坏或发生写入都会 fail closed。

### 7.5 Manual 显式绝对 cache root 的只读复用

当每个节点已经在同一绝对 node-local 路径保留了对应 `node<N>` artifact 时，可显式复用。下列命令假设调用前已正确导出 `NODE_RANK`、`MASTER_ADDR`：

```bash
: "${NODE_RANK:?set NODE_RANK on each node}"
: "${MASTER_ADDR:?set MASTER_ADDR to rank-0 host}"

LOCAL_CACHE_ROOT='/tmp/openpi-comet/tiger/b1k-reviewed-cache' \
FORCE_LOAD_CACHE=1 \
NUM_NODES=4 \
GPUS_PER_NODE=8 \
NODE_RANK="${NODE_RANK}" \
MASTER_ADDR="${MASTER_ADDR}" \
MASTER_PORT=29514 \
bash "${LAUNCHER}"
```

只有 `LOCAL_CACHE_ROOT` 而没有 managed/explicit run ID 时，launcher 仅允许 `FORCE_LOAD_CACHE=1`，并从 absolute root 推导 rank-consistent run identity；normal/prepare manual run 不允许借此隐式复用，必须提供新的显式 `OPENPI_HF_CACHE_RUN_ID`。

### 7.6 运维约束

- `PREPARE_HF_CACHE_ONLY=1` 与 `FORCE_LOAD_CACHE=1` 互斥，launcher 和 trainer 都会拒绝同时启用；
- 不承诺跨 job 的 `/tmp` 复用：`/tmp` 是 node-local 且受节点/容器生命周期影响；
- multi-node strict 复用要求**每个节点**都拥有与其 `node<N>` 对应的 prepared cache；只有一个节点有 cache 不足以启动；
- launcher preflight 只验证外围本地目录和 primitive，故意不提前触碰 HF datasets tree；
- normal/prepare manual run 的 run ID 必须唯一且所有 rank 一致，不能用 PID 或各节点独立生成的时间值；
- strict 模式不 repair。需要修复时，退出 strict，并以新的正常/prepare run 明确重建。

## 8. Tests、证据与限制

### 8.1 已完成审计命令与结果

审计使用项目已知、未受 shared base Miniconda 变更影响的解释器，覆盖五个相关测试文件。完整命令形态如下：

```bash
/mnt/bn/navigation-hl/mlx/users/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python -m pytest \
  /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/tests/test_behavior_dataset_hf_cache_sync.py \
  /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/tests/test_train_accelerate_manifest.py \
  /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/tests/test_pi05_ki_joint_query_config.py \
  /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/tests/test_train_accelerate_checkpoint_policy.py \
  /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/tests/test_train_accelerate_diagnostics.py
```

审计记录结果：

```text
141 passed in 82.64s
```

### 8.2 覆盖范围

- 真实 2-process Gloo/default `PrefixStore`：retryable failure 后新 generation 恢复、non-retryable abort、strict-missing 与 mkdir setup failure 的对称传播；
- 真实 `TCPStore` + `PrefixStore`：repeated generation wait 不被旧 generation key 满足；
- 4 nodes × 8 local ranks：32-rank thread + FakeStore 协议模拟，验证每节点一个 builder 和全局 readiness；
- strict：完整 tiny Parquet cache 的只读复用、cache root/Arrow 缺失、legacy marker 拒绝、保持相同 size/mtime 的文件中部 corruption 仍被 full SHA-256 拒绝；
- world1/local branches：run identity、configured root、PG/world 环境一致性和无 barrier 构建路径；
- trainer/launcher：prepare/force 互斥、train+val prepare 顺序、`probe=0`、只导出 path 不预触碰 datasets tree、manual identity/absolute root 规则；
- 同一测试集合还覆盖 full B1K config、checkpoint policy 和 diagnostics，防止 cache 变更破坏相邻启动契约。

### 8.3 明确限制

- 本次审计没有新增 GPU smoke，也没有执行完整 full B1K training run；
- 真实 Gloo spawned 测试只有 2 processes；4×8/32-rank 证据来自 thread/FakeStore 模拟，不等价于完整 4×8 GPU 训练；
- `torch.distributed.distributed_c10d._get_default_store()` 是 PyTorch private API，升级 PyTorch 时需要重新验证；
- `request_id` 的源身份使用 path + size/mtime/ctime/mode，而非源 Parquet 内容 SHA；它适合当前 metadata identity 协议，但不能证明源内容在保留全部 stat 字段时未变化；
- prepared Arrow artifact 使用全文件 SHA-256，可检测 cache 文件 corruption，但不覆盖 Arrow 外部的懒加载视频；
- 每个节点都依赖自己的 node-local `/tmp` 和对应 `node<N>`，节点替换、容器重建、清理或生命周期结束都会使 cache 不可用；
- 不承诺跨 job `/tmp` reuse。显式 absolute `LOCAL_CACHE_ROOT` + strict 只是受约束的只读入口，不把生命周期不确定的本地盘升级为持久存储保证。

## 9. Why Not 与 Caveats

| 备选方案 | 未采用原因 / tradeoff |
|---|---|
| 把 decoded video pixels 写入 Arrow | 会改变当前数据边界并扩大 cache；当前 Arrow 只承载结构化帧数据及外部视频引用/索引，视频按需懒解码 |
| 所有 rank 并发 `load_dataset` | 增加同一节点 filelock 与构建竞争，无法明确谁发布完整 artifact；当前每节点只允许 local rank 0 构建 |
| 仅 global rank 0 构建 | 无法写入其他物理节点的 node-local `/tmp` |
| 全局共享一份 Arrow cache | 多个节点 builder 可能竞争同一路径；当前分布式 shared-cache mode fail closed |
| 长 NCCL barrier 或 `main_process_first()` | 缺少 artifact identity、完整性、failure consensus 和 generation isolation，并有长 I/O 触发 watchdog 的风险 |
| 只信任 HF `_fingerprint` 或 ready 文件存在 | 不能证明所需 artifact 的路径 containment、完整大小、mtime 和内容 digest；strict 使用 request marker + manifest + full SHA-256 |
| strict 发现缺失后自动重建 | 会破坏“已验证只读复用”的语义，并让部分 rank 进入 build、部分 rank 进入 read；当前统一 fail closed |
| 每 rank 一份 Arrow | 存储和构建成本过高；当前是每节点一份。不过其代价仍是**每个节点各自保存一份 prepared Arrow copy**，多节点磁盘占用约随节点数增长 |

该设计优先保证启动协议可判定、失败对称和 strict 复用可验证，代价是自定义协调代码、每节点一份存储，以及对 rank topology 和 node-local lifecycle 的显式假设。任何性能收益都应通过对应环境的真实测量确认；现有测试只证明协议行为与兼容路径，不支持额外的吞吐或训练效果结论。

更新日期：2026-07-29
