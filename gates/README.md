# MoMA-VLA 正式训练准入 gate

离线、全量、可重跑。数据被重新生成之后原样再跑一次即可。

## 分工（重要，不要合并成一句「已覆盖」）

| 层 | 谁做 | 什么时候判 | 判什么 |
|-|-|-|-|
| **数据 gate（本目录）** | 本 worker | 训练前，离线，全量一次 | 当前这份数据不越界、结构自洽、无 oracle 泄漏、分片够分 |
| **运行时 hard error** | 训练代码 worker | 训练中，逐样本 | 真出现超长/EOS 丢失/未命中区间时立刻报错而不是静默截断 |

两层不能互相替代。文档 3.4.5 要求「未命中区间 / token 超长 / EOS 丢失」是 hard error，
而训练侧 `openpi/models/tokenizer.py:515-519`（prompt 超 512）与 `:550-557`（subtask 超 128）
目前**都只是 `logging.warning` 后截断**，「未命中区间」则没有任何代码路径。也就是说这三项
现在是**靠数据恰好不越界而满足的，不是靠代码保证的**。

## 怎么跑

```bash
PYTHONPATH=/mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet-hier-moma/src \
/mnt/bn/behavior-data-hl/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python \
  moma_pretrain_gate.py \
  --data-root /mnt/bn/behavior-data-hl/chenjunting/data/2025-challenge-demos/derived/fixed_compact_memory_annotations \
  --frames-meta-root /mnt/bn/behavior-data-hl/chenjunting/data/2025-challenge-demos \
  --world-size 32 --num-workers 16 \
  --out gate_report_full.json
```

- 必须带 `PYTHONPATH=<worktree>/src`，否则 `openpi` 的 editable install 会指向主树、静默跑旧代码。
- env 必须是 `openpi-comet-nas`（另两个 conda env 没有 `sentencepiece`）。
- `--world-size` / `--num-workers` / `--frames-meta-root` **必传，不提供默认值**，理由见下。
- 只读，不写数据目录。

### 机器成本（这台机很容易被压垮，实测数字放在这里）

| 步骤 | 耗时 |
|-|-:|
| `import openpi.models.tokenizer` | **294 秒**（load 80 时） |
| glob 一万个 episode 文件 | 47 秒 |
| 全量扫描本身 | 空载 24 进程 40 秒 |

所以：

- 反复跑 gate 时用 `threeway_inproc.py` 那种**单进程 harness**，把 gate 当模块 import 一次；起 20 次子进程等于 98 分钟纯 import。
- 用 `--file-list` 跳过重复 glob。
- 脚本在**导入 numpy 之前**就把 `OMP_NUM_THREADS` 等五个变量设成 1。这台机 `nproc` 报 119 而 cgroup 只给 8 核，BLAS 线程池按 119 开、多进程一乘就是几千个线程。
- `--jobs` 默认按 **cgroup 配额**取（上限 8），不是 `os.cpu_count()`。

退出码：

| 码 | 含义 |
|-|-|
| 0 | PASS |
| 1 | FAIL（有 hard failure；带 `--fail-on-soft` 时 soft failure 也算） |
| 3 | CANNOT-ASSESS（枚举到 0 个文件，或有 episode 不可读）——不会把空值渲染成读数 |

其他入口：

```bash
# 检测器自检（正对照：构造坏样本，断言每个检查都会点亮；含一条干净样本的负对照）
... moma_pretrain_gate.py --self-test

# 三向验证 + 变异测试（单进程 harness，只 import 一次）
PYTHONPATH=<worktree>/src <env-python> threeway_inproc.py [输出目录]

# subtask_max_len 作用域守卫（可单独跑，也已接进 gate 的 --config-scope-expect）
PYTHONPATH=<worktree>/src <env-python> config_scope_guard.py --dump        # 打印当前解析值
PYTHONPATH=<worktree>/src <env-python> config_scope_guard.py --expect scope_expectations.json

# 「同一个 vocab 文件、不同 sentencepiece 库版本，是不是同一把尺子」
... ruler_compare.py dump --tag sp020 --out /tmp/a.jsonl        # 训练侧 env
PYTHONPATH=/tmp/b1k_tokenizer_py313 <py313> ruler_compare.py dump --tag sp022 --out /tmp/b.jsonl
... ruler_compare.py compare --a /tmp/a.jsonl --b /tmp/b.jsonl
```

## 主要参数（阈值全部可调，这是三向验证的前提）

| 参数 | 默认 | 说明 |
|-|-|-|
| `--data-root` | fixed_compact_memory_annotations | 数据根目录，参数化以便在重生成后的副本上先跑 |
| `--frames-meta-root` | `.../2025-challenge-demos` | 视频语料 LeRobot root（需含 `meta/episodes.jsonl`）。annotation 是它的 derived 层，本身**没有任何帧/state 数据**，必须联表才能判 frame 覆盖。传空串显式跳过 |
| `--expect-episodes` | 10000 | |
| `--expect-intervals` | 261353 | **传 `-1` 表示只报实测值和差值、不做相等判定** |
| `--subtask-max-len` | **必传，无默认值** | 那次 run 生效的上限。取值仍在决策中（128 / 160 / 192），写死任何一个候选值都等于让 gate 在假设的上限上给出 PASS。⚠️ `Pi05SubtaskConfig` 的**类默认值**是 128 且被在飞实验共用，抬高只能在 MoMA 配置实例上覆盖 —— 作用域由 `--config-scope-expect` 单独断言 |
| `--prompt-max-len` | **必传，无默认值** | 同上（512 / 320 决策中） |
| `--action-dim` | **23** | tokenize 时 state 的真实维度（`extract_state_from_proprio`）。**不是** `Pi05SubtaskConfig.action_dim=32` —— 那是 padded 宽度，而 padding 发生在 tokenize 之后，进不了 `State:` 文本。用 32 会把 prompt 预算高估（190 vs 154），且两者不能线性折算 |
| `--world-size` / `--num-workers` | **必传，无默认值** | 只能来自那次 run 的**生效值**。`pretrain_config.py:411` 是 16、`train_config.py:99` 默认是 2、单机启动脚本 `:51` 又覆盖回 2 —— 不要从配置文件读 |
| `--chunk-size` | 250 | `dataset.py:1334` 硬编码，决定分片单元数 |
| `--max-uncovered-frames` | 0 | 按 `[0, L)` 采样时没有 Memory 区间覆盖的帧数上限 |
| `--max-dead-chunks` | 0 | 整块落在标注覆盖之外的 chunk 数上限 |
| `--max-bridge-anomalies` | 0 | bridge 结构不变量（归纳自数据）的违例上限 |
| `--chunk-stats-json` | 无 | 训练侧 `memory_chunk_stats()` 落盘的 json；期望值从这里读，不写死在 gate 里 |
| `--baseline-report` | 无 | 上一次的 gate 报告，用于逐项报出重生成后的变化量 |
| `--expect-overhang-episodes` | -1 | `last_end > length` 的 episode 数期望值；clamp 后应为 0，这是「量的是新数据」最可判定的凭据 |
| `--config-scope-expect` | 无 | `subtask_max_len` 作用域期望文件 |
| `--file-list` | 无 | 预生成的文件清单，跳过每次 47 秒的 glob |
| `--file-stride` | 1 | 跨 task 无偏取子集（阈值扫描用），不要用有偏的 `--limit-files` |
| `--max-fallback-intervals` | -1 | 组合型 primitive fallback 文案命中区间上限；-1 = 只报不判 |
| `--sample-episodes` | 100 | 深度抽查的 episode 数（每个 task 均摊，保证跨 task 覆盖，不用 `head`） |
| `--skip-count-checks` | off | 在子集上跑时关掉总数硬检查 |

## 检查清单

hard（失败即非零退出）：

- `READABLE_EPISODES` — 读数不足直接渲染成 CANNOT-ASSESS，不当测量值
- `NEGATIVE_CONTROL_SENTINELS` — 两个必然为 0 的哨兵（字面量 + 正则）
- `POSITIVE_CONTROL_COVERAGE` — 每一行 target 都含 `Memory:` 且 token 长度 > 0，证明扫描确实覆盖了每一行
- `EPISODE_FILE_COUNT` / `TASK_LAYOUT` / `INTERVAL_TOTAL` / `INTERVAL_NONEMPTY`
- `INTERVAL_STRUCTURE` — 连续、无倒置、`memory_idx` 递增、覆盖范围等于 `valid_duration`、schema 字段正确
- `TARGET_TOKEN_MAX` —— 判定用**生产口径** `tokenize_memory`（保留换行）；文档 3.4.4 口径 `tokenize_subtask` 另列一条 `TARGET_TOKEN_MAX_DOC_PATH` 只报不判
- `MEMORY_CODEC_ACCEPTS_ALL_TARGETS` —— 每行都要能过 `validate_field_structure`
- `TARGET_OVER_MAX_COUNT` / `EOS_PRESENT_AND_SUPERVISED` / `LOSS_MASK_CONTRACT`
- `PROMPT_TOKEN_BUDGET` — prompt 是**从右截断**的，而 3.4.3 把 `Previous memory:` 放末尾，溢出时被吃掉的正好是 memory 文本
- `STATE_IS_DISCRETIZED_NOT_FRAME_COUNTER`
- `ACTION_QUERY_NOT_IN_CE_TEXT` / `MODEL_TARGET_RECONSTRUCTION`
- `NO_ORACLE_LEAK`
- `MEMORY_CHAIN_ORDERING` / `BRIDGE_LABEL_VALUE_SET`
- `FRAME_TO_INTERVAL_HIT`
- `EPISODE_JOIN_1TO1` / `FRAME_COVERAGE_VS_VIDEO` / `NO_DEAD_CHUNKS`
- `SHARDING_EVERY_WORKER_NONEMPTY` / `SHARDING_AFTER_DEAD_CHUNK_EXCLUSION`
- `CHUNK_STATS_AGREE_WITH_TRAINING_SIDE` / `CLAMPED_RANGE_HITS_100PCT` / `NO_ANNOTATED_FRAME_DROPPED`
- `FALLBACK_PHRASE_TAXONOMY_COMPLETE`
- `SUBTASK_MAX_LEN_SCOPE`（提供 `--config-scope-expect` 时）

soft（只告警）：

- `BRIDGE_STRUCTURAL_INVARIANT` — 归纳自数据、不是文档写的，所以不拦训练
- `ANNOTATION_WITHIN_VIDEO` / `SHARDING_TASK_LEVEL_WOULD_FAIL`
- `FALLBACK_PHRASE_COUNT` / `COMPACT_MEMORY_DISTRIBUTION`

## 纪律上做了什么

- **正对照**：`--self-test` 用构造的坏样本逐项点亮每个检测器，另有一条干净样本作负对照（必须零 violation）。
- **负对照哨兵**：两个必然为 0 的哨兵跟着每次全量扫描一起算；非 0 说明匹配本身坏了。
- **不用 `head` 抽样**：文件按 task/episode 排序，`head -N` 会全部落在同一个 task。深度抽查按 task 均摊取样，其余检查全量扫。
- **先界定集合规模**：先打印 `task_dirs` / `episode_files`，再谈命中数；枚举到 0 个文件直接 CANNOT-ASSESS。
- **版本可追溯**：报告里记 `manifest.json` 的 mtime/md5/episodes/intervals，以及跨偏移取样 16 个文件的 md5 摘要，说清「量的是哪个版本的数据」。
- **尺子可追溯**：记 SentencePiece 模型路径、md5、vocab_size、库版本、Python 版本。

## 目录内容

| 文件 | 作用 |
|-|-|
| `moma_pretrain_gate.py` | gate 本体 |
| `ruler_compare.py` | 两个 sentencepiece 版本的 token id 逐条比对 |
| `threeway_inproc.py` | 三向验证 + 变异测试（单进程，只 import 一次） |
| `_expected_counts.py` | 独立于 gate 的预期条数统计（不拿 gate 输出当预期） |
| `config_scope_guard.py` | `subtask_max_len` 作用域守卫 |
| `three_way_validation.sh` | 旧的多进程版本，保留作参考；这台机上不要用（每次子进程 import 要 294 秒） |
| `gate_report_full.json` | 全量实测报告 |
| `selftest_report.json` | 检测器自检结果 |
| `ruler_compare_result.json` | 尺子比对结果 |
| `threeway/` | 三向验证的全部中间产物与汇总表 |
| `FINDINGS.md` | 实测结论与需要裁决的问题 |
| `distribution_baselines.json` | 分布观察项的版本化登记基线 |
| `verify_v2_inplace_fix.py` | 原地修复的逐文件逐字段验证 |
| `scope_expectations.json` | `subtask_max_len` 作用域期望（MoMA 配置建好后要加一条） |

## 版本控制

`moma_handoff_20260907/` **不在任何 git 仓库里**（`git rev-parse` 报 not a git repository）。
所以本目录的交付**没有 commit hash**。要纳入版本控制需要另行指定仓库 ——
`openpi-comet-hier-moma` 由训练代码的 worker 拥有，本 worker 不写它。
