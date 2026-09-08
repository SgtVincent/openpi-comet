# MoMA-VLA 训练准入 gate — 实测结论

数据版本：`fixed_compact_memory_annotations`，`manifest.json` mtime `2026-09-07T05:29:22`、md5 `8c772547325b99086b07e3ec195a73f1`、`episodes=10000`、`intervals=261353`、`schema_version=b1k_fixed_compact_memory_v1`。
帧语料：`/mnt/bn/behavior-data-hl/chenjunting/data/2025-challenge-demos`，`meta/episodes.jsonl` 10,000 行、总帧数 119,094,660。
尺子：`SubtaskTokenizer`，SentencePiece 模型 md5 `1420adc9856720a559e8a87284b195e2`、vocab 257,152、库版本 0.2.0、Python 3.11.6。

以下所有数字都是**全量实测**（10,000 episode / 261,353 区间），不是抽样。

---

## 1. token 长度

| 量 | min | p50 | p90 | p95 | p99 | p999 | max | 上限 | 余量 |
|-|-|-|-|-|-|-|-|-|-|
| `planner_target_text` / `tokenize_subtask`（文档 3.4.4 口径） | 39 | 59 | 70 | 74 | 83 | 91 | **101** | 128 | 27 |
| `planner_target_text` / `tokenize_memory`（**生产口径**，保留换行） | 43 | 63 | 74 | 78 | 87 | 95 | **105** | 160 | **55** |
| `model_target_text`（含 `Action Query:`，与 il_lib 审计同口径） | 42 | 62 | 73 | 77 | 86 | 94 | **104** | — | — |
| `Task/State/Previous memory` prefix（**state 23 维，实测口径**） | 117 | 126 | 131 | 132 | 136 | 146 | **154** | 320 | **166** |
| 同上（state 32 维，**保守上界，高估**） | 153 | 162 | 167 | 168 | 172 | 182 | 190 | 512 | 322 |
| `fixed_compact_memory` 字段 | 13 | 22 | 27 | 29 | **32** | 41 | **49** | 观察项 | 与文档记录 32/49 一致 |

- 超长 0 条（两条口径都是）、EOS 丢失 0 条、EOS 未被监督 0 条、BOS 被监督 0 条。全部在真实 tokenizer 输出上逐条断言。
- **261,353/261,353 行都能过 `MemoryTextCodec.validate_field_structure`**（五个标签、顺序固定、标签不得出现在字段正文里），0 条被拒。这一条**换算量不到，只能真跑**。
- 🔴 **「+4」是实测值，不是转换公式，而且它的成立范围很窄。** 全量 261,353 行**逐行**测得差值分布 `{4: 261353}`，只有一个取值。但成立的前提是**目标文本不含下划线**：`tokenize_subtask`（`tokenizer.py:533`）既压换行**又**把 `_` 换成空格，而 `tokenize_memory` → `MemoryTextCodec.encode(text.strip())` **两样都不做**。两条清洗是两套。
  ⚠️ 数据侧正在用 API 重新生成 8,233 个 phrase 签名。**新文本一旦含下划线，「+4」立刻失效，而失效方式是静默的** —— 任何还在用「文档口径 + 4」的地方会给出错的数字。所以：
  - 引用 105 时必须写「v1 数据上的实测值」，不要当成 101 + 4
  - gate 里有 `TWO_PATH_DELTA_INVARIANT`（hard）：逐行差值集合必须仍是 `{4}` **且**目标文本不含下划线；自检里有对应的正对照（插一个下划线，检查必须报错）
- **按 `transition_type` 分组的 max**（token 压力集中在 bridge 上）：

| transition_type | 条数 | 生产口径 max | 文档口径 max |
|-|-:|-:|-:|
| `normal` | 235,429 | 92 | 88 |
| `inter_primitive_bridge` | 6,481 | 103 | 99 |
| `intra_primitive_skill_bridge` | 19,442 | **105** | **101** |
| `repaired_parent_primitive` | 1 | 62 | 58 |

  全体 max 105 来自 `intra_primitive_skill_bridge`，而 `normal` 只到 92 —— **差 13 个 token**。
  bridge 文案 `transitioning from A to B` 一行里点两个 primitive，所以长在那一侧。
  非 bridge 样本占 90.1%，它们离任何候选上限都很远。

- **候选上限三档（生产口径 / 文档口径的越界条数与余量）**：

| 上限 | 生产口径越界 | 文档口径越界 | 生产口径余量 | 文档口径余量 |
|-:|-:|-:|-:|-:|
| 128 | 0 | 0 | 23 | 27 |
| 160 | 0 | 0 | 55 | 59 |
| 192 | 0 | 0 | 87 | 91 |

- **文档 3.4.4 写的是 `tokenize_subtask`，实现走 `tokenize_memory`**。这个不一致要回流进文档，不能靠换函数掩掉。生产路径判据以 **105 / 余量 23** 为准。
- 需要注意的接线现状：`transforms.py:467-473` 只有在 item 带 `memory_text` 时才走 `tokenize_memory`，而 `memory_text` 由 `PromptFromLeRobotItem.include_memory_text` 决定是否保留 —— 全仓 `include_memory_text=True` 的赋值点为 **0**（只有 `transforms.py:414/416` 两处错误提示字符串，以及一个测试文件）。正对照 `include_subtask_text=True` 命中测试文件，哨兵 0，搜索集合 178 个 py 文件。⇒ **「设计上该走 `tokenize_memory`」成立，「当前接线实际会走」不成立**，两者要分开陈述。

## 2. 尺子同一性

同一个模型文件、两个 `sentencepiece` 版本（训练侧 0.2.0 / il_lib 审计侧 0.2.2），在同一批 **522,706 条文本**（261,353 × 2）上逐条比对 token id 序列的 sha1：

- token id 序列不一致 **0** 条
- token 数不一致 **0** 条
- 两侧行集合完全相同（`only_in_a=0` / `only_in_b=0`）
- 正对照：首条两侧 n 都是 58、id 前缀相同；负对照：注入伪造 key，比对器确实报出差异

⇒ 两个库版本在这份语料上是同一把尺子。本 gate 量出的 `model_target_text` 七个分位数与 `token_length_audit.json` 逐项相同。

## 3. 分片

当前代码唯一的分片单元是 `self.chunks`（`dataset.py:287` → `:1334`），公式 `Σ ceil(episode_length / 250)`。函数名与 docstring 讲 keyframe/GOP，但代码里没有任何 keyframe 检测。

| 单元 | 数量 | 512 个 global worker 下每人最少 | 空 worker | 余量 |
|-|-|-|-|-|
| chunk（全部） | 481,383 | 940 | 0 | 940× |
| **chunk（剔除 dead 后）** | **476,889** | **931** | **0** | **931×** |
| memory interval | 261,353 | 510 | 0 | 510× |
| episode | 10,000 | 19 | 0 | 19.5× |
| task 目录 | 50 | 0 | 462 | 0.098× ❌ |

rank-blind 失效条件（`dataset.py:1013-1026`）：`range(g, N, W)` 为空当且仅当 `g = rank*P + worker_id >= N`，此时落进 `:1025` 的 `range(worker_id, N, P)` —— 该行**不含 rank**，不同 rank 上同号 worker 拿到逐位相同的 chunk 集合。各处 RNG 种子都带 `g`（`:1020/:1027/:1030`），所以两个 rank 读的是同一批 chunk 的**不同排列**，日志和指标上看不出异常。

`num_workers` 的三个来源不一致，gate 因此把它做成**必传、无默认值**：

| 来源 | 值 |
|-|-|
| `pretrain_config.py:411/470/535`（三个 Pi05Subtask 配置） | 16 |
| `train_config.py:99`（TrainConfig 默认） | 2 |
| `run_pi05_b1k_skill-pt50_pretrain_lr1e-4_2ep.sh:51`（单机启动脚本，会覆盖配置） | 2 |
| `conda_run_..._multinode_...sh:78`（多机启动脚本） | 16 |

## 4. chunk 覆盖：与训练侧逐个相等

annotation 只覆盖 `valid_duration`，而 chunk 按 `range(0, L, 250)` 切整段视频。定义：`live_e = min(cov_e, L)`；`dead ⟺ ce <= cov_s 或 cs >= live_e`；`clamped = [max(cs,cov_s), min(ce,live_e))`。

| 量 | 本 gate | 训练侧 `memory_chunk_stats()` |
|-|-|-|
| 总 chunk | 481,383 | 481,383 |
| dead chunk | 4,494 | 4,494 |
| 保留 chunk | 476,889 | 476,889 |
| 跨边界被夹取的 chunk | 11,427 | 11,427 |
| 有 ≥1 dead chunk 的 episode | 2,849 | 2,849 |
| 无 Memory 覆盖的帧 | 1,349,597（头 274,657 / 尾 1,074,940） | 1,349,597 |
| 标注伸出视频结尾 | 83 episode / 2,820 帧 | 83 / 2,820 |

两边独立实现、独立计算，结果逐个相等。

- `clipped = 11,427` 数的是 **chunk 窗口**，不是区间也不是帧：头部夹取 3,945 + 尾部夹取 7,482 + 两端同 chunk 0，涉及 8,448 个 episode。它和「83 个越界 episode」「2,820 帧」是不同量纲，不要当同一个量比较。
- **不要剔除部分覆盖的 chunk**：那样会丢掉 **1,674,631 帧有标注的数据**，比 dead chunk 覆盖的 1,349,597 无标注帧还多 1.24 倍。gate 的 `NO_ANNOTATED_FRAME_DROPPED` 就是挡这个反方向漏的。
- 剔除 + 夹取之后，采样范围内 frame → interval 命中率 **100%**（6,751 个探测，0 未命中，0 个空夹取范围）。
- ⚠️ `memory_chunks_clipped` **不能**当「上游换没换数据」的凭据：数据侧 clamp 前后 dead 和 clipped 一个数都不变，因为 `min(cov_e, L)` 本来就把越界吃掉了。可用的凭据是「`last_end > length` 的 episode 数从 83 变成 0」。

## 5. oracle 泄漏

全量 0 条。检测的形态：帧计数器 `Frame N of M`、`of <多位数>`、`annotation_index`、`segments`、`frame_duration`、`memory_idx`、`memory_signature`、task ID、episode ID、`episode_length|total_frames|num_frames|task_duration`；原始 object handle 用**该 episode 自己的 `object_id`/`manipulating_object_id` 列表**逐个比对（原串与下划线换空格两种形态）。

handle 判据限定为「至少含一个下划线的全小写数字串」（`radio_89` / `coffee_table_koagbh_0`），裸词不算 —— 不这样限定会把 `object_id` 里的 `robot` 一词判成泄漏，在 73 个文件的抽样里就产生了 86 条假警报。

`State:` 段实测是 32 个 `[-1,255]` 整数（256 档离散化），不是 P0 那批的 `Frame N of M`。

## 6. 三个互不重叠的独立缺陷

| 缺陷 | 规模 | 涉及 task |
|-|-|-|
| 组合型 primitive fallback 文案 | 35,945 区间 / 1,387 episode | 9 个：0004 0008 0011 0025 0026 0040 0042 0046 0049 |
| `intra_primitive_skill_bridge` 标签与结构矛盾 | 96 区间 / 92 episode | 10 个：0002 0012 0013 0015 0019 0020 0027 0029 0030 0043 |
| caption 下标错位（他人发现） | — | 32 个 |

前两者的 task 集合**交集为空**，同时带这两种缺陷的 episode **0 个**。所以它们互不掩盖，重生成 fallback 数据不会顺带改掉 bridge 标签矛盾，反之亦然。

## 7. fallback 文案分类表的性质

`7630833c-749` 给的 8 个串在 `current_primitive`/`next_primitive` 上完备（含 ` + ` 却不属于这 8 串的条数 = **0**）。但：

- **完备性断言不能扩到 `model_target_text`**：那里有 3,794 条 ` + ` 是 LLM 摘要把加号当「和」用（`Collected trash can + 2 soda cans`），会变成假警报。
- **8 个串互有子串包含关系，计数不可相加**。逐个删串暴露的独占命中量（全量）：

| 删掉的串 | 暴露条数 |
|-|-|
| `pick up from + place on` | 7,368 |
| `pick up from + place in` | 3,753 |
| `unknown primitive` | 22 |
| `pick up from + chop` | 21 |
| `pick up from + pour` | 4 |
| `pick up from + place on next to` | 0（被 `place on` 吸收） |
| `pick up from + place in next to` | 0（被 `place in` 吸收） |
| `pick up from + chop + place on next to` | 0（被 `chop` / `place on` 吸收） |

⇒ 要按串统计 fallback 分布，必须按**最长优先**归类（像 `memory_text.py` 的 `_LABEL_RE` 那样），否则短串会把长串的量吸走。

## 8. 需要裁决的

1. **`intra_primitive_skill_bridge` 的 96 条结构矛盾**：标签说「同一 primitive 内的 skill 过渡」，但相邻两行 `current_primitive` 变了。判据是从数据归纳的、不是文档写的，所以记软告警。例：`task-0002/episode_00020250.json#6`，上一行 primitive 是 pillar candle、本行变成 cauldron，而本行 skill 文案讲的是 candle。看起来与源标注的 primitive overlap（manifest 记 1,360 次）有关。
2. **`subtask_max_len=128` 的余量**：生产口径 max 105、余量 23。补 phrase 模板后预估 108–111（余量 17–20），加上 caption 退回模板的增长还会更少。这个上限本身是否要抬高，需要在拿到重生成后的实测值时决定。

## 9. 分工

gate 在**数据上**判（离线、全量、可重跑）；训练代码在**运行时**判（在线、逐样本、hard error）。文档 3.4.5 要求的三项 hard error 里：

| 项 | 运行时状态 |
|-|-|
| prompt 超 512 | 已加（`fd32815`） |
| 未命中区间 | 已加（`memory_annotation.py` + chunk 级剔除 `936b34e`） |
| subtask 超 128 | 已加（`936b34e`，`tokenize_memory` 无条件抛错，错误信息带各字段 token 数） |

在这三项落地之前，它们是**靠数据恰好不越界而满足的，不是靠代码保证的**。两层谁也不替代谁。


---

# 附：三向验证与变异测试结果（交付 ②）

跑法：`threeway_inproc.py`，单进程 harness（gate 只 import 一次），子集 `--file-stride 10`
= 1,000 个 episode / 26,304 个区间、跨全部 50 个 task。预期条数由 `_expected_counts.py`
独立算出并**先落盘再跑 gate**（不拿 gate 自己的输出当预期）。

**`THREEWAY_VERDICT: mismatches=0  rows=27  total_seconds=284`**

| 检查 | 方向 | 参数 | 预期 | 实测 |
|-|-|-|-|-|
| SELF_TEST | 正对照 | `--self-test` | 全过 | 17/17 |
| TARGET_OVER_MAX_COUNT | 当前/放宽 | `--subtask-max-len 160` | 0 | 0 |
| TARGET_OVER_MAX_COUNT | 放宽 | `--subtask-max-len 256` | 0 | 0 |
| TARGET_OVER_MAX_COUNT | 调紧 | `--subtask-max-len 90` | **114** | **114** |
| TARGET_OVER_MAX_COUNT | 调紧 | `--subtask-max-len 83` | **576** | **576** |
| PROMPT_TOKEN_BUDGET | 当前 | `--prompt-max-len 512` | 0 | 0 |
| PROMPT_TOKEN_BUDGET | 放宽 | `--prompt-max-len 1024` | 0 | 0 |
| PROMPT_TOKEN_BUDGET | 调紧 | `--prompt-max-len 180` | **18** | **18** |
| PROMPT_TOKEN_BUDGET | 调紧 | `--prompt-max-len 162` | **11,721** | **11,721** |
| SHARDING_AFTER_DEAD_CHUNK_EXCLUSION | 当前 | 32×16（live=47,917） | PASS | PASS，min/worker=93 |
| 同上 | 放宽 | 8×1 | PASS | PASS，min=5,989 |
| 同上 | **边界** | 47,917×1 | PASS | PASS，min=1 |
| 同上 | **越界一格** | 47,918×1 | FAIL | FAIL，min=0 |
| 同上 | 调紧 | 95,834×1 | FAIL | FAIL，min=0 |
| CHUNK_COUNTS_VS_INDEPENDENT | 交叉核对 | gate vs 独立统计 | dead=500 live=47,917 | 相同 |
| CHUNK_STATS_AGREE_WITH_TRAINING_SIDE | 当前 | dropped=500 | PASS | PASS |
| 同上 | **注入偏差** | dropped=506 | FAIL | FAIL |
| COUNT_CHECKS | 当前 | 1000 / 26,304 | PASS | PASS |
| COUNT_CHECKS | 调紧 | 1000 / 26,303 | FAIL | FAIL |
| COUNT_CHECKS | 调紧 | 999 / 26,304 | FAIL | FAIL |
| COUNT_CHECKS | 放宽 | intervals=-1 | PASS | PASS |
| FALLBACK_PHRASE_TAXONOMY_COMPLETE | 当前 | 完整分类表 | 0 | 0 |
| 同上 | **变异分类表** | 逐个删 8 个串 | ≥1 个能杀死 | 2/8 能杀死（子集） |
| MUTATION | 负对照 | 未变异副本 | 不报红 | INTERVAL_STRUCTURE passed |
| MUTATION | 变异 | `rows[1].start += 7` | KILLED | KILLED |
| MUTATION | 源文件未动 | md5 | 前后相同 | `254183785b...` 相同 |
| DATA_UNTOUCHED | 全树指纹 | 10,000 个文件 | 相同 | 相同 |

## 三向验证过程中被抓出来的三个真问题

这三个都是「跑通了但结论是错的」，不做三向就不会暴露：

1. **第一版变异分类表没杀死检查**：我删的是 `pick up from + place in next to`，而更短的
   `pick up from + place in` 仍在表里、把它的命中全吸走 ⇒ 暴露 0 条，检查存活。改成逐串
   扫描才拿到「哪些串能单独杀死检查」那张表（全量 5/8，子集 2/8）。
2. **预期与实测口径不一致**：`TARGET_OVER_MAX_COUNT` 改成判生产口径后，我的独立统计还在
   算文档口径 ⇒ 阈值 83 处预期 252、实测 576。修法是让独立统计**按 `.strip()` 单独编码**
   一份生产口径长度，而不是给文档口径 +4 —— 因为 `tokenize_memory` 不仅不压换行，也**不做
   下划线 → 空格**，两条清洗是两套，+4 只是这批数据上恰好成立。
   附带结果：独立统计出的逐行差值取值集合是 **{4}**，只有一个取值 ⇒ +4 在这批数据上确实成立，
   但它是**测出来的**不是假设的。
3. **截断的缓存文件被当成完整的**：上一轮 harness 被 kill 时正在写文件清单，留下 6,416 行的
   截断文件，而 `exists()` 照样为真。断言 `len(清单) == fingerprint 的 n_files` 当场抓住。
   已改成**临时文件 + `os.replace` 原子写**，复用前必须与本轮实测文件数对账。

## 环境读数（数字必须带环境，否则不可比）

开发机重启前后同一批子集扫描：

| | load（1/5/15 分钟） | 进程/线程 | 单次子集扫描 | `import openpi.models.tokenizer` |
|-|-|-|-|-|
| 重启前 | 90 / 80 / 70 | 25 / 17,771 | **137–184 秒** | **294 秒** |
| 重启后 | 26 / 29 / 46 | 60 / 394 | **13–14 秒** | **26.5 秒** |

同一份代码、同一批数据，**慢了约 10 倍**。所以此前记录的 import 耗时里有多少是环境争抢、
多少是文件系统固有成本，只有带上环境读数才分得开。harness 现在每条测量都打印
`load / procs / threads`。


## 附：`subtask_max_len` 作用域守卫的三向验证

`config_scope_guard.py --expect scope_expectations.json`，环境 `load=26.0/26.5/36.2 threads≈394`。

当前状态（**PASS**）：类默认值 128、三个既有 Pi05SubtaskConfig 全是 128、没有未登记的配置。

只会通过的守卫没有信息量，所以逐条做了正对照 —— 每个变异只动一处期望，验证它点亮的是
**对应那一条**断言而不是别的：

| 变异 | 预期 | 实测 |
|-|-|-|
| 期望类默认值 = 160 | FAIL 在 `class_default` | FAIL：`class_default: expected=160 observed=128` |
| 期望某个在飞实验 = 160 | FAIL 在那一条配置 | FAIL：`pi05_b1k_skill-pt50_...: expected=160 observed=128` |
| 期望表里漏登记一个配置 | FAIL 在 `no_unlisted` | FAIL：`no_unlisted_Pi05SubtaskConfig: observed=['pi05_b1k_skill-pt12_...']` |

三条全部命中各自的断言、退出码 1。

MoMA 专用训练配置**还没建**，所以期望表里现在只有三个既有配置、全是 128。配置建好后要
在 `scope_expectations.json` 的 `per_config_subtask_max_len` 里加它一条（160 或 192），
**其余三条保持 128、`class_default_subtask_max_len` 永远保持 128**。


---

# 附：全量实测（生产口径已升级为实测值）

环境：`load=24.6/25.8/33.8 procs≈57 threads=459`，`--jobs 2`，全量 10,000 episode /
261,353 区间 **143 秒**。`verdict=PASS`（3 条 WARN，见下）。

对照：同一份代码在开发机重启前（load≈90、threads≈17,771）跑同样的子集要 137–184 秒、
`import openpi.models.tokenizer` 要 294 秒；重启后子集 13 秒、import 26.5 秒。**所以任何
耗时数字都必须带环境读数**，否则「环境变好了」会被算成「优化生效了」。

## 三条 WARN 的性质

1. `BRIDGE_STRUCTURAL_INVARIANT` = 96 —— 归纳自数据、非文档规定，等数据侧解释，不阻塞。
2. `FRAME_COVERAGE_VS_VIDEO` = 1,349,597 帧 —— 处置已定为在 dataloader 侧剔除 dead chunk +
   夹取 anchor，所以降为观察项；判定由 `CHUNK_STATS_AGREE_WITH_TRAINING_SIDE` /
   `SHARDING_AFTER_DEAD_CHUNK_EXCLUSION` / `CLAMPED_RANGE_HITS_100PCT` /
   `NO_ANNOTATED_FRAME_DROPPED` 四条 hard 承担，全部 PASS。
3. `ANNOTATION_WITHIN_VIDEO` = 83 episode / 2,820 帧。**按最后一个区间的 transition_type
   分组：83 个全部是 `normal`，没有一个落在 bridge 上。**

另有一条 `TARGET_TOKEN_MAX_NO_REGRESSION` 报 `delta=+4`，那是**口径变更造成的，不是数据变化**：
我传的基线 101 是文档口径的数，而这条检查现在比的是生产口径 max 105。下一次跑请用
`--baseline-target-max 105`，或者直接用 `--baseline-report` 读上一份报告（它逐字段对齐口径）。

## 变异测试里「不可用作变异」的三个串

fallback 分类表的 8 个串里，**只有 5 个能单独杀死完备性检查**（全量口径）：
`place on` 7,368 / `place in` 3,753 / `unknown primitive` 22 / `chop` 21 / `pour` 4。
另外 3 个 —— `place on next to`、`place in next to`、`chop + place on next to` —— 的命中被更短的
串完全包含，**删掉它们暴露 0 条、检查存活**。⇒ 这 3 个在变异测试里是无效变异，以后不要用。


---

# 附：新 MoMA 配置的登记，以及登记时查出的两件事

## 1. 🔴 两个 MoMA 配置定义了，但没有并入 `_CONFIGS` ⇒ 按名字选不中

`src/openpi/training/moma_memory_config.py` 里的 `MOMA_MEMORY_CONFIGS` 是一个模块级
tuple，含两个 `TrainConfig`。运行时实测其生效值（构造出来打印，不是读字面量）：

| 配置名 | subtask_max_len | max_token_len | planner_stride | subtask_source | num_workers | steps | **in `_CONFIGS_DICT`** |
|-|-:|-:|-:|-|-:|-:|-|
| `pi05_moma_memory_b1k-k5_smoke` | 192 | 320 | 5 | `annotations_memory` | 2 | 200 | **False** |
| `pi05_moma_memory_b1k-k5` | 192 | 320 | 5 | `annotations_memory` | 2 | 30,000 | **False** |

`train_config.py:307-313` 的 `_CONFIGS` 只拼了 5 个模块，**不含 `moma_memory_config`**。
实测 `_CONFIGS_DICT` 共 55 个配置，**含 "moma" 的 0 个**（正对照：
`pi05_subtask_b1k-pt50_...` 在里面）。后果分两条路径，性质不同：

- **`cli()`**（`train_config.py:322`，两个训练入口 `train_accelerate.py:6340` /
  `train_pytorch.py:1210` 用的都是它）：tyro 从 `_CONFIGS_DICT` 生成可选项 ⇒ 这两个名字
  **根本不是合法选项**。
- **`get_config()`**（`train_config.py:325-330`）：未注册的名字**只打一条
  `logging.warning` 然后返回 `pi05_b1k-base`**。实测三个输入
  —— 两个 MoMA 名字和一个哨兵 `zzz_no_such_config_qqq` —— **返回完全相同的对象**，
  即「配置没注册」与「名字拼错了」在输出上不可区分。
  唯一挡住它的是 `train_accelerate.py:515-519`：`pristine = get_config(config.name)` 之后
  显式比对 `pristine.name != config.name` 并 `raise ValueError(...refusing fallback)`。
  ⇒ **那个调用点是安全的；`get_config` 本身对其他调用者仍是一个静默回退的坑。**

⇒ gate 里把 `per_config_registered` 对这两个配置声明为 **True**（它们*应该*处于的状态），
所以这条现在**是红的**。这是有意的：把「配置定义了但选不中」变成一条不会被忘掉的红灯。
`moma_memory_config` 并入 `_CONFIGS` 之后它会自动转绿。

## 2. ⚠️ 我此前说的「三个在飞配置」是错的，实际是 33 个

之前我只遍历 `pretrain_config._PRETRAIN_CONFIGS`（3 个）。改成遍历
`train_config._CONFIGS_DICT` 之后，实测共 **35 个** `Pi05SubtaskConfig` 配置：
**33 个已注册的在飞配置全是 `subtask_max_len=128` / `max_token_len=512`**，
外加两个 MoMA 的 192 / 320。漏掉的 32 个里包含全部 `pi05_ki_joint_*`（含 skillbridge）
与 make_pizza SFT 系列。

⇒ **「改类默认值会波及多少实验」这个量，我此前报小了一个数量级。** 现在登记表逐个列出
全部 35 个。

## 3. 字段名：`max_token_len`，不是 `prompt_max_len`

prompt 预算那个旋钮的**配置字段名是 `max_token_len`**。`prompt_max_len` 只是
`SubtaskTokenizer` 的构造参数名，生产在 `data_config.py:274` 做
`prompt_max_len=model_config.max_token_len` 的映射。按 `prompt_max_len` 去改配置会崩。
登记表按 `max_token_len` 写，并在注释里写明这个映射。

## 4. 守卫的语义与四条正对照（全部重跑并各自命中）

语义是「**每个配置的上限必须等于登记表为它声明的值**」，不是「必须等于 128」——
所以上限改成 192 时改的是登记表，gate 代码不动。

| 变异 | 预期命中的断言 | 实测 |
|-|-|-|
| 期望类默认值 = 160 | `class_default` | FAIL `class_default: expected=160 observed=128` |
| 期望某在飞实验 = 160 | 那一条配置的 `subtask_max_len` | FAIL `pi05_b1k_skill-pt50_...subtask_max_len: expected=160 observed=128` |
| 登记表漏登记一个配置 | `no_unlisted` | FAIL `no_unlisted_Pi05SubtaskConfig: observed=['pi05_b1k_skill-pt12_...']` |
| 期望 MoMA 的 `max_token_len` = 512 | 那一条的 `max_token_len` | FAIL `pi05_moma_memory_b1k-k5.max_token_len: expected=512 observed=320` |

四条各自命中自己的断言，没有互相串台。`class_default_subtask_max_len` 单列一条、
**永远保持 128** —— 那才是「防止有人全局抬高、静默改掉 33 个在飞实验」的那道闸。


---

# 附：v2 metadata clamp 修复的验收，以及分布观察项的版本化登记基线

## 1. 三步验收（全部 rc=0）

| 步骤 | verdict | 关键数 |
|-|-|-|
| `verify_v2_inplace_fix.py` | **PASS** | changed 83/83、unchanged 9,917/9,917、`memory_annotation` 变化 0、顶层只有 `meta_data` 变、只有允许的两键变、`valid_duration[1]==last_end` 0 条不符、manifest `1448cd306ce6`→`50726ae508ef` |
| 主 gate（基线 `state23`） | **PASS，hard=[]** | `coverage_end_mismatch` 83→**0**、`coverage_start_mismatch` **0**、`ANNOTATION_WITHIN_VIDEO` 0 帧/0 episode |
| 定向 diff（基线 v2 首跑） | **PASS** | 登记的 12 个量**全 0 变化** |

⚠️ **「定向 diff 12 项全 0」不等于「修复生效」**：`coverage_end_mismatch` 不在那 12 个被 diff 的量里。
修复生效由主 gate 的 83→0 证明。两者合起来才完整：**一个证明该变的变了，一个证明不该变的一个都没变。**

v1@23 → v2 的口径对齐差异（`prompt_max` 这次是正确的 +7，不是早先混读 32 维基线得到的 −29）：
生产口径 target max 105→**153**（对 192 余量 39）、p99 83→99、prompt max 154→**161**、
compact max 49→**56**、越界 episode 83→**0**、fallback 区间 35,945→**2,167**、bridge 异常 96→94；
`intervals` / dead / clipped / uncovered 四项不变。

## 2. 分布观察项改成版本化登记基线，而不是跟着数据走的常量

`gates/distribution_baselines.json`，按**该数据版本 manifest.json 的 md5** 索引
（目录名会被原地改动骗过，md5 不会）：

| manifest md5 | 版本 | status | compact p99 / max |
|-|-|-|-:|
| `8c772547…` | v1 | superseded | 32 / 49 |
| `1448cd30…` | v2 首跑（metadata 未 clamp） | superseded | 32 / 56 |
| `50726ae5…` | **v2 metadata-clamped** | **accepted** | **32 / 56** |

v2 条目里写明增量来源：max 从 49 涨到 56 来自**已批准的** phrase 模板补齐
（`source_unresolved_phrase_templates` 29,300→221、composed 29,079），同批次生产口径
`planner_target` max=153、对 `subtask_max_len=192` 仍有 39 余量，所以这个增长不构成预算风险。

**为什么不直接把阈值改成 56**：那样每次数据一变就把阈值改成新值，gate 永远 PASS、检不出任何漂移，
变成跟着数据走的橡皮图章。登记基线的语义是「这个版本经过验收、分布应当是这些值」，
所以**同一版本内再涨仍会报警**。未登记的版本一律 **fail-closed 记 NOT-MEASURED**，不自动学习。

## 3. 四向验证（每条都跑了真实 gate，全部命中预期）

| 情形 | 登记值 | 实测 max | 预期 | 实测 |
|-|-:|-:|-|-|
| v2 已登记 | 56 | 56 | PASS | **PASS** |
| 调紧登记值 | **55** | 56 | WARN | **WARN** |
| 放宽登记值 | **57** | 56 | PASS | **PASS** |
| 该版本从登记表删掉 | — | — | fail-closed / NOT-MEASURED | **hard failure，threshold=NOT-MEASURED**，并列出已登记版本 |

⚠️ 关于「未来 max 涨到 57 在 v2 基线下仍应 WARN」：这条**没有伪造 max=57 的数据去测**，
它走的是与「调紧到 55」**完全相同的比较**（实测 > 登记 ⇒ WARN），后者已实测命中。
把它标成「由同一代码路径的实测推出」，而不是「已直接验证」。

fail-closed 那次跑在子集上，所以 `EPISODE_JOIN_1TO1` 也同时红了（子集必然如此）；
要看的是 `COMPACT_MEMORY_DISTRIBUTION` 从软告警**升级成 hard failure** 且 threshold 显示
`NOT-MEASURED` —— 它没有拿别的版本的值去卡，也没有把当前值学成阈值。
