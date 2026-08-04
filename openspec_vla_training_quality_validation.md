# VLA 数据清洗质量验证：纠偏后的当前计划

更新时间：2026-07-02

## 当前执行目标

本轮重启会话后，已成功创建新的中型 `/goal`：

继续推进 VLA 数据清洗质量验证纠偏后的新目标：先修实验可检出性，完成 task/profile filter-dose scan、equal-dose random-drop 对照规范、独立/净化 held-out 或闭环指标契约；优先验证 qwen-manip suspicious subset，而不是继续用极小剂量 strict structural R1 作为清洗策略证据；持续更新 WjGad 飞书文档与 OpenSpec。

## 核心结论

R1 `il_lib bcrnn_proprio` attempt 4 已完成，但它不能作为“清洗策略有效/无效”的正负证据。正确结论是：

**R1 实验不具备检出力：under-powered + 被测对象错位 + 指标错位 + 无独立 test。**

R1 实际测到的是：“在 `sorting_household_items` 中，54 个 demo 里只有 1 个 demo (`00270670`) 的局部 strict-structural 片段被跳过时，同分布 held-out 开环 L1/GMM 是否一致变好”。这不是主清洗方案的验证，也不能反推 qwen-manip、DA、物理一致性或 artifact-driven success 过滤是否有用。

## Multi-Agent Review 汇总

| 维度 | Review 判断 | 修正动作 |
| --- | --- | --- |
| 实验剂量 | attempt 4 的 raw/cleaned 差异只覆盖 demo `00270670` frames `[4124,4331)`，在 54 demos 中扰动极小。 | 不再把 R1 写成 cleaned negative；下一轮必须放大清洗剂量。 |
| 被测对象 | R1 只测 `offline_hard_exclude.json` strict structural filter。 | strict structural 只作为第 0 层 sanity，不代表完整清洗策略。 |
| 指标方向 | val L1/GMM 是未清洗同分布 demo 上的开环 imitation metric。 | 主指标改为 cleaned/verified held-out 或闭环 rollout 成功率/物理有效性；开环 loss 仅辅助。 |
| test split | `test/* == val/*`，不是独立 test。 | 下一轮必须建立独立 test，或明确只称 validation。 |
| 方差 | 4 seed 不足以做结论，GMM mean 被 seed 42 离群差异主导；L1 差异小于约 1%。 | 报 per-seed 分布和显著性，不用 mean delta 单独下结论。 |
| 文档 drift | 文档混有“未启动训练”“negative for cleaned”“R2 ready”等旧状态。 | 飞书文档重构为当前判断 + 下一轮计划；历史日志只保留索引。 |

## R1 当前可支持和不可支持的结论

R1 可以支持：

- attempt 3 是 no-op contrast，因为 `max_num_demos=20` 没包含 filtered demo `00270670`。
- attempt 4 是 non-no-op contrast，因为 `max_num_demos=54` 包含 `00270670`。
- strict structural exclusion 这一个极小剂量 ablation 没有在当前开环 held-out L1/GMM 上表现出一致改善。
- 不能从 R1 推广 official `segment_filter_path` 或 hard filter profile。

R1 不可以支持：

- 不能说完整 VLA 清洗策略是负向或无效。
- 不能说 qwen-manip 平滑性、DA、物理一致性、artifact-driven success 过滤无效，因为 R1 没测这些。
- 不能说 cleaned 数据训练质量更差，因为当前指标和假设方向不匹配，且剂量过小。
- 不能声称有独立 test 或泛化结论，因为 `test/*` 实际等于 validation metrics。

## R1 Attempt 4 结果保留为诊断证据

| metric | cleaned wins | mean delta cleaned - raw | 解读 |
| --- | ---: | ---: | --- |
| final val L1 | 1/4 | `+0.029933` | 差异小，不能检出清洗效应 |
| final val GMM loss | 1/4 | `+1.354017` | mean 受 seed 42 离群影响大 |
| best val L1 | 1/4 | `+0.068814` | 不构成 cleaned 改善证据 |
| best val GMM loss | 0/4 | `+2.092133` | 不构成策略负证据，只说明当前 ablation/metric 不支持提升 |

结果文件：

- `outputs/vla_training_quality_validation_model_restart/r1_il_lib_bcrnn_proprio/results/r1_attempt4_metrics_summary.json`
- `outputs/vla_training_quality_validation_model_restart/r1_il_lib_bcrnn_proprio/results/r1_attempt4_metrics_summary.md`

## 修正后的验证逻辑

验证问题改写为：

**在足够清洗剂量、随机等量丢弃对照、独立/净化 held-out 或闭环 rollout 指标下，清洗后的数据是否比 raw 数据更适合训练策略。**

必须同时满足三个条件，才进入下一轮训练结论：

1. 剂量足够：cleaned 与 raw 的差异覆盖多个 demo / 多个高置信问题片段，而不是 1 个 demo 的局部窗口。
2. 对照正确：加入 `random_drop_equal_dose`，把“样本变少”与“数据变干净”分离。
3. 指标对齐：以闭环 rollout 成功率、物理有效性、action replay / state replay 一致性、cleaned verified held-out 为主；开环 L1/GMM 只做辅助。

## 下一轮计划

### R1-fix：先修实验能否测

| 项 | 要求 |
| --- | --- |
| filter dose scan | 已生成；统计每个 task/profile 的被过滤 demo 数、窗口数、frame 占比，选择 effect size 可检出的任务/子集 |
| equal-dose random control | 已生成 plan-only 规范；为每个 candidate cleaned condition 规划 3 个随机等量丢弃 baseline |
| split policy | 建立 train / val / independent test，或只报告 val，不再把 `test/*` 写成独立泛化 |
| metric contract | 明确主指标和辅助指标；禁止用未清洗 held-out L1/GMM 单独评判 artifact 清洗 |

R1-fix 当前产物：

| 产物 | 路径 | 状态 |
| --- | --- | --- |
| filter-dose scan | `outputs/vla_training_quality_validation_model_restart/r1_fix_filter_dose_scan/FILTER_DOSE_SCAN.md` | 完成，read-only |
| filter-dose JSON/CSV | `outputs/vla_training_quality_validation_model_restart/r1_fix_filter_dose_scan/filter_dose_scan.json` / `.csv` | 完成，read-only |
| random-drop equal-dose plan | `outputs/vla_training_quality_validation_model_restart/r1_fix_random_drop_plan/RANDOM_DROP_EQUAL_DOSE_PLAN.md` | 完成，plan-only |
| random-drop JSON/CSV | `outputs/vla_training_quality_validation_model_restart/r1_fix_random_drop_plan/random_drop_equal_dose_plan.json` / `.csv` | 完成，plan-only |
| held-out split candidate | `outputs/vla_training_quality_validation_model_restart/r1_fix_heldout_split_candidate/heldout_split_candidate.md` | 完成，candidate-only |
| held-out split contract | `outputs/vla_training_quality_validation_model_restart/r1_fix_heldout_split_contract/heldout_split_contract.md` | 完成，`ready_for_metrics` |
| qwen training validation plan | `outputs/vla_training_quality_validation_model_restart/r1_qwen_training_validation_plan/QWEN_TRAINING_VALIDATION_PLAN.md` | 完成，plan-only；blocked pending reviewer/threshold |
| qwen/numeric candidate packet | `outputs/vla_training_quality_validation_model_restart/r2_qwen_numeric_candidate_packet/QWEN_NUMERIC_CANDIDATE_PACKET.md` | 完成，pre-approval only；5 个 ready candidates，未写 `segment_filter_path`，未启动训练 |

R1-fix 当前发现：

- strict structural 在 `sorting_household_items` 只有 1 row / 1 demo / 207 frames，约 `0.0065%` task frames；`max_num_demos=20` 是 no-op，`max_num_demos=54` 才刚纳入 filtered demo。这验证了 R1 剂量过小。
- 现有非结构 ablation 中，剂量较大的候选是 `make_pizza`：`narrow_eval_hard_reject_ablation` 和 `aggressive_repair_or_review_ablation` 都影响 5 demos / 731 frames。但这些仍混有 pipeline / metric / protocol 风险，只能作为 sensitivity，不是 admission evidence。
- random-drop plan 为 4 个候选生成 12 个 control specs，每个候选 3 个随机等量对照；这些只是计划，不是可直接训练的 `segment_filter_path`。
- held-out split contract 已验证 4 个 `sorting_household_items` clean/non-overlap examples，可作为 metrics split 前置契约；它不启动训练、不实例化 dataset、不证明 cleaned 更好。
- qwen-manip training validation plan 已定义 raw / qwen-cleaned-candidate / 3 个 random-drop-equal-dose controls 的比较契约，但仍 blocked：34-row qwen reviewer handoff 有 10 行待人工 verdict，threshold config 尚未 accepted。
- qwen/numeric trajectory candidate packet 已把 first500 的 47 个 offline-clean qwen review rows 拆成 qwen smoothness、jerk/accel/residual/delta dominant、high anomaly fraction、score-floor review-only 等 profile；当前生成 5 个 pre-approval launch candidates，每个只含 raw / cleaned-candidate / 3 个 random-drop-equal-dose specs，仍需显式批准后才能物化 filter 或启动训练。

### R1-qwen：测真正的过滤器

优先验证主方案第 1 层，而不是继续放大 strict structural hard filter：

- qwen-manip smoothing / residual / accel / jerk scorer 产出 first-pass suspicious subset。
- 对 high-confidence suspicious rows 做 cleaned vs raw vs random-drop-equal-dose。
- 保留 strict structural filter 作为第 0 层 sanity，只单独汇报，不代表完整策略。
- 当前 realdata first500 qwen screening 有 487 segments、49 个 qwen suspect，其中 47 个是 offline-clean 的 qwen smoothness review candidate；这些是候选来源，不是训练过滤标签。
- 新的 pre-approval 候选优先级：`spraying_for_bugs` high-anomaly/qwen smoothness，`clean_boxing_gloves` qwen smoothness，`spraying_for_bugs` accel-dominant，`make_microwave_popcorn` jerk-dominant；这些都是 candidate-only，不是 admission label。

### R2：闭环或 VLA-family confirmation

只有 R1-fix/R1-qwen 给出可解释的正信号后，才进入 R2：

- `il_lib`：继续作为轻量 first screen，但需使用正确剂量与随机对照。
- `openpi-comet`：只允许 pi0/pi05 flow-matching action loss；冻结 LLM/vision prefix，只训练 action expert 与必要 projection/time 参数。
- 主指标优先闭环 rollout success / physical validity；如果资源不足，先做 cleaned verified held-out + replay/DA evidence。

## 当前 Todo

- [x] 废弃 VLM2 作为当前证据路径。
- [x] 完成 `il_lib bcrnn_proprio` R1 attempt 4，并纠正其结论边界。
- [x] 将 R1 结果重写为 `underpowered_metric_misaligned_no_evidence_for_cleaning_strategy`。
- [x] 做 task/profile filter-dose scan，找可检出的清洗剂量。
- [x] 生成 equal-dose random-drop control 规范。
- [x] 建立 independent / cleaned verified held-out split 方案。
- [x] 启动 qwen-manip first-pass suspicious subset 的训练验证计划。
- [x] 生成 qwen/numeric trajectory candidate schema、dose scan、pre-approval launch packet。
- [ ] 显式批准后，才物化训练 filter / random-drop controls 并启动训练验证。
- [ ] 只有上述 gate 通过后，再考虑 pi0/pi05 frozen-action-expert R2。

## Guardrails

- 不使用 VLM2，不引用旧 VLM2 R1/R2 指标作为当前结论。
- 不修改 raw data root。
- 不把 review queue bucket、replay bucket、visual review status、manual review status 直接写进训练过滤标签。
- 不推广 official `segment_filter_path` 或 hard filter profile，除非新一轮 paired evidence 支持且用户明确批准。
- 不停止当前机器上的 GPU keepalive/occupy 进程。
