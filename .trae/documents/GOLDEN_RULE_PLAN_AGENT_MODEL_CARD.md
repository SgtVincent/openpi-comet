# Golden Rule Plan Agent Model Card

## 1. 一句话定义

Golden Rule Plan Agent 是一个 **用 GT skill plan 驱动 VLA 执行** 的评测代理：它不让模型自己决定“下一步做什么”，而是直接从 demo annotation 读取 skill 序列，只评估 **给定 skill prompt 下的执行上界**。

当前主要目标不是通用规划，而是回答一个更窄但更关键的问题：

> 如果高层 plan 是对的，当前 checkpoint 的低层执行到底能做到什么程度？

---

## 2. 当前实验对象

- **checkpoint**:
  `/mnt/bn/navigation-hl/mlx/users/chenjunting/checkpoints/openpi_comet/pi05_skill_pt12_pretrain_4x8_20260325_065102/30997`
- **task**: `turning_on_radio`
- **demo**: `00000010`
- **主运行脚本**: `openpi-comet/scripts/run_golden_rule_e2e_test.sh:1`
- **主要模式**: evaluator 与 policy server 分离的 **remote websocket** 模式

---

## 3. 这个 agent 到底做什么

### 3.1 输入

1. **任务级 prompt**（来自 task mapping）
2. **GT plan**（来自 demo annotation）
3. **当前 observation**
4. **可选 prompt 控制项**
   - `SKILL_PROMPT_TEMPLATE`
   - `SKILL_PROMPT_DETAIL_MAP_JSON`
   - `SKILL_PROMPT_OVERRIDE`

### 3.2 输出

1. 当前 skill 对应的 policy prompt
2. 动作序列
3. skill 完成/超时后的 plan advancement
4. 每个 skill 的成功结果、diagnostics、视频

### 3.3 它**不**做什么

1. 不做开放式 task decomposition
2. 不做真正的 reasoning-based next-skill selection（golden rule 模式下绕过）
3. 不代表完整 agent 能力，只代表 **“已知正确 plan 下的执行能力上界”**

---

## 4. 系统结构

## 4.1 GT 计划加载

GT plan 由 `GTPlanLoader` 从 demo annotation 中读取，支持多种 annotation 路径格式，包括：

- `{annotations}/{task_name}/{demo_id}.json`
- `{annotations}/task-{idx:04d}/{demo_id}.json`
- `episode_{demo_id}.json` 文件名变体

关键实现：

- annotation path 兼容逻辑：`BEHAVIOR-1K/OmniGibson/omnigibson/learning/gt_plan_loader.py:86`
- 计划加载与排序：`BEHAVIOR-1K/OmniGibson/omnigibson/learning/gt_plan_loader.py:151`

这意味着 Golden Rule 模式本质上是：

> 读取 demo 中已经标好的 skill 边界与 skill description，然后按顺序驱动 policy。

## 4.2 Server 侧执行包装器

真正驱动 policy 的核心是 `GoldenRulePolicyWrapper`：

- 类定义：`openpi-comet/src/openpi/shared/golden_rule_policy.py:16`
- skill prompt 解析：`openpi-comet/src/openpi/shared/golden_rule_policy.py:159`
- remote advance signal 消费：`openpi-comet/src/openpi/shared/golden_rule_policy.py:262`
- skill 前进 + session 轮换：`openpi-comet/src/openpi/shared/golden_rule_policy.py:286`

它做了几件关键事：

1. 当 `fine_grained_level == 2` 时，不走 VLM reasoner，而是直接从 plan loader 取当前 skill。
2. 把 skill description 映射成最终给模型的 prompt。
3. 在 skill 完成/超时后前进到下一个 skill。
4. 在每次前进 skill 时调用 `rotate_session()`，强制切断上一 skill 的 runtime session memory。

## 4.3 B1K wrapper 的 session 隔离

底层 session 轮换逻辑在 `B1KPolicyWrapper`：

- session 元数据与 active session 机制：`openpi-comet/src/openpi/shared/eval_b1k_wrapper.py:33`
- `reset()`：`openpi-comet/src/openpi/shared/eval_b1k_wrapper.py:129`
- `rotate_session()`：`openpi-comet/src/openpi/shared/eval_b1k_wrapper.py:141`
- 单测：`openpi-comet/src/openpi/shared/eval_b1k_wrapper_test.py:79`

`rotate_session()` 的作用不是 reset task，而是：

1. 为模型切换一个新的 `_session_id`
2. 尝试 `clear_session(old_session_id)`
3. 清空 action queue / last_action / step_counter
4. 重新设置 active session 与 streaming state

这是为了解决 **跨 skill streaming memory 污染** 的假设问题。

## 4.4 Evaluator 侧 skill 驱动与状态恢复

Evaluator 侧由 `GoldenRuleEvaluator` 管理 skill 成功判定、skill 切换、状态恢复和日志记录：

- 类定义：`BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_golden_rule.py:82`
- policy 包装逻辑：`BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_golden_rule.py:142`
- episode setup / plan 注入：`BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_golden_rule.py:194`
- skill 成功判定：`BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_golden_rule.py:322`
- step 内 skill advancement / restore / remote notify：`BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_golden_rule.py:365`
- episode diagnostics 汇总：`BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_golden_rule.py:465`

当前 evaluator 侧有两个关键修复：

1. **state restore 成功后刷新 obs**  
   `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_golden_rule.py:451`

2. **remote websocket 模式下显式通知 server 前进 plan**  
   `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_golden_rule.py:456`

---

## 5. Remote websocket 模式的数据流

当前主实验走的是 remote 模式，而不是本地把 openpi wrapper 直接 import 进 evaluator。

### 5.1 Server 启动

server 入口：`openpi-comet/scripts/serve_golden_rule.py:1`

关键位置：

- 从 BEHAVIOR 侧 lazy import `GTPlanLoader`：`openpi-comet/scripts/serve_golden_rule.py:103`
- 创建 `GoldenRulePolicyWrapper`：`openpi-comet/scripts/serve_golden_rule.py:168`

### 5.2 Eval 启动

e2e 脚本：`openpi-comet/scripts/run_golden_rule_e2e_test.sh:1`

关键参数：

- checkpoint、demo、timeout：`openpi-comet/scripts/run_golden_rule_e2e_test.sh:13`
- prompt ablation 相关环境变量：`openpi-comet/scripts/run_golden_rule_e2e_test.sh:21`
- `RESTORE_AT_EACH_PRIMITIVE_START / MAX_LEN / ACTION_HORIZON`：`openpi-comet/scripts/run_golden_rule_e2e_test.sh:29`
- server 启动命令：`openpi-comet/scripts/run_golden_rule_e2e_test.sh:78`
- evaluator 启动命令：`openpi-comet/scripts/run_golden_rule_e2e_test.sh:162`

### 5.3 远端同步机制

最初 remote 模式的核心 bug 是：

> evaluator 知道 skill 已经完成，但 server 侧 wrapper 的 plan_loader 并不会自动前进。

现在修复后，流程是：

1. evaluator 检测 skill 完成
2. 如果是本地 wrapper，则直接 `_advance_plan()`
3. 如果是 remote wrapper，则给 obs 注入 `golden_rule_advance_plan=True`
4. server 在下一次 `act()` 时消费该 control signal 并 `_advance_plan()`

这部分实现对应：

- evaluator 发送信号：`BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_golden_rule.py:456`
- wrapper 消费信号：`openpi-comet/src/openpi/shared/golden_rule_policy.py:274`

日志里已经能看到它实际生效，例如：

- `eval_logs/golden_rule_e2e_turning_on_radio_00000010_remote_sync_restore_each_rotatesession_full/server.log:28`
- `eval_logs/golden_rule_e2e_turning_on_radio_00000010_remote_sync_restore_each_rotatesession_full/server.log:32`
- `eval_logs/golden_rule_e2e_turning_on_radio_00000010_remote_sync_restore_each_rotatesession_full/server.log:36`

---

## 6. Prompt 设计

Golden Rule 模式下，模型最终拿到的不是任务级自由 prompt，而是“任务 + 当前 skill”的组合 prompt。

基础逻辑位于：

- `openpi-comet/src/openpi/shared/golden_rule_policy.py:122`
- `openpi-comet/src/openpi/shared/golden_rule_policy.py:159`

支持三类 prompt 控制：

1. **`skill_prompt_template`**：用模板包裹 skill prompt
2. **`skill_prompt_detail_map_json`**：按 skill 名字替换成更细的文字描述
3. **`skill_prompt_override`**：完全替换 skill prompt

这使它不仅是评测代理，也是一个可做 prompt ablation 的 skill-conditioned execution harness。

---

## 7. 评测输出与证据

## 7.1 当前最重要的成功 run

成功 run（4/4）对应 metrics：

- `eval_logs/golden_rule_e2e_turning_on_radio_00000010_remote_sync_restore_each_rotatesession_full/metrics/golden_rule_turning_on_radio_aggregate.json:1`

关键结果：

- `skill_success_rate = 1.0`
- `endtoend_success_rate = 1.0`
- `press` skill 成功，耗时 **401 steps**

对应视频：

```text
/mnt/bn/behavior-data-hl/chenjunting/repo/eval_logs/golden_rule_e2e_turning_on_radio_00000010_remote_sync_restore_each_rotatesession_full/videos/turning_on_radio_golden_rule_demo00000010.mp4
```

## 7.2 当前最关键的失败 run

### A. full-sequence replicate 失败（3/4）

metrics：

- `eval_logs/golden_rule_e2e_turning_on_radio_00000010_remote_sync_restore_each_rotatesession_full_replicate2/metrics/golden_rule_turning_on_radio_00000010.json:1`

现象：

- `move to` 成功
- `pick up from` 成功
- `press` 超时，耗时 **1088 steps**
- `place on` 最后仍可 `success_env`

视频：

```text
/mnt/bn/behavior-data-hl/chenjunting/repo/eval_logs/golden_rule_e2e_turning_on_radio_00000010_remote_sync_restore_each_rotatesession_full_replicate2/videos/turning_on_radio_golden_rule_demo00000010.mp4
```

### B. isolated press 失败（0/1）

metrics：

- `eval_logs/golden_rule_e2e_turning_on_radio_00000010_press_diag_topbutton_h32_resetrotate/metrics/golden_rule_turning_on_radio_00000010.json:1`

现象：

- 单独评 press，仍然 **1088 steps timeout**
- 说明问题不只是 full-sequence 上下文污染

视频：

```text
/mnt/bn/behavior-data-hl/chenjunting/repo/eval_logs/golden_rule_e2e_turning_on_radio_00000010_press_diag_topbutton_h32_resetrotate/videos/turning_on_radio_golden_rule_demo00000010.mp4
```

### C. 其他失败视频

```text
/mnt/bn/behavior-data-hl/chenjunting/repo/eval_logs/golden_rule_e2e_turning_on_radio_00000010_remote_sync_restore_each_rotatesession_holdpress_full/videos/turning_on_radio_golden_rule_demo00000010.mp4
/mnt/bn/behavior-data-hl/chenjunting/repo/eval_logs/golden_rule_e2e_turning_on_radio_00000010_press_diag_topbutton_h32_rerun/videos/turning_on_radio_golden_rule_demo00000010.mp4
/mnt/bn/behavior-data-hl/chenjunting/repo/eval_logs/golden_rule_e2e_turning_on_radio_00000010_remote_sync_restore_each_rotatesession_resetrotate_full/videos/turning_on_radio_golden_rule_demo00000010.mp4
```

---

## 8. 当前结论：已经排除了什么

基于当前代码修改、metrics 和视频，已经基本排除以下解释：

1. **不是 remote skill plan 不同步**  
   `server.log` 中已看到 remote advance signal 被 server 消费。

2. **不是 state restore 没刷新 obs**  
   evaluator 已在 restore 后显式刷新 obs：`BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_golden_rule.py:451`

3. **不是 success checker 本身坏了**  
   teacher-forcing / replay 路径能成功复现 success。

4. **不是单纯 action horizon 问题**  
   isolated press 在不同设置下仍可失败。

5. **不是单纯 session memory 污染**  
   `rotate_session()` 已加入，但 isolated press 仍失败。

因此当前最可信的判断是：

> `press the top button on the radio` 这个 primitive 本身存在明显控制不稳定性，尤其是在持握 radio 过程中，几何关系容易漂移，最终无法稳定形成有效按压姿态。

---

## 9. 这个 model 的优点

1. **把高层规划误差和低层执行误差分离开了**  
   很适合做“执行上界”评测。

2. **工程闭环完整**  
   从 GT plan loader、policy wrapper、evaluator 到视频输出都是连通的。

3. **支持 remote / local 两种模式**  
   当前主用 remote websocket，更接近真实部署结构。

4. **支持 prompt ablation**  
   能直接研究 skill wording 对执行的影响。

5. **支持诊断模式**  
   可以只跑单个 `diagnostic_skill_idx` 来隔离 primitive 问题。

---

## 10. 当前主要缺点 / 风险

1. **它不是通用 agent**  
   一旦去掉 GT plan，它不能代表完整任务求解能力。

2. **press 这类接触型 primitive 很脆弱**  
   当前 `turning_on_radio` 的瓶颈几乎都集中在这里。

3. **成功样本存在一定随机性**  
   同样设置下可以出现 4/4 和 3/4 两种结果。

4. **目前更多是 execution harness，不是稳定产品**  
   还在快速迭代 prompt / session / restore / diagnostic 逻辑。

---

## 11. 我建议你 review 代码时重点看什么

### 11.1 如果你关心“plan 是怎么喂给模型的”

- `openpi-comet/src/openpi/shared/golden_rule_policy.py:159`
- `openpi-comet/scripts/serve_golden_rule.py:185`
- `BEHAVIOR-1K/OmniGibson/omnigibson/learning/gt_plan_loader.py:151`

### 11.2 如果你关心“remote 模式为什么以前不同步、现在怎么同步”

- `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_golden_rule.py:419`
- `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_golden_rule.py:456`
- `openpi-comet/src/openpi/shared/golden_rule_policy.py:274`

### 11.3 如果你关心“为什么会做 session rotate”

- `openpi-comet/src/openpi/shared/eval_b1k_wrapper.py:141`
- `openpi-comet/src/openpi/shared/golden_rule_policy.py:286`
- `openpi-comet/src/openpi/shared/eval_b1k_wrapper_test.py:79`

### 11.4 如果你关心“为什么判定当前问题更像控制不稳定而不是 checker bug”

- 成功 metrics：
  `eval_logs/golden_rule_e2e_turning_on_radio_00000010_remote_sync_restore_each_rotatesession_full/metrics/golden_rule_turning_on_radio_aggregate.json:24`
- 失败 metrics：
  `eval_logs/golden_rule_e2e_turning_on_radio_00000010_remote_sync_restore_each_rotatesession_full_replicate2/metrics/golden_rule_turning_on_radio_00000010.json:14`
- isolated press 失败 metrics：
  `eval_logs/golden_rule_e2e_turning_on_radio_00000010_press_diag_topbutton_h32_resetrotate/metrics/golden_rule_turning_on_radio_00000010.json:4`

---

## 12. 下一轮最可能有效的方向

如果你看完视频后认可当前诊断，我认为下一轮值得优先试的方向是：

1. **缩短 press 的 replan 周期**  
   例如更小的 `MAX_LEN` / `ACTION_HORIZON`

2. **继续强化 press prompt 的执行约束**  
   更强调 hold 稳定、button 对准、press 时不要丢物体

3. **做更强的 press-only 诊断**  
   保持其它变量不变，只对 press skill 高频实验

4. **必要时按 skill 重启 server**  
   作为“比 rotate_session 更强隔离”的对照实验

---

## 13. 当前定位

我对这个系统的当前定义是：

> **Golden Rule Plan Agent = 一个用于测量 skill-conditioned execution ceiling 的 GT-plan-driven agent harness。**

它已经足够好用来：

- 做 skill-level 上界评测
- 做 prompt ablation
- 做 primitive 失败定位
- 做 success checker / restore / remote sync 的交叉验证

但它还没有证明：

- 模型已经具备稳定的接触型 manipulation 能力
- 完整 end-to-end agent 在开放规划下同样可靠

这也是为什么现在最值得你 review 的，不只是“代码对不对”，而是：

1. 代码路径是否把 planning/execution 分离得足够干净
2. `press` 失败究竟更像 prompt 问题、控制问题，还是观测/动作闭环问题
3. 成功 run 和失败 run 的视频里，最早出现分叉的时刻到底在哪

