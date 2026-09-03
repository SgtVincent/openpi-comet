# Hierarchical-MoMA-VLA P1 实施报告（item 1、item 2）

- 分支：`feat/hierarchical-moma-vla-p1`，base `41df7a6`
- 工作树：`/mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet-hier-moma`
- 提交：`b17417f`（item 1）、`484f5ac`（item 2）、`5225ac9`（补 `PolicyRecorder` 的 reset 转发）
- 范围：只做设计文档第八节的前两项。item 3（HeldHierarchy-K rollout，K=1/2/5/10）与 item 4（按 planner anchor 构造 held-hierarchy 训练样本）**未开始**，没有训练、没有用 GPU。

> 运行任何测试或脚本前必须设 `PYTHONPATH=<worktree>/src`。`openpi` 是 editable 安装且硬编码指向主树，不设就会改工作树、跑主树的旧代码，而且不报错。`scripts/hier/run_tests.sh` 已经把这件事写进脚本，并在跑之前断言 `openpi.__file__` 解析到本工作树，不符就以 97 退出。

---

## 一、`encode_prefix` 的实际顺序与注意力掩码

设计文档第九节把这一条列为待确认项。结论如下，每条都给出可执行代码的行号（注释不算证据），并且我逐条复核过被引用的行。

### 1.1 prefix 的拼接顺序

只有三段：

```
[ 图像 token（每个相机一段，按 images 列表顺序） | prompt 文本 token | subtask token ]
```

- 图像与语言：`encode_prefix` 调 `model.embed_prefix(...)`（`action_experts/subtask_expert.py:78-80`）；`embed_prefix` 先在相机循环里追加图像（`pi0_pytorch.py:294` 循环、`:303` `embs.append(img_emb)`），再追加语言（`:317` `embs.append(lang_emb)`），最后 `:324` 拼接。
- subtask 第三段追加：`subtask_expert.py:45` `torch.cat([prefix_embs, subtask_embs], dim=1)`，掩码在 `:46` 与 `:64`。
- 短路分支：`subtask_tokens is None or subtask_mask is None or not torch.any(subtask_mask)` 时整段不追加（`subtask_expert.py:38-39`）。

### 1.2 与设计假设不一致的地方

**prefix 里没有独立的 state token 段。** 设计第 4.3、4.4 节的伪代码把 `state` 写成 `encode_prefix(instruction, images, state, hierarchy_tokens)` 的一个独立入参，第九节也按"image、state、subtask 三组 token"来提问。实际不是这样：

- `embed_prefix` 根本没有 `state` 参数（`pi0_pytorch.py:283-284`）。
- pi05 在 suffix 里也跳过了 state 投影：`pi0_pytorch.py:340` `if not self.pi05:` 把 `:341-358` 整段守住。
- state 实际是在更上游被离散成 256 档、拼成字符串写进 prompt 的：`tokenizer.py:504` `np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1`、`:505` `state_str = " ".join(...)`、`:506` `full_prompt = f"Task: {cleaned_text}, State: {state_str};\nSubtask: "`、`:507` 编码。配置项 `pi05_subtask_config.py:46 discrete_state_input: bool = True`。

**所以 `encode_prefix(instruction, images, state, hierarchy_tokens)` 这个签名并不存在。** 后来的人不要照着伪代码去实现。真实签名是 `encode_prefix(*, model, images, img_masks, lang_tokens, lang_masks, subtask_tokens=None, subtask_mask=None)`（`subtask_expert.py:67-77`），state 早已在 `lang_tokens` 里面。

### 1.3 注意力掩码：分块的 prefix-LM

一维 `att_masks` 是 `[0]*(N_img+N_lang) + [1]*N_subtask`：

- 图像段全 0：`pi0_pytorch.py:307` `att_masks += [0] * num_img_embs`
- 语言段全 0：`pi0_pytorch.py:322` `att_masks += [0] * num_lang_embs`
- subtask 段全 1，因为 `encode_prefix` 把 `causal=True` 写死（`subtask_expert.py:88`），走 `:48-49` 的 `torch.ones_like` 分支

二维掩码规则在 `pi0_pytorch.py:84-87`：`cumsum = torch.cumsum(att_masks, dim=1)`，`att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]`，再与 padding 有效性相与。即 query `i` 能看 key `j` 当且仅当 `cumsum[j] <= cumsum[i]`。

由此得到：

| query \\ key | 图像 | prompt（含 state） | subtask |
|---|---|---|---|
| 图像 | 可见 | 可见 | **不可见** |
| prompt | 可见 | 可见 | **不可见** |
| subtask | 可见 | 可见 | 仅可见自身及之前（严格因果） |

图像与 prompt 合成一个双向块；subtask 每个 token 各开一个块，段内严格因果。

**这个结构正好符合设计需要，不需要改。** 分层的自回归分解（memory → primitive → skill → next）要求 hierarchy 段内因果，而 hierarchy 能看到图像和文本、图像和文本看不到 hierarchy —— 这意味着观测编码不受 hierarchy 内容影响。现有掩码就是这样。

### 1.4 位置编码与 KV 交接

- position_ids 是拼接后的**全局** cumsum：`subtask_expert.py:91` `torch.cumsum(prefix_pad_masks, dim=1) - 1`。这一行在 subtask 段拼接之后执行，所以覆盖整个扩展后的 prefix，不是分段计算。
- `past_key_values=None` 写死在 `subtask_expert.py:98`，`use_cache=True` 在 `:100`。**当前每次推理都重建 prefix，所以"动作看的是旧图像"这个风险目前不存在。**
- action expert 通过共享 KV 读 prefix：`pi05_subtask.py:241` 拿到 ctx → `subtask_expert.py:283` → `pi0_pytorch.py:482-526` `denoise_step`，其中 `:497` 让 suffix 无限制地看全部非 padding 的 prefix，`:513` 用 `PreserveCacheLen` 保护缓存长度。

### 1.5 顺带记录：一条死分支

`_embed_conditioning_subtask` 的 `causal=False` 分支（`subtask_expert.py:50-63`，含一段关于双向 subtask 会污染 CE 目标的长注释）在整棵树里没有任何调用点：8 个调用点全传 `causal=True`（`subtask_expert.py:88`、`:192`，`pi05_ki_joint_fast.py:278`、`:467`，`pi05_ki_joint_query.py:356`、`:536`、`:812`、`:1059`）。

**只记录，不删。** 那段注释记录的是设计意图，删掉有它自己的代价，而且不在本次任务范围内。

---

## 二、特殊 token 的选择，以及对 checkpoint 兼容性的影响

### 2.1 三个选项与各自代价

**选项 A：复用词表里已有的保留槽位**（选中）

PaliGemma 的 SentencePiece 词表里本来就有 99 个保留槽位 `<unused0>`..`<unused98>`，id 7..105，`vocab_size()` = 257,152。这是我实测枚举出来的，不是推断。

- 代价：模型看到的文本里，标签是 `<unused0>` 这种不可读的形式。已解决：磁盘上的数据和日志保留可读的 `<MEM>` 写法，只有交给 tokenizer 的那个字符串才转成槽位写法，转换由 `HierarchyTagCodec` 一处负责。
- 词表不变 ⇒ embedding 矩阵不变 ⇒ **不破坏任何 checkpoint**。
- 需要 8 个，有 99 个，余量充足。

**选项 B：扩词表 + 给已有 checkpoint 写迁移**（未选）

- 会同时改 `embed_tokens.weight` 和与之绑定的 `lm_head.weight` 的行数。
- **`strict=False` 挡不住这个问题。** `safetensors/torch.py:205` 永远是 `model.load_state_dict(state_dict, strict=False)`，它自己的 `strict` 参数只在 `:213` 用来决定要不要对缺失/多余的 key 报错；而形状不匹配的 `RuntimeError` 抛在 `torch/nn/modules/module.py:2592-2597`，在 `if strict:` 块（`:2576-2590`）**外面**。所以四条加载路径全部会炸：`train_accelerate.py:4944`（KI 变体本来是 `strict=False`）、`train_pytorch.py:823`、`checkpoint_utils.py:21`（硬 `strict=True`）、`pi05_ki_joint_trainer.py:1082`（默认 `strict=True`）。

  这一条我实跑复现过，不是只读代码推的。构造一个和 PaliGemma 同形状的小模型（`embed_tokens` 与 `lm_head` 权重绑定），以 vocab=64 存盘，再用 vocab=72（相当于加了 8 个标签）去加载：

  ```
  strict=True : RuntimeError -> size mismatch for embed_tokens.weight:
                copying a param with shape torch.Size([64, 8]) ... current model is torch.Size([72, 8]).
  strict=False: RuntimeError -> （同一条报错，逐字相同）
  对照：vocab=64 加载 vocab=64，strict 两种取值都正常
  ```
  两种 `strict` 取值报同一个错，对照组正常 —— 所以"KI 变体用的是 `strict=False`，应该不受影响"这个想法是错的。
- 还有一个更隐蔽的问题：`gemma_pytorch.py:150` 把 `image_token_index` 设成 `257152`，正好是最后一个合法 id 再加 1 的哨兵值。词表一涨，这个哨兵就变成一个真实可寻址的行，图像占位符会和一个真 token 重名。
- 词表大小在代码里是写死的字面量，至少五处要同步改：`gemma_pytorch.py:149`、`:150`、`:159`、`:174`，以及 JAX 侧 `models/gemma.py:41`。
- 全树没有任何 `resize_token_embeddings` 或 `new_num_tokens` 调用（rg 自身返回码 1、0 命中；正对照 `load_pytorch_weights` 返回码 0、7 命中，证明搜索有效）。也就是说迁移路径要从零写。

**选项 C：标签就当普通文本，先不注册特殊 token**（未选）

我实测了它的两个代价，都真实存在：

- 长度：8 个标签作为普通文本共 24 个 token（每个 3 个，例如 `<MEM>` → `['<', 'MEM', '>']`），用槽位则是 8 个。每条样本多花 16 个 token。
- 漂移：标签的切分**会随上下文变化**。`<MEM>` 单独编码是 `['<', 'MEM', '>']`，前面有一个空格就变成 `['▁<', 'MEM', '>']` —— 首个 token 的 id 变了。而槽位写法在任何前缀下都恒为单个 id 7。

### 2.2 选择与理由

**选 A。** 它在每个维度上都不差于另外两个：不破坏 checkpoint、标签边界原子且不漂移、token 数只有三分之一、编解码可完整还原。选项 B 的全部成本换来的能力，选项 A 已经具备。

### 2.3 对 checkpoint 兼容性的结论

**本次改动不破坏 checkpoint 兼容性。** 词表大小仍是 257,152，embedding 与 lm_head 的形状没动，四条加载路径都不受影响。这一点有测试守着（`test_no_vocabulary_growth`），谁要是改成扩词表，测试会先失败。

关于"我们最近加的严格加载校验会大声拦住扩词表"这件事：值得说清楚方向。它确实会拦，但它拦的是形状不匹配，而形状不匹配本来就在 `if strict:` 之外无条件抛错。所以严格加载校验在这里是把一个本来就会炸的事情提前暴露出来，属于按预期工作，不是需要绕开的障碍。我没有绕开它，我是选了一条不触发它的实现路径。

**如果将来确实需要扩词表**，迁移至少要做这些：把旧 checkpoint 的 embedding 按行拷进新矩阵并给新行做初始化；保持 `lm_head.weight` 与 `embed_tokens.weight` 的绑定关系不断（现有 checkpoint 把这个绑定在 safetensors 的 `__metadata__` 里去重存储，见 `pi05_ki_joint_query_config.py:57-62`）；同步五处写死的词表字面量；把 `image_token_index` 挪到新的越界位置。这些我都没有做，因为选了 A。

---

## 三、实现了什么

### 3.1 item 1：注册标签、压缩模型文本

新增 `src/openpi/models/hierarchy_tokens.py`：

- `HIERARCHY_TAG_TO_SLOT`：8 个标签到 `<unused0>`..`<unused7>` 的映射。这是一份落到磁盘和 checkpoint 里的约定，注释里写明只能在末尾追加、不能重新编号。
- `build_hierarchy_text(...)`：按 memory → primitive → skill → next 的顺序生成模型文本。不含 `annotation_index`、`segments`、原始物体 handle、任务 id、帧号。字段内部的空白会被压成单个空格，换行只作字段分隔符；字段内容里如果出现标签字符会直接报错，防止标注内容伪造字段边界。
- `HierarchyTagCodec`：负责可读形式与槽位形式的互转、编码、解码。构造时就检查 tokenizer 是否真的把这 8 个槽位编成单 token，不满足直接报错，避免悄悄退化成多 token 标签。

`SubtaskTokenizer` 新增 `tokenize_hierarchy` 与 `decode_hierarchy`（`src/openpi/models/tokenizer.py`），掩码语义与 `tokenize_subtask` 完全一致（BOS、因果 ar_mask、BOS 之后才算 CE 目标、按 `subtask_max_len` 补齐或截断），只是标签换成槽位。超长时的警告文字写明"截断会丢掉尾部标签并破坏监督目标"，而不是只说一句截断了。

**压缩了多少（用设计 3.1 节那段例子实测）：**

| 写法 | token 数 |
|---|---|
| 文档 3.1 节原样（`<MEM>` 里带换行），当普通文本 | 69 |
| 本实现的文本（字段内换行压成空格），当普通文本 | 66 |
| 本实现 + 保留槽位标签（含 BOS/EOS） | **52** |

**一处对文档的有意偏离**：文档 3.1 节的例子在 `<MEM>` 标签内部换行。本实现把字段内部的空白压成单个空格，换行只用作字段之间的分隔符。理由是换行如果既能出现在字段内、又是字段分隔符，一条标注里的换行就能伪造出一个字段边界。这属于 item 1 要求的"压缩 canonical text"，但确实和文档给的样例长得不一样，所以在这里写明，免得后来的人以为是实现漏了。

### 3.2 item 2：`previous_hierarchy_tokens` 输入通路与 hierarchy token 缓存

新增 `src/openpi/models/hierarchy_cache.py`：

- `HierarchyTokenCache`：**只存 token id**。拒绝 `past_key_values` 这类载荷（按类型名匹配 `cache`/`past_key_values` 等），拒绝浮点张量，拒绝 `None` 和空数组。`invalidate(reason)` 必须给理由 —— rollout 日志里一次没有理由的失效和一个 bug 分不出来。没有 TTL、没有隐式过期：忘记失效的代价必须是一个明确的报错，而不是一次安静的过期读取。
- `observation_fingerprint(images, lang_tokens)` 与 `assert_prefix_fresh(...)`：第 4.5 节风险的检查点。
- `store` 的语义是覆盖（对应设计 2.3 节"最小 commit 就是覆盖赋值"），`generation` 计数 Planner tick 次数。

`Policy` 里那三个从来没被读写过的字段（`policy.py:57-59` 的 `_cached_subtask_prompt` / `_cached_subtask_tokens` / `_cached_subtask_text`）现在有了真实实现：后两个变成读缓存的属性，并新增 `Policy.reset(reason)` 用于 episode 边界清空。跨 episode 残留的 hierarchy 一定是过期的，而下游没有任何环节会发现，所以做成显式方法。

这个 `reset` 落在一个**已经存在的钩子**上：`BasePolicy.reset()` 本来就声明为空实现（`packages/openpi-client/src/openpi_client/base_policy.py:10-12`），而 `ActionChunkBroker` 已经在调 `self._policy.reset()`（`action_chunk_broker.py:48`，不带参数）。所以失效逻辑不需要新增调用点，现有运行时就会触发 —— 这也是 `reason` 必须保留默认值的原因。

顺带修掉一个真实缺口：`PolicyRecorder` 是个包装类，它继承了 `BasePolicy` 那个空的 `reset()` 而**没有转发给内层 policy**。也就是说，一旦把 `Policy` 包进 `PolicyRecorder`，hierarchy 失效就被悄悄废掉了：内层 policy 永远收不到 episode 结束的通知，held hierarchy 会漏进下一个 episode，而且没有任何地方会报。已加上转发（`policy.py:190-203`）并配了测试。保护必须对**每一个**使用方都可达，只对没被包装的那个可达不算数。

### 3.3 conditioning 通道：hierarchy 并入 subtask 段，不新开一条

hierarchy 文本写进已有的 `subtask_*` 槽位，不是第三条并行通道。这棵树里已经有两条互不相连的 GT plan 通道（`subtask_tokens` 和 prompt 文本），并且此前查出 golden-rule 路径**一次都没有**碰过 `subtask_tokens`，而当时大家以为它在给模型做 conditioning。再加第三条只会把同一个问题放大。

兼容性：纯 subtask 字符串走 `tokenize_subtask`，逐字节和以前一样（有测试 `test_subtask_text_still_works_unchanged` 守着）。hierarchy 文本在 token 层面可区分，因为它以保留槽位 `<MEM>` 的 id 开头，所以用 subtask 文本训出来的 checkpoint 不会被悄悄改变含义。

### 3.4 把三级静默丢字段改成出错即停

现状是三级连环，全程不报错：

1. `transforms.py:395-396`：`include_subtask_text=False` 时 `result.pop("subtask_text", None)`
2. `transforms.py:435-442`：字段没了就造一个全零、mask 全 False 的 subtask
3. `subtask_expert.py:38-39`：看到 `not torch.any(subtask_mask)` 直接短路，整段从 prefix 消失

结果是训练/推理悄悄变成无 conditioning，但 loss 照常输出。

改法（**默认行为一律不变**）：

- `PromptFromLeRobotItem` 增加 `include_hierarchy_text` 与 `require_hierarchy_text`。要求 hierarchy 却又配置成丢弃它，属于配置矛盾，直接报错；要求 hierarchy 但字段缺失，也报错。
- `TokenizeSubtaskInputs` 增加 `require_hierarchy`。为真且没有 `hierarchy_text` 时**报错**，不再往下走那个造零张量的分支。

这和严格加载校验是同一个思路：必需的输入缺失是错误，不是默认值。

### 3.5 KV 路径没有做任何优化

按要求保持 P1 的做法：只缓存 hierarchy token id，每个 chunk 用最新观测重算 prefix KV。

顺带记一个**将来的**可能性，现在不做：因为图像和文本在掩码上看不到 subtask 段（见 1.3 的表），理论上存在只缓存前两段 KV、每次只重算 hierarchy 段的方案。它需要改掩码形状（从 `[B,N,N]` 变成 `[B,N_new,L+N_new]`）、加位置偏移、处理缓存长度被原地修改的问题。这正是旧观测重新混进来的典型路径，**不在 P1 范围内，现在不要做**。

---

## 四、测试，以及每个测试在什么情况下会失败

新增 45 个测试，分三个文件。每个测试的 docstring 里都写了 "FAILS IF"。

### 4.1 明确要求的三个测试

**（1）复用了含旧观测的 prefix KV 就失败**
`tests/test_hierarchy_cache.py::test_reusing_prefix_kv_from_a_stale_observation_raises`
用观测 A 编码 prefix，然后拿这个 prefix 去跑观测 B 的动作 —— 这就是"跨 chunk 缓存整个 KV"在代码里的样子。
失败条件：守卫被删、指纹不再打戳或不再比较、有人重新引入跨 chunk 的 prefix KV 复用。此时不会抛异常，模型会安静地基于旧图像和旧 state 出动作。
配套的反向对照 `test_fresh_prefix_kv_is_accepted` 保证守卫不会误伤正确路径 —— 没有它的话，一个无条件抛错的守卫也能通过上面那个测试。
另有 `test_fingerprint_changes_when_the_image_changes` / `..._state_changes` / `..._is_stable_for_the_same_observation` 三个，保证指纹本身是敏感且确定的。少了这几个，一个恒定指纹会让上面所有守卫测试变成空转还照样通过。

**（2）`include_subtask_text=False` 悄悄丢掉 hierarchy 就失败**
`tests/test_hierarchy_transforms.py::test_require_hierarchy_text_rejects_the_contradictory_config`、`test_require_hierarchy_text_raises_when_the_field_is_absent`、`test_require_hierarchy_raises_instead_of_fabricating_zeros`
第三个断言的是**抛错**，不是断言得到一个零张量 —— 零张量恰恰就是那个静默故障，去断言它等于把 bug 固化下来。
另有 `test_encode_prefix_really_does_drop_an_all_false_mask`，实测证明第三级短路确实存在（全 False mask 下 prefix 长度和不传 subtask 时完全一样）。它的作用是：如果哪天短路没有了，说明我上面这套推理的前提变了，应该重新审视而不是让结论留着。

**（3）特殊 token 编解码往返完整保留标签边界**
`tests/test_hierarchy_tokens.py::test_special_token_round_trip_preserves_tag_boundaries_exactly`
除了逐字节相等，还断言每个标签在 id 流里恰好占 1 个 id、恰好出现 1 次。
`test_round_trip_is_stable_across_surrounding_context` 补上下文不变性：标签 id 不能随前缀变化。

### 4.2 其余测试覆盖的点

- 选择本身的正对照 `test_raw_text_tags_would_drift_and_cost_more`：如果普通文本标签其实既不贵也不漂移，那选项 C 的否决理由就不成立，这个决定应当重新讨论而不是默默保留。
- `test_no_vocabulary_growth`：词表一旦增长就失败。
- `test_reserved_slots_do_not_collide_with_fast_action_tokens`：槽位 id 若落进 FAST 动作 token 区间就失败。
- 缓存类：拒绝 KV 载荷、拒绝浮点张量、拒绝 None/空、冷缓存读取报错、失效必须带理由、覆盖语义与 generation 计数、`Policy.reset` 清空。
- `test_policy_recorder_forwards_reset_to_the_wrapped_policy`：`PolicyRecorder` 一旦不转发 `reset`，就失败。这条测的是"保护对每个使用方都可达"，不是缓存本身的逻辑。
- 兼容性：`test_subtask_text_still_works_unchanged`、`test_legacy_zero_fabrication_still_happens_when_not_required`、`test_guard_is_a_noop_for_callers_that_do_not_opt_in`。

### 4.3 变异测试：证明这些测试真的会失败

只说"这个测试会失败"是不够的，所以我逐个把守卫改坏，确认对应测试确实红，然后还原。

| 变异 | 结果 |
|---|---|
| M1 删掉 `compute_velocity_infer` 里的 `assert_prefix_fresh` | `test_reusing_prefix_kv_from_a_stale_observation_raises` 失败 |
| M2 `encode_prefix` 不再打指纹戳 | 上面那个 + `test_encode_prefix_stamps_the_observation_fingerprint` 失败 |
| M3 指纹退化成常量 | 3 个失败（含 stale-KV 那个） |
| M4 指纹只看图像、忽略语言/state | `test_fingerprint_changes_when_the_state_changes` 失败 |
| M5 `require_hierarchy` 失效 | `test_require_hierarchy_raises_instead_of_fabricating_zeros` 失败 |
| M6 把 `<MEM>` 改回普通文本、不用保留槽位 | 5 个失败 |
| M7 缓存接受 KV 载荷 | `test_cache_refuses_a_kv_payload` 失败 |
| M8 `PolicyRecorder` 不转发 `reset` | `test_policy_recorder_forwards_reset_to_the_wrapped_policy` 失败 |

8 个变异全部命中预期的测试，还原后回到 45 全绿。每次变异脚本执行完都核对过树里没有残留（搜 `MUTANT` 返回码 1、0 命中；同目录正对照有命中）。

M4 只挂掉 1 个测试是符合预期的：stale-KV 那个测试里图像也变了，所以只看图像的指纹仍能发现。这说明测试之间的职责是分开的，`test_fingerprint_changes_when_the_state_changes` 才是专门盯 state 那一路的。

---

## 五、测试数量：改动前 / 改动后

跑法：受 32 GiB 的 cgroup 限制（`/sys/fs/cgroup/memory.max`），一个 pytest 进程把所有重型 torch 模块 import 进来会被 OOM 杀掉（实测退出码 137）。所以按**每个文件一个进程**跑。前后两次用同一个解析脚本 `scripts/hier/parse_results.py` 统计，保证可比。

改动前的数字是对 base commit `41df7a6` 的**纯净解包**（`git archive 41df7a6 | tar -x -C /tmp/basetree`）跑出来的，不是在我改过的工作树上跑的。第一次基线跑到一半我才意识到它被污染了：那次是后台跑的，跑到后面几个文件时我已经改了 `src/`，而每个文件是独立进程、会 import 到改后的代码。那次结果已作废并重跑。

| | 文件数 | passed | failed | errors | skipped |
|---|---|---|---|---|---|
| 改动前（`41df7a6` 纯净解包） | 55 | 740 | 10 | 0 | 12 |
| 改动后（本分支） | 58 | **785** | 10 | 0 | 12 |

差值完全对得上：文件 +3（我新增的 3 个测试文件），passed +45（正好是我新增的 45 个测试），failed 与 skipped 一个没动。740 + 45 = 785。

那 10 个失败在改动前后是**同一批测试、逐条相同**（用 `diff` 比对过 FAILED 行，结果 IDENTICAL）：

- `tests/test_pi05_ki_a100_bf16_formal.py` 2 个：还是那个 `Numba needs NumPy 2.3 or less. Got NumPy 2.4`
- `tests/test_skill_bridge_integration.py` 8 个：`TestBridgeConfigPaths` 检查 checkpoint / 数据 / 输出目录是否存在，这些路径在本机不存在

### 上面这张表为什么排除了 2 个文件（这一条必须说清楚）

第一次算出来的原始数字是「改动前 795 passed / 23 failed，改动后 853 passed / 10 failed」，看起来像是我的改动顺手修好了 13 个测试。**不是。** 那 13 个是我基线跑的位置造成的假象，与本次改动无关：

- `tests/test_pi05_ki_v100_fp32_formal.py` 11 个：报 `FATAL: REPO_ROOT is not a git worktree: /tmp/basetree`。我的基线是 `git archive` 解出来的纯文件，没有 `.git`，启动脚本因此拒绝运行。
- `scripts/test_run_skill_metric_multinode_sweep.py` 2 个：报 `refusing to write formal evaluation outputs under /tmp`，正好撞上我把基线放在 `/tmp` 下。

两者都是「基线放在 /tmp」的副作用，不是真实的既有失败，更不是我修好的。所以这 2 个文件在**两边同时**排除，上面那张表才是同口径的比较。原始数字一并记在这里，免得有人后来只看到 795→853 就以为多了 13 个修复。

### 没能跑的部分

以下 4 个文件在收集阶段就失败，原因早于本分支，与本次改动无关。前后两次**用同样的方式排除**，所以计数可比。**这部分我没有跑，不能算通过。**

- `scripts/test_b1k_openpi.py`、`tests/test_lean_b1k_stride12.py`：`ImportError: Numba needs NumPy 2.3 or less. Got NumPy 2.4`（环境里 numba 与 numpy 版本不兼容）
- `packages/openpi-client/src/openpi_client/image_tools_test.py`、`msgpack_numpy_test.py`：`openpi-client` 是**第二个** editable 安装，仍然指向主树，pytest 报 `import file mismatch`（指向 `openpi-comet/packages/...`）。本次只 pin 了 `openpi`，没 pin `openpi_client`。

另外两个只在「改动后」这一侧存在、无法与基线比较的项：

- `scripts/test_preprocess_b1k_structured_memory.py`：这是我接手前就已经放在工作树里的 P0 未跟踪文件，base commit 里没有。为了让差值只反映本次改动，它在改动后那一侧被排除，我也没有提交它。
- 上面说的那 2 个位置相关的文件（`test_pi05_ki_v100_fp32_formal.py`、`test_run_skill_metric_multinode_sweep.py`）：**在本分支上是全过的**（分别 19 passed、47 passed），只是拿不到可比的基线数字。


---

## 六、已验证 / 未验证

### 已验证（我自己跑过，或读了可执行代码行）

- 1.1–1.5 全部结论。子代理给的引用我抽查了 10 处，全部字面命中；`if not self.pi05:`、prompt 模板、三级静默丢字段、`load_strict` 名单、`checkpoint_utils.py:21` 的硬 `strict=True`、`gemma_pytorch.py` 四处词表字面量，都是我自己复核的。
- 词表 257,152；99 个保留槽位 `<unused0>`..`<unused98>`（id 7..105）—— 实际加载 tokenizer 枚举得到。
- 保留槽位是 user-defined、在文本中原子匹配（`a<unused0>b` → `['a','<unused0>','b']`），编解码可完整还原。
- 普通文本标签每个 3 token、共 24 个，且首 token id 随前缀空格变化。
- 全树没有 `resize_token_embeddings` / `new_num_tokens`（rg 自身返回码 1；正对照返回码 0、7 命中）。注意这里我特意没有在管道后读 `$?`，那读到的会是管道最后一个进程的返回码。
- 44 个新测试全绿；7 个变异全部命中预期测试；变异后树里无残留。
- 我的改动没有给已有文件增加新的 ruff 报错（逐文件对比 base：`transforms.py` 1→1、`policy.py` 0→0、`tokenizer.py` 1→1、`subtask_expert.py` 5→5、`pi05_subtask.py` 5→5）。两个新增的 src 文件 ruff 全通过。
- `policy.py` 只删了两行（那两个死字段），模型初始化那段没动 —— 用 `git diff -U0` 逐行确认过。

### 未验证（明确列出，不与上面混在一起）

- **没有跑过真实模型的端到端推理。** `test_encode_prefix_stamps_the_observation_fingerprint` 等用的是 stub 模型（被测的 `encode_prefix` / `compute_velocity_infer` 是真代码，但 PaliGemma 主干是 stub）。指纹守卫在真实 GPU 推理下的行为**未验证**。本任务不用 GPU，所以没做。
- **没有训练过，也没有跑过 rollout。** hierarchy 文本在真实训练里的 CE 收敛情况、对动作质量的影响，全部未测。
- **扩词表会抛什么错，异常类型与报错形状已实跑复现，但没有在真实 checkpoint 上跑过。** 我用一个同构的小模型（tied embedding/lm_head）复现了 `RuntimeError: size mismatch`，并确认 `strict=True/False` 行为一致。**但**具体的 key 字符串（`paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight` 之类）是从模型构造代码推断的，我没有真的拿一个 pi05 checkpoint 去触发一次。
- **没有验证 DeepSpeed/Accelerate 的 `accelerator.load_state` 在形状变化时的行为。** 上面只覆盖了 safetensors 那条路径。
- **多相机时图像段内部的先后顺序没有追。** 它取决于 `observation.images.values()` 的字典顺序（`pi0_pytorch.py:262`）。我没有针对具体配置确认相机 key 的顺序。目前的实现不依赖这个顺序，但如果将来要按相机做 per-camera 的事情，需要先确认。
- **`hierarchy_text` 字段的上游数据生产没有做。** 本次只做了消费侧（transform、tokenizer、缓存）。P0 产出的 JSONL 里叫什么字段、怎么接进 LeRobot dataset，属于 item 4 的范围，未开始。
- **指纹的碰撞概率没有量化。** 每个张量采样最多 512 个元素、取 sum 与 abs-max。理论上两个不同观测可能撞上同一个指纹。我没有做碰撞率测量；对本用途（发现"整段观测换了一批"）够用，但它不是密码学意义上的摘要。
- **设计第九节其余待确认项没有回答**：每次 policy call 实际消费多少 action、是否用 temporal ensemble、K 按 chunk 还是物理秒、组合 primitive fallback 怎么处理、Action Expert 该读完整 hierarchy 还是只读 current+next skill。其中最后一条会影响 item 3/4 的做法。
