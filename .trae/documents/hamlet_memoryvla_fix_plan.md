## OOM 诊断与当前状态：HAMLET & MemoryVLA 实现中的显存问题

本文档最初基于 commit `0bb8b12d` 的逐行审查，定位了 HAMLET / MemoryVLA 中导致 OOM 与显存持续增长的关键问题。

截至 `2026-04-29`，该文档已同步为“诊断 + 当前落地状态”版本：

- 第一、二部分保留原始问题诊断，说明为什么当时会 OOM
- 第三、四、五部分更新为当前代码状态，记录哪些修复已经落地、哪些建议尚未采纳
- 当前工作区已经额外补上 `Hamlet` 注册加载推理路径中的 dtype mismatch 修复，并已通过真实 checkpoint 的 `load + infer` 验证

"随 step 增长 memory 持续增加" 这个现象主要是实现 bug 导致的，而不是 memory baseline 的预期行为。

---

### 一、OOM 主因：`embed_prefix` 中多了一次完整 VLM Forward Pass

这是最严重的问题，直接导致**训练时峰值显存翻倍**。

#### HAMLET (`pi0_hamlet.py`)

```python
# Pi05WithHamlet.embed_prefix() 内部调用链:
def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
    # ① 调用父类 embed_prefix → 得到 prefix_embs (仅做 SigLIP 编码 + embedding)
    prefix_embs, prefix_pad_masks, prefix_att_masks = super().embed_prefix(...)
    
    # ② ❌ BUG: 额外做了一次完整 VLM forward pass 来"编码" moment tokens
    current_moment_tokens = self._encode_current_moment_tokens(
        prefix_embs, prefix_pad_masks, prefix_att_masks
    )
    # _encode_current_moment_tokens 内部：
    #   - 把 moment_tokens 拼到 prefix_embs 后面
    #   - 调用 self.paligemma_with_expert.forward(...) ← 完整 VLM forward!
    #   - 取最后 4 个 token 的 hidden states 作为 moment tokens
    ...
```

然后在 `GemmaTokenExpert.compute_velocity_train()` 中：

```python
def compute_velocity_train(self, *, model, ...):
    # ③ 调用 model.embed_prefix() → 内部已经做了一次 VLM forward (上面的②)
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(...)
    suffix_embs, ... = model.embed_suffix(...)
    
    # ④ 又做一次完整的 VLM forward (prefix + suffix 一起)
    (_, suffix_out), _ = model.paligemma_with_expert.forward(
        inputs_embeds=[prefix_embs, suffix_embs], ...
    )
```

**结果**：每个训练 step 做了 **2 次完整 VLM forward pass**（一次在 `_encode_current_moment_tokens` 中，一次在 `compute_velocity_train` 中），而且第一次的 activation 全部保留在计算图中用于反向传播。

#### MemoryVLA (`pi0_memoryvla.py`) — 同样的问题

```python
def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
    prefix_embs, prefix_pad_masks, prefix_att_masks = super().embed_prefix(...)
    
    # ❌ 额外完整 VLM forward pass
    prefix_hidden = self._encode_prefix_hidden(prefix_embs, prefix_pad_masks, prefix_att_masks)
    # _encode_prefix_hidden 内部调用 self.paligemma_with_expert.forward(...)
    
    current_tokens = prefix_hidden.mean(dim=1, keepdim=True)
    memory_tokens, gate = self.memoryvla(current_tokens, update_memory=not self.training)
    ...
```

同样是 **2 次完整 VLM forward pass per step**。

#### 论文设计对比

| 设计点 | HAMLET 论文 | 你的实现 |
|---|---|---|
| Moment token 编码 | 将 moment tokens 拼到 VLM 输入序列，**在同一次 forward 中一起处理** | 先做一次独立 VLM forward 编码 moment tokens，再做正常训练 forward |
| VLM forward 次数/step | **1 次** | **2 次** |
| 额外显存开销 | ~2%（仅多 4 个 token） | ~**100%**（多一次完整 forward 的 activation） |

| 设计点 | MemoryVLA 论文 | 你的实现 |
|---|---|---|
| 工作记忆提取 | 由**独立的轻量编码器**（SE-bottleneck + DINOv2/SigLIP）提取，**不过主 VLM** | 用完整 VLM forward 编码全部 prefix → mean pool |
| VLM forward 次数/step | **1 次** | **2 次** |

#### 显存估算

pi0.5 (PaliGemma 3B) 单次 prefix forward 的 activation 占用（bf16, seq_len≈500, batch=8）:

- 约 **8-12 GB per GPU**
- 两次 forward = **16-24 GB** 仅用于 prefix activations
- 加上模型参数 (~6GB)、优化器状态 (~12GB with Adam)、suffix forward、梯度 = **轻松超过 80 GB**

---

### 二、内存管理 Bug（导致"显存随 step 增长"）

#### Bug #1: `update_memory=not self.training` 逻辑反转

**HAMLET (`pi0_hamlet.py:L106`)**:
```python
memory_tokens = self.hamlet_memory(current_moment_tokens, update_memory=not self.training)
```

**MemoryVLA (`pi0_memoryvla.py:L82`)**:
```python
memory_tokens, gate = self.memoryvla(current_tokens, update_memory=not self.training)
```

- `self.training = True` → `update_memory = False` → **训练时永远不更新记忆**
- 这意味着训练时记忆模块完全不起作用（历史为空），模型退化为 baseline + 额外参数

**按论文应该是**：训练时**也要更新记忆**（HAMLET 论文明确说 end-to-end fine-tuning 时更新 history buffer）。但由于 history buffer 存的是 detached tensor，不影响梯度计算，应当是 `update_memory=True`（始终更新）。

#### Bug #2: `_session_memory_state` 字典在训练时泄漏 tensor 引用

```python
# pi0_hamlet.py L108-109:
if self._active_session_id is not None:
    self._session_memory_state[self._active_session_id] = self.hamlet_memory.get_runtime_state()
```

这个 dict 存储了每个 session 的 `history_buffer` tensor 引用。虽然训练时 `_active_session_id` 默认是 `None` 所以不会执行，但如果某些训练入口先调用了 `set_active_session()`，就会导致**每步都往 dict 里写入新的 tensor 引用**，阻止 GC 回收。

#### Bug #3: `torch.cat` 拼接历史导致显存碎片化

**HAMLET `_append_history()`**:
```python
def _append_history(self, current_moment_tokens: torch.Tensor) -> None:
    current_step = current_moment_tokens.detach().unsqueeze(1)
    if self.history_buffer is None:
        self.history_buffer = current_step
    elif int(self.history_count.item()) < self.history_length:
        self.history_buffer = torch.cat([self.history_buffer, current_step], dim=1)  # ← 每次创建新 tensor
    else:
        self.history_buffer = torch.cat([self.history_buffer[:, 1:], current_step], dim=1)  # ← 每次创建新 tensor
```

**MemoryVLA `SingleStreamMemoryBank.update()`**:
```python
self.memory_bank = torch.cat([self.memory_bank[:, :count], item.unsqueeze(1)], dim=1)  # ← 每次创建新 tensor
```

每次 `torch.cat` 都会分配新的 CUDA 内存块。旧 tensor 虽然会被 GC 回收，但 CUDA memory allocator 的碎片化会导致 `memory_reserved` 持续增长，即使 `memory_allocated` 稳定。

此外，MemoryVLA 在 similarity merge 分支还有一个 `.clone()`：
```python
updated = self.memory_bank.clone()  # ← 额外副本
```

---

### 三、修复方案与当前落地状态

#### Fix 1（核心）：消除额外 VLM Forward Pass

**当前状态：已落地**

- `Pi05WithHamlet.embed_prefix()` 已不再调用额外的 `paligemma_with_expert.forward()`
- `Pi05WithMemoryVLA.embed_prefix()` 也已不再调用额外的 prefix-only `paligemma_with_expert.forward()`
- 两者当前都改为基于 `prefix_embs` 的 masked summary + 轻量线性投影来生成 memory 输入
- 因此，“每个训练 step 做两次完整 VLM forward” 这一 OOM 主因已经被移除

**HAMLET 方案 A（推荐）—— 在同一次 forward 中编码 moment tokens**：

```python
# pi0_hamlet.py - 修改 embed_prefix
def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
    prefix_embs, prefix_pad_masks, prefix_att_masks = super().embed_prefix(...)
    
    # 不做额外 VLM forward！直接用 prefix_embs 的 mean pool 作为 moment token 输入
    batch_size = prefix_embs.shape[0]
    moment_tokens = self.moment_token_pool(batch_size).to(
        device=prefix_embs.device, dtype=prefix_embs.dtype
    )
    
    # 用轻量 cross-attention 或 mean-pool 代替完整 VLM forward
    # 例如：moment tokens attend to prefix_embs via 1-layer cross-attention
    current_moment_tokens = self.moment_cross_attn(
        query=moment_tokens, key=prefix_embs, value=prefix_embs
    )  # 新增一个轻量 nn.MultiheadAttention
    
    # history memory (不变)
    memory_tokens = self.hamlet_memory(current_moment_tokens, update_memory=True)
    memory_tokens = self.memory_to_prefix_proj(memory_tokens)
    
    # 拼接到 prefix
    prefix_embs = torch.cat([prefix_embs, memory_tokens], dim=1)
    prefix_pad_masks = torch.cat([prefix_pad_masks, ...], dim=1)
    prefix_att_masks = torch.cat([prefix_att_masks, ...], dim=1)
    return prefix_embs, prefix_pad_masks, prefix_att_masks
```

**HAMLET 方案 B（更接近论文）—— 把 moment tokens 直接拼入主 forward 的输入序列**：

在 `embed_prefix` 中只做拼接，让主 forward（`compute_velocity_train` 中的 `paligemma_with_expert.forward`）来"顺便"编码 moment tokens。训练后从主 forward 的输出中抽取 moment token 位置的 hidden state。但这需要修改 action_expert 的接口，侵入性较大。

**MemoryVLA 修复**：同理，用**轻量编码器**（1-2 层 MLP 或 cross-attention）代替完整 VLM forward 来生成 memory item：

```python
def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
    prefix_embs, prefix_pad_masks, prefix_att_masks = super().embed_prefix(...)
    
    # 轻量编码代替完整 VLM forward
    current_summary = prefix_embs.mean(dim=1, keepdim=True)  # 直接对 embeddings mean pool
    # 或用 1-layer attention: self.summary_attn(query=learned_query, kv=prefix_embs)
    
    memory_tokens, gate = self.memoryvla(current_summary, update_memory=True)
    ...
```

#### Fix 2：修复 update_memory 条件

```python
# 两个文件都改为始终更新：
memory_tokens = self.hamlet_memory(current_moment_tokens, update_memory=True)
# 以及
memory_tokens, gate = self.memoryvla(current_tokens, update_memory=True)
```

训练时需要更新记忆才能让模型学会使用历史信息。`detach()` 已经保证了 buffer 不参与梯度计算。

**当前状态：尚未按该建议落地**

- 当前代码仍然使用：

```python
memory_tokens = self.hamlet_memory(current_moment_tokens, update_memory=not self.training)
memory_tokens, gate = self.memoryvla(current_tokens, update_memory=not self.training)
```

- 也就是说：
  - 训练态不会跨 batch 持久化 runtime memory
  - 推理态会更新 runtime memory
- 这是当前实现与本文档原始建议之间最大的未对齐点
- 从工程角度看，当前实现至少避免了训练期 history/bank 在 batch 之间累积带来的额外不确定性；但从论文意图角度看，它仍不等价于“训练时也使用持久记忆”

#### Fix 3：预分配固定大小 buffer，消除 torch.cat

**HAMLET**：
```python
def __init__(self, ...):
    # 预分配固定大小 buffer
    self.register_buffer(
        "history_buffer",
        torch.zeros(1, history_length, num_moment_tokens, feature_dim),
        persistent=False,
    )
    self.register_buffer("history_count", torch.tensor(0), persistent=False)
    self._write_ptr = 0  # 环形指针

def _append_history(self, current_moment_tokens: torch.Tensor) -> None:
    B = current_moment_tokens.shape[0]
    if self.history_buffer.shape[0] != B:
        self.history_buffer = torch.zeros(
            B, self.history_length, self.num_moment_tokens, self.feature_dim,
            device=current_moment_tokens.device, dtype=current_moment_tokens.dtype,
        )
        self._write_ptr = 0
    
    # 原地写入，不分配新内存
    self.history_buffer[:, self._write_ptr].copy_(current_moment_tokens.detach())
    self._write_ptr = (self._write_ptr + 1) % self.history_length
    self.history_count = torch.tensor(
        min(int(self.history_count.item()) + 1, self.history_length),
        device=current_moment_tokens.device,
    )
```

**MemoryVLA**：类似用固定大小 bank + 环形/索引写入。

**当前状态：主体已落地**

- `HamletMemoryAdapter` 现在已使用：
  - `history_buffer`
  - `history_count`
  - `history_write_index`
  - 原地 `copy_()` 写入
- `SingleStreamMemoryBank` 现在已使用：
  - `memory_bank`
  - `memory_count`
  - `memory_write_index`
  - 原地 `copy_()` / 原地 merge
- 因此，原先由 `torch.cat` 导致的每步新 tensor 分配问题已经基本移除
- 需要注意：`HAMLET` 在读取满 buffer 的历史顺序时仍会在 `_get_history_steps()` 中通过一次 `torch.cat` 重排视图，但这已经不是“每步 append 时不断扩容新 buffer” 的旧问题

#### Fix 4：清理 session state dict（训练时）

```python
# 训练时不使用 session 机制：
def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
    ...
    memory_tokens = self.hamlet_memory(current_moment_tokens, update_memory=True)
    # 删除训练路径中对 _session_memory_state 的写入
    # if self._active_session_id is not None:  ← 训练时不需要
    #     self._session_memory_state[...] = ...
    ...
```

**当前状态：已基本落地**

- 当前代码不会在训练态写 `_session_memory_state`
- 仅在 `not self.training` 且存在 active session 时才会写回 session memory
- 因此，这个问题在当前实现里已被规避

---

### 四、当前实现状态总结

#### 已落地修复

1. **去掉额外 VLM forward**
   - HAMLET / MemoryVLA 都已改为 masked summary 路径

2. **固定 buffer + 原地写入**
   - HAMLET: `history_write_index`
   - MemoryVLA: `memory_write_index`

3. **训练态不写 session state**
   - 仅推理态写 `_session_memory_state`

4. **MemoryVLA 的 DDP/static-graph 与 autograd 细节修复**
   - `GatedMemoryFusion` 在空 memory 时仍保持相同参数路径
   - `retrieve()` 使用 detached snapshot，避免 update 影响 backward

5. **Hamlet 注册加载推理 dtype mismatch 修复**
   - `prefix_summary` 先 cast 到 `prefix_summary_proj.weight.dtype`
   - 再投影并 cast 回 `prefix_embs.dtype`
   - 该修复已通过真实 checkpoint 的 `test_registered_model_loading_inference` 验证

#### 尚未按原始建议落地的点

1. **训练态 runtime memory 仍不持久化**
   - 当前仍是 `update_memory=not self.training`
   - 因此本文档原先“训练时也更新记忆”的建议还没有落地

2. **HAMLET 仍未实现更接近论文的主-forward joint token encoding**
   - 当前是 masked summary + linear projection 的轻量近似
   - 不是把 moment tokens 直接并入主 VLM forward 再取其 hidden state

3. **MemoryVLA 仍是单流 MVP**
   - 还没有进入更完整的双流 cognitive branch 版本

---

### 五、验证结果（当前状态）

#### 代码级验证

- `pytest -q src/openpi/models_pytorch/memory_baselines/hamlet_memory_test.py src/openpi/models_pytorch/pi0_hamlet_test.py src/openpi/models_pytorch/memory_baselines/memoryvla_memory_test.py src/openpi/models_pytorch/pi0_memoryvla_test.py`
- `python scripts/test_pi05_hamlet.py`
- `python scripts/test_pi05_memoryvla.py`

#### 注册加载 + 单步推理验证

使用真实 `make_pizza` SFT checkpoint 运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/test_registered_model_loading_inference.py \
  --config pi05_hamlet_b1k-make_pizza_lr1e-4_5ep_sft \
  --ckpt-dir outputs/checkpoints/pi05_hamlet_b1k-make_pizza_lr1e-4_5ep_sft_baseckpt_5ep_20260418_005748 \
  --device cuda:0 \
  --default-prompt "make pizza"

CUDA_VISIBLE_DEVICES=0 python -u scripts/test_registered_model_loading_inference.py \
  --config pi05_memoryvla_b1k-make_pizza_lr1e-4_5ep_sft \
  --ckpt-dir outputs/checkpoints/pi05_memoryvla_b1k-make_pizza_lr1e-4_5ep_sft_baseckpt_5ep_20260418_162427 \
  --device cuda:0 \
  --default-prompt "make pizza"
```

结果：

- `Hamlet`
  - 初始出现 `mat1 and mat2 must have the same dtype` 错误
  - 修复 `prefix_summary_proj` 输入 dtype 后通过
  - checkpoint step: `299795`
  - `actions_shape=[32, 23]`
  - `infer_ms ~= 702.09`

- `MemoryVLA`
  - 直接通过
  - checkpoint step: `149895`
  - `actions_shape=[32, 23]`
  - `infer_ms ~= 892.68`

#### 8-GPU / 10-step 回归结论

- OOM 主因“第二次完整 VLM forward”移除后，8-GPU / 10-step full-finetune 已可正常完成
- HAMLET 峰值从修复前约 `72GB` 降到约 `37GB`
- MemoryVLA 峰值从修复前的 OOM / static-graph / inplace 系列错误，收敛到约 `35.6GB`

---

### 六、修复后的合理 batch size 预估

| 配置 | 修复前（2x VLM forward） | 修复后（1x VLM forward） |
|---|---|---|
| 模型参数 (bf16) | ~6 GB | ~6 GB |
| Adam 优化器状态 (fp32) | ~12 GB | ~12 GB |
| 1x VLM forward activation | ~10 GB (batch=8) | ~10 GB |
| 额外 VLM forward activation | ~10 GB | **0** |
| Suffix forward + 梯度 | ~8 GB | ~8 GB |
| Memory 模块 | <0.5 GB | <0.5 GB |
| **总计** | **~46 GB** | **~37 GB** |
| **80 GB A100 余量** | 34 GB (紧张) | 43 GB (舒适) |

修复后建议的 batch size:

| batch_per_gpu | 估计显存 | 建议 |
|---|---|---|
| 8 | ~37 GB | 修复后可稳定运行 |
| 12 | ~47 GB | 开 gradient checkpointing 后可行 |
| 16 | ~57 GB | 需要 gradient checkpointing + bf16 |

---

### 七、Bug 汇总表（按当前状态更新）

| # | 严重度 | 问题 | 影响 | 修复 |
|---|---|---|---|---|
| **1** | **P0** | HAMLET 额外完整 VLM forward | 已修复 | 已改为 masked summary + linear projection |
| **2** | **P0** | MemoryVLA 额外完整 VLM forward | 已修复 | 已改为 masked summary + linear projection |
| **3** | **P1** | `update_memory=not self.training` 使训练时不持久化 runtime memory | 仍存在 | 当前实现保留，尚未切到 `update_memory=True` |
| **4** | **P2** | `torch.cat` 扩容历史 / bank 导致碎片化 | 已基本修复 | 已改为固定 buffer + 写指针原地写入 |
| **5** | **P2** | MemoryVLA merge 分支额外副本 / backward 版本问题 | 已修复 | 已改为当前 in-place / detached snapshot 方案 |
| **6** | **P3** | `_session_memory_state` 训练时泄漏 tensor 引用 | 已修复 | 当前仅推理态写 session state |
| **7** | **P1** | HAMLET 注册加载推理路径中的 dtype mismatch | 已修复 | 已在真实 checkpoint 上验证通过 |

**当前结论**：

- 造成 OOM 的主因已经被移除
- 造成显存持续增长的主要 runtime buffer 管理问题已经基本修复
- 当前最主要的“设计未对齐”问题是：训练态仍未持久化 runtime memory，因此它与文档原始建议、以及更强形式的论文训练设定仍不完全一致
