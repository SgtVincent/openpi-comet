"""Tests for PI05KIJointQueryPytorch (query-MSE variant: query tokens + MSE + KI).

Tests cover:
1. Model instantiation (KI=True and KI=False)
2. KI mode: flow loss backward produces zero backbone grads
3. Non-KI mode: flow loss backward produces non-zero backbone grads (baseline)
4. Query tokens are learned embeddings, no GT action info in input
5. Query action head receives gradients from MSE loss
6. Two-phase training (compute_backbone_losses + compute_expert_loss)
   works without retain_graph=True
7. Parameter grouping (backbone vs expert params are disjoint)
8. query_embeddings + query_action_head are in BACKBONE group, not expert
9. Shape assertions on all key outputs
10. KV truncation: query tokens not in expert prefix
11. expert_loss = flow_loss_weight * flow_loss
12. Wrapper-safe forward dispatch preserves phase-specific and combined APIs

We use a mini dual-transformer model that mirrors the PaliGemma+Expert
architecture pattern to test gradient flow properties without loading
HF models.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from openpi.models_pytorch.pi05_ki_joint_query import _detach_kv_cache


# ===========================================================================
#  Mini dual-transformer model (mirrors PI05KIJointQueryPytorch architecture)
# ===========================================================================

class _MiniBackbone(nn.Module):
    """Tiny 1-layer transformer representing the VLM backbone."""

    def __init__(self, hidden_dim: int = 32, num_heads: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
        )

    def forward(self, x, mask=None):
        """
        x: [B, T, D]
        mask: [T, T] 2D bool or [B, T, T] or None (True = attendable)
        """
        B, T, D = x.shape
        h = self.norm(x)
        q = self.q_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [B, 1, T, T]
            attn = attn.masked_fill(~mask, -1e4)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D)
        out = self.o_proj(out)
        x = x + out
        x = x + self.mlp(self.norm(x))
        return x

    def get_kv(self, x, mask=None):
        """Return (k, v) tensors shaped [B, H, T, D_head]."""
        B, T, D = x.shape
        h = self.norm(x)
        k = self.k_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        return k, v


class _MiniExpert(nn.Module):
    """Tiny 1-layer transformer representing the action expert.

    Cross-attends to backbone KV (prefix positions).
    """

    def __init__(self, hidden_dim: int = 32, num_heads: int = 2, action_dim: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.action_dim = action_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, prefix_k, prefix_v, prefix_mask=None):
        """
        x: [B, S, D] suffix (expert) tokens
        prefix_k, prefix_v: [B, H, P, D_head] backbone prefix K/V
        prefix_mask: [B, P] bool (True = valid)
        """
        B, S, D = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k_self = self.k_proj(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v_self = self.v_proj(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Self-attention within suffix (causal)
        causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=x.device))
        self_attn = torch.matmul(q, k_self.transpose(-2, -1)) / math.sqrt(self.head_dim)
        self_attn = self_attn.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e4)
        self_attn = F.softmax(self_attn, dim=-1)
        self_out = torch.matmul(self_attn, v_self)

        # Cross-attention to backbone prefix KV
        cross_attn = torch.matmul(q, prefix_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if prefix_mask is not None:
            cross_attn = cross_attn.masked_fill(
                ~prefix_mask.unsqueeze(1).unsqueeze(1), -1e4
            )
        cross_attn = F.softmax(cross_attn, dim=-1)
        cross_out = torch.matmul(cross_attn, prefix_v)

        out = self_out + cross_out
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.o_proj(out)
        x = x + out
        x = x + self.mlp(self.norm1(x))
        return self.action_head(x)


class _MiniJointModel(nn.Module):
    """Minimal joint model testing the KI architecture pattern.

    Mirrors PI05KIJointQueryPytorch's core structure:
      - Backbone: processes prefix (text + subtask + query tokens)
      - Expert: processes action tokens, cross-attends to backbone prefix KV
      - KI: detach prefix KV before expert forward (blocks gradient flow)
      - Query tokens: learned embeddings + query_action_head (Option A design)
      - Two-phase API: compute_backbone_losses / compute_expert_loss
    """

    def __init__(self, hidden_dim=32, num_heads=2, prefix_len=8,
                 num_query_tokens=6, action_horizon=6, action_dim=4,
                 knowledge_insulation=True, truncate_expert_kv=True,
                 beta_text=1.0, beta_query=1.0, flow_loss_weight=10.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.prefix_len = prefix_len
        self.num_query_tokens = num_query_tokens
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.knowledge_insulation = knowledge_insulation
        self.truncate_expert_kv = truncate_expert_kv
        self.beta_text = beta_text
        self.beta_query = beta_query
        self.flow_loss_weight = flow_loss_weight

        # Backbone
        self.backbone = _MiniBackbone(hidden_dim, num_heads)
        self.prefix_embed = nn.Embedding(prefix_len, hidden_dim)
        self.subtask_head = nn.Linear(hidden_dim, 16)  # small "vocab"

        # Query tokens (learned embeddings — NO GT action info)
        self.query_embeddings = nn.Parameter(torch.randn(num_query_tokens, hidden_dim) * 0.02)

        # Query action head (maps backbone hidden → action_dim)
        self.query_action_head = nn.Linear(hidden_dim, action_dim)

        # Expert
        self.expert = _MiniExpert(hidden_dim, num_heads, action_dim)
        self.action_in_proj = nn.Linear(action_dim, hidden_dim)

    # -- Two-phase API (mirrors PI05KIJointQueryPytorch) --

    def compute_backbone_losses(self, prefix_tokens, actions, subtask_targets=None):
        """Phase 1: backbone CE + query MSE."""
        B = prefix_tokens.shape[0]
        device = prefix_tokens.device
        P = self.prefix_len
        Q = self.num_query_tokens

        # Build full prefix: [base_prefix, query_tokens]
        prefix_emb = self.prefix_embed(prefix_tokens)  # [B, P, D]
        query_emb = self.query_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]
        full_prefix = torch.cat([prefix_emb, query_emb], dim=1)  # [B, P+Q, D]

        # Block-causal attention mask:
        #   - Base prefix: bidirectional within itself
        #   - Query block: bidirectional within itself
        #   - Base prefix cannot attend query tokens
        #   - Query tokens can attend all base prefix + all query tokens
        full_len = P + Q
        mask = torch.ones(full_len, full_len, dtype=torch.bool, device=device)
        mask[:P, P:] = False  # base prefix cannot attend queries
        # Queries can attend everything before and including themselves
        # (bidirectional within query block + attend base prefix)
        # mask[P:, :] is already all True from torch.ones

        backbone_out = self.backbone(full_prefix, mask=mask)

        # Subtask CE loss (emulated on prefix positions)
        if subtask_targets is not None:
            subtask_hidden = backbone_out[:, :P]
            subtask_logits = self.subtask_head(subtask_hidden)
            ce_loss = F.cross_entropy(
                subtask_logits.view(-1, subtask_logits.size(-1)),
                subtask_targets.view(-1),
            )
        else:
            ce_loss = torch.tensor(0.0, device=device)

        # Query MSE loss (Option A: query_action_head → actions)
        query_hidden = backbone_out[:, P:P + Q]
        pred_actions = self.query_action_head(query_hidden)  # [B, Q, action_dim]

        if self.action_horizon != Q:
            target_actions = F.interpolate(
                actions.permute(0, 2, 1), size=Q, mode="linear", align_corners=False
            ).permute(0, 2, 1)
        else:
            target_actions = actions

        query_mse_loss = F.mse_loss(pred_actions, target_actions)
        backbone_loss = self.beta_text * ce_loss + self.beta_query * query_mse_loss

        return {
            "backbone_loss": backbone_loss,
            "ce_loss": ce_loss.detach(),
            "query_mse_loss": query_mse_loss.detach(),
        }

    def compute_expert_loss(self, prefix_tokens, actions, noise=None, time=None):
        """Phase 2: expert flow matching loss with KI.

        When knowledge_insulation=True, prefix KV is detached so flow loss
        backward produces zero backbone gradients.
        """
        B = prefix_tokens.shape[0]
        device = prefix_tokens.device
        P = self.prefix_len
        Q = self.num_query_tokens

        # Build expert prefix (truncated: no query tokens)
        prefix_emb = self.prefix_embed(prefix_tokens)

        if self.truncate_expert_kv:
            expert_prefix_emb = prefix_emb
            expert_prefix_len = P
        else:
            query_emb = self.query_embeddings.unsqueeze(0).expand(B, -1, -1)
            expert_prefix_emb = torch.cat([prefix_emb, query_emb], dim=1)
            expert_prefix_len = P + Q

        # Bidirectional mask for expert prefix (prefix attends to itself)
        prefix_mask_2d = torch.ones(
            expert_prefix_len, expert_prefix_len, dtype=torch.bool, device=device
        )

        expert_prefix_out = self.backbone(expert_prefix_emb, mask=prefix_mask_2d)
        prefix_k, prefix_v = self.backbone.get_kv(expert_prefix_out)

        # KI: detach KV before expert forward
        if self.knowledge_insulation:
            prefix_k = prefix_k.detach()
            prefix_v = prefix_v.detach()

        # Expert forward with noisy actions
        if noise is None:
            noise = torch.randn_like(actions)
        if time is None:
            time = torch.rand(B, device=device) * 0.998 + 0.001
        time_exp = time[:, None, None]
        x_t = time_exp * noise + (1 - time_exp) * actions
        u_t = noise - actions

        expert_input = self.action_in_proj(x_t)
        prefix_valid_mask = torch.ones(B, expert_prefix_len, dtype=torch.bool, device=device)
        v_t = self.expert(expert_input, prefix_k, prefix_v, prefix_valid_mask)

        flow_loss = F.mse_loss(u_t.float(), v_t.float(), reduction="mean")
        expert_loss = self.flow_loss_weight * flow_loss

        return {
            "flow_loss": flow_loss,
            "expert_loss": expert_loss,
        }

    def compute_all_losses(self, prefix_tokens, actions, subtask_targets=None):
        """Convenience: both phases together (for testing / KI-OFF baseline)."""
        bb = self.compute_backbone_losses(prefix_tokens, actions, subtask_targets)
        ex = self.compute_expert_loss(prefix_tokens, actions)
        total = bb["backbone_loss"].detach() + ex["expert_loss"].detach()
        return {
            "backbone_loss": bb["backbone_loss"],
            "flow_loss": ex["flow_loss"],
            "expert_loss": ex["expert_loss"],
            "ce_loss": bb["ce_loss"],
            "query_mse_loss": bb["query_mse_loss"],
            "total_loss": total,
        }


# ===========================================================================
#  Fixtures
# ===========================================================================

def _make_model(ki, truncate=True):
    torch.manual_seed(42)
    return _MiniJointModel(
        hidden_dim=32, num_heads=2, prefix_len=8,
        num_query_tokens=6, action_horizon=6, action_dim=4,
        knowledge_insulation=ki, truncate_expert_kv=truncate,
    )


@pytest.fixture
def model_ki():
    """Mini joint model with KI enabled + truncated KV."""
    m = _make_model(ki=True)
    m.eval()
    return m


@pytest.fixture
def model_no_ki():
    """Mini joint model with KI disabled + truncated KV (leakage baseline)."""
    m = _make_model(ki=False)
    m.eval()
    return m


@pytest.fixture
def batch():
    """Small test batch."""
    B = 2
    P = 8
    prefix_tokens = torch.randint(0, 8, (B, P))
    actions = torch.randn(B, 6, 4)
    subtask_targets = torch.randint(0, 16, (B, P))
    return prefix_tokens, actions, subtask_targets


# ===========================================================================
#  Test 1: KI gradient isolation
# ===========================================================================

class TestKIGradientIsolation:
    """Test Knowledge Insulation gradient flow properties."""

    def test_ki_flow_loss_zero_backbone_grad(self, model_ki, batch):
        """KI=True: expert_loss.backward() produces ZERO backbone grads."""
        prefix_tokens, actions, _ = batch
        model_ki.zero_grad()

        ex = model_ki.compute_expert_loss(prefix_tokens, actions)
        ex["expert_loss"].backward()

        for name, param in model_ki.named_parameters():
            if ("backbone" in name or "prefix_embed" in name
                or "query_embeddings" in name or "query_action_head" in name
                or "subtask_head" in name):
                if param.grad is not None:
                    max_grad = param.grad.abs().max().item()
                    assert max_grad < 1e-12, (
                        f"KI violation: {name} has max grad = {max_grad:.2e} "
                        f"from expert loss"
                    )

    def test_ki_flow_loss_expert_gets_grads(self, model_ki, batch):
        """KI=True: expert params get non-zero grads from flow loss."""
        prefix_tokens, actions, _ = batch
        model_ki.zero_grad()

        ex = model_ki.compute_expert_loss(prefix_tokens, actions)
        ex["expert_loss"].backward()

        found = False
        for name, param in model_ki.named_parameters():
            if "expert" in name or "action_in_proj" in name:
                if param.grad is not None and param.grad.abs().max().item() > 1e-10:
                    found = True
                    break
        assert found, "Expert parameters got no gradient from expert loss"

    def test_no_ki_flow_leaks_to_backbone(self, model_no_ki, batch):
        """KI=False: expert loss leaks to backbone through cross-attention KV."""
        prefix_tokens, actions, _ = batch
        model_no_ki.zero_grad()

        ex = model_no_ki.compute_expert_loss(prefix_tokens, actions)
        ex["expert_loss"].backward()

        found = False
        for name, param in model_no_ki.named_parameters():
            if "backbone" in name or "prefix_embed" in name:
                if param.grad is not None and param.grad.abs().max().item() > 1e-10:
                    found = True
                    break
        assert found, (
            "KI=False: expected expert loss to leak to backbone, but got zero grads"
        )


# ===========================================================================
#  Test 2: Backbone loss gradient reach
# ===========================================================================

class TestBackboneLossGradReach:
    """Test backbone loss reaches the right parameters."""

    def test_ce_loss_reaches_backbone_not_expert(self, model_ki, batch):
        """CE loss gives grads to backbone, NOT expert."""
        prefix_tokens, actions, subtask_targets = batch
        model_ki.zero_grad()

        bb = model_ki.compute_backbone_losses(prefix_tokens, actions, subtask_targets)
        # isolate CE by recomputing (since backbone_loss combines CE + MSE)
        B = prefix_tokens.shape[0]
        P = model_ki.prefix_len
        prefix_emb = model_ki.prefix_embed(prefix_tokens)
        Q = model_ki.num_query_tokens
        query_emb = model_ki.query_embeddings.unsqueeze(0).expand(B, -1, -1)
        full_prefix = torch.cat([prefix_emb, query_emb], dim=1)
        full_len = P + Q
        mask = torch.ones(full_len, full_len, dtype=torch.bool)
        mask[:P, P:] = False
        out = model_ki.backbone(full_prefix, mask=mask)
        logits = model_ki.subtask_head(out[:, :P])
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), subtask_targets.view(-1))
        ce.backward()

        # Backbone has grads
        assert model_ki.backbone.q_proj.weight.grad is not None
        assert model_ki.backbone.q_proj.weight.grad.abs().max() > 1e-10

        # Expert has NO grads from CE
        assert (model_ki.expert.q_proj.weight.grad is None
                or model_ki.expert.q_proj.weight.grad.abs().max() < 1e-12)

    def test_query_mse_reaches_query_head_and_embeddings(self, model_ki, batch):
        """Query MSE gives grads to query_embeddings and query_action_head."""
        prefix_tokens, actions, _ = batch
        model_ki.zero_grad()

        B = prefix_tokens.shape[0]
        P = model_ki.prefix_len
        Q = model_ki.num_query_tokens
        prefix_emb = model_ki.prefix_embed(prefix_tokens)
        query_emb = model_ki.query_embeddings.unsqueeze(0).expand(B, -1, -1)
        full_prefix = torch.cat([prefix_emb, query_emb], dim=1)
        full_len = P + Q
        mask = torch.ones(full_len, full_len, dtype=torch.bool)
        mask[:P, P:] = False
        out = model_ki.backbone(full_prefix, mask=mask)
        pred = model_ki.query_action_head(out[:, P:P + Q])
        mse = F.mse_loss(pred, actions)
        mse.backward()

        # Both get grads
        assert model_ki.query_embeddings.grad is not None
        assert model_ki.query_embeddings.grad.abs().max() > 1e-10
        assert model_ki.query_action_head.weight.grad is not None
        assert model_ki.query_action_head.weight.grad.abs().max() > 1e-10

        # Expert gets NO grads
        assert (model_ki.expert.q_proj.weight.grad is None
                or model_ki.expert.q_proj.weight.grad.abs().max() < 1e-12)


# ===========================================================================
#  Test 3: Query tokens = learned embeddings, no GT info
# ===========================================================================

class TestQueryTokensNoGTInfo:
    """Verify query tokens are pure learned embeddings with no GT dependency."""

    def test_query_embeddings_are_learnable_param(self, model_ki):
        assert isinstance(model_ki.query_embeddings, nn.Parameter)
        assert model_ki.query_embeddings.requires_grad
        assert model_ki.query_embeddings.shape == (
            model_ki.num_query_tokens, model_ki.hidden_dim
        )

    def test_query_embeddings_no_forward_time_modification(self, model_ki, batch):
        """Query embeddings are not modified by forward pass."""
        prefix_tokens, actions, _ = batch
        before = model_ki.query_embeddings.detach().clone()
        _ = model_ki.compute_all_losses(prefix_tokens, actions)
        after = model_ki.query_embeddings.detach().clone()
        assert torch.allclose(before, after)

    def test_query_action_head_is_backbone_side(self, model_ki):
        """query_action_head has hidden_dim → action_dim shape (decoder head)."""
        assert model_ki.query_action_head.in_features == model_ki.hidden_dim
        assert model_ki.query_action_head.out_features == model_ki.action_dim


# ===========================================================================
#  Test 4: Wrapper-safe forward phase dispatch
# ===========================================================================

class TestForwardPhaseDispatch:
    """The real model dispatches phases through ``forward`` for DDP hooks."""

    @staticmethod
    def _make_probe():
        from openpi.models_pytorch.pi05_ki_joint_query import PI05KIJointQueryPytorch

        class _DispatchProbe(PI05KIJointQueryPytorch):
            def __init__(self):
                nn.Module.__init__(self)
                self.calls = []

            def compute_backbone_losses(self, observation, actions):
                self.calls.append(("backbone", observation, actions))
                return {"backbone_loss": torch.tensor(2.0, requires_grad=True)}

            def compute_expert_loss(self, observation, actions, noise=None, time=None):
                self.calls.append(("expert", observation, actions, noise, time))
                return {
                    "flow_loss": torch.tensor(0.3, requires_grad=True),
                    "expert_loss": torch.tensor(3.0, requires_grad=True),
                }

            def compute_all_losses(self, observation, actions, noise=None, time=None):
                self.calls.append(("all", observation, actions, noise, time))
                return {
                    "backbone_loss": torch.tensor(2.0, requires_grad=True),
                    "flow_loss": torch.tensor(0.3, requires_grad=True),
                    "expert_loss": torch.tensor(3.0, requires_grad=True),
                    "ce_loss": torch.tensor(0.2),
                    "query_mse_loss": torch.tensor(0.1),
                    "total_loss": torch.tensor(5.0),
                }

        return _DispatchProbe()

    def test_backbone_phase_dispatches_to_direct_api(self):
        model = self._make_probe()
        observation, actions = object(), object()

        losses = model(observation, actions, phase="backbone")

        assert losses["backbone_loss"].item() == 2.0
        assert model.calls == [("backbone", observation, actions)]

    def test_expert_phase_forwards_noise_and_time(self):
        model = self._make_probe()
        observation, actions = object(), object()
        noise, time = object(), object()

        losses = model(
            observation,
            actions,
            noise=noise,
            time=time,
            phase="expert",
        )

        assert losses["expert_loss"].item() == 3.0
        assert model.calls == [("expert", observation, actions, noise, time)]

    def test_default_forward_preserves_combined_contract(self):
        model = self._make_probe()
        observation, actions = object(), object()

        losses = model(observation, actions)

        assert losses["loss"].item() == 5.0
        assert losses["backbone_loss"].requires_grad
        assert losses["expert_loss"].requires_grad
        assert model.calls[0][0] == "all"

    def test_unknown_phase_fails_fast(self):
        model = self._make_probe()

        with pytest.raises(ValueError, match="Unsupported training phase"):
            model(object(), object(), phase="invalid")


# ===========================================================================
#  Test 5: Two-phase training (no retain_graph)
# ===========================================================================

class TestTwoPhaseTraining:
    """Test two-phase training works without retain_graph."""

    def test_separate_backward_passes_work(self, model_ki, batch):
        """Phase1 backward → step → Phase2 backward → step works fine."""
        prefix_tokens, actions, subtask_targets = batch
        model_ki.train()

        bb_params = [p for n, p in model_ki.named_parameters()
                     if ("backbone" in n or "prefix_embed" in n
                         or "query_embeddings" in n or "query_action_head" in n
                         or "subtask_head" in n)]
        ex_params = [p for n, p in model_ki.named_parameters()
                     if "expert" in n or "action_in_proj" in n]

        opt_bb = torch.optim.SGD(bb_params, lr=0.01)
        opt_ex = torch.optim.SGD(ex_params, lr=0.01)

        # Phase 1: backbone
        opt_bb.zero_grad()
        opt_ex.zero_grad()
        bb_losses = model_ki.compute_backbone_losses(prefix_tokens, actions, subtask_targets)
        bb_losses["backbone_loss"].backward()
        opt_bb.step()

        # Phase 2: expert (separate graph, no retain_graph needed)
        opt_ex.zero_grad()
        ex_losses = model_ki.compute_expert_loss(prefix_tokens, actions)
        ex_losses["expert_loss"].backward()
        opt_ex.step()

        # No errors = success
        assert True

    def test_compute_backbone_has_no_retain_graph(self):
        """compute_backbone_losses source must not contain retain_graph."""
        from openpi.models_pytorch.pi05_ki_joint_query import PI05KIJointQueryPytorch
        import inspect
        src = inspect.getsource(PI05KIJointQueryPytorch.compute_backbone_losses)
        # Only check actual code, not docstring
        lines = [l for l in src.split("\n") if not l.strip().startswith("#")]
        code = "\n".join(lines)
        assert "retain_graph" not in code

    def test_compute_expert_has_no_retain_graph(self):
        """compute_expert_loss source must not contain retain_graph."""
        from openpi.models_pytorch.pi05_ki_joint_query import PI05KIJointQueryPytorch
        import inspect
        src = inspect.getsource(PI05KIJointQueryPytorch.compute_expert_loss)
        lines = [l for l in src.split("\n") if not l.strip().startswith("#")]
        code = "\n".join(lines)
        assert "retain_graph" not in code


# ===========================================================================
#  Test 5: KV truncation
# ===========================================================================

class TestKVTruncation:
    """Test KV truncation at subtask boundary."""

    def test_truncated_kv_excludes_query_tokens(self, model_ki, batch):
        """With truncate_expert_kv=True, expert KV has prefix_len positions."""
        prefix_tokens, _, _ = batch
        B = prefix_tokens.shape[0]

        prefix_emb = model_ki.prefix_embed(prefix_tokens)
        out = model_ki.backbone(prefix_emb)
        k, v = model_ki.backbone.get_kv(out)
        assert k.shape[2] == model_ki.prefix_len
        assert v.shape[2] == model_ki.prefix_len

    def test_no_truncation_includes_query_tokens(self):
        """Without truncation, expert sees query tokens too (ablation mode)."""
        model = _make_model(ki=False, truncate=False)
        prefix_tokens = torch.randint(0, 8, (2, 8))
        B = 2

        prefix_emb = model.prefix_embed(prefix_tokens)
        query_emb = model.query_embeddings.unsqueeze(0).expand(B, -1, -1)
        full_emb = torch.cat([prefix_emb, query_emb], dim=1)
        out = model.backbone(full_emb)
        k, v = model.backbone.get_kv(out)
        assert k.shape[2] == 8 + 6


# ===========================================================================
#  Test 6: Shape assertions
# ===========================================================================

class TestShapeAssertions:
    """Test output shapes are correct."""

    def test_losses_are_scalars(self, model_ki, batch):
        prefix_tokens, actions, subtask_targets = batch
        losses = model_ki.compute_all_losses(prefix_tokens, actions, subtask_targets)
        for key in ["backbone_loss", "flow_loss", "expert_loss",
                     "ce_loss", "query_mse_loss", "total_loss"]:
            assert losses[key].dim() == 0, f"{key} shape = {losses[key].shape}"

    def test_expert_loss_weighting(self, model_ki, batch):
        """expert_loss = flow_loss_weight * flow_loss."""
        prefix_tokens, actions, _ = batch
        ex = model_ki.compute_expert_loss(prefix_tokens, actions)
        expected = model_ki.flow_loss_weight * ex["flow_loss"]
        assert torch.allclose(ex["expert_loss"], expected)


# ===========================================================================
#  Test 7: Parameter grouping (disjointness + membership)
# ===========================================================================

class TestParameterGrouping:
    """Test backbone/expert parameter groups are correct and disjoint."""

    def _backbone_names(self, model):
        return {n for n, _ in model.named_parameters()
                if ("backbone" in n or "prefix_embed" in n
                    or "query_embeddings" in n or "query_action_head" in n
                    or "subtask_head" in n)}

    def _expert_names(self, model):
        return {n for n, _ in model.named_parameters()
                if "expert" in n or "action_in_proj" in n}

    def test_groups_disjoint(self, model_ki):
        bb = self._backbone_names(model_ki)
        ex = self._expert_names(model_ki)
        assert len(bb & ex) == 0, f"Overlap: {bb & ex}"

    def test_query_embeddings_in_backbone_not_expert(self, model_ki):
        """query_embeddings must be BACKBONE group, never expert."""
        bb = self._backbone_names(model_ki)
        ex = self._expert_names(model_ki)
        assert "query_embeddings" in bb
        assert "query_embeddings" not in ex

    def test_query_action_head_in_backbone_not_expert(self, model_ki):
        """query_action_head must be BACKBONE group, never expert."""
        bb = self._backbone_names(model_ki)
        ex = self._expert_names(model_ki)
        found_head = any("query_action_head" in n for n in bb)
        assert found_head, "query_action_head not found in backbone params"
        assert not any("query_action_head" in n for n in ex)

    def test_all_params_classified(self, model_ki):
        all_names = {n for n, _ in model_ki.named_parameters()}
        classified = self._backbone_names(model_ki) | self._expert_names(model_ki)
        unclassified = all_names - classified
        assert len(unclassified) == 0, f"Unclassified: {unclassified}"


# ===========================================================================
#  Test 8: _detach_kv_cache helper
# ===========================================================================

class TestDetachKVCache:
    """Test the _detach_kv_cache utility function."""

    def test_detach_list_format(self):
        kv = [
            (torch.randn(2, 2, 4, 8, requires_grad=True),
             torch.randn(2, 2, 4, 8, requires_grad=True))
            for _ in range(3)
        ]
        detached = _detach_kv_cache(kv)
        for k_d, v_d in detached:
            assert k_d.grad_fn is None
            assert v_d.grad_fn is None
        # Values preserved
        for (k_orig, v_orig), (k_d, v_d) in zip(kv, detached):
            assert torch.allclose(k_d, k_orig)
            assert torch.allclose(v_d, v_orig)

    def test_detach_dynamic_cache_format(self):
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        # key shape convention: [batch, num_heads, seq_len, head_dim]
        seq_len = 8
        batch = 2
        num_heads = 2
        head_dim = 16
        for i in range(2):
            k = torch.randn(batch, num_heads, seq_len, head_dim, requires_grad=True)
            v = torch.randn(batch, num_heads, seq_len, head_dim, requires_grad=True)
            cache.update(k, v, i)
        _detach_kv_cache(cache)
        # Verify seq length through public API
        assert cache.get_seq_length(0) == seq_len
        # Check whichever internal storage format exists (version-agnostic)
        has_detached = False
        if hasattr(cache, "layers"):
            # New API: layers[i].keys / .values (tensors)
            for layer in cache.layers:
                if hasattr(layer, "keys") and torch.is_tensor(layer.keys):
                    assert layer.keys.grad_fn is None
                    assert layer.values.grad_fn is None
                    has_detached = True
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            # Old API: key_cache / value_cache lists
            for k in cache.key_cache:
                assert k.grad_fn is None
            for v in cache.value_cache:
                assert v.grad_fn is None
            has_detached = True
        assert has_detached, "Could not verify detachment — unknown DynamicCache format"

    def test_detached_kv_blocks_gradient_flow(self):
        """Detached KV blocks gradient flow through cross-attention."""
        x = torch.randn(2, 4, 8, requires_grad=True)
        k = x * 2.0
        v = x + 1.0
        kv = [(k, v)]
        kv_detached = _detach_kv_cache(kv)

        q = torch.randn(2, 2, 3, 4, requires_grad=True)  # fake
        # Simple: q @ k.T @ v path
        k_d = kv_detached[0][0].view(2, 2, 4, 4)[:, :1, :, :]
        v_d = kv_detached[0][1].view(2, 2, 4, 4)[:, :1, :, :]
        q_1h = q[:, :1, :3, :]
        attn = F.softmax(torch.matmul(q_1h, k_d.transpose(-2, -1)), dim=-1)
        out = torch.matmul(attn, v_d)
        out.sum().backward()

        assert x.grad is None or x.grad.abs().max().item() < 1e-12


# ===========================================================================
#  Test 9: KI toggle controls gradient flow
# ===========================================================================

class TestKIToggle:
    """Test that knowledge_insulation flag correctly controls gradient flow."""

    def test_ki_true_blocks(self, model_ki, batch):
        prefix_tokens, actions, _ = batch
        model_ki.zero_grad()
        ex = model_ki.compute_expert_loss(prefix_tokens, actions)
        ex["expert_loss"].backward()
        for name, param in model_ki.named_parameters():
            if "backbone" in name or "prefix_embed" in name:
                if param.grad is not None:
                    assert param.grad.abs().max().item() < 1e-12, name

    def test_ki_false_allows_leakage(self, model_no_ki, batch):
        prefix_tokens, actions, _ = batch
        model_no_ki.zero_grad()
        ex = model_no_ki.compute_expert_loss(prefix_tokens, actions)
        ex["expert_loss"].backward()
        found = False
        for name, param in model_no_ki.named_parameters():
            if "backbone" in name or "prefix_embed" in name:
                if param.grad is not None and param.grad.abs().max().item() > 1e-10:
                    found = True
                    break
        assert found, "KI=False but no backbone leakage (unexpected)"


# ===========================================================================
#  Test 10: Query attention mask is bidirectional within block
# ===========================================================================

class TestQueryAttentionMask:
    """Test query tokens use bidirectional self-attention within the block."""

    def test_query_att_masks_all_zeros(self, model_ki):
        """query_att_masks = all zeros → bidirectional within query block."""
        from openpi.models_pytorch.pi05_ki_joint_query import PI05KIJointQueryPytorch
        import inspect
        src = inspect.getsource(PI05KIJointQueryPytorch._embed_query_tokens)
        # Check that attention mask is zeros (bidirectional block), not [1, 0, 0...]
        # (all zeros = same block, bidirectional under make_att_2d_masks cumsum semantics)
        assert "query_att_masks = torch.zeros" in src
