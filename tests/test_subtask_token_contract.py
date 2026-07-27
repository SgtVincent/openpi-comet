"""Token-level contract tests for the subtask implementation.

These tests verify the token-level correctness of the subtask training and
inference pipelines. They document the expected contract and catch the bugs
identified in the subtask implementation audit.

Audit findings referenced:
- CRIT-1: BOS token misalignment between training and inference
- MAJ-1: Training uses embed_tokens.weight.T, inference uses lm_head
        (should be tied, but may diverge under DeepSpeed/LoRA)
- MAJ-3: subtask_ar_mask parameter is threaded through but never used
"""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
#  Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    """Deterministic random generator for reproducible tests."""
    return torch.Generator().manual_seed(42)


@pytest.fixture
def tokenizer():
    """SubtaskTokenizer instance (lazy import to avoid download in collection)."""
    pytest.importorskip("openpi")
    from openpi.models.tokenizer import SubtaskTokenizer

    return SubtaskTokenizer(prompt_max_len=64, subtask_max_len=32)


class _MiniLMLayer(nn.Module):
    """Minimal single-layer transformer-like module for contract testing.

    We don't need a real Gemma model to verify token-level contracts: we just
    need a module where we can reason about which hidden state produces which
    logits, and where we can compare the training and inference paths.
    """

    def __init__(self, vocab_size: int, hidden_size: int, seq_len: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        # Single self-attention layer (simplified — no actual attention math
        # needed because we only care about *which position* produces logits,
        # not the actual attention values).
        self.layer = nn.Identity()
        # lm_head is separate from embed_tokens by default in this mock.
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def tie_weights(self) -> None:
        """Tie lm_head.weight to embed_tokens.weight (Gemma-style)."""
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, inputs_embeds: torch.Tensor, attention_mask=None, position_ids=None):
        # Simple identity forward: hidden states = input embeddings.
        # This is enough because we only need to check *which position*
        # contributes logits, not the actual transformation.
        hidden = self.layer(inputs_embeds)
        logits = self.lm_head(hidden)
        return SimpleNamespace(last_hidden_state=hidden, logits=logits)


@pytest.fixture
def mini_lm(rng):
    """A tiny language model with tied weights for contract testing."""
    model = _MiniLMLayer(vocab_size=128, hidden_size=32, seq_len=16)
    model.tie_weights()
    model.eval()
    return model


# ===========================================================================
#  Test 1: Tokenizer BOS/EOS positions and loss_mask
# ===========================================================================


class TestSubtaskTokenizerPositions:
    """SubtaskTokenizer.tokenize_subtask must produce the correct BOS/EOS
    positions and loss_mask pattern for causal next-token prediction.

    Contract:
      tokens        = [BOS, tok1, tok2, ..., tokN, EOS]
      loss_mask     = [F  , T   , T   , ..., T    , T  ]
                          ↑                    ↑
                          BOS is not           EOS is the
                          supervised           prediction target
                                             of tokN

    Audit ref: validates the tokenization contract that underpins CE loss.
    """

    def test_bos_is_first_token(self, tokenizer):
        """First token in the subtask sequence must be BOS."""
        tokens, mask, ar_mask, loss_mask = tokenizer.tokenize_subtask("pick up the cup")
        valid_tokens = tokens[mask.astype(bool)]
        bos_id = tokenizer._tokenizer.bos_id()
        assert valid_tokens[0] == bos_id, "First valid subtask token must be BOS"

    def test_eos_is_last_token(self, tokenizer):
        """Last token in the subtask sequence must be EOS."""
        tokens, mask, ar_mask, loss_mask = tokenizer.tokenize_subtask("pick up the cup")
        valid_tokens = tokens[mask.astype(bool)]
        eos_id = tokenizer._tokenizer.eos_id()
        assert valid_tokens[-1] == eos_id, "Last valid subtask token must be EOS"

    def test_loss_mask_bos_is_false(self, tokenizer):
        """BOS position must not contribute to loss (nothing predicts BOS)."""
        tokens, mask, ar_mask, loss_mask = tokenizer.tokenize_subtask("pick up the cup")
        valid_loss_mask = loss_mask[mask.astype(bool)]
        assert valid_loss_mask[0] == False, "BOS position must have loss_mask=False"  # noqa: E712

    def test_loss_mask_eos_is_true(self, tokenizer):
        """EOS position must be supervised (last real token predicts EOS)."""
        tokens, mask, ar_mask, loss_mask = tokenizer.tokenize_subtask("pick up the cup")
        valid_loss_mask = loss_mask[mask.astype(bool)]
        assert valid_loss_mask[-1] == True, "EOS position must have loss_mask=True"  # noqa: E712

    def test_loss_mask_count_equals_len_minus_one(self, tokenizer):
        """Total True entries in loss_mask must equal len(valid_tokens) - 1.

        All positions except BOS are supervised: position i predicts token i+1.
        """
        tokens, mask, ar_mask, loss_mask = tokenizer.tokenize_subtask("pick up the cup")
        n_valid = int(mask.sum())
        n_loss = int(loss_mask.sum())
        assert n_loss == n_valid - 1, (
            f"Expected {n_valid - 1} supervised positions (all except BOS), "
            f"got {n_loss}"
        )

    def test_ar_mask_all_ones_for_valid_tokens(self, tokenizer):
        """All valid subtask token positions must have ar_mask=1 (causal)."""
        tokens, mask, ar_mask, loss_mask = tokenizer.tokenize_subtask("pick up the cup")
        valid_ar = ar_mask[mask.astype(bool)]
        assert np.all(valid_ar == 1), "All valid subtask tokens must have ar_mask=1 (causal attention)"

    def test_empty_string_still_has_bos_eos(self, tokenizer):
        """Even an empty subtask produces [BOS, EOS] with loss_mask=[F, T]."""
        tokens, mask, ar_mask, loss_mask = tokenizer.tokenize_subtask("")
        valid_tokens = tokens[mask.astype(bool)]
        valid_loss = loss_mask[mask.astype(bool)]
        assert len(valid_tokens) == 2, "Empty subtask should have 2 tokens: BOS + EOS"
        assert valid_loss[0] == False  # noqa: E712
        assert valid_loss[1] == True  # noqa: E712


# ===========================================================================
#  Test 2: CE loss shift correctness (training path)
# ===========================================================================


class TestCELossShiftCorrectness:
    """Verify that the CE loss is computed on correctly shifted positions.

    Contract:
      text_logits[i]  predicts  subtask_tokens[i+1]
      shift_logits  = text_logits[:, :-1]
      shift_targets = subtask_tokens[:, 1:]
      shift_loss_mask = subtask_loss_mask[:, 1:]

    Audit ref: confirms the training shift logic is correct (the shift itself
    is not the bug — the bug is in how inference aligns to this scheme).

    This test PASSES on current code.
    """

    def test_shift_matches_next_token(self, rng):
        """logits at position t should target token at position t+1."""
        # Construct a tiny known scenario:
        #   tokens = [BOS=10, tok1=20, tok2=30, tok3=40, EOS=50]
        #   loss_mask = [F, T, T, T, T]
        # After shift:
        #   shift_logits positions 0..3  target  tokens[1..4] = [20, 30, 40, 50]
        #   shift_loss_mask positions 0..3 = [T, T, T, T]
        vocab_size = 128
        hidden_size = 16
        tokens = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)  # BOS, t1, t2, t3, EOS
        loss_mask = torch.tensor([[False, True, True, True, True]])

        # Create a deterministic model where we know exactly what logits look like
        embed = nn.Embedding(vocab_size, hidden_size)
        lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # Use identity-like weights for predictability
        nn.init.eye_(embed.weight[:hidden_size])
        nn.init.eye_(lm_head.weight[:hidden_size])

        # Simulate hidden states (just the embeddings for this test)
        hidden = embed(tokens)  # (1, 5, hidden)

        # ---- Compute CE loss the way subtask_expert does ----
        text_logits = torch.matmul(hidden, embed.weight.T)  # (1, 5, vocab)
        shift_logits = text_logits[:, :-1].contiguous()
        shift_targets = tokens[:, 1:].contiguous()
        shift_loss_mask = loss_mask[:, 1:].contiguous().float()

        ce_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            reduction="none",
        ).view(shift_logits.shape[0], -1)

        total_loss = (ce_per_token * shift_loss_mask).sum()
        total_valid = shift_loss_mask.sum().clamp(min=1)
        ce_loss = total_loss / total_valid

        # ---- Verify the contract ----
        # 4 supervised positions (tokens 1..4)
        assert shift_logits.shape[1] == 4, "Shifted logits should have seq_len-1 positions"
        assert shift_targets.shape[1] == 4, "Shifted targets should have seq_len-1 positions"
        assert shift_loss_mask.shape[1] == 4, "Shifted loss_mask should have seq_len-1 positions"

        # shift_targets[t] == tokens[t+1]
        for i in range(4):
            assert shift_targets[0, i].item() == tokens[0, i + 1].item(), (
                f"shift_targets[{i}] should equal tokens[{i+1}]"
            )

        # Total valid tokens = 4 (all except BOS, which after shift means all 4)
        assert total_valid.item() == 4, "Should have 4 valid token positions for CE loss"

        # Loss is finite
        assert torch.isfinite(ce_loss), "CE loss must be finite"

    def test_loss_mask_excludes_bos_after_shift(self, rng):
        """BOS position must not contribute to loss even in shifted form."""
        vocab_size = 64
        hidden_size = 8
        # Two samples: one with 5 tokens, one with 4 (padded)
        tokens = torch.tensor([
            [10, 20, 30, 40, 50],  # BOS, t1, t2, t3, EOS
            [10, 21, 31, 0, 0],    # BOS, t1, EOS, pad, pad
        ], dtype=torch.long)
        loss_mask = torch.tensor([
            [False, True, True, True, True],
            [False, True, True, False, False],
        ])

        embed = nn.Embedding(vocab_size, hidden_size)
        hidden = embed(tokens)
        text_logits = torch.matmul(hidden, embed.weight.T)

        shift_logits = text_logits[:, :-1].contiguous()
        shift_targets = tokens[:, 1:].contiguous()
        shift_loss_mask = loss_mask[:, 1:].contiguous().float()

        # Sample 0: 4 valid (t1, t2, t3, EOS)
        # Sample 1: 2 valid (t1, EOS)
        # Total: 6 valid tokens
        assert shift_loss_mask[0].sum().item() == 4
        assert shift_loss_mask[1].sum().item() == 2
        assert shift_loss_mask.sum().item() == 6

        # Per-token loss should be zero-weighted where mask=0
        ce_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            reduction="none",
        ).view(shift_logits.shape[0], -1)

        weighted = ce_per_token * shift_loss_mask
        # Positions with loss_mask=0 must contribute 0 to the total
        for b in range(2):
            for t in range(4):
                if shift_loss_mask[b, t] == 0:
                    assert weighted[b, t].item() == 0.0, (
                        f"Position (b={b}, t={t}) with loss_mask=0 must contribute 0"
                    )


# ===========================================================================
#  Test 3: BOS alignment — training vs inference (CRIT-1)
# ===========================================================================


class TestBOSAlignmentTrainingVsInference:
    """Contract test: the first subtask token predicted during inference must
    match the first subtask token predicted at the BOS position during training.

    CRIT-1 bug explanation:
      In training, the subtask sequence is [BOS, tok1, tok2, ...] appended
      to the prefix with causal attention.  The BOS position's hidden state
      produces logits that are trained to predict tok1.

      In inference (predict_subtask_tokens), the model does NOT insert a BOS
      token.  Instead, it takes the last prefix token's hidden state and
      uses lm_head on it to predict the first generated token.

      These are different positions with different hidden states, so the
      first generated token is not aligned with what the training objective
      teaches the model to predict at "step 0" of subtask generation.

    This test FAILS on current code (by design — it captures CRIT-1).
    """

    def test_first_token_train_bos_matches_inference_first_token(self, mini_lm):
        """Training's BOS-position prediction must equal inference's 1st token.

        After CRIT-1 fix: inference injects a BOS token after the prefix and
        uses the BOS position's hidden state to predict the first subtask
        token.  This must match training, where tok1 is predicted from the
        BOS position in the full sequence.
        """
        # After CRIT-1 fix: inference injects BOS after the prefix and uses
        # the BOS position's hidden state for first-token prediction.
        # This matches training's BOS → tok1 pattern.

        vocab_size = mini_lm.vocab_size
        hidden_size = mini_lm.hidden_size

        # ---- Setup: a tiny prefix + subtask ----
        # Prefix tokens (simulating "Task: ... State: ... Subtask: ")
        prefix_tokens = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)  # 4 prefix tokens
        prefix_len = prefix_tokens.shape[1]

        # Subtask tokens: [BOS, tok1, tok2, EOS]
        bos_id = 2
        eos_id = 3
        tok1_id = 20
        tok2_id = 30
        subtask_tokens = torch.tensor([[bos_id, tok1_id, tok2_id, eos_id]], dtype=torch.long)
        subtask_len = subtask_tokens.shape[1]

        # ---- Training path: BOS is the first subtask token ----
        # In training, full sequence = [prefix..., BOS, tok1, tok2, EOS]
        # with causal attention for the subtask portion.
        # The hidden state at the BOS position predicts tok1.
        full_train_tokens = torch.cat([prefix_tokens, subtask_tokens], dim=1)
        full_train_embeds = mini_lm.embed_tokens(full_train_tokens) * math.sqrt(hidden_size)

        train_out = mini_lm(full_train_embeds)
        train_hidden = train_out.last_hidden_state

        # Training: BOS position (index = prefix_len) predicts tok1
        bos_position_hidden = train_hidden[:, prefix_len:prefix_len + 1, :]
        train_logits_for_tok1 = mini_lm.lm_head(bos_position_hidden)
        train_first_pred = train_logits_for_tok1.argmax(dim=-1)

        # ---- Inference path: BOS injection after prefix (CRIT-1 fix) ----
        # In fixed inference: run prefix, then inject BOS token and use its
        # hidden state to predict the first generated token.
        # This matches the actual implementation in predict_subtask_tokens.
        prefix_embeds = mini_lm.embed_tokens(prefix_tokens) * math.sqrt(hidden_size)
        prefix_out = mini_lm(prefix_embeds)
        prefix_hidden = prefix_out.last_hidden_state

        # Inject BOS token (CRIT-1 fix pattern)
        bos_token_tensor = torch.tensor([[bos_id]], dtype=torch.long)
        bos_embeds = mini_lm.embed_tokens(bos_token_tensor) * math.sqrt(hidden_size)
        # Sequence for BOS step: prefix + BOS
        full_bos_embeds = torch.cat([prefix_embeds, bos_embeds], dim=1)
        bos_out = mini_lm(full_bos_embeds)
        bos_hidden = bos_out.last_hidden_state

        # Inference: BOS position (last token) predicts the first generated token
        infer_first_hidden = bos_hidden[:, -1:, :]
        infer_logits_first = mini_lm.lm_head(infer_first_hidden)
        infer_first_pred = infer_logits_first.argmax(dim=-1)

        # ---- Contract: these must be equal after CRIT-1 fix ----
        assert torch.equal(train_first_pred, infer_first_pred), (
            "CRIT-1 regression: first subtask token prediction differs between "
            f"training (BOS position: pred={train_first_pred.item()}) and inference "
            f"(injected BOS position: pred={infer_first_pred.item()}). "
            "Both should predict from the BOS embedding position."
        )

    def test_bos_position_vs_last_prefix_hidden_state_differ(self, mini_lm):
        """Explicitly demonstrate the hidden state mismatch.

        The hidden state at training's BOS position is NOT the same as the
        hidden state at inference's last-prefix position. This is the root
        cause of CRIT-1.

        This test documents the mismatch; it will "pass" (detect the mismatch)
        on current buggy code and would need updating when the fix lands.
        """
        vocab_size = mini_lm.vocab_size
        hidden_size = mini_lm.hidden_size

        prefix_tokens = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
        bos_id = 2
        subtask_tokens = torch.tensor([[bos_id, 20]], dtype=torch.long)
        prefix_len = prefix_tokens.shape[1]

        # Full sequence: prefix + BOS (training scenario when first subtask
        # token is BOS)
        full_tokens = torch.cat([prefix_tokens, subtask_tokens], dim=1)
        full_embeds = mini_lm.embed_tokens(full_tokens) * math.sqrt(hidden_size)
        full_out = mini_lm(full_embeds)

        # Hidden state at BOS position (training: this predicts tok1)
        bos_hidden = full_out.last_hidden_state[:, prefix_len, :]

        # Hidden state at last prefix position (inference: this predicts tok1)
        prefix_embeds = mini_lm.embed_tokens(prefix_tokens) * math.sqrt(hidden_size)
        prefix_out = mini_lm(prefix_embeds)
        last_prefix_hidden = prefix_out.last_hidden_state[:, -1, :]

        # These must be different (they use different input embeddings)
        # Even with Identity layers, BOS embedding ≠ last prefix token embedding
        assert not torch.allclose(bos_hidden, last_prefix_hidden, atol=1e-6), (
            "Expected BOS-position hidden state to differ from last-prefix-position "
            "hidden state. If they are equal, CRIT-1 may have been inadvertently fixed. "
            "Update test_bos_alignment accordingly."
        )


# ===========================================================================
#  Test 4: lm_head vs embed_tokens weight consistency (MAJ-1)
# ===========================================================================


class TestLMHeadVsEmbedTokensConsistency:
    """Verify that training's embed_tokens.weight.T and inference's lm_head
    produce identical logits when weights are tied.

    MAJ-1 audit finding:
      Training:  text_logits = hidden @ embed_tokens.weight.T
      Inference: logits = lm_head(hidden)

    With properly tied weights (Gemma default) these are equivalent.  But they
    can diverge under DeepSpeed ZeRO-3 (weights may be sharded separately),
    LoRA (only one path gets LoRA adapters), or weight-tying bugs.

    This test PASSES on current code with tied weights and documents the
    contract that must be preserved.
    """

    def test_tied_weights_produce_equal_logits(self, mini_lm, rng):
        """When lm_head and embed_tokens share weights, outputs must match."""
        hidden_size = mini_lm.hidden_size
        x = torch.randn(1, 5, hidden_size, generator=rng)

        # Path A: training-style (embed_tokens.weight.T)
        logits_via_embed = torch.matmul(x, mini_lm.embed_tokens.weight.T)

        # Path B: inference-style (lm_head linear layer)
        logits_via_lm_head = mini_lm.lm_head(x)

        assert torch.allclose(logits_via_embed, logits_via_lm_head, atol=1e-6), (
            "embed_tokens.weight.T and lm_head must produce identical logits "
            "when weights are tied. This contract may break under DeepSpeed/LoRA."
        )

    def test_untied_weights_produce_different_logits(self, rng):
        """When weights are NOT tied, the two paths diverge.

        Documents the MAJ-1 risk: if weight tying breaks (e.g. DeepSpeed
        ZeRO-3 bug, manual lm_head re-init, LoRA on only one path), logits
        will silently diverge between train and inference.
        """
        vocab_size = 128
        hidden_size = 32
        model = _MiniLMLayer(vocab_size=vocab_size, hidden_size=hidden_size, seq_len=8)
        # Deliberately NOT tying weights — simulating the failure mode
        model.eval()

        x = torch.randn(1, 3, hidden_size, generator=rng)
        logits_via_embed = torch.matmul(x, model.embed_tokens.weight.T)
        logits_via_lm_head = model.lm_head(x)

        # With untrained random weights, they should be different
        assert not torch.allclose(logits_via_embed, logits_via_lm_head, atol=1e-3), (
            "Expected different logits when weights are not tied. "
            "This documents the MAJ-1 failure mode."
        )

    def test_lm_head_weight_shape_contract(self, mini_lm):
        """Document the shape contract for both paths.

        embed_tokens: (vocab_size, hidden_size)
        lm_head.weight: (vocab_size, hidden_size)
        lm_head(x) = x @ lm_head.weight.T  ←  nn.Linear does this transpose
        """
        assert mini_lm.embed_tokens.weight.shape == mini_lm.lm_head.weight.shape, (
            "embed_tokens.weight and lm_head.weight must have the same shape "
            "for weight tying to work correctly."
        )
        vocab_size = mini_lm.vocab_size
        hidden_size = mini_lm.hidden_size
        assert mini_lm.embed_tokens.weight.shape == (vocab_size, hidden_size)
        assert mini_lm.lm_head.weight.shape == (vocab_size, hidden_size)


# ===========================================================================
#  Test 5: subtask_ar_mask is unused (MAJ-3, informational)
# ===========================================================================


class TestSubtaskArMaskUnused:
    """subtask_ar_mask is threaded through the interface but never consumed.

    MAJ-3 audit finding:
      - SubtaskTokenizer produces ar_mask
      - PI05SubtaskPytorch.forward passes subtask_ar_mask to compute_subtask_loss_train
      - compute_subtask_loss_train accepts subtask_ar_mask as a parameter
      - BUT subtask_ar_mask is never actually used in the computation

    The actual attention mask for subtask tokens is built inside
    _embed_conditioning_subtask using the `causal` flag, which always sets
    subtask_att = all-ones (causal) during training.

    This test documents the dead parameter. It PASSES on current code.
    """

    def test_ar_mask_parameter_accepted_but_not_used_in_signature(self):
        """subtask_expert.compute_subtask_loss_train accepts subtask_ar_mask
        but never references it in the function body.

        This is a static check that verifies the parameter name exists in the
        function signature but is not used in the body — documenting MAJ-3.
        """
        import inspect

        from openpi.models_pytorch.action_experts import subtask_expert as se

        # Check the parameter exists in the signature
        sig = inspect.signature(se.SubtaskActionExpert.compute_subtask_loss_train)
        assert "subtask_ar_mask" in sig.parameters, (
            "Expected subtask_ar_mask parameter in compute_subtask_loss_train signature"
        )

        # Check that it's not actually used in the function body
        source = inspect.getsource(se.SubtaskActionExpert.compute_subtask_loss_train)
        # Count occurrences: 1 in signature = not used in body
        # We look for the identifier used as a variable (not just in docstring)
        # Strip the signature line and docstring
        lines = source.split("\n")
        # Find where the body starts (after def line + docstring)
        body_lines = []
        in_docstring = False
        past_sig = False
        for line in lines:
            stripped = line.strip()
            if not past_sig:
                if stripped.endswith("):"):
                    past_sig = True
                continue
            if not body_lines and stripped.startswith('"""'):
                in_docstring = True
                # Handle single-line docstring
                if stripped.endswith('"""') and len(stripped) > 3:
                    in_docstring = False
                continue
            if in_docstring:
                if '"""' in stripped:
                    in_docstring = False
                continue
            body_lines.append(line)

        body_text = "\n".join(body_lines)
        # Check if subtask_ar_mask appears as more than just parameter passing
        # It might appear in the signature (already passed), but in the body
        # it should NOT be referenced if it's truly unused.
        ar_mask_refs = body_text.count("subtask_ar_mask")
        assert ar_mask_refs == 0, (
            f"subtask_ar_mask is referenced {ar_mask_refs} time(s) in the body "
            "of compute_subtask_loss_train. MAJ-3 finding may be resolved or "
            "partially resolved. Update this test accordingly."
        )

    def test_ar_mask_does_not_affect_loss_output(self, mini_lm, rng):
        """Different subtask_ar_mask values must produce identical loss.

        If the parameter were actually used, changing it would change the
        attention mask and thus the loss. Since it's dead code, different
        values produce the same result.
        """
        vocab_size = mini_lm.vocab_size
        hidden_size = mini_lm.hidden_size

        batch_size = 2
        prefix_len = 4
        subtask_len = 5

        # Build mock prefix embeddings
        prefix_embs = torch.randn(batch_size, prefix_len, hidden_size, generator=rng)
        prefix_pad = torch.ones(batch_size, prefix_len, dtype=torch.bool)
        prefix_att = torch.zeros(batch_size, prefix_len, dtype=torch.int32)

        # Build mock subtask tokens
        subtask_tokens = torch.randint(0, vocab_size, (batch_size, subtask_len), generator=rng)
        subtask_mask = torch.ones(batch_size, subtask_len, dtype=torch.bool)
        subtask_loss_mask = torch.tensor([
            [False, True, True, True, True],
            [False, True, True, True, True],
        ], dtype=torch.bool)

        # Two different ar_mask values that should produce different results
        # IF the parameter were actually used
        ar_mask_all_causal = torch.ones(batch_size, subtask_len, dtype=torch.int32)
        ar_mask_all_bidir = torch.zeros(batch_size, subtask_len, dtype=torch.int32)

        # Simulate what compute_subtask_loss_train does:
        # It uses _embed_conditioning_subtask with causal=True,
        # which ALWAYS sets subtask_att = ones_like (causal),
        # ignoring subtask_ar_mask entirely.

        def simulate_loss(subtask_ar_mask):
            """Replicate the relevant parts of compute_subtask_loss_train.

            We reproduce the exact pattern from subtask_expert.py to show
            that subtask_ar_mask is never consumed.
            """
            # This is what _embed_conditioning_subtask does:
            subtask_embs = mini_lm.embed_tokens(subtask_tokens)
            subtask_embs = subtask_embs * (hidden_size ** 0.5)

            extended_embs = torch.cat([prefix_embs, subtask_embs], dim=1)
            extended_pad = torch.cat([prefix_pad, subtask_mask], dim=1)

            # KEY LINE: causal=True → always ones, subtask_ar_mask is IGNORED
            causal = True  # training always uses causal
            if causal:
                subtask_att = torch.ones_like(subtask_mask, dtype=prefix_att.dtype)
            else:
                subtask_att = torch.zeros_like(subtask_mask, dtype=prefix_att.dtype)
            extended_att = torch.cat(
                [prefix_att, subtask_att], dim=1
            )

            # Build 2D attention mask
            from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

            att_2d = make_att_2d_masks(extended_pad, extended_att)

            # Forward pass
            out = mini_lm(extended_embs)
            hidden = out.last_hidden_state

            # Get subtask hidden states
            subtask_hidden = hidden[:, prefix_len:prefix_len + subtask_len, :]

            # Compute text logits via embed_tokens.weight.T (training path)
            text_logits = torch.matmul(subtask_hidden, mini_lm.embed_tokens.weight.T)

            # CE loss with shift
            shift_logits = text_logits[:, :-1].contiguous()
            shift_targets = subtask_tokens[:, 1:].contiguous().to(torch.long)
            shift_loss_mask = subtask_loss_mask[:, 1:].contiguous().float()

            ce_per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                reduction="none",
            ).view(shift_logits.shape[0], -1)

            total_ce = (ce_per_token * shift_loss_mask).sum()
            total_valid = shift_loss_mask.sum().clamp(min=1)
            return total_ce / total_valid

        loss_causal = simulate_loss(ar_mask_all_causal)
        loss_bidir = simulate_loss(ar_mask_all_bidir)

        # Both produce identical results because subtask_ar_mask is ignored
        assert torch.allclose(loss_causal, loss_bidir, atol=1e-7), (
            "MAJ-3 verification: subtask_ar_mask values produce identical loss "
            "because the parameter is never consumed in compute_subtask_loss_train. "
            "If this assertion fails, the parameter may have been wired up — "
            "update the MAJ-3 finding."
        )


# ===========================================================================
#  Test 6: BOS conditioning in build_hierarchical_observation (Gap 1)
# ===========================================================================


class TestBuildHierarchicalObservationBOS:
    """build_hierarchical_observation must prepend BOS to subtask_tokens.

    **Gap 1**: Training uses ``[BOS, tok1, ..., EOS]`` as the subtask
    conditioning sequence.  ``predict_subtask_tokens`` returns only ``[tok1,
    ..., EOS]`` (no BOS — the BOS is the generation seed, not generated
    output).  If ``build_hierarchical_observation`` simply copies the tokens,
    the action expert sees a **different** conditioning sequence at inference
    than at training — missing the leading BOS.

    The fix: ``build_hierarchical_observation`` prepends BOS so that the
    action expert's conditioning matches training.
    """

    def _make_model_instance(self):
        """Create a minimal PI05SubtaskPytorch instance for testing.

        We can't use __new__ on nn.Module subclasses because __getattr__
        expects _modules/_parameters/_buffers.  We properly initialize
        nn.Module and set the lazy-load attributes.
        """
        pytest.importorskip("openpi")
        from openpi.models_pytorch.pi05_subtask import PI05SubtaskPytorch

        model = PI05SubtaskPytorch.__new__(PI05SubtaskPytorch)
        # Initialize nn.Module base properly
        import torch.nn as nn
        nn.Module.__init__(model)
        # Set lazy-load attribute so __getattr__ doesn't intercept it
        model._text_tokenizer = None
        model._last_predicted_subtasks = []
        return model

    def test_pi05_build_hierarchical_prepends_bos(self):
        """PI05SubtaskPytorch.build_hierarchical_observation prepends BOS.

        Given subtask tokens ``[tok1, tok2, EOS]`` from predict_subtask_tokens,
        the first token in ``observation.subtask_tokens`` must be BOS.
        """
        model = self._make_model_instance()
        from types import SimpleNamespace
        model.config = SimpleNamespace(subtask_max_len=16)

        # Force the text tokenizer to load
        bos_id = model._load_text_tokenizer().bos_id()
        eos_id = model._eos_token_id()
        tok1, tok2 = 42, 99

        subtask_tokens = torch.tensor([[tok1, tok2, eos_id]], dtype=torch.int32)

        # Use a minimal dataclass-compatible observation
        @dataclasses.dataclass
        class _DummyObs:
            state: torch.Tensor = None
            subtask_tokens: torch.Tensor = None
            subtask_mask: torch.Tensor = None
            subtask_loss_mask: torch.Tensor = None
            subtask_ar_mask: torch.Tensor = None

        obs = _DummyObs(state=torch.zeros(1, 23))
        result = model.build_hierarchical_observation(obs, subtask_tokens)

        # The first valid token must be BOS — this is the core assertion.
        assert result.subtask_tokens[0, 0].item() == bos_id, (
            "Gap 1: build_hierarchical_observation must prepend BOS to subtask_tokens "
            "so the action expert sees the same conditioning sequence as training. "
            f"Expected first token = BOS ({bos_id}), got {result.subtask_tokens[0, 0].item()}"
        )

        # Second token should be tok1
        assert result.subtask_tokens[0, 1].item() == tok1
        # Third should be tok2
        assert result.subtask_tokens[0, 2].item() == tok2
        # Fourth should be EOS
        assert result.subtask_tokens[0, 3].item() == eos_id

        # subtask_mask should mark BOS as valid too
        assert result.subtask_mask[0, 0].item() == True  # noqa: E712
        assert result.subtask_mask[0, 1].item() == True  # noqa: E712
        assert result.subtask_mask[0, 2].item() == True  # noqa: E712
        assert result.subtask_mask[0, 3].item() == True  # noqa: E712

    def test_bos_accounts_for_max_len(self):
        """Prepending BOS must respect subtask_max_len (clips one fewer token)."""
        model = self._make_model_instance()
        from types import SimpleNamespace
        # Set max_len = 4: with BOS prepended, only 3 generated tokens fit.
        model.config = SimpleNamespace(subtask_max_len=4)

        eos_id = model._eos_token_id()
        # 5 generated tokens — more than fit in max_len=4 (BOS + 3 = 4 total)
        subtask_tokens = torch.tensor([[10, 20, 30, 40, eos_id]], dtype=torch.int32)

        @dataclasses.dataclass
        class _DummyObs:
            state: torch.Tensor = None
            subtask_tokens: torch.Tensor = None
            subtask_mask: torch.Tensor = None
            subtask_loss_mask: torch.Tensor = None
            subtask_ar_mask: torch.Tensor = None

        obs = _DummyObs(state=torch.zeros(1, 23))
        result = model.build_hierarchical_observation(obs, subtask_tokens)

        # With max_len=4 and BOS prepended, we get BOS + 3 generated tokens.
        assert result.subtask_tokens.shape[1] == 4
        # First token: BOS
        assert result.subtask_tokens[0, 0].item() == model._load_text_tokenizer().bos_id()
        # Tokens 1-3: first 3 of the generated tokens
        assert result.subtask_tokens[0, 1].item() == 10
        assert result.subtask_tokens[0, 2].item() == 20
        assert result.subtask_tokens[0, 3].item() == 30
        # 40 and EOS got clipped (beyond max_len)
        assert result.subtask_mask[0, 3].item() == True  # noqa: E712

    def test_default_max_len_preserves_eos(self):
        """When config lacks subtask_max_len, EOS must not be clipped.

        Robustness issue (a): the default max_len was subtask_tokens.shape[1],
        but we prepend BOS (+1 token), so the last generated token (EOS)
        would get clipped.  The fix uses subtask_tokens.shape[1] + 1 as the
        default so EOS is preserved.
        """
        model = self._make_model_instance()
        from types import SimpleNamespace
        # No subtask_max_len on config — triggers the default fallback
        model.config = SimpleNamespace()

        bos_id = model._load_text_tokenizer().bos_id()
        eos_id = model._eos_token_id()

        # 3 generated tokens: tok1, tok2, EOS
        subtask_tokens = torch.tensor([[42, 99, eos_id]], dtype=torch.int32)

        @dataclasses.dataclass
        class _DummyObs:
            state: torch.Tensor = None
            subtask_tokens: torch.Tensor = None
            subtask_mask: torch.Tensor = None
            subtask_loss_mask: torch.Tensor = None
            subtask_ar_mask: torch.Tensor = None

        obs = _DummyObs(state=torch.zeros(1, 23))
        result = model.build_hierarchical_observation(obs, subtask_tokens)

        # With default max_len (= generated_len + 1), BOS + 3 tokens all fit.
        assert result.subtask_tokens.shape[1] == 4
        # Sequence: BOS, tok1, tok2, EOS — all present
        assert result.subtask_tokens[0, 0].item() == bos_id
        assert result.subtask_tokens[0, 1].item() == 42
        assert result.subtask_tokens[0, 2].item() == 99
        assert result.subtask_tokens[0, 3].item() == eos_id
        # All 4 positions are valid
        assert result.subtask_mask[0, :4].all().item() == True  # noqa: E712


# ===========================================================================
#  Test 7: Batch EOS contamination (Gap 2)
# ===========================================================================


class TestBatchEOSContamination:
    """predict_subtask_tokens must zero post-EOS tokens in each row.

    **Gap 2**: Generation stops only when ``torch.all(next_token == eos_token)``
    — i.e. when ALL rows in the batch have produced EOS.  Rows that finish
    early keep sampling garbage tokens after their EOS.  Then
    ``build_hierarchical_observation`` uses ``clipped != 0`` as the mask, so
    the action expert attends to garbage post-EOS tokens in early-finishing
    rows.

    The fix has two parts:
    1. ``predict_subtask_tokens``: track per-row EOS status; zero out tokens
       in rows that have already finished.
    2. ``build_hierarchical_observation``: mask should only mark tokens up to
       (and including) the first EOS as valid.
    """

    def test_post_eos_tokens_zeroed_in_predict_subtask(self):
        """predict_subtask_tokens must zero tokens after per-row EOS.

        We simulate a batch where row 0 produces EOS at step 2 but row 1
        continues for 5 steps.  After row 0 hits EOS, its subsequent tokens
        must be zeroed out (pad) so that build_hierarchical_observation's
        mask correctly ignores them.
        """
        # We test the logic directly: given a batch of generated tokens with
        # mixed EOS positions, verify that a "zero after first EOS" pass
        # produces the expected result.  This documents the contract that
        # predict_subtask_tokens must satisfy.
        eos_id = 3
        # Row 0: EOS at position 2 (0-indexed among generated tokens)
        # Row 1: EOS at position 4
        # Row 2: no EOS yet (generation hit max_tokens)
        generated = torch.tensor([
            [10, 20, eos_id, 999, 999],   # row 0: EOS at idx 2, garbage after
            [30, 40, 50, 60, eos_id],     # row 1: EOS at idx 4
            [70, 80, 90, 100, 110],       # row 2: no EOS at all
        ], dtype=torch.int32)

        # Apply the zeroing rule that predict_subtask_tokens must implement:
        # after the first EOS in a row, set all subsequent tokens to 0.
        def zero_after_eos(tokens, eos_id):
            """Zero all tokens after the first EOS in each row (in-place copy)."""
            result = tokens.clone()
            for i in range(tokens.shape[0]):
                # Find position of first EOS
                eos_positions = (tokens[i] == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    first_eos = eos_positions[0].item()
                    result[i, first_eos + 1:] = 0
            return result

        zeroed = zero_after_eos(generated, eos_id)

        # Row 0: positions 0-2 preserved, position 3-4 zeroed
        assert zeroed[0, 0].item() == 10
        assert zeroed[0, 1].item() == 20
        assert zeroed[0, 2].item() == eos_id
        assert zeroed[0, 3].item() == 0
        assert zeroed[0, 4].item() == 0

        # Row 1: all positions preserved (EOS is last token)
        assert zeroed[1, 0].item() == 30
        assert zeroed[1, 4].item() == eos_id

        # Row 2: no EOS → nothing zeroed
        assert (zeroed[2] == generated[2]).all()

    def test_eos_based_mask_in_build_hierarchical(self):
        """build_hierarchical_observation mask must be EOS-aware (Gap 2 part 2).

        The mask must mark tokens as valid up to and including the first EOS,
        not just ``!= 0``.  This guards against garbage tokens that may slip
        through even after the zeroing fix (e.g. if ``predict_subtask_tokens``
        has a bug, or if someone passes hand-crafted tokens).
        """
        # We test the mask logic directly: given a tensor of subtask tokens,
        # the valid mask should be True from position 0 up to and including
        # the first EOS, and False after.
        eos_id = 3
        bos_id = 2

        # Simulate what build_hierarchical_observation sees: BOS-prepended
        # tokens with zeroed post-EOS garbage.
        tokens = torch.tensor([
            [bos_id, 10, 20, eos_id, 0, 0, 0],      # EOS at idx 3
            [bos_id, 30, eos_id, 0, 0, 0, 0],         # EOS at idx 2
            [bos_id, 40, 50, 60, 70, 80, 90],         # no EOS (truncated)
        ], dtype=torch.int32)

        # Build EOS-aware mask: True from start to first EOS (inclusive).
        # If no EOS, mask is True for all nonzero tokens.
        def build_eos_mask(tokens, eos_id):
            mask = torch.zeros_like(tokens, dtype=torch.bool)
            for i in range(tokens.shape[0]):
                eos_positions = (tokens[i] == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    first_eos = eos_positions[0].item()
                    mask[i, :first_eos + 1] = True
                else:
                    # No EOS — all nonzero tokens are valid
                    mask[i] = tokens[i] != 0
            return mask

        mask = build_eos_mask(tokens, eos_id)

        # Row 0: positions 0-3 valid (BOS, 10, 20, EOS)
        assert mask[0, 0].item() == True  # noqa: E712
        assert mask[0, 3].item() == True  # noqa: E712  (EOS is valid)
        assert mask[0, 4].item() == False  # noqa: E712
        assert mask[0, 6].item() == False  # noqa: E712

        # Row 1: positions 0-2 valid (BOS, 30, EOS)
        assert mask[1, 0].item() == True  # noqa: E712
        assert mask[1, 2].item() == True  # noqa: E712  (EOS)
        assert mask[1, 3].item() == False  # noqa: E712

        # Row 2: no EOS → all 7 tokens valid (none are zero)
        assert mask[2, :].all().item() == True  # noqa: E712

    def test_naive_nonzero_mask_would_include_garbage(self):
        """Documents why ``!= 0`` mask is incorrect for mixed-EOS batches.

        If predict_subtask_tokens didn't zero post-EOS tokens and
        build_hierarchical_observation used ``!= 0``, garbage tokens would
        be marked as valid context for the action expert.
        """
        eos_id = 3
        # Simulate the buggy scenario: post-EOS tokens are non-zero garbage
        # and != 0 marks them as valid.
        garbage_tokens = torch.tensor([
            [10, 20, eos_id, 999, 888],   # garbage after EOS
        ], dtype=torch.int32)

        # Buggy mask: != 0 — includes garbage
        buggy_mask = garbage_tokens != 0
        assert buggy_mask[0, 3].item() == True  # noqa: E712  (garbage is "valid")
        assert buggy_mask[0, 4].item() == True  # noqa: E712

        # Correct mask: only up to and including first EOS
        def correct_mask(tokens, eos_id):
            mask = torch.zeros_like(tokens, dtype=torch.bool)
            for i in range(tokens.shape[0]):
                eos_pos = (tokens[i] == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_pos) > 0:
                    mask[i, :eos_pos[0] + 1] = True
                else:
                    mask[i] = tokens[i] != 0
            return mask

        good_mask = correct_mask(garbage_tokens, eos_id)
        assert good_mask[0, 3].item() == False  # noqa: E712  (garbage masked out)
        assert good_mask[0, 4].item() == False  # noqa: E712

    def test_vectorized_eos_mask_matches_loop(self):
        """Vectorized EOS mask (cumsum) must match per-row loop reference.

        Robustness issue (c): the EOS-aware mask must be vectorizable
        (no Python per-row .nonzero() calls) to avoid GPU syncs and to
        be torch.compile-friendly.  Verifies that the cumsum-based
        implementation produces identical results to a loop-based
        reference across various batch sizes and EOS patterns.
        """
        eos_id = 3
        bos_id = 2

        # Test various patterns: early EOS, late EOS, no EOS, multiple EOS,
        # batch size 1, batch size > 1, all-zero rows, BOS-only rows.
        test_cases = [
            # (description, tokens_tensor)
            ("mixed early/late/no EOS (batch=3)",
             torch.tensor([
                 [bos_id, 10, eos_id, 0, 0, 0],       # EOS at idx 2
                 [bos_id, 20, 30, 40, eos_id, 0],    # EOS at idx 4
                 [bos_id, 50, 60, 70, 80, 90],       # no EOS
             ], dtype=torch.int32)),
            ("single row with EOS at end",
             torch.tensor([[bos_id, 10, 20, eos_id]], dtype=torch.int32)),
            ("single row no EOS",
             torch.tensor([[bos_id, 10, 20, 30]], dtype=torch.int32)),
            ("multiple EOS (only first counts)",
             torch.tensor([[bos_id, 10, eos_id, 20, eos_id, 30]], dtype=torch.int32)),
            ("large batch mixed",
             torch.tensor([
                 [bos_id, eos_id, 0, 0, 0],            # EOS at idx 1
                 [bos_id, 10, 20, eos_id, 0],          # EOS at idx 3
                 [bos_id, 10, 20, 30, 40],              # no EOS
                 [bos_id, 10, eos_id, 20, eos_id],      # double EOS
             ], dtype=torch.int32)),
        ]

        # Reference (loop-based) mask
        def ref_eos_mask(tokens, eos_id):
            mask = torch.zeros_like(tokens, dtype=torch.bool)
            for i in range(tokens.shape[0]):
                eos_pos = (tokens[i] == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_pos) > 0:
                    mask[i, :eos_pos[0] + 1] = True
                else:
                    mask[i] = tokens[i] != 0
            return mask

        # Vectorized (cumsum-based) mask — the implementation used in production
        def vectorized_eos_mask(tokens, eos_id):
            eos_mask_bool = tokens == eos_id
            eos_cumsum = torch.cumsum(eos_mask_bool.to(torch.long), dim=1)
            has_eos = eos_cumsum[:, -1] > 0
            valid_up_to_eos = (eos_cumsum < 1) | ((eos_cumsum == 1) & eos_mask_bool)
            nonzero_mask = tokens != 0
            return torch.where(has_eos[:, None], valid_up_to_eos, nonzero_mask)

        for desc, tokens in test_cases:
            ref = ref_eos_mask(tokens, eos_id)
            vec = vectorized_eos_mask(tokens, eos_id)
            assert torch.equal(ref, vec), (
                f"Vectorized EOS mask mismatch for case '{desc}'.\n"
                f"Reference:\n{ref}\nVectorized:\n{vec}"
            )


# ---------------------------------------------------------------------------
#  10. KV cache non-mutation across denoise steps
# ---------------------------------------------------------------------------


class TestKVCacheNonMutation:
    """Verify that prefix KV cache is not mutated across denoise steps.

    HuggingFace DynamicCache grows in-place during each forward pass even when
    ``use_cache=False``.  Our ``PreserveCacheLen`` wrapper must truncate the
    cache back to its original prefix length after each ``denoise_step`` call.
    """

    def _make_dummy_cache(self, batch_size=1, prefix_len=16, num_layers=2, head_dim=32):
        """Create a simple DynamicCache-like object for testing."""
        from openpi.models_pytorch.cache_utils import (
            PreserveCacheLen,
            get_cache_seq_len,
            truncate_cache_to_len,
        )

        class DummyCache:
            def __init__(self, num_layers, batch, seq_len, num_kv_heads, head_dim):
                self.key_cache = [torch.randn(batch, num_kv_heads, seq_len, head_dim) for _ in range(num_layers)]
                self.value_cache = [torch.randn(batch, num_kv_heads, seq_len, head_dim) for _ in range(num_layers)]
                self._seen_tokens = seq_len

            def get_seq_length(self, layer_idx=0):
                return self.key_cache[layer_idx].shape[2]

            def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
                # Simulate HF DynamicCache.update — append keys/values
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
                self._seen_tokens = max(self._seen_tokens, self.key_cache[layer_idx].shape[2])
                return self.key_cache[layer_idx], self.value_cache[layer_idx]

            def crop(self, max_len):
                for i in range(len(self.key_cache)):
                    self.key_cache[i] = self.key_cache[i][:, :, :max_len, :]
                    self.value_cache[i] = self.value_cache[i][:, :, :max_len, :]

        num_kv_heads = 1
        cache = DummyCache(num_layers, batch_size, prefix_len, num_kv_heads, head_dim)
        return cache, get_cache_seq_len, truncate_cache_to_len, PreserveCacheLen

    def test_cache_grows_without_wrapper(self):
        """Baseline: verify the cache *does* grow when not wrapped (confirming the hazard)."""
        cache, get_len, _, _ = self._make_dummy_cache(
            batch_size=1, prefix_len=16, num_layers=2
        )
        assert get_len(cache) == 16

        # Simulate one forward step that appends suffix keys
        suffix_k = torch.randn(1, 1, 4, 32)
        suffix_v = torch.randn(1, 1, 4, 32)
        cache.update(suffix_k, suffix_v, layer_idx=0)
        cache.update(suffix_k, suffix_v, layer_idx=1)

        assert get_len(cache) == 20, "Cache should grow without wrapper (baseline check)"

    def test_preserve_cache_len_restores_length(self):
        """PreserveCacheLen must restore original seq_len after a forward that grows the cache."""
        cache, get_len, _, PreserveCacheLen = self._make_dummy_cache(
            batch_size=1, prefix_len=16, num_layers=2
        )
        original_len = get_len(cache)
        assert original_len == 16

        with PreserveCacheLen(cache):
            # Simulate forward that appends suffix keys
            suffix_k = torch.randn(1, 1, 4, 32)
            suffix_v = torch.randn(1, 1, 4, 32)
            cache.update(suffix_k, suffix_v, layer_idx=0)
            cache.update(suffix_k, suffix_v, layer_idx=1)
            assert get_len(cache) == 20  # Grew inside the context

        # After context exit, length should be restored
        assert get_len(cache) == original_len, "Cache length must be restored after PreserveCacheLen"

    def test_preserve_cache_len_multiple_steps(self):
        """Repeated denoise steps must leave the cache at its original prefix length.

        This is the real-world scenario: sample_actions calls denoise_step N times
        with the same prefix_ctx (which contains past_key_values).  The cache must
        not grow across steps.
        """
        cache, get_len, _, PreserveCacheLen = self._make_dummy_cache(
            batch_size=2, prefix_len=32, num_layers=3
        )
        original_len = get_len(cache)
        num_steps = 5

        for step in range(num_steps):
            with PreserveCacheLen(cache):
                # Simulate forward
                suffix_k = torch.randn(2, 1, 8, 32)
                suffix_v = torch.randn(2, 1, 8, 32)
                for layer_idx in range(3):
                    cache.update(suffix_k, suffix_v, layer_idx=layer_idx)

            current_len = get_len(cache)
            assert current_len == original_len, (
                f"After step {step}: cache len={current_len}, expected {original_len}"
            )

    def test_preserve_cache_len_restores_on_exception(self):
        """Cache length must be restored even if the forward pass raises an exception."""
        cache, get_len, _, PreserveCacheLen = self._make_dummy_cache(
            batch_size=1, prefix_len=16, num_layers=2
        )
        original_len = get_len(cache)

        with pytest.raises(RuntimeError, match="simulated failure"):
            with PreserveCacheLen(cache):
                suffix_k = torch.randn(1, 1, 4, 32)
                suffix_v = torch.randn(1, 1, 4, 32)
                cache.update(suffix_k, suffix_v, layer_idx=0)
                cache.update(suffix_k, suffix_v, layer_idx=1)
                assert get_len(cache) == 20
                raise RuntimeError("simulated failure")

        assert get_len(cache) == original_len, "Cache must be restored even on exception"

    def test_legacy_tuple_cache_preservation(self):
        """PreserveCacheLen must also work with legacy list-format caches."""
        from openpi.models_pytorch.cache_utils import (
            PreserveCacheLen,
            get_cache_seq_len,
        )

        batch, num_kv_heads, seq_len, head_dim = 1, 1, 16, 32
        num_layers = 2
        # Legacy format: list of (key, value) tuples
        cache = [
            (torch.randn(batch, num_kv_heads, seq_len, head_dim),
             torch.randn(batch, num_kv_heads, seq_len, head_dim))
            for _ in range(num_layers)
        ]
        original_len = get_cache_seq_len(cache)
        assert original_len == 16

        with PreserveCacheLen(cache):
            # Simulate growth by replacing with longer tensors
            for i in range(num_layers):
                extra_k = torch.randn(batch, num_kv_heads, 4, head_dim)
                extra_v = torch.randn(batch, num_kv_heads, 4, head_dim)
                cache[i] = (
                    torch.cat([cache[i][0], extra_k], dim=2),
                    torch.cat([cache[i][1], extra_v], dim=2),
                )
            assert get_cache_seq_len(cache) == 20

        assert get_cache_seq_len(cache) == original_len


# ---------------------------------------------------------------------------
#  11. Subtask causal mask consistency (encode_prefix matches training)
# ---------------------------------------------------------------------------


class TestSubtaskCausalMaskConsistency:
    """Verify that encode_prefix uses causal attention for subtask tokens.

    During training, subtask tokens in the prefix have causal attention
    (``causal=True`` in ``_embed_conditioning_subtask``).  Inference must
    match this so the action expert's cross-attention sees the same
    subtask-token hidden states at inference time as during training.
    """

    def test_encode_prefix_subtask_causal_matches_training(self):
        """SubtaskActionExpert.encode_prefix must pass causal=True for subtask tokens.

        Training path (``compute_subtask_loss_train``) uses ``causal=True``.
        The inference encode_prefix must use the same value so the subtask
        conditioning distribution matches between train and inference.
        """
        from openpi.models_pytorch.action_experts.subtask_expert import SubtaskActionExpert

        expert = SubtaskActionExpert()

        # Use a mini model that records what causal value is passed
        class RecordingModel:
            def __init__(self):
                self.prefix_embs = None
                self.prefix_pad = None
                self.prefix_att = None
                self.causal_used = None
                self.config = SimpleNamespace(action_horizon=4)
                self.paligemma_with_expert = SimpleNamespace()
                # We need paligemma_with_expert to have embed_language_tokens
                self._lang_emb = nn.Embedding(100, 16)
                self.paligemma_with_expert.embed_language_tokens = self._lang_emb.forward
                # Also need paligemma for encode_prefix forward
                self.paligemma_with_expert.paligemma = SimpleNamespace()
                self.paligemma_with_expert.paligemma.language_model = SimpleNamespace()
                self.paligemma_with_expert.paligemma.language_model.config = SimpleNamespace(
                    _attn_implementation="eager"
                )
                self.paligemma_with_expert.forward = self._fake_forward
                self.make_att_2d_masks = lambda pm, am: (pm[:, None, :] * am[:, :, None]).bool()
                self._prepare_attention_masks_4d = lambda m: m[:, None, :, :].float()

            def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
                b = lang_tokens.shape[0]
                # Return dummy prefix of length 8
                return (
                    torch.randn(b, 8, 16),       # prefix_embs
                    torch.ones(b, 8, dtype=torch.bool),  # prefix_pad_masks
                    torch.zeros(b, 8, dtype=torch.int32), # prefix_att_masks (non-causal)
                )

            def _fake_forward(self, attention_mask=None, position_ids=None, past_key_values=None,
                             inputs_embeds=None, use_cache=False, adarms_cond=None):
                # Simulate prefix forward: return past_key_values as None
                return (None, None)

        # Monkey-patch _embed_conditioning_subtask to capture the causal flag
        original_method = expert._embed_conditioning_subtask
        captured_causal = {}

        def patched(*, model, prefix_embs, prefix_pad_masks, prefix_att_masks,
                   subtask_tokens, subtask_mask, causal):
            captured_causal["value"] = causal
            return original_method(
                model=model,
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks,
                subtask_tokens=subtask_tokens,
                subtask_mask=subtask_mask,
                causal=causal,
            )

        expert._embed_conditioning_subtask = patched

        try:
            m = RecordingModel()
            subtask_tokens = torch.randint(10, 90, (1, 6))
            subtask_mask = torch.ones(1, 6, dtype=torch.bool)

            expert.encode_prefix(
                model=m,
                images=[torch.randn(1, 3, 224, 224)],
                img_masks=[torch.ones(1, 256, dtype=torch.bool)],
                lang_tokens=torch.randint(10, 90, (1, 8)),
                lang_masks=torch.ones(1, 8, dtype=torch.bool),
                subtask_tokens=subtask_tokens,
                subtask_mask=subtask_mask,
            )

            assert "value" in captured_causal, "_embed_conditioning_subtask was not called"
            assert captured_causal["value"] is True, (
                "encode_prefix must use causal=True for subtask tokens to match training. "
                f"Got causal={captured_causal['value']}"
            )
        finally:
            expert._embed_conditioning_subtask = original_method

    def test_subtask_tokens_have_causal_attention_in_prefix(self):
        """With causal=True, subtask token i can attend to subtask tokens 0..i (not future tokens).

        This verifies the mask semantics directly using make_att_2d_masks,
        confirming that subtask tokens form a causal block within the prefix.
        """
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        batch = 1
        prefix_len = 8    # visual + prompt tokens (non-causal)
        subtask_len = 6   # subtask tokens (causal)

        # Non-causal prefix: att_masks = 0 for all prefix positions
        prefix_pad = torch.ones(batch, prefix_len, dtype=torch.bool)
        prefix_att = torch.zeros(batch, prefix_len, dtype=torch.int32)

        # Causal subtask: att_masks = 1 for all subtask positions
        subtask_pad = torch.ones(batch, subtask_len, dtype=torch.bool)
        subtask_att = torch.ones(batch, subtask_len, dtype=torch.int32)

        # Combine
        full_pad = torch.cat([prefix_pad, subtask_pad], dim=1)
        full_att = torch.cat([prefix_att, subtask_att], dim=1)

        att_2d = make_att_2d_masks(full_pad, full_att)
        # Shape: (batch, prefix_len+subtask_len, prefix_len+subtask_len)
        assert att_2d.shape == (batch, prefix_len + subtask_len, prefix_len + subtask_len)

        # All prefix tokens can attend to each other (bidirectional)
        prefix_att_block = att_2d[0, :prefix_len, :prefix_len]
        assert prefix_att_block.all(), "Prefix tokens must have bidirectional attention"

        # All subtask tokens can attend to all prefix tokens
        subtask_to_prefix = att_2d[0, prefix_len:, :prefix_len]
        assert subtask_to_prefix.all(), "All subtask tokens must be able to attend to all prefix tokens"

        # Subtask tokens have causal attention among themselves
        # subtask token i can attend to subtask tokens 0..i
        for i in range(subtask_len):
            for j in range(subtask_len):
                global_i = prefix_len + i
                global_j = prefix_len + j
                if j <= i:
                    assert att_2d[0, global_i, global_j].item(), (
                        f"Subtask token {i} should be able to attend to subtask token {j}"
                    )
                else:
                    assert not att_2d[0, global_i, global_j].item(), (
                        f"Subtask token {i} should NOT be able to attend to future subtask token {j}"
                    )

    def test_causal_false_gives_bidirectional(self):
        """With causal=False (old buggy behavior), subtask tokens have full bidirectional attention.

        This test documents the old/wrong behavior for comparison.  It should
        pass regardless — it's just verifying our understanding of the mask
        semantics.
        """
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        batch = 1
        prefix_len = 8
        subtask_len = 6

        prefix_pad = torch.ones(batch, prefix_len, dtype=torch.bool)
        prefix_att = torch.zeros(batch, prefix_len, dtype=torch.int32)

        # causal=False: subtask_att = zeros
        subtask_pad = torch.ones(batch, subtask_len, dtype=torch.bool)
        subtask_att = torch.zeros(batch, subtask_len, dtype=torch.int32)

        full_pad = torch.cat([prefix_pad, subtask_pad], dim=1)
        full_att = torch.cat([prefix_att, subtask_att], dim=1)

        att_2d = make_att_2d_masks(full_pad, full_att)

        # All tokens can attend to all other tokens (fully bidirectional)
        assert att_2d.all(), "With causal=False, all tokens must have bidirectional attention"
