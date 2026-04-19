"""Subtask action expert for PI05_SUBTASK.

Extends GemmaTokenExpert to compute BOTH:
1. Flow matching velocity field (continuous actions) via the action expert
2. Text logits for subtask prediction via the VLM backbone's language head

This implements the dual-output architecture described in the π0.5 paper,
where the model simultaneously predicts text tokens (subtask) and continuous
action chunks within a single forward pass.
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from openpi.models_pytorch.action_experts.base import ActionExpert
from openpi.models_pytorch.dtype_utils import align_tensors_to_reference_dtype
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks


class SubtaskActionExpert(ActionExpert):
    """Action expert that returns both flow matching velocity and text logits."""

    def _embed_conditioning_subtask(
        self,
        *,
        model: nn.Module,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        subtask_tokens: torch.Tensor | None,
        subtask_mask: torch.Tensor | None,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if subtask_tokens is None or subtask_mask is None or not torch.any(subtask_mask):
            return prefix_embs, prefix_pad_masks, prefix_att_masks

        subtask_embs = model.paligemma_with_expert.embed_language_tokens(subtask_tokens)
        emb_dim = subtask_embs.shape[-1]
        subtask_embs = subtask_embs * (emb_dim**0.5)

        prefix_embs = torch.cat([prefix_embs, subtask_embs], dim=1)
        prefix_pad_masks = torch.cat([prefix_pad_masks, subtask_mask], dim=1)

        if causal:
            subtask_att = torch.ones_like(subtask_mask, dtype=prefix_att_masks.dtype)
        else:
            subtask_att = torch.zeros_like(subtask_mask, dtype=prefix_att_masks.dtype)
        prefix_att_masks = torch.cat([prefix_att_masks, subtask_att], dim=1)
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    def encode_prefix(
        self,
        *,
        model: nn.Module,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        subtask_tokens: torch.Tensor | None = None,
        subtask_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._embed_conditioning_subtask(
            model=model,
            prefix_embs=prefix_embs,
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks,
            subtask_tokens=subtask_tokens,
            subtask_mask=subtask_mask,
            causal=False,
        )
        prefix_att_2d_masks = model.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)

        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return {
            "prefix_pad_masks": prefix_pad_masks,
            "past_key_values": past_key_values,
        }

    def compute_velocity_train(
        self,
        *,
        model: nn.Module,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        x_t: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Standard flow matching velocity computation (same as GemmaTokenExpert)."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(state, x_t, time)

        prefix_embs, suffix_embs = align_tensors_to_reference_dtype(
            model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight,
            prefix_embs,
            suffix_embs,
            context="language model",
        )

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = model.make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = model._prepare_attention_masks_4d(att_2d_masks)

        (_, suffix_out), _ = model.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = suffix_out[:, -model.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return model.action_out_proj(suffix_out)

    def compute_subtask_loss_train(
        self,
        *,
        model: nn.Module,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        x_t: torch.Tensor,
        time: torch.Tensor,
        subtask_tokens: torch.Tensor,
        subtask_mask: torch.Tensor,
        subtask_ar_mask: torch.Tensor,
        subtask_loss_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute both flow matching loss and text CE loss in a single pass.

        Architecture:
        - Prefix: [image_tokens, prompt+state_tokens, subtask_tokens]
        - Suffix: [action_tokens] (processed by action expert)

        The subtask tokens are appended to the prefix with causal attention,
        enabling the VLM backbone to predict them autoregressively.
        After the joint forward pass:
        - Text logits from the VLM backbone predict subtask tokens -> CE loss
        - Action expert output predicts flow velocity -> MSE loss

        Returns dict with 'flow_loss', 'ce_loss', and 'v_t' tensors.
        """
        # === Build prefix: images + prompt/state + subtask tokens ===
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        extended_prefix_embs, extended_prefix_pad_masks, extended_prefix_att_masks = self._embed_conditioning_subtask(
            model=model,
            prefix_embs=prefix_embs,
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks,
            subtask_tokens=subtask_tokens,
            subtask_mask=subtask_mask,
            causal=True,
        )
        subtask_len = subtask_tokens.shape[1]

        # === Build suffix: action tokens ===
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(state, x_t, time)

        extended_prefix_embs, suffix_embs = align_tensors_to_reference_dtype(
            model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight,
            extended_prefix_embs,
            suffix_embs,
            context="language model",
        )
        extended_prefix_att_masks = extended_prefix_att_masks.to(dtype=suffix_att_masks.dtype)

        # === Combined forward pass ===
        pad_masks = torch.cat([extended_prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([extended_prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = model._prepare_attention_masks_4d(att_2d_masks)

        (prefix_out, suffix_out), _ = model.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[extended_prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        # === Flow matching loss (from action expert output) ===
        action_out = suffix_out[:, -model.config.action_horizon :]
        action_out = action_out.to(dtype=torch.float32)
        v_t = model.action_out_proj(action_out)

        # === Text CE loss (from VLM backbone output on subtask tokens) ===
        # The prefix_out contains hidden states for [images, prompt, subtask].
        # We need the hidden states corresponding to subtask token positions.
        prefix_len_no_subtask = prefix_embs.shape[1]
        subtask_hidden = prefix_out[:, prefix_len_no_subtask:prefix_len_no_subtask + subtask_len]

        # Project to vocabulary logits using the VLM's language model head
        subtask_hidden = subtask_hidden.to(
            dtype=model.paligemma_with_expert.paligemma.language_model.embed_tokens.weight.dtype
        )
        text_logits = torch.matmul(
            subtask_hidden,
            model.paligemma_with_expert.paligemma.language_model.embed_tokens.weight.T,
        )  # (batch, subtask_len, vocab_size)

        # Compute CE loss: predict next subtask token
        # Shift: logits[t] predicts token[t+1]
        shift_logits = text_logits[:, :-1].contiguous()  # (batch, subtask_len-1, vocab)
        shift_targets = subtask_tokens[:, 1:].contiguous().to(dtype=torch.long)  # (batch, subtask_len-1)
        shift_loss_mask = subtask_loss_mask[:, 1:].contiguous().float()  # (batch, subtask_len-1)

        # Compute per-token CE loss
        ce_loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            reduction="none",
        ).view(shift_logits.shape[0], -1)  # (batch, subtask_len-1)

        # Global mean CE loss over all valid tokens in the batch.
        # Previously we computed per-sample mean then averaged across batch (mean-of-means),
        # which amplified variance when batch composition varied (e.g., some samples had no
        # valid subtask tokens, producing ce_loss=0 that skewed the batch mean).
        # Now we sum all per-token losses and divide by total valid tokens directly,
        # which is the standard NLP practice for consistent loss statistics.
        total_ce_loss = (ce_loss_per_token * shift_loss_mask).sum()
        total_valid_tokens = shift_loss_mask.sum().clamp(min=1)
        ce_loss = total_ce_loss / total_valid_tokens

        return {
            "v_t": v_t,
            "ce_loss": ce_loss,  # scalar
        }

    def compute_velocity_infer(
        self,
        *,
        model: nn.Module,
        prefix_ctx: dict[str, Any],
        state: torch.Tensor,
        x_t: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Standard inference velocity computation."""
        prefix_pad_masks = prefix_ctx["prefix_pad_masks"]
        past_key_values = prefix_ctx["past_key_values"]
        return model.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time)
