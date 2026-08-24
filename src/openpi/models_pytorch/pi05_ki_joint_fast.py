"""pi0.5-KI Variant A: FAST discrete action tokens with a cross-entropy backbone objective.

This is the paper-accurate Knowledge Insulation recipe. The sibling
``pi05_ki_joint_query`` implements Variant B, which replaces the discrete
objective with learned action queries regressed by mean-squared error.

Why the two differ, per the KI paper
------------------------------------
The paper's central claim is that the *form* of the backbone action target
matters, not merely its presence: continuous adapters feed gradients from
freshly initialized weights straight into the pretrained VLM, which
"can degrade both their ability to interpret language commands and the overall
performance". Discrete next-token prediction sidesteps that because the signal
is "unaffected by the uninitialized weights of the action expert". Its own
ablation ranks FAST > naive tokenization > continuous actions alone.

Design
------
Sequence layout (backbone phase):

    [images | prompt+state | subtask (causal) | FAST action tokens (causal)]

* Action tokens are ground truth under teacher forcing, embedded through the
  EXISTING vocabulary embedding and scored against its tied transpose, so this
  variant adds **no new parameters**. Variant B by contrast owns
  ``query_embeddings`` and ``query_action_head``.
* The action segment is CAUSAL, matching the paper ("FAST action tokens attend
  to this prefix and auto-regressively on previous action tokens") and unlike
  Variant B's bidirectional query block, which is parallel-decoded and so has
  no autoregressive structure to respect.
* ``truncate_expert_kv`` keeps the flow expert from attending to the action
  tokens. For Variant B that boundary was architectural; here it is a hard
  correctness requirement, because the tokens ARE ground truth. Without it the
  expert would read the answer during training and find it missing at
  inference. The paper states the same constraint: expert embeddings
  "do not attend to FAST action tokens to avoid information leakage between the
  two representations of actions".
* Action tokens are training-only. Inference still samples through the flow
  expert, exactly as in Variant B.

Everything else -- the flow-matching expert phase, knowledge insulation,
subtask cross-entropy, optimizer grouping, checkpointing -- is inherited
unchanged from Variant B, so an A/B comparison isolates the backbone action
objective.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from openpi.models_pytorch.attn_impl import resolve_attn_impl
from openpi.models_pytorch.pi05_ki_joint_query import PI05KIJointQueryPytorch

logger = logging.getLogger("openpi")


class PI05KIJointFastPytorch(PI05KIJointQueryPytorch):
    """Variant A: discrete FAST action tokens supervised with cross-entropy."""

    # No learned query embeddings and no query action head: the action target
    # is discrete and reuses the vocabulary embedding plus its tied transpose.
    _uses_learned_query_tokens = False

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        # beta_action replaces beta_query as the backbone action-loss weight.
        # Falling back to beta_query keeps a shared config factory usable.
        self.beta_action: float = float(
            getattr(config, "beta_action", getattr(config, "beta_query", 1.0))
        )
        self.action_token_max_len: int = int(getattr(config, "action_token_max_len", 64))

        if not self.truncate_expert_kv:
            # Variant B could tolerate truncate_expert_kv=False as an ablation
            # because its query tokens carry no ground truth. Here they do, so
            # the same setting would leak the target into the expert and create
            # a train/inference mismatch that silently degrades the policy.
            raise ValueError(
                "PI05KIJointFastPytorch requires truncate_expert_kv=True. The FAST action "
                "tokens are teacher-forced ground truth, so letting the flow expert attend "
                "to them leaks the target during training while they are absent at inference."
            )

        logger.info(
            "PI05KIJointFastPytorch initialized: KI=%s, beta_text=%.3f, beta_action=%.3f, "
            "action_token_max_len=%d, truncate_kv=%s, flow_loss_weight=%.3f",
            self.knowledge_insulation,
            self.beta_text,
            self.beta_action,
            self.action_token_max_len,
            self.truncate_expert_kv,
            self.flow_loss_weight,
        )

    # ------------------------------------------------------------------
    #  Variant B hooks that must not be reachable here
    # ------------------------------------------------------------------

    def _embed_query_tokens(self, **kwargs):
        raise NotImplementedError(
            "PI05KIJointFastPytorch has no learned query tokens; it embeds discrete FAST "
            "action tokens instead. This is a Variant B code path."
        )

    def _compute_query_mse_loss(self, **kwargs):
        raise NotImplementedError(
            "PI05KIJointFastPytorch supervises actions with cross-entropy over FAST tokens, "
            "not with query MSE. This is a Variant B code path."
        )

    # ------------------------------------------------------------------
    #  Action-token embedding
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_action_token_fields(observation):
        """Pull the Variant A action-token fields off an observation."""
        return (
            getattr(observation, "action_tokens", None),
            getattr(observation, "action_token_mask", None),
            getattr(observation, "action_token_loss_mask", None),
        )

    def _embed_action_tokens(
        self,
        *,
        action_tokens: Tensor,
        action_token_mask: Tensor,
        target_dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embed the FAST action tokens for the backbone sequence.

        Returns ``(embs, pad_masks, att_masks)`` where ``att_masks`` is all ones
        so every action position opens its own block. That yields causal
        attention within the segment and, because the first entry is 1, keeps
        the whole segment invisible to the prefix -- the same block-causal
        invariant the query block relies on.
        """
        embs = self.paligemma_with_expert.embed_language_tokens(action_tokens.to(dtype=torch.long))
        # PaliGemma scales token embeddings by sqrt(hidden_dim).
        embs = embs * (embs.shape[-1] ** 0.5)
        embs = embs.to(dtype=target_dtype)

        pad_masks = action_token_mask.to(dtype=torch.bool)
        # ones_like -> the cumulative block id advances on every physical slot,
        # matching _embed_conditioning_subtask's causal convention.
        att_masks = torch.ones_like(pad_masks, dtype=torch.bool)
        return embs, pad_masks, att_masks

    def _compute_action_ce_loss(
        self,
        *,
        prefix_out: Tensor,
        action_segment_start: int,
        action_tokens: Tensor,
        action_token_loss_mask: Tensor,
    ) -> Tensor:
        """Next-token cross-entropy over the FAST action segment.

        Mirrors ``_compute_ce_loss`` exactly (tied embedding transpose, shift by
        one, masked mean) so the subtask and action objectives stay comparable.
        """
        shift_logits, shift_targets, shift_loss_mask = self._action_token_predictions(
            prefix_out=prefix_out,
            action_segment_start=action_segment_start,
            action_tokens=action_tokens,
            action_token_loss_mask=action_token_loss_mask,
        )

        per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            reduction="none",
        ).view(shift_logits.shape[0], -1)

        total = (per_token * shift_loss_mask).sum()
        denom = shift_loss_mask.sum().clamp(min=1)
        return total / denom

    def _compute_action_token_accuracy_from_hidden(
        self,
        *,
        prefix_out: Tensor,
        action_segment_start: int,
        action_tokens: Tensor,
        action_token_loss_mask: Tensor,
    ) -> Tensor:
        """Compute masked next-token accuracy for the FAST action segment."""
        shift_logits, shift_targets, shift_loss_mask = self._action_token_predictions(
            prefix_out=prefix_out,
            action_segment_start=action_segment_start,
            action_tokens=action_tokens,
            action_token_loss_mask=action_token_loss_mask,
        )
        correct = (shift_logits.argmax(dim=-1) == shift_targets).float()
        total_valid = shift_loss_mask.sum().clamp(min=1)
        return (correct * shift_loss_mask).sum() / total_valid

    def _action_token_predictions(
        self,
        *,
        prefix_out: Tensor,
        action_segment_start: int,
        action_tokens: Tensor,
        action_token_loss_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return shifted FAST logits, targets, and mask shared by eval metrics."""
        seg_len = action_tokens.shape[1]
        hidden = prefix_out[:, action_segment_start : action_segment_start + seg_len]

        embed_weight = self.paligemma_with_expert.paligemma.language_model.embed_tokens.weight
        hidden = hidden.to(dtype=embed_weight.dtype)
        logits = torch.matmul(hidden, embed_weight.T)

        # Row t predicts token t+1.
        shift_logits = logits[:, :-1].contiguous()
        shift_targets = action_tokens[:, 1:].contiguous().to(dtype=torch.long)
        shift_loss_mask = action_token_loss_mask[:, 1:].contiguous().float()
        return shift_logits, shift_targets, shift_loss_mask

    # ------------------------------------------------------------------
    #  Phase 1: backbone losses (subtask CE + action-token CE)
    # ------------------------------------------------------------------

    def compute_backbone_losses(self, observation, actions) -> dict[str, Tensor]:
        """Subtask cross-entropy plus FAST action-token cross-entropy.

        Structurally identical to Variant B's phase 1, except the trailing
        segment holds discrete ground-truth action tokens scored by CE rather
        than learned queries regressed by MSE.
        """
        assert actions.dim() == 3, f"actions must be [B, T, D], got shape {actions.shape}"

        images, img_masks, lang_tokens, lang_masks, _state = self._preprocess_observation(
            observation, train=True
        )

        subtask_tokens = getattr(observation, "subtask_tokens", None)
        subtask_mask = getattr(observation, "subtask_mask", None)
        subtask_loss_mask = getattr(observation, "subtask_loss_mask", None)
        has_subtask = (
            subtask_tokens is not None
            and subtask_mask is not None
            and subtask_loss_mask is not None
            and subtask_loss_mask.any()
        )

        action_tokens, action_token_mask, action_token_loss_mask = self._extract_action_token_fields(
            observation
        )
        if action_tokens is None or action_token_mask is None or action_token_loss_mask is None:
            raise ValueError(
                "PI05KIJointFastPytorch requires action_tokens / action_token_mask / "
                "action_token_loss_mask on the observation. Use "
                "transforms.TokenizeSubtaskAndActionInputs in the data pipeline."
            )

        # ---- Prefix: images + prompt/state + subtask ----
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_base_len = prefix_embs.shape[1]

        if has_subtask:
            prefix_embs, prefix_pad_masks, prefix_att_masks = (
                self.action_expert._embed_conditioning_subtask(  # noqa: SLF001
                    model=self,
                    prefix_embs=prefix_embs,
                    prefix_pad_masks=prefix_pad_masks,
                    prefix_att_masks=prefix_att_masks,
                    subtask_tokens=subtask_tokens,
                    subtask_mask=subtask_mask,
                    causal=True,
                )
            )
        prefix_after_subtask_len = prefix_embs.shape[1]

        # ---- Append the discrete action-token segment ----
        act_embs, act_pad_masks, act_att_masks = self._embed_action_tokens(
            action_tokens=action_tokens,
            action_token_mask=action_token_mask,
            target_dtype=prefix_embs.dtype,
        )
        full_embs = torch.cat([prefix_embs, act_embs], dim=1)
        full_pad_masks = torch.cat([prefix_pad_masks, act_pad_masks], dim=1)
        full_att_masks = torch.cat([prefix_att_masks, act_att_masks], dim=1)

        full_att_2d = self.make_att_2d_masks(full_pad_masks, full_att_masks)
        full_position_ids = torch.cumsum(full_pad_masks, dim=1) - 1
        full_att_2d_4d = self._prepare_attention_masks_4d(full_att_2d)

        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (  # noqa: SLF001
            resolve_attn_impl()
        )
        (prefix_out, _), _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_4d,
            position_ids=full_position_ids,
            past_key_values=None,
            inputs_embeds=[full_embs, None],
            use_cache=False,
        )

        ce_loss = self._compute_ce_loss(
            prefix_out=prefix_out,
            prefix_base_len=prefix_base_len,
            subtask_tokens=subtask_tokens,
            subtask_loss_mask=subtask_loss_mask,
            has_subtask=has_subtask,
        )

        action_ce_loss = self._compute_action_ce_loss(
            prefix_out=prefix_out,
            action_segment_start=prefix_after_subtask_len,
            action_tokens=action_tokens,
            action_token_loss_mask=action_token_loss_mask,
        )

        backbone_loss = self.beta_text * ce_loss + self.beta_action * action_ce_loss

        return {
            "backbone_loss": backbone_loss,
            "ce_loss": ce_loss.detach(),
            "action_ce_loss": action_ce_loss.detach(),
        }

    # ------------------------------------------------------------------
    #  Evaluation: FAST-specific backbone metrics + shared expert metrics
    # ------------------------------------------------------------------

    def compute_eval_metrics(
        self,
        observation,
        actions,
        *,
        compute_flow_l1: bool = False,
        num_denoise_steps: int = 10,
        flow_l1_seed: int = 42,
        deterministic_flow: bool = False,
    ) -> dict[str, Tensor]:
        """Compute validation metrics without entering learned-query code.

        Variant A reports its discrete action objective as ``action_ce_loss``
        and ``action_token_accuracy``. It deliberately does not emit Variant B's
        ``query_mse_loss`` or ``query_l1`` keys: FAST has neither learned query
        tokens nor a query action head, so either name would misstate the metric.

        Args:
            deterministic_flow: if True, make the flow-matching metrics
                reproducible by drawing the ``(noise, time)`` pair from a fixed
                seed instead of the global RNG and preprocessing images with
                ``train=False``. Without it ``flow_loss`` / ``expert_loss`` /
                ``total_loss`` carry a random component that does NOT shrink as
                the validation subset grows.

                This must stay behaviourally identical to
                ``PI05KIJointQueryPytorch.compute_eval_metrics``: the trainer
                calls both variants through the same ``is_pi05_ki_joint`` branch
                and passes this flag unconditionally, so any asymmetry here would
                make the two arms differ in validation determinism and become a
                confound in the A/B comparison rather than a code difference.
        """
        (
            backbone_loss,
            ce_loss,
            action_ce_loss,
            subtask_accuracy,
            action_token_accuracy,
        ) = self._compute_backbone_eval_metrics(observation, actions)

        # ---- Expert forward ----
        # Mirrors Variant B exactly. NOTE torch.manual_seed() reseeds BOTH the CPU
        # and all CUDA generators while torch.get/set_rng_state() covers only the
        # CPU one, so the CUDA state is saved and restored explicitly; otherwise
        # this would leak a CUDA reseed into the training RNG stream.
        if deterministic_flow:
            cpu_state = torch.get_rng_state()
            cuda_states = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
            try:
                torch.manual_seed(flow_l1_seed)
                noise = self.sample_noise(actions.shape, actions.device)
                time = self.sample_time(actions.shape[0], actions.device)
            finally:
                torch.set_rng_state(cpu_state)
                if cuda_states is not None:
                    torch.cuda.set_rng_state_all(cuda_states)
            expert_losses = self.compute_expert_loss(
                observation, actions, noise=noise, time=time, train_preprocess=False
            )
        else:
            expert_losses = self.compute_expert_loss(observation, actions)
        expert_loss = expert_losses["expert_loss"]
        flow_loss = expert_losses["flow_loss"]

        result = {
            "total_loss": backbone_loss.detach() + expert_loss.detach(),
            "backbone_loss": backbone_loss.detach(),
            "expert_loss": expert_loss.detach(),
            "ce_loss": ce_loss.detach(),
            "action_ce_loss": action_ce_loss.detach(),
            "flow_loss": flow_loss.detach(),
            "subtask_accuracy": subtask_accuracy.detach(),
            "action_token_accuracy": action_token_accuracy.detach(),
            "flow_mse": flow_loss.detach(),
        }

        if compute_flow_l1:
            result["flow_l1"] = self._compute_flow_l1_euler(
                observation=observation,
                actions=actions,
                num_steps=num_denoise_steps,
                seed=flow_l1_seed,
            ).detach()

        return result

    def _compute_backbone_eval_metrics(
        self, observation, actions
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run one FAST backbone pass for loss and accuracy metrics."""
        assert actions.dim() == 3, f"actions must be [B, T, D], got shape {actions.shape}"

        images, img_masks, lang_tokens, lang_masks, _state = self._preprocess_observation(
            observation, train=False
        )

        subtask_tokens = getattr(observation, "subtask_tokens", None)
        subtask_mask = getattr(observation, "subtask_mask", None)
        subtask_loss_mask = getattr(observation, "subtask_loss_mask", None)
        has_subtask = (
            subtask_tokens is not None
            and subtask_mask is not None
            and subtask_loss_mask is not None
            and subtask_loss_mask.any()
        )

        action_tokens, action_token_mask, action_token_loss_mask = self._extract_action_token_fields(
            observation
        )
        if action_tokens is None or action_token_mask is None or action_token_loss_mask is None:
            raise ValueError(
                "PI05KIJointFastPytorch requires action_tokens / action_token_mask / "
                "action_token_loss_mask on the observation. Use "
                "transforms.TokenizeSubtaskAndActionInputs in the data pipeline."
            )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_base_len = prefix_embs.shape[1]

        if has_subtask:
            prefix_embs, prefix_pad_masks, prefix_att_masks = (
                self.action_expert._embed_conditioning_subtask(  # noqa: SLF001
                    model=self,
                    prefix_embs=prefix_embs,
                    prefix_pad_masks=prefix_pad_masks,
                    prefix_att_masks=prefix_att_masks,
                    subtask_tokens=subtask_tokens,
                    subtask_mask=subtask_mask,
                    causal=True,
                )
            )
        action_segment_start = prefix_embs.shape[1]

        action_embs, action_pad_masks, action_att_masks = self._embed_action_tokens(
            action_tokens=action_tokens,
            action_token_mask=action_token_mask,
            target_dtype=prefix_embs.dtype,
        )
        full_embs = torch.cat([prefix_embs, action_embs], dim=1)
        full_pad_masks = torch.cat([prefix_pad_masks, action_pad_masks], dim=1)
        full_att_masks = torch.cat([prefix_att_masks, action_att_masks], dim=1)

        full_att_2d = self.make_att_2d_masks(full_pad_masks, full_att_masks)
        full_position_ids = torch.cumsum(full_pad_masks, dim=1) - 1
        full_att_2d_4d = self._prepare_attention_masks_4d(full_att_2d)

        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (  # noqa: SLF001
            resolve_attn_impl()
        )
        (prefix_out, _), _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_4d,
            position_ids=full_position_ids,
            past_key_values=None,
            inputs_embeds=[full_embs, None],
            use_cache=False,
        )

        ce_loss = self._compute_ce_loss(
            prefix_out=prefix_out,
            prefix_base_len=prefix_base_len,
            subtask_tokens=subtask_tokens,
            subtask_loss_mask=subtask_loss_mask,
            has_subtask=has_subtask,
        )
        subtask_accuracy = self._compute_subtask_accuracy_from_hidden(
            prefix_out=prefix_out,
            prefix_base_len=prefix_base_len,
            subtask_tokens=subtask_tokens,
            subtask_loss_mask=subtask_loss_mask,
            has_subtask=has_subtask,
        )
        action_ce_loss = self._compute_action_ce_loss(
            prefix_out=prefix_out,
            action_segment_start=action_segment_start,
            action_tokens=action_tokens,
            action_token_loss_mask=action_token_loss_mask,
        )
        action_token_accuracy = self._compute_action_token_accuracy_from_hidden(
            prefix_out=prefix_out,
            action_segment_start=action_segment_start,
            action_tokens=action_tokens,
            action_token_loss_mask=action_token_loss_mask,
        )
        backbone_loss = self.beta_text * ce_loss + self.beta_action * action_ce_loss

        return backbone_loss, ce_loss, action_ce_loss, subtask_accuracy, action_token_accuracy
