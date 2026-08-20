"""π0.5-KI joint query training model (query-MSE variant: action query tokens + MSE + KI).

Implements the π0.5-KI joint query training architecture with Knowledge Insulation (KI).
query-MSE variant uses learned action query tokens in the backbone sequence, trained via
MSE between a query action head and ground-truth continuous actions.

Design choice (corrected from initial proposal)
=================================================
Initial proposal used ``stop_grad(P(actions))`` as MSE target, but ``P`` would
never receive gradients — a learnable parameter that never learns is a bug.

Corrected design (Option A):
  - **Query tokens**: learned embeddings (no GT action info in input)
  - **Query action head**: ``query_action_head: hidden_dim → action_dim``
    maps backbone hidden states at query positions directly to action space
  - **MSE loss**: ``MSE(query_action_head(h_query), gt_actions)`` — both
    predictions and targets live in the same action space; the head receives
    gradients, learning to decode the backbone's action representations
  - **No phantom target projection** — the query tokens + action head form the
    complete action prediction pathway on the backbone side

This is cleaner and analogous to Variant A's FAST CE loss:
  - Var A: backbone predicts discrete tokens → CE vs GT tokens
  - Var B: backbone predicts continuous actions → MSE vs GT actions

Architecture summary
====================
Backbone (PaliGemma) losses:
  - CE(subtask_tokens) — subtask text prediction (our extension)
  - MSE(query_action_head(h_query), actions) — action query prediction

Expert (Gemma) loss:
  - MSE(u_t, v_t) — flow matching velocity prediction

Knowledge Insulation:
  - KV cache truncated at subtask boundary (query tokens invisible to expert)
  - When ``knowledge_insulation=True``, truncated KV is detached before expert
    forward, preventing flow loss gradients from reaching backbone params.
  - When ``knowledge_insulation=False``, KV is not detached (baseline leakage).

Two-phase design (for memory-efficient training)
=================================================
To avoid ``retain_graph=True`` (which doubles activation memory) AND to allow
freeing the backbone graph before computing the expert pass, we expose two
separate public methods:

1. **``compute_backbone_losses(obs, actions)``** — full backbone sequence
   [images, prompt, subtask, query_tokens] → CE + query_MSE → backward → step
   → (graph freed when loss goes out of scope)

2. **``compute_expert_loss(obs, actions, noise, time)``** — prefix-only
   [images, prompt, subtask] → KV → expert flow loss → backward → step

   When ``knowledge_insulation=True``, the KV is detached, so this pass has
   no gradient connection to backbone parameters — safe to call at any time.

   When ``knowledge_insulation=False`` (baseline), the KV is NOT detached,
   so ``flow_loss.backward()`` will backprop through the full backbone graph.
   In this case, call ``compute_all_losses()`` which builds both graphs
   together, then call both backwards before stepping either optimizer.

A convenience ``compute_all_losses()`` is provided for testing/debugging and
for the KI-OFF baseline case where both graphs must exist simultaneously.

Query attention mask semantics
==============================
Query tokens use bidirectional self-attention within the query block (all
query positions can attend to all other query positions).  Since query tokens
contain NO ground-truth action information (they're pure learned embeddings),
bidirectional attention is safe and actually beneficial — it lets all query
positions share information like DETR object queries.

The block is causal with respect to preceding tokens (subtask, prompt, images):
query positions can attend all earlier prefix tokens, but earlier prefix
tokens cannot attend query positions (consistent with block-causal architecture).

Config fields (read via getattr on the existing config object)
==============================================================
- knowledge_insulation: bool = False — if True, detach expert KV (KI ON)
- beta_text: float = 1.0 — subtask CE loss weight (0 = core paper, no subtask)
- beta_query: float = 1.0 — action query MSE loss weight
- num_query_tokens: int = action_horizon — number of learned query tokens
- query_emb_dim: int | None = None — query embedding dim; None = VLM hidden dim
- truncate_expert_kv: bool = True — if True, expert sees prefix only (no queries)
- flow_loss_weight: float = 10.0 — expert loss weight (alpha, for logging/scaling)
"""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from openpi.models_pytorch.cache_utils import get_cache_seq_len
from openpi.models_pytorch.pi05_subtask import PI05SubtaskPytorch
from openpi.models_pytorch.attn_impl import resolve_attn_impl

logger = logging.getLogger("openpi")


# ---------------------------------------------------------------------------
#  KV cache helpers
# ---------------------------------------------------------------------------

def _validate_num_query_tokens(num_query_tokens: int) -> int:
    """Validate the learned-query block size.

    The query block must hold at least one token: ``_embed_query_tokens`` marks
    its first entry to open a dedicated attention block, and ``L_query`` has no
    meaning without a query position to supervise.
    """
    if num_query_tokens < 1:
        raise ValueError(
            f"num_query_tokens must be >= 1, got {num_query_tokens}. "
            "Set beta_query=0.0 to disable the query objective instead."
        )
    return num_query_tokens


def _detach_kv_cache(past_key_values: Any) -> Any:
    """Detach all tensors in a KV cache.

    Supports:
    - HuggingFace DynamicCache (new API: layers[i].keys / layers[i].values as tensors)
    - HuggingFace DynamicCache (old API: key_cache / value_cache lists)
    - Legacy list-of-tuples format: [(k, v), ...]
    - Legacy tuple-of-tuples format

    Returns the same cache object with all key/value tensors detached in-place.
    """
    # New-style DynamicCache: layers[i].keys / layers[i].values (tensors, not callable)
    if hasattr(past_key_values, "layers") and isinstance(past_key_values.layers, (list, tuple)):
        for layer in past_key_values.layers:
            if hasattr(layer, "keys") and isinstance(layer.keys, Tensor):
                layer.keys = layer.keys.detach()
            if hasattr(layer, "values") and isinstance(layer.values, Tensor):
                layer.values = layer.values.detach()
        return past_key_values

    # Old-style DynamicCache: key_cache / value_cache lists
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for i in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[i] = past_key_values.key_cache[i].detach()
            past_key_values.value_cache[i] = past_key_values.value_cache[i].detach()
        return past_key_values

    # Legacy list format: [(k, v), ...]
    if isinstance(past_key_values, list):
        return [(k.detach(), v.detach()) for k, v in past_key_values]

    # Legacy tuple format
    if isinstance(past_key_values, tuple):
        return tuple((k.detach(), v.detach()) for k, v in past_key_values)

    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)}")


# ---------------------------------------------------------------------------
#  π0.5-KI joint query Model (query-MSE variant)
# ---------------------------------------------------------------------------

class PI05KIJointQueryPytorch(PI05SubtaskPytorch):
    #: Whether this class creates the learned query-token parameters. The
    #: Variant A subclass sets this False: its backbone target is discrete FAST
    #: action tokens embedded through the existing vocabulary, so it needs no
    #: query embeddings and no query action head.
    _uses_learned_query_tokens: bool = True

    """π0.5-KI joint query training model — query-MSE variant: action query tokens + MSE + KI.

    Extends :class:`PI05SubtaskPytorch` with:
    - Learned action query embeddings inserted into the backbone sequence
    - Query action head maps query hidden states → action space
    - MSE(query_action_head(h_query), gt_actions) as backbone action loss
    - KV truncation at subtask boundary (query tokens invisible to expert)
    - Knowledge Insulation toggle (detach KV before expert forward)
    - Two-phase design: ``compute_backbone_losses`` + ``compute_expert_loss``
      for memory-efficient dual-optimizer training

    Parameters (read from config via getattr with defaults):
        knowledge_insulation: If True, flow loss produces zero backbone grads.
        beta_text: Subtask CE loss weight (default 1.0).
        beta_query: Action query MSE loss weight (default 1.0).
        num_query_tokens: Number of learned query tokens (default = action_horizon).
        query_emb_dim: Query embedding dim (default = VLM hidden dim).
        truncate_expert_kv: If True, truncate KV at subtask boundary (default True).
        flow_loss_weight: Expert loss weight / alpha (default 10.0).
    """

    def __init__(
        self,
        config,
        *,
        alpha: float = 10.0,
        action_expert_name: str = "subtask",
        action_expert_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            config,
            alpha=alpha,
            action_expert_name=action_expert_name,
            action_expert_kwargs=action_expert_kwargs,
        )

        # ---- Config fields with defaults ----
        self.knowledge_insulation: bool = bool(getattr(config, "knowledge_insulation", False))
        self.beta_text: float = float(getattr(config, "beta_text", 1.0))
        self.beta_query: float = float(getattr(config, "beta_query", 1.0))
        self.truncate_expert_kv: bool = bool(getattr(config, "truncate_expert_kv", True))
        self.flow_loss_weight: float = float(getattr(config, "flow_loss_weight", alpha))

        num_query_tokens = _validate_num_query_tokens(
            int(getattr(config, "num_query_tokens", config.action_horizon))
        )
        self.num_query_tokens: int = num_query_tokens

        # For first implementation, require num_query_tokens == action_horizon.
        # Silent interpolation can hide bugs and makes loss semantics ambiguous.
        # If different lengths are needed later, make it an explicit option.
        if num_query_tokens != config.action_horizon:
            logger.warning(
                "num_query_tokens (%d) != action_horizon (%d). "
                "This is experimental — query MSE will use linear interpolation.",
                num_query_tokens,
                config.action_horizon,
            )

        # Determine VLM hidden dimension from the actual model weights
        self._vlm_hidden_dim = self.paligemma_with_expert.paligemma.language_model.embed_tokens.weight.shape[1]

        query_emb_dim = getattr(config, "query_emb_dim", None)
        if query_emb_dim is None:
            query_emb_dim = self._vlm_hidden_dim
        self.query_emb_dim: int = int(query_emb_dim)

        action_dim = config.action_dim
        self._action_dim = action_dim

        # ---- Learned query embeddings ----
        # Guarded so the Variant A subclass (FAST discrete action tokens + CE)
        # can inherit every shared phase without carrying dead query-MSE
        # parameters in its state_dict or handing ZeRO gradient-free tensors.
        # Defaults to True, so this variant is unaffected.
        if not self._uses_learned_query_tokens:
            self._action_dim = action_dim
            self._log_init_summary(action_dim)
            return

        # Shape: [num_query_tokens, query_emb_dim]
        # These are learned embeddings placed after subtask tokens in the
        # backbone sequence.  They contain NO ground-truth action information
        # — pure learned vectors, analogous to DETR object queries.
        self.query_embeddings = nn.Parameter(
            torch.empty(self.num_query_tokens, self.query_emb_dim)
        )
        nn.init.normal_(self.query_embeddings, mean=0.0, std=self._vlm_hidden_dim ** -0.5)

        # Query embeddings may be a different dim than VLM hidden; if so,
        # we need a projection to map them into the VLM sequence.
        if self.query_emb_dim != self._vlm_hidden_dim:
            self.query_to_vlm_proj = nn.Linear(self.query_emb_dim, self._vlm_hidden_dim, bias=False)
        else:
            self.query_to_vlm_proj = nn.Identity()

        # ---- Query action head ----
        # Maps backbone hidden states at query positions to action space.
        # This is the "decoder" that converts the backbone's learned action
        # representations into actual action predictions.
        # Analogous to Var A's embedding matrix (which projects hidden → vocab).
        self.query_action_head = nn.Linear(self._vlm_hidden_dim, action_dim, bias=True)
        nn.init.zeros_(self.query_action_head.bias)
        nn.init.xavier_uniform_(self.query_action_head.weight, gain=0.01)

        self._log_init_summary(action_dim)

    def _log_init_summary(self, action_dim: int) -> None:
        logger.info(
            "PI05KIJointQueryPytorch initialized: KI=%s, beta_text=%.3f, beta_query=%.3f, "
            "num_query_tokens=%d, query_emb_dim=%d, action_dim=%d, truncate_kv=%s, "
            "flow_loss_weight=%.3f",
            self.knowledge_insulation,
            self.beta_text,
            self.beta_query,
            self.num_query_tokens,
            self.query_emb_dim,
            action_dim,
            self.truncate_expert_kv,
            self.flow_loss_weight,
        )

    # ------------------------------------------------------------------
    #  Phase 1: Backbone losses (CE + query MSE)
    # ------------------------------------------------------------------

    def compute_backbone_losses(
        self,
        observation,
        actions,
    ) -> dict[str, Tensor]:
        """Compute backbone-side losses: subtask CE + action query MSE.

        This is **Phase 1** of the two-phase training loop.  Call this,
        call ``backbone_loss.backward()``, step the backbone optimizer,
        then let the computation graph go out of scope to free memory
        before running Phase 2 (expert loss).

        Returns dict with:
            - ``backbone_loss``: scalar, beta_text * CE + beta_query * MSE
            - ``ce_loss``: scalar, subtask CE (detached, for logging)
            - ``query_mse_loss``: scalar, action query MSE (detached, for logging)
        """
        # ---- Shape assertions ----
        assert actions.dim() == 3, f"actions must be [B, T, D], got shape {actions.shape}"
        assert actions.shape[2] == self._action_dim, (
            f"action_dim mismatch: expected {self._action_dim}, got {actions.shape[2]}"
        )

        # ---- Preprocess observation ----
        images, img_masks, lang_tokens, lang_masks, _state = self._preprocess_observation(
            observation, train=True
        )

        # ---- Subtask info ----
        subtask_tokens = getattr(observation, "subtask_tokens", None)
        subtask_mask = getattr(observation, "subtask_mask", None)
        subtask_loss_mask = getattr(observation, "subtask_loss_mask", None)

        has_subtask = (
            subtask_tokens is not None
            and subtask_mask is not None
            and subtask_loss_mask is not None
            and subtask_loss_mask.any()
        )

        # ---- Build full prefix: [images, prompt/state, subtask, query_tokens] ----
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_base_len = prefix_embs.shape[1]

        if has_subtask:
            prefix_embs, prefix_pad_masks, prefix_att_masks = (
                self.action_expert._embed_conditioning_subtask(
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

        # Add query tokens
        query_embs, query_pad_masks, query_att_masks = self._embed_query_tokens(
            batch_size=prefix_embs.shape[0],
            device=prefix_embs.device,
            target_dtype=prefix_embs.dtype,
        )
        full_prefix_embs = torch.cat([prefix_embs, query_embs], dim=1)
        full_prefix_pad_masks = torch.cat([prefix_pad_masks, query_pad_masks], dim=1)
        full_prefix_att_masks = torch.cat([prefix_att_masks, query_att_masks], dim=1)

        # ---- Build 2D attention masks ----
        full_prefix_att_2d_masks = self.make_att_2d_masks(full_prefix_pad_masks, full_prefix_att_masks)
        full_prefix_position_ids = torch.cumsum(full_prefix_pad_masks, dim=1) - 1
        full_prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(full_prefix_att_2d_masks)

        # ---- Run backbone forward (PaliGemma first stream only) ----
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = resolve_attn_impl()
        (prefix_out, _), _ = self.paligemma_with_expert.forward(
            attention_mask=full_prefix_att_2d_masks_4d,
            position_ids=full_prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[full_prefix_embs, None],
            use_cache=False,
        )

        # ---- CE loss ----
        ce_loss = self._compute_ce_loss(
            prefix_out=prefix_out,
            prefix_base_len=prefix_base_len,
            subtask_tokens=subtask_tokens,
            subtask_loss_mask=subtask_loss_mask,
            has_subtask=has_subtask,
        )

        # ---- Query MSE loss ----
        query_mse_loss = self._compute_query_mse_loss(
            prefix_out=prefix_out,
            prefix_after_subtask_len=prefix_after_subtask_len,
            actions=actions,
            observation=observation,
        )

        # Backbone total: weighted sum
        backbone_loss = self.beta_text * ce_loss + self.beta_query * query_mse_loss

        return {
            "backbone_loss": backbone_loss,
            "ce_loss": ce_loss.detach(),
            "query_mse_loss": query_mse_loss.detach(),
        }

    # ------------------------------------------------------------------
    #  Phase 2: Expert flow loss
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _no_gc_on_backbone(self):
        """Temporarily disable gradient checkpointing on the backbone language model.

        PaliGemmaWithExpertModel.forward() forces gradient checkpointing ON
        in training mode (gemma_pytorch.py), which disables ``use_cache`` and
        returns ``past_key_values=None``.  We need KV cache for the expert's
        cross-attention, so we temporarily disable GC during prefix encoding.

        Also saves/restores ``config.use_cache`` for robustness.
        """
        lm = self.paligemma_with_expert.paligemma.language_model
        old_gc = getattr(lm, "gradient_checkpointing", False)
        old_use_cache = getattr(lm.config, "use_cache", True)
        try:
            if old_gc:
                lm.gradient_checkpointing = False
            lm.config.use_cache = True
            yield
        finally:
            if old_gc:
                lm.gradient_checkpointing = old_gc
            lm.config.use_cache = old_use_cache

    def compute_expert_loss(
        self,
        observation,
        actions,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute expert-side flow matching loss.

        This is **Phase 2** of the two-phase training loop.

        When ``knowledge_insulation=True`` (recommended):
          - The prefix KV is **detached** before expert forward.
          - ``flow_loss.backward()`` produces **zero backbone gradients**.
          - Safe to call after Phase 1 backward + step (backbone graph freed).
          - Prefix KV is built under ``torch.no_grad()`` for memory efficiency
            (valid because KI guarantees no flow grads reach backbone anyway).

        When ``knowledge_insulation=False`` (baseline leakage):
          - The prefix KV is NOT detached.
          - ``flow_loss.backward()`` backprops through the full backbone.
          - Two-phase sequential (backbone → expert) with a single optimizer
            step is valid; gradients accumulate on shared params.

        Returns dict with:
            - ``flow_loss``: scalar, raw flow matching MSE
            - ``expert_loss``: scalar, ``flow_loss_weight * flow_loss`` (for backward)
        """
        # ---- Preprocess observation ----
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=True
        )

        subtask_tokens = getattr(observation, "subtask_tokens", None)
        subtask_mask = getattr(observation, "subtask_mask", None)
        has_subtask = (
            subtask_tokens is not None
            and subtask_mask is not None
            and subtask_mask.any()
        )

        batch_size = actions.shape[0]
        device = actions.device

        if noise is None:
            noise = self.sample_noise(actions.shape, device)
        if time is None:
            time = self.sample_time(batch_size, device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # ---- Build expert prefix (images + prompt + subtask, NO query tokens) ----
        # KI=ON: build the entire prefix (embed + encode + cache) under no_grad()
        #   for memory efficiency — no graph is kept for vision/language backbone.
        #   Valid because flow grads never reach backbone via KI anyway.
        # KI=OFF: build with grad so flow loss backprops through full backbone.
        _grad_ctx = torch.no_grad() if self.knowledge_insulation else contextlib.nullcontext()
        with _grad_ctx:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )

            if has_subtask:
                prefix_embs, prefix_pad_masks, prefix_att_masks = (
                    self.action_expert._embed_conditioning_subtask(
                        model=self,
                        prefix_embs=prefix_embs,
                        prefix_pad_masks=prefix_pad_masks,
                        prefix_att_masks=prefix_att_masks,
                        subtask_tokens=subtask_tokens,
                        subtask_mask=subtask_mask,
                        causal=True,
                    )
                )

            # If truncation is disabled (ablation), include query tokens in expert prefix
            if not self.truncate_expert_kv:
                query_embs, query_pad_masks, query_att_masks = self._embed_query_tokens(
                    batch_size=batch_size,
                    device=device,
                    target_dtype=prefix_embs.dtype,
                )
                prefix_embs = torch.cat([prefix_embs, query_embs], dim=1)
                prefix_pad_masks = torch.cat([prefix_pad_masks, query_pad_masks], dim=1)
                prefix_att_masks = torch.cat([prefix_att_masks, query_att_masks], dim=1)

            # ---- Encode prefix through backbone to get KV cache ----
            prefix_att_2d_masks = self.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

            self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = resolve_attn_impl()

            # Temporarily disable GC so use_cache works (GC forces use_cache=False)
            with self._no_gc_on_backbone():
                _, past_key_values = self.paligemma_with_expert.forward(
                    attention_mask=prefix_att_2d_masks_4d,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=True,
                )

            if past_key_values is None:
                raise RuntimeError(
                    "compute_expert_loss: past_key_values is None after prefix encoding. "
                    "This usually means use_cache was overridden by gradient checkpointing. "
                    "Check that _no_gc_on_backbone properly disabled GC and that "
                    "paligemma.language_model.config.use_cache is True."
                )

            # Verify KV length matches prefix length (sanity check)
            kv_seq_len = get_cache_seq_len(past_key_values, layer_idx=0)
            expected_prefix_len = int(prefix_pad_masks.shape[1])
            assert kv_seq_len == expected_prefix_len, (
                f"KV cache length mismatch: {kv_seq_len} vs {expected_prefix_len}"
            )

            # ---- Knowledge Insulation: detach KV if enabled ----
            # (For KI=ON we built cache under no_grad, but double-detach is harmless.)
            if self.knowledge_insulation:
                past_key_values = _detach_kv_cache(past_key_values)

        # ---- Run action expert forward with the prefix KV ----
        # NOTE: expert suffix forward runs OUTSIDE the no_grad context so we get
        # gradients on expert parameters. prefix_pad_masks is non-leaf but that's
        # fine — it's just used as an attention mask, not as a learnable parameter.
        v_t = self.action_expert.compute_velocity_infer(
            model=self,
            prefix_ctx={
                "prefix_pad_masks": prefix_pad_masks,
                "past_key_values": past_key_values,
            },
            state=state,
            x_t=x_t,
            time=time,
        )

        flow_loss = F.mse_loss(u_t.float(), v_t.float(), reduction="mean")
        expert_loss = self.flow_loss_weight * flow_loss

        return {
            "flow_loss": flow_loss,
            "expert_loss": expert_loss,
        }

    # ------------------------------------------------------------------
    #  Convenience: both phases together (for testing / KI-OFF baseline)
    # ------------------------------------------------------------------

    def compute_all_losses(
        self,
        observation,
        actions,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute all losses in one call (convenience for testing).

        .. warning::
            For KI=OFF baseline training, this builds both graphs together
            so that both backward calls work.  Memory usage is ~2x the
            two-phase approach.

        .. warning::
            For KI=ON training, prefer two separate calls:
            ``compute_backbone_losses`` → backward → step → free graph →
            ``compute_expert_loss`` → backward → step.
            This saves peak memory because the backbone graph is freed
            before computing the expert pass.

        Returns dict with:
            - ``backbone_loss``: scalar, weighted backbone loss (attached)
            - ``flow_loss``: scalar, raw flow MSE (attached)
            - ``expert_loss``: scalar, weighted expert loss (attached)
            - ``ce_loss``: scalar, subtask CE (detached)
            - ``query_mse_loss``: scalar, query MSE (detached)
            - ``total_loss``: scalar, backbone_loss + expert_loss (detached)
        """
        backbone_losses = self.compute_backbone_losses(observation, actions)
        expert_losses = self.compute_expert_loss(observation, actions, noise=noise, time=time)

        total_loss = backbone_losses["backbone_loss"].detach() + expert_losses["expert_loss"].detach()

        return {
            "backbone_loss": backbone_losses["backbone_loss"],
            "flow_loss": expert_losses["flow_loss"],
            "expert_loss": expert_losses["expert_loss"],
            "ce_loss": backbone_losses["ce_loss"],
            "query_mse_loss": backbone_losses["query_mse_loss"],
            "total_loss": total_loss,
        }

    def compute_eval_metrics(
        self,
        observation,
        actions,
        *,
        compute_flow_l1: bool = False,
        num_denoise_steps: int = 10,
        flow_l1_seed: int = 42,
    ) -> dict[str, Tensor]:
        """Compute evaluation/validation metrics (no backward pass).

        Runs one backbone forward pass and one expert forward pass in eval
        mode and returns a dictionary of scalar metrics.  The backbone pass
        is reused for CE loss, subtask accuracy, query MSE, and query L1
        to avoid redundant computation.

        All tensors in the returned dict are detached scalars on the
        same device as the inputs.

        Args:
            observation: observation batch
            actions: ground truth action batch [B, horizon, dim]
            compute_flow_l1: if True, also compute flow_l1 via Euler
                integration (slow path, epoch-end only).  Uses fixed-seed
                noise for determinism.
            num_denoise_steps: number of Euler steps for flow_l1 (default 10)
            flow_l1_seed: seed for fixed noise in flow_l1 computation

        Returns dict with:
            - ``total_loss``: scalar, backbone_loss + expert_loss
            - ``backbone_loss``: scalar, weighted backbone loss
            - ``expert_loss``: scalar, weighted expert loss
            - ``ce_loss``: scalar, subtask CE loss
            - ``query_mse_loss``: scalar, action query MSE loss
            - ``flow_loss``: scalar, raw flow matching MSE
            - ``subtask_accuracy``: scalar, teacher-forced subtask token accuracy
              (argmax of CE logits vs GT, masked by loss_mask)
            - ``query_l1``: scalar, query token action L1 (mean over valid positions)
            - ``flow_mse``: scalar, same as flow_loss (velocity MSE, alias)
            - ``flow_l1``: (only if compute_flow_l1=True) scalar, Euler-integrated
              action L1 vs ground truth (fixed-seed noise, deterministic)
        """
        # ---- Shared backbone forward ----
        # We do one backbone pass and extract all metrics from it.
        # This duplicates some logic from compute_backbone_losses but
        # avoids running the backbone 3x (CE + accuracy + query metrics).
        (
            backbone_loss,
            ce_loss,
            query_mse_loss,
            subtask_accuracy,
            query_l1,
        ) = self._compute_backbone_eval_metrics(observation, actions)

        # ---- Expert forward ----
        ex_losses = self.compute_expert_loss(observation, actions)
        ex_loss = ex_losses["expert_loss"]
        flow_loss = ex_losses["flow_loss"]

        total_loss = backbone_loss.detach() + ex_loss.detach()

        result = {
            "total_loss": total_loss,
            "backbone_loss": backbone_loss.detach(),
            "expert_loss": ex_loss.detach(),
            "ce_loss": ce_loss.detach(),
            "query_mse_loss": query_mse_loss.detach(),
            "flow_loss": flow_loss.detach(),
            "subtask_accuracy": subtask_accuracy.detach(),
            "query_l1": query_l1.detach(),
            "flow_mse": flow_loss.detach(),  # alias for clarity
        }

        # ---- Slow path: Euler integration flow L1 (epoch-end only) ----
        if compute_flow_l1:
            flow_l1 = self._compute_flow_l1_euler(
                observation=observation,
                actions=actions,
                num_steps=num_denoise_steps,
                seed=flow_l1_seed,
            )
            result["flow_l1"] = flow_l1.detach()

        return result

    def _compute_backbone_eval_metrics(self, observation, actions) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Single backbone forward for eval: returns (bb_loss, ce_loss, query_mse, subtask_acc, query_l1).

        All return values are detached scalar tensors except bb_loss which
        has the computation graph attached (for potential gradient use).
        """
        # ---- Preprocess observation ----
        images, img_masks, lang_tokens, lang_masks, _state = self._preprocess_observation(
            observation, train=False
        )

        # ---- Subtask info ----
        subtask_tokens = getattr(observation, "subtask_tokens", None)
        subtask_mask = getattr(observation, "subtask_mask", None)
        subtask_loss_mask = getattr(observation, "subtask_loss_mask", None)

        has_subtask = (
            subtask_tokens is not None
            and subtask_mask is not None
            and subtask_loss_mask is not None
            and subtask_loss_mask.any()
        )

        # ---- Build full prefix ----
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_base_len = prefix_embs.shape[1]

        if has_subtask:
            prefix_embs, prefix_pad_masks, prefix_att_masks = (
                self.action_expert._embed_conditioning_subtask(
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

        # Add query tokens
        query_embs, query_pad_masks, query_att_masks = self._embed_query_tokens(
            batch_size=prefix_embs.shape[0],
            device=prefix_embs.device,
            target_dtype=prefix_embs.dtype,
        )
        full_prefix_embs = torch.cat([prefix_embs, query_embs], dim=1)
        full_prefix_pad_masks = torch.cat([prefix_pad_masks, query_pad_masks], dim=1)
        full_prefix_att_masks = torch.cat([prefix_att_masks, query_att_masks], dim=1)

        full_prefix_att_2d_masks = self.make_att_2d_masks(full_prefix_pad_masks, full_prefix_att_masks)
        full_prefix_position_ids = torch.cumsum(full_prefix_pad_masks, dim=1) - 1
        full_prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(full_prefix_att_2d_masks)

        # ---- Run backbone forward ----
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = resolve_attn_impl()
        (prefix_out, _), _ = self.paligemma_with_expert.forward(
            attention_mask=full_prefix_att_2d_masks_4d,
            position_ids=full_prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[full_prefix_embs, None],
            use_cache=False,
        )

        # ---- CE loss + subtask accuracy ----
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

        # ---- Query MSE + L1 ----
        query_mse_loss = self._compute_query_mse_loss(
            prefix_out=prefix_out,
            prefix_after_subtask_len=prefix_after_subtask_len,
            actions=actions,
            observation=observation,
        )

        query_l1 = self._compute_query_l1_from_hidden(
            prefix_out=prefix_out,
            prefix_after_subtask_len=prefix_after_subtask_len,
            actions=actions,
            observation=observation,
        )

        # ---- Backbone total: weighted sum ----
        backbone_loss = self.beta_text * ce_loss + self.beta_query * query_mse_loss

        return backbone_loss, ce_loss, query_mse_loss, subtask_accuracy, query_l1

    def _compute_subtask_accuracy_from_hidden(
        self,
        *,
        prefix_out: Tensor,
        prefix_base_len: int,
        subtask_tokens: Tensor | None,
        subtask_loss_mask: Tensor | None,
        has_subtask: bool,
    ) -> Tensor:
        """Compute teacher-forced subtask accuracy from backbone hidden states.

        Args:
            prefix_out: [B, seq_len, hidden_dim] backbone output
            prefix_base_len: length of prefix before subtask tokens
            subtask_tokens: [B, subtask_len] ground truth subtask tokens
            subtask_loss_mask: [B, subtask_len] loss mask (1=predict, 0=ignore)
            has_subtask: whether subtask tokens are present

        Returns:
            Scalar accuracy tensor (0.0 if no subtask tokens).
        """
        if not has_subtask:
            return torch.tensor(0.0, device=prefix_out.device)

        subtask_len = subtask_tokens.shape[1]
        subtask_hidden = prefix_out[:, prefix_base_len : prefix_base_len + subtask_len]

        # Project to vocabulary
        embed_weight = self.paligemma_with_expert.paligemma.language_model.embed_tokens.weight
        subtask_hidden = subtask_hidden.to(dtype=embed_weight.dtype)
        text_logits = torch.matmul(subtask_hidden, embed_weight.T)

        # Shift: logits[t] predicts token[t+1]
        shift_logits = text_logits[:, :-1].contiguous()
        shift_targets = subtask_tokens[:, 1:].contiguous().to(dtype=torch.long)
        shift_loss_mask = subtask_loss_mask[:, 1:].contiguous().float()

        # Accuracy
        preds = shift_logits.argmax(dim=-1)
        correct = (preds == shift_targets).float()
        masked_correct = correct * shift_loss_mask
        total_valid = shift_loss_mask.sum().clamp(min=1)
        accuracy = masked_correct.sum() / total_valid

        return accuracy

    def _compute_query_l1_from_hidden(
        self,
        *,
        prefix_out: Tensor,
        prefix_after_subtask_len: int,
        actions: Tensor,
        observation,
    ) -> Tensor:
        """Compute query action L1 from backbone hidden states.

        Args:
            prefix_out: [B, seq_len, hidden_dim] backbone output
            prefix_after_subtask_len: length of prefix after subtask tokens
            actions: [B, action_horizon, action_dim] ground truth actions
            observation: observation with action_is_pad (optional)

        Returns:
            Scalar L1 tensor (mean over valid positions).
        """
        batch_size = prefix_out.shape[0]
        device = prefix_out.device

        query_hidden = prefix_out[:, prefix_after_subtask_len : prefix_after_subtask_len + self.num_query_tokens]
        head_dtype = self.query_action_head.weight.dtype
        query_hidden_aligned = query_hidden.to(dtype=head_dtype)
        pred_actions = self.query_action_head(query_hidden_aligned).float()

        # Target actions (with interpolation if needed)
        action_horizon = actions.shape[1]
        if action_horizon != self.num_query_tokens:
            target_actions = F.interpolate(
                actions.permute(0, 2, 1).float(),
                size=self.num_query_tokens,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)
        else:
            target_actions = actions.float()
        target_actions = target_actions.to(device=device)

        # Mask from action_is_pad
        loss_mask = getattr(observation, "action_is_pad", None)
        if loss_mask is not None:
            valid_mask = (~loss_mask.bool()).float()
            if action_horizon != self.num_query_tokens:
                valid_mask = F.interpolate(
                    valid_mask.unsqueeze(1),
                    size=self.num_query_tokens,
                    mode="nearest",
                ).squeeze(1)
            loss_mask_3d = valid_mask.unsqueeze(-1).expand(-1, -1, self._action_dim)
        else:
            loss_mask_3d = torch.ones(
                batch_size, self.num_query_tokens, self._action_dim,
                dtype=torch.float32, device=device,
            )

        # L1 (mean over valid positions, in fp32)
        l1_per_pos = (pred_actions - target_actions).abs()
        masked_l1 = l1_per_pos * loss_mask_3d
        total_valid = loss_mask_3d.sum().clamp(min=1)
        mean_l1 = masked_l1.sum() / total_valid

        return mean_l1

    def _compute_flow_l1_euler(
        self,
        *,
        observation,
        actions: Tensor,
        num_steps: int = 10,
        seed: int = 42,
    ) -> Tensor:
        """Compute flow L1 via Euler integration of predicted velocity field.

        Starts from fixed-seed noise (for deterministic validation) and
        integrates the predicted velocity field for ``num_steps`` Euler
        steps to arrive at predicted actions, then computes L1 distance
        to ground truth actions.

        Args:
            observation: observation batch
            actions: ground truth actions [B, action_horizon, action_dim]
            num_steps: number of Euler integration steps (default 10)
            seed: random seed for fixed noise generation

        Returns:
            Scalar float tensor: mean L1 over batch and valid positions.
        """
        batch_size = actions.shape[0]
        device = actions.device

        # Fixed-seed noise for deterministic validation
        rng_state = torch.get_rng_state()
        try:
            torch.manual_seed(seed)
            noise = torch.randn_like(actions)
        finally:
            torch.set_rng_state(rng_state)

        # Build prefix KV cache for expert (same as compute_expert_loss)
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=False
        )

        subtask_tokens = getattr(observation, "subtask_tokens", None)
        subtask_mask = getattr(observation, "subtask_mask", None)
        has_subtask = (
            subtask_tokens is not None
            and subtask_mask is not None
            and subtask_mask.any()
        )

        with torch.no_grad():
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )

            if has_subtask:
                prefix_embs, prefix_pad_masks, prefix_att_masks = (
                    self.action_expert._embed_conditioning_subtask(
                        model=self,
                        prefix_embs=prefix_embs,
                        prefix_pad_masks=prefix_pad_masks,
                        prefix_att_masks=prefix_att_masks,
                        subtask_tokens=subtask_tokens,
                        subtask_mask=subtask_mask,
                        causal=True,
                    )
                )

            if not self.truncate_expert_kv:
                query_embs, query_pad_masks, query_att_masks = self._embed_query_tokens(
                    batch_size=batch_size,
                    device=device,
                    target_dtype=prefix_embs.dtype,
                )
                prefix_embs = torch.cat([prefix_embs, query_embs], dim=1)
                prefix_pad_masks = torch.cat([prefix_pad_masks, query_pad_masks], dim=1)
                prefix_att_masks = torch.cat([prefix_att_masks, query_att_masks], dim=1)

            prefix_att_2d_masks = self.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

            self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = resolve_attn_impl()
            with self._no_gc_on_backbone():
                _, past_key_values = self.paligemma_with_expert.forward(
                    attention_mask=prefix_att_2d_masks_4d,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=True,
                )

            if past_key_values is None:
                raise RuntimeError("_compute_flow_l1_euler: past_key_values is None")

            if self.knowledge_insulation:
                past_key_values = _detach_kv_cache(past_key_values)

            prefix_ctx = {
                "prefix_pad_masks": prefix_pad_masks,
                "past_key_values": past_key_values,
            }

            # Euler integration: start from noise (t=1), step to t=0
            dt = -1.0 / num_steps
            x_t = noise
            t = 1.0
            for _ in range(num_steps):
                time_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float32)
                v_t = self.action_expert.compute_velocity_infer(
                    model=self,
                    prefix_ctx=prefix_ctx,
                    state=state,
                    x_t=x_t,
                    time=time_tensor,
                )
                x_t = x_t + dt * v_t
                t += dt

            # x_t now holds predicted actions at t≈0
            pred_actions = x_t.float()
            target_actions = actions.float()

            # Mask by action_is_pad if available
            loss_mask = getattr(observation, "action_is_pad", None)
            action_horizon = actions.shape[1]
            if loss_mask is not None:
                valid_mask = (~loss_mask.bool()).float()
                # If action_horizon != num_query_tokens, we shouldn't interpolate
                # for action L1 — action_horizon matches the GT action shape.
                # But pred_actions from flow has same shape as actions, so no resize needed.
                loss_mask_3d = valid_mask.unsqueeze(-1).expand(-1, -1, self._action_dim)
            else:
                loss_mask_3d = torch.ones(
                    batch_size, action_horizon, self._action_dim,
                    dtype=torch.float32, device=device,
                )

            # L1 (mean over valid positions)
            l1_per_pos = (pred_actions - target_actions).abs()
            masked_l1 = l1_per_pos * loss_mask_3d
            total_valid = loss_mask_3d.sum().clamp(min=1)
            mean_l1 = masked_l1.sum() / total_valid

            return mean_l1

    # ------------------------------------------------------------------
    #  Loss computation helpers
    # ------------------------------------------------------------------

    def _compute_ce_loss(
        self,
        *,
        prefix_out: Tensor,
        prefix_base_len: int,
        subtask_tokens: Tensor | None,
        subtask_loss_mask: Tensor | None,
        has_subtask: bool,
    ) -> Tensor:
        """Compute subtask CE loss from backbone hidden states."""
        if not has_subtask:
            return torch.tensor(0.0, device=prefix_out.device)

        subtask_len = subtask_tokens.shape[1]
        subtask_hidden = prefix_out[:, prefix_base_len : prefix_base_len + subtask_len]

        # Project to vocabulary using the tied embed_tokens weight
        embed_weight = self.paligemma_with_expert.paligemma.language_model.embed_tokens.weight
        subtask_hidden = subtask_hidden.to(dtype=embed_weight.dtype)
        text_logits = torch.matmul(subtask_hidden, embed_weight.T)

        # Shift: logits[t] predicts token[t+1]
        shift_logits = text_logits[:, :-1].contiguous()
        shift_targets = subtask_tokens[:, 1:].contiguous().to(dtype=torch.long)
        shift_loss_mask = subtask_loss_mask[:, 1:].contiguous().float()

        ce_loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            reduction="none",
        ).view(shift_logits.shape[0], -1)

        total_ce = (ce_loss_per_token * shift_loss_mask).sum()
        total_valid = shift_loss_mask.sum().clamp(min=1)
        return total_ce / total_valid

    def _compute_query_mse_loss(
        self,
        *,
        prefix_out: Tensor,
        prefix_after_subtask_len: int,
        actions: Tensor,
        observation,
    ) -> Tensor:
        """Compute MSE loss between query action head predictions and GT actions.

        Mixed-precision safe:
          1. Align query hidden to query_action_head weight dtype
          2. Compute prediction
          3. Cast prediction and target to float32
          4. Compute MSE in float32 for numerical stability

        If ``action_is_pad`` is available on the observation, it is used to
        mask out padded action positions from the loss.
        """
        batch_size = prefix_out.shape[0]
        device = prefix_out.device

        # Hidden states at query token positions
        query_hidden = prefix_out[:, prefix_after_subtask_len : prefix_after_subtask_len + self.num_query_tokens]
        assert query_hidden.shape[1] == self.num_query_tokens, (
            f"Expected {self.num_query_tokens} query positions, got {query_hidden.shape[1]}"
        )
        assert query_hidden.shape[2] == self._vlm_hidden_dim, (
            f"Expected hidden_dim={self._vlm_hidden_dim}, got {query_hidden.shape[2]}"
        )

        # Align to head weight dtype (handles bf16/fp16/fp32 mixed precision)
        head_dtype = self.query_action_head.weight.dtype
        query_hidden_aligned = query_hidden.to(dtype=head_dtype)

        # Project to action space
        pred_actions = self.query_action_head(query_hidden_aligned)
        assert pred_actions.shape == (batch_size, self.num_query_tokens, self._action_dim), (
            f"Expected pred shape [{batch_size}, {self.num_query_tokens}, {self._action_dim}], "
            f"got {pred_actions.shape}"
        )

        # Cast to float32 for MSE computation
        pred_actions_f32 = pred_actions.float()

        # Target actions
        action_horizon = actions.shape[1]
        if action_horizon != self.num_query_tokens:
            # Linear interpolation along time dimension (experimental)
            target_actions = F.interpolate(
                actions.permute(0, 2, 1).float(),  # [B, D, T]
                size=self.num_query_tokens,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)  # [B, num_query_tokens, D]
        else:
            target_actions = actions.float()

        target_actions = target_actions.to(device=device)

        # Build loss mask from action_is_pad if available
        loss_mask = getattr(observation, "action_is_pad", None)
        if loss_mask is not None:
            # action_is_pad: [B, action_horizon], bool (True = padded)
            valid_mask = (~loss_mask.bool()).float()
            if action_horizon != self.num_query_tokens:
                valid_mask = F.interpolate(
                    valid_mask.unsqueeze(1),  # [B, 1, T]
                    size=self.num_query_tokens,
                    mode="nearest",
                ).squeeze(1)  # [B, num_query_tokens]
            # Expand to action_dim
            loss_mask_3d = valid_mask.unsqueeze(-1).expand(-1, -1, self._action_dim)
        else:
            loss_mask_3d = torch.ones(
                batch_size, self.num_query_tokens, self._action_dim,
                dtype=torch.float32, device=device,
            )

        # Masked MSE loss (mean over valid positions, computed in fp32)
        sq_error = (pred_actions_f32 - target_actions) ** 2
        total_sq_error = (sq_error * loss_mask_3d).sum()
        total_valid = loss_mask_3d.sum().clamp(min=1.0)
        mse_loss = total_sq_error / total_valid

        return mse_loss

    # ------------------------------------------------------------------
    #  Query token embedding helper
    # ------------------------------------------------------------------

    def _embed_query_tokens(
        self,
        *,
        batch_size: int,
        device: torch.device,
        target_dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embed the learned action query tokens for the backbone sequence.

        Query tokens are pure learned embeddings — they contain NO ground-truth
        action information.  The backbone must learn to represent actions at
        these positions via the query MSE loss.

        Attention mask semantics:
          - Query tokens use **bidirectional** self-attention within the query
            block.  Since they contain no GT info, bidirectional is safe and
            beneficial (like DETR object queries).
          - Query positions can attend all earlier prefix tokens (images,
            prompt, subtask).
          - Earlier prefix tokens CANNOT attend query positions (block-causal).

        Mixed-precision safe: query_embeddings are cast to the projection weight
        dtype, then output is cast to target_dtype (matching prefix embeddings).

        Returns (query_embs, query_pad_masks, query_att_masks):
            - query_embs: [B, num_query_tokens, vlm_hidden_dim]
            - query_pad_masks: [B, num_query_tokens], all True
            - query_att_masks: [B, num_query_tokens], first entry opens a new
              attention block, the rest join it bidirectionally
        """
        # Step 1: query embeddings to query_to_vlm_proj weight dtype
        if isinstance(self.query_to_vlm_proj, nn.Identity):
            proj_dtype = target_dtype
        else:
            proj_dtype = self.query_to_vlm_proj.weight.dtype

        query_base = self.query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        query_base = query_base.to(dtype=proj_dtype, device=device)

        # Step 2: project to VLM hidden dim (identity if same dim)
        query_embs = self.query_to_vlm_proj(query_base)

        # Step 3: scale by sqrt(dim) to match language token embedding scaling
        # (PaliGemma multiplies text embeddings by sqrt(hidden_dim))
        query_embs = query_embs * math.sqrt(self._vlm_hidden_dim)

        # Step 4: cast to target dtype (matching prefix embeddings)
        query_embs = query_embs.to(dtype=target_dtype)

        query_pad_masks = torch.ones(
            batch_size, self.num_query_tokens, dtype=torch.bool, device=device
        )

        # The query tokens form a single bidirectional block that the prefix
        # must not be able to see.
        #
        # make_att_2d_masks compares cumulative sums: writing b(t) =
        # cumsum(att_masks)[t], row i attends column j iff b(j) <= b(i) and
        # both are valid. A token whose att_mask is 1 therefore opens a new
        # block that preceding blocks cannot see, while a token whose att_mask
        # is 0 joins the preceding block bidirectionally. Marking only the
        # FIRST query token keeps every query mutually visible while making
        # the whole group invisible to every prefix position.
        #
        # Leaving all entries at 0 (the earlier behaviour) gave the query block
        # the same b(t) as the LAST PHYSICAL subtask slot, letting that row
        # attend forward into the queries. That row is normally padding (real
        # subtasks are far shorter than subtask_max_len) and is masked out by
        # pad_2d_masks, and even when the segment is exactly full the row is
        # dropped by the next-token shift in _compute_ce_loss, so the observed
        # numerical impact was ~0. It was still a block-causal violation, and
        # it becomes a real one that silently contaminates every supervised CE
        # row if the subtask segment is ever conditioned bidirectionally --
        # see the guard in SubtaskActionExpert._embed_conditioning_subtask.
        #
        # This matches upstream openpi, whose action tokens use the identical
        # pattern: ar_mask += [True] + ([False] * (action_horizon - 1)).
        query_att_masks = torch.zeros(
            batch_size, self.num_query_tokens, dtype=torch.bool, device=device
        )
        query_att_masks[:, 0] = True

        return query_embs, query_pad_masks, query_att_masks

    # ------------------------------------------------------------------
    #  Standard forward (compatibility)
    # ------------------------------------------------------------------

    def forward(
        self,
        observation,
        actions,
        noise=None,
        time=None,
        *,
        phase: str = "all",
    ) -> dict[str, Tensor]:
        """Dispatch a wrapper-safe training phase.

        Calling phase-specific work through ``forward`` is required when the
        model is wrapped by DDP, Accelerate, or DeepSpeed: wrapper reducer and
        autocast hooks only run for normal module calls.  The direct
        :meth:`compute_backbone_losses` and :meth:`compute_expert_loss` APIs are
        retained for unwrapped models, unit tests, and compatibility.

        Args:
            observation: input observation batch.
            actions: ground-truth action batch.
            noise: optional flow-matching noise for the expert phase.
            time: optional flow-matching time for the expert phase.
            phase: ``"backbone"`` for CE + query MSE, ``"expert"`` for
                flow matching, or ``"all"`` for the legacy combined forward.

        Returns:
            A phase-specific loss dictionary.  The default ``"all"`` result
            includes the legacy combined ``loss`` key.

        Raises:
            ValueError: if ``phase`` is not one of the supported values.
        """
        if phase == "backbone":
            return self.compute_backbone_losses(observation, actions)
        if phase == "expert":
            return self.compute_expert_loss(
                observation,
                actions,
                noise=noise,
                time=time,
            )
        if phase != "all":
            raise ValueError(
                f"Unsupported training phase {phase!r}; expected 'backbone', 'expert', or 'all'."
            )

        losses = self.compute_all_losses(observation, actions, noise=noise, time=time)
        combined_loss = losses["backbone_loss"] + losses["expert_loss"]

        return {
            "loss": combined_loss,
            "flow_loss": losses["flow_loss"].detach(),
            "ce_loss": losses["ce_loss"].detach(),
            "backbone_loss": losses["backbone_loss"],
            "query_mse_loss": losses["query_mse_loss"].detach(),
            "expert_loss": losses["expert_loss"],
            "total_loss": losses["total_loss"],
        }

    # ------------------------------------------------------------------
    #  Inference methods (delegate to parent class)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_actions_hierarchical(
        self,
        device,
        observation,
        noise=None,
        num_steps=10,
        *,
        max_subtask_tokens: int = 64,
        temperature: float = 0.0,
    ) -> Tensor:
        """Hierarchical inference: predict subtask → generate actions.

        Query tokens are training-only — not used during inference.
        Action generation is identical to PI05SubtaskPytorch.
        """
        return super().sample_actions_hierarchical(
            device,
            observation,
            noise=noise,
            num_steps=num_steps,
            max_subtask_tokens=max_subtask_tokens,
            temperature=temperature,
        )

    @torch.no_grad()
    def predict_subtask_tokens(
        self,
        observation,
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
        min_tokens: int = 1,
    ) -> torch.Tensor:
        """Predict subtask tokens from observation (same as parent class)."""
        return super().predict_subtask_tokens(
            observation,
            max_tokens=max_tokens,
            temperature=temperature,
            min_tokens=min_tokens,
        )

    # ------------------------------------------------------------------
    #  Parameter grouping helpers (for dual optimizer setup)
    # ------------------------------------------------------------------

    def get_backbone_param_names(self) -> set[str]:
        """Return names of all backbone-side parameters.

        Includes: vision encoder, PaliGemma language model, query embeddings,
        query action head, query_to_vlm_proj (if not Identity).
        """
        names = set()
        for name, _ in self.named_parameters():
            if name.startswith("paligemma_with_expert.paligemma."):
                names.add(name)
            elif name == "query_embeddings":
                names.add(name)
            elif name.startswith("query_to_vlm_proj."):
                if not isinstance(self.query_to_vlm_proj, nn.Identity):
                    names.add(name)
            elif name.startswith("query_action_head."):
                names.add(name)
        return names

    def get_expert_param_names(self) -> set[str]:
        """Return names of all expert-side parameters.

        Includes: gemma_expert transformer, action_in_proj, action_out_proj,
        time_mlp_in/out.
        """
        names = set()
        for name, _ in self.named_parameters():
            if name.startswith("paligemma_with_expert.gemma_expert."):
                names.add(name)
            elif name.startswith("action_in_proj."):
                names.add(name)
            elif name.startswith("action_out_proj."):
                names.add(name)
            elif name.startswith("time_mlp_in."):
                names.add(name)
            elif name.startswith("time_mlp_out."):
                names.add(name)
        return names

    def get_backbone_params(self) -> list[nn.Parameter]:
        """Return all backbone-side parameters (see get_backbone_param_names)."""
        names = self.get_backbone_param_names()
        return [p for n, p in self.named_parameters() if n in names]

    def get_expert_params(self) -> list[nn.Parameter]:
        """Return all expert-side parameters (see get_expert_param_names)."""
        names = self.get_expert_param_names()
        return [p for n, p in self.named_parameters() if n in names]
