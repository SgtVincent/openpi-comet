"""PI05 subtask PyTorch model."""

import dataclasses
import logging
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

import openpi.shared.download as download
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

import sentencepiece

logger = logging.getLogger("openpi")


class PI05SubtaskPytorch(PI0Pytorch):
    """PI05 subtask model with combined text CE + flow matching loss.

    Inherits from PI0Pytorch and adds:
    - Subtask token embedding and processing in the VLM backbone
    - Text logits computation via VLM's embedding matrix (tied weights)
    - Combined loss: ce_loss + alpha * flow_matching_loss
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
            action_expert_name=action_expert_name,
            action_expert_kwargs=action_expert_kwargs,
        )
        self.alpha = alpha
        self._text_tokenizer: sentencepiece.SentencePieceProcessor | None = None
        self._last_predicted_subtasks: list[str] = []

    def _load_text_tokenizer(self) -> sentencepiece.SentencePieceProcessor:
        if self._text_tokenizer is None:
            path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
            with path.open("rb") as f:
                self._text_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        return self._text_tokenizer

    def _eos_token_id(self) -> int:
        return self._load_text_tokenizer().eos_id()

    def _has_subtask_conditioning(self, observation) -> bool:
        subtask_mask = getattr(observation, "subtask_mask", None)
        return subtask_mask is not None and bool(torch.any(subtask_mask).item())

    def decode_subtask_tokens(self, token_batch: torch.Tensor) -> list[str]:
        sp = self._load_text_tokenizer()
        bos_token = sp.bos_id()
        eos_token = self._eos_token_id()
        results = []
        for row in token_batch.detach().cpu().tolist():
            tokens = []
            for token in row:
                if token == 0:
                    continue
                if token == bos_token:
                    continue
                if token == eos_token:
                    break
                tokens.append(int(token))
            results.append(sp.decode(tokens))
        return results

    def build_hierarchical_observation(self, observation, subtask_tokens: torch.Tensor):
        batch_size = subtask_tokens.shape[0]
        max_len = getattr(self.config, "subtask_max_len", subtask_tokens.shape[1])
        device = subtask_tokens.device

        padded_tokens = torch.zeros(batch_size, max_len, dtype=subtask_tokens.dtype, device=device)
        padded_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
        padded_loss_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
        padded_ar_mask = torch.zeros(batch_size, max_len, dtype=torch.int32, device=device)

        clipped = subtask_tokens[:, :max_len]
        clipped_len = clipped.shape[1]
        padded_tokens[:, :clipped_len] = clipped
        padded_mask[:, :clipped_len] = clipped != 0

        # During action generation the predicted subtask is fixed context, not an AR target.
        return dataclasses.replace(
            observation,
            subtask_tokens=padded_tokens,
            subtask_mask=padded_mask,
            subtask_loss_mask=padded_loss_mask,
            subtask_ar_mask=padded_ar_mask,
        )

    def forward(self, observation, actions, noise=None, time=None) -> dict[str, Tensor]:
        """Compute combined loss: CE(subtask) + alpha * flow_matching(actions).

        Returns a dict with:
        - 'loss': combined scalar loss (for backward)
        - 'flow_loss': flow matching MSE loss
        - 'ce_loss': text cross-entropy loss
        - 'total_loss': same as 'loss' for logging
        """
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=True
        )

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Get subtask tokens from observation
        subtask_tokens = getattr(observation, "subtask_tokens", None)
        subtask_mask = getattr(observation, "subtask_mask", None)
        subtask_ar_mask = getattr(observation, "subtask_ar_mask", None)
        subtask_loss_mask = getattr(observation, "subtask_loss_mask", None)

        has_subtask = (
            subtask_tokens is not None
            and subtask_mask is not None
            and subtask_loss_mask is not None
            and subtask_loss_mask.any()
        )

        if has_subtask:
            # Use the subtask expert for the combined forward pass.
            result = self.action_expert.compute_subtask_loss_train(
                model=self,
                images=images,
                img_masks=img_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                state=state,
                x_t=x_t,
                time=time,
                subtask_tokens=subtask_tokens,
                subtask_mask=subtask_mask,
                subtask_ar_mask=subtask_ar_mask,
                subtask_loss_mask=subtask_loss_mask,
            )
            v_t = result["v_t"]
            ce_loss = result["ce_loss"].mean()

            # Flow matching loss
            flow_loss = F.mse_loss(u_t, v_t, reduction="mean")

            # Combined loss: CE + alpha * flow_matching (Equation 1)
            combined_loss = ce_loss + self.alpha * flow_loss

            return {
                "loss": combined_loss,
                "flow_loss": flow_loss.detach(),
                "ce_loss": ce_loss.detach(),
            }
        else:
            # Fallback: flow matching only (no subtask supervision)
            v_t = self.action_expert.compute_velocity_train(
                model=self,
                images=images,
                img_masks=img_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                state=state,
                x_t=x_t,
                time=time,
            )
            flow_loss = F.mse_loss(u_t, v_t, reduction="mean")

            return {
                "loss": flow_loss,
                "flow_loss": flow_loss.detach(),
                "ce_loss": torch.tensor(0.0, device=flow_loss.device),
            }

    @torch.no_grad()
    def _sample_actions_with_conditioning(self, device, observation, noise=None, num_steps=10) -> Tensor:
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        prefix_ctx = self.action_expert.encode_prefix(
            model=self,
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            subtask_tokens=getattr(observation, "subtask_tokens", None),
            subtask_mask=getattr(observation, "subtask_mask", None),
        )

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.action_expert.compute_velocity_infer(
                model=self,
                prefix_ctx=prefix_ctx,
                state=state,
                x_t=x_t,
                time=expanded_time,
            )
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    @torch.no_grad()
    def sample_actions(
        self,
        device,
        observation,
        noise=None,
        num_steps=10,
        *,
        max_subtask_tokens: int = 64,
        temperature: float = 0.0,
    ) -> Tensor:
        if self._has_subtask_conditioning(observation):
            return self._sample_actions_with_conditioning(device, observation, noise=noise, num_steps=num_steps)
        return self.sample_actions_hierarchical(
            device,
            observation,
            noise=noise,
            num_steps=num_steps,
            max_subtask_tokens=max_subtask_tokens,
            temperature=temperature,
        )

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
        generated_tokens = self.predict_subtask_tokens(
            observation,
            max_tokens=max_subtask_tokens,
            temperature=temperature,
        )
        self._last_predicted_subtasks = self.decode_subtask_tokens(generated_tokens)
        conditioned_observation = self.build_hierarchical_observation(observation, generated_tokens)
        return self._sample_actions_with_conditioning(device, conditioned_observation, noise=noise, num_steps=num_steps)

    @torch.no_grad()
    def predict_subtask_tokens(
        self,
        observation,
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        images, img_masks, lang_tokens, lang_masks, _ = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = self.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        prefix_out = self.paligemma_with_expert.paligemma.language_model.forward(
            inputs_embeds=prefix_embs,
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            use_cache=True,
        )
        prefix_hidden = prefix_out.last_hidden_state
        past_kv = prefix_out.past_key_values

        batch_size = prefix_embs.shape[0]
        device = prefix_embs.device
        embed_weight = self.paligemma_with_expert.paligemma.language_model.embed_tokens.weight
        emb_dim = embed_weight.shape[1]

        seq_indices = torch.arange(prefix_hidden.shape[1], device=device).unsqueeze(0)
        last_pos = torch.max(
            torch.where(prefix_pad_masks, seq_indices, torch.full_like(seq_indices, -1)),
            dim=1,
        ).values
        last_hidden = prefix_hidden[torch.arange(batch_size, device=device), last_pos][:, None, :]
        logits = torch.matmul(last_hidden.to(embed_weight.dtype), embed_weight.T)

        eos_token = self._eos_token_id()
        generated_tokens = []
        next_pos = prefix_pad_masks.sum(dim=-1).to(torch.int64)

        for step in range(max_tokens):
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)

            generated_tokens.append(next_token)
            if torch.all(next_token == eos_token):
                break

            token_emb = self.paligemma_with_expert.embed_language_tokens(next_token) * math.sqrt(emb_dim)
            gen_mask = torch.ones(batch_size, step + 1, dtype=torch.bool, device=device)
            full_mask = torch.cat([prefix_pad_masks, gen_mask], dim=1)
            full_mask_4d = self._prepare_attention_masks_4d(full_mask[:, None, :])

            out = self.paligemma_with_expert.paligemma.language_model.forward(
                inputs_embeds=token_emb,
                attention_mask=full_mask_4d,
                position_ids=next_pos[:, None],
                past_key_values=past_kv,
                use_cache=True,
            )
            last_hidden = out.last_hidden_state
            past_kv = out.past_key_values
            logits = torch.matmul(last_hidden.to(embed_weight.dtype), embed_weight.T)
            next_pos = next_pos + 1

        if not generated_tokens:
            return torch.zeros(batch_size, 0, dtype=torch.int32, device=device)
        return torch.cat(generated_tokens, dim=1).to(dtype=torch.int32)

    @torch.no_grad()
    def predict_subtask(self, observation, *, max_tokens: int = 64, temperature: float = 0.0) -> list[str]:
        generated_tokens = self.predict_subtask_tokens(
            observation,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        results = self.decode_subtask_tokens(generated_tokens)
        self._last_predicted_subtasks = results
        return results
