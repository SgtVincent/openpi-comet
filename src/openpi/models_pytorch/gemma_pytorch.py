import math
from typing import Literal

import pytest
import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma
from transformers.models.siglip import modeling_siglip


def _sharded_lecun_normal_(tensor: torch.Tensor, fan_in: int) -> None:
    if tensor.numel() == 0:
        return
    variance = 1.0 / max(1, fan_in)
    std = math.sqrt(variance) / 0.87962566103423978
    modeling_siglip.trunc_normal_tf_(tensor, std=std)


def _sharded_default_flax_embed_init_(tensor: torch.Tensor, fan_in: int) -> None:
    if tensor.numel() == 0:
        return
    std = math.sqrt(1.0 / max(1, fan_in))
    with torch.no_grad():
        tensor.normal_(std=std)


def _sharded_xavier_uniform_(tensor: torch.Tensor, fan_in: int, fan_out: int) -> None:
    if tensor.numel() == 0:
        return
    bound = math.sqrt(6.0 / max(1, fan_in + fan_out))
    with torch.no_grad():
        tensor.uniform_(-bound, bound)


def _fan_in_from_module(module: nn.Module) -> int:
    if isinstance(module, nn.Embedding):
        return int(module.embedding_dim)
    if isinstance(module, nn.Linear):
        return int(module.in_features)
    if isinstance(module, nn.Conv2d):
        kernel_h, kernel_w = module.kernel_size
        return int((module.in_channels // module.groups) * kernel_h * kernel_w)
    raise TypeError(f"Unsupported module type for sharded init: {type(module)}")


def _patch_siglip_init_for_zero3() -> None:
    current = modeling_siglip.SiglipPreTrainedModel._init_weights
    if getattr(current, "_openpi_zero3_safe", False):
        return

    original_init_weights = current

    def _patched_init_weights(self, module):
        if isinstance(module, nn.Embedding) and module.weight.ndim < 2:
            _sharded_default_flax_embed_init_(module.weight, _fan_in_from_module(module))
            return
        if isinstance(module, modeling_siglip.SiglipAttention) and module.q_proj.weight.ndim < 2:
            embed_dim = int(module.q_proj.in_features)
            proj_dim = int(module.q_proj.out_features)
            for proj in (module.q_proj, module.k_proj, module.v_proj, module.out_proj):
                _sharded_xavier_uniform_(proj.weight, embed_dim, proj_dim)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)
            return
        if isinstance(module, modeling_siglip.SiglipMLP) and module.fc1.weight.ndim < 2:
            _sharded_xavier_uniform_(module.fc1.weight, module.fc1.in_features, module.fc1.out_features)
            _sharded_xavier_uniform_(module.fc2.weight, module.fc2.in_features, module.fc2.out_features)
            if module.fc1.bias is not None:
                nn.init.normal_(module.fc1.bias, std=1e-6)
            if module.fc2.bias is not None:
                nn.init.normal_(module.fc2.bias, std=1e-6)
            return
        if isinstance(module, (nn.Linear, nn.Conv2d)) and module.weight.ndim < 2:
            _sharded_lecun_normal_(module.weight, _fan_in_from_module(module))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            return
        return original_init_weights(self, module)

    _patched_init_weights._openpi_zero3_safe = True  # type: ignore[attr-defined]
    modeling_siglip.SiglipPreTrainedModel._init_weights = _patched_init_weights


def _patch_gemma_init_for_zero3() -> None:
    current = modeling_gemma.GemmaPreTrainedModel._init_weights
    if getattr(current, "_openpi_zero3_safe", False):
        return

    original_init_weights = current

    def _patched_init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear) and module.weight.ndim < 2:
            if module.weight.numel() > 0:
                with torch.no_grad():
                    module.weight.normal_(mean=0.0, std=std)
            if module.bias is not None and module.bias.numel() > 0:
                module.bias.data.zero_()
            return
        if isinstance(module, nn.Embedding) and module.weight.ndim < 2:
            if module.weight.numel() > 0:
                with torch.no_grad():
                    module.weight.normal_(mean=0.0, std=std)
                if module.padding_idx is not None and 0 <= module.padding_idx < module.weight.shape[0]:
                    module.weight.data[module.padding_idx].zero_()
            return
        return original_init_weights(self, module)

    _patched_init_weights._openpi_zero3_safe = True  # type: ignore[attr-defined]
    modeling_gemma.GemmaPreTrainedModel._init_weights = _patched_init_weights


def _patch_gemma_rmsnorm_repr_for_zero3() -> None:
    current = modeling_gemma.GemmaRMSNorm.extra_repr
    if getattr(current, "_openpi_zero3_safe", False):
        return

    def _patched_extra_repr(self):
        weight = getattr(self, "weight", None)
        shape_repr = tuple(weight.shape) if weight is not None else ("partitioned",)
        return f"{shape_repr}, eps={self.eps}"

    _patched_extra_repr._openpi_zero3_safe = True  # type: ignore[attr-defined]
    modeling_gemma.GemmaRMSNorm.extra_repr = _patched_extra_repr


_patch_siglip_init_for_zero3()
_patch_gemma_init_for_zero3()
_patch_gemma_rmsnorm_repr_for_zero3()


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float16":
            self.to(dtype=torch.float16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # Attention computation
                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )
                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Match the downstream MLP compute dtype to avoid extra implicit casts.
                    target_dtype = layer.mlp.up_proj.weight.dtype
                    if out_emb.dtype != target_dtype:
                        out_emb = out_emb.to(dtype=target_dtype)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
