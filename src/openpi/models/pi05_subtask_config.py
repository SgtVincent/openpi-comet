"""Configuration for the PI05 subtask model.

This config extends the existing PI0.5 PyTorch path to support the combined loss:
    L = CE(text_logits, subtask_tokens) + alpha * flow_matching_loss

The model produces both:
1. Text logits for subtask prediction (supervised by skill_description from B1K)
2. Continuous action output via flow matching action expert
"""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils


PALIGEMMA_VOCAB_SIZE = 257_152


@dataclasses.dataclass(frozen=True)
class Pi05SubtaskConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 512

    # Subtask token sequence max length.
    subtask_max_len: int = 128

    # Combined loss trade-off parameter (alpha in Eq. 1).
    alpha: float = 10.0

    # Pi05 settings.
    pi05: bool = True
    discrete_state_input: bool = True

    # Vocab size for text logits head.
    vocab_size: int = PALIGEMMA_VOCAB_SIZE

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05_SUBTASK

    @override
    def create(self, rng: at.KeyArrayLike):
        raise NotImplementedError(
            "PI05_SUBTASK JAX model is not implemented. Use the PyTorch model via train_pytorch.py."
        )

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                subtask_tokens=jax.ShapeDtypeStruct([batch_size, self.subtask_max_len], jnp.int32),
                subtask_mask=jax.ShapeDtypeStruct([batch_size, self.subtask_max_len], bool),
                subtask_loss_mask=jax.ShapeDtypeStruct([batch_size, self.subtask_max_len], jnp.bool_),
                subtask_ar_mask=jax.ShapeDtypeStruct([batch_size, self.subtask_max_len], jnp.int32),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        filters = []
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")

        if "lora" in self.paligemma_variant:
            filters.append(gemma_params_filter)
            if "lora" not in self.action_expert_variant:
                filters.append(nnx.Not(action_expert_params_filter))
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))
            return nnx.All(*filters)

        return nnx.Nothing