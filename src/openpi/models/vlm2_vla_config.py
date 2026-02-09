import dataclasses
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import gemma as _gemma
from openpi.models import model as _model
import openpi.shared.array_typing as at

if TYPE_CHECKING:
    from openpi.models.model import BaseModel


@dataclasses.dataclass(frozen=True)
class VLM2VLAConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore

    pi05: bool = True
    discrete_state_input: bool = None  # type: ignore

    vggt_pretrained: str | None = "checkpoints/openpi_comet/vggt/model.pt"
    vggt_load_strict: bool = False
    vggt_enable_track: bool = False

    freeze_vggt_backbone: bool = True
    freeze_image_encoder: bool = True

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 256)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", True)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        raise NotImplementedError("VLM2VLAConfig is intended for PyTorch VLM2 training.")

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
                pcd_xyz=jax.ShapeDtypeStruct([batch_size, 16, 2025, 3], jnp.float32),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec
