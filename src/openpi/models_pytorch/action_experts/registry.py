from typing import Any

from openpi.models_pytorch.action_experts.base import ActionExpert
from openpi.models_pytorch.action_experts.gemma_token_expert import GemmaTokenExpert


def create_action_expert(
    name: str,
    *,
    vlm_width: int,
    action_dim: int,
    pi05: bool,
    kwargs: dict[str, Any] | None = None,
) -> ActionExpert:
    kwargs = {} if kwargs is None else dict(kwargs)
    if name == "gemma_token":
        return GemmaTokenExpert()
    if name == "subtask":
        from openpi.models_pytorch.action_experts.subtask_expert import SubtaskActionExpert

        return SubtaskActionExpert()
    if name == "il_moe_velocity":
        from openpi.models_pytorch.action_experts.il_moe_velocity_expert import IllibMoeVelocityFieldExpert

        return IllibMoeVelocityFieldExpert(vlm_width=vlm_width, state_dim=action_dim, action_dim=action_dim, **kwargs)
    raise ValueError(f"Unknown action expert: {name}")

