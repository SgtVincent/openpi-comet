"""π0.5-KI joint query Training Config (query-MSE variant).

Lightweight config dataclass for the π0.5-KI joint query variant model.
Extends the existing Pi05SubtaskConfig pattern with π0.5-KI joint query-specific fields.

Note: this is a standalone config module — it does NOT modify the core
``ModelType`` enum or other shared config infrastructure.  The model is
instantiated directly via ``PI05KIJointQueryPytorch(config)``.
"""

from __future__ import annotations

import dataclasses

from openpi.models.pi05_subtask_config import Pi05SubtaskConfig


@dataclasses.dataclass(frozen=True)
class Pi05KIJointQueryConfig(Pi05SubtaskConfig):
    """Config for π0.5-KI joint query training — query-MSE variant (query tokens + MSE + KI).

    Extends :class:`Pi05SubtaskConfig` with π0.5-KI joint query-specific fields.

    Architecture variant:
      - Backbone action supervision: learned query tokens + MSE (query-MSE variant)
      - Expert loss: flow matching MSE
      - Knowledge Insulation: configurable via ``knowledge_insulation`` flag

    New fields vs Pi05SubtaskConfig:
      - knowledge_insulation: if True, flow loss grads do NOT reach backbone
      - beta_text: subtask CE loss weight (0 = core paper, no subtask)
      - beta_query: action query MSE loss weight
      - num_query_tokens: number of learned query tokens (default = action_horizon)
      - query_emb_dim: query embedding dim (None = VLM hidden dim)
      - truncate_expert_kv: if True, expert sees no query tokens in prefix
      - flow_loss_weight: expert loss weight (alpha)
      - pi05_ki_joint_query: flag indicating this is a pi0.5-KI query variant config
    """

    # ===== Knowledge Insulation =====
    knowledge_insulation: bool = True

    # ===== Loss weights =====
    beta_text: float = 1.0        # subtask CE weight (our extension)
    beta_query: float = 1.0       # query MSE weight
    flow_loss_weight: float = 10.0  # alpha / expert loss weight

    # ===== Action query tokens (query-MSE variant) =====
    num_query_tokens: int = 0     # 0 = use action_horizon
    query_emb_dim: int | None = None  # None = use VLM hidden dim

    # ===== KV truncation =====
    truncate_expert_kv: bool = True  # must be True for correctness

    # ===== Architecture flag =====
    pi05_ki_joint_query: bool = True

    def __post_init__(self):
        # Default num_query_tokens to action_horizon
        if self.num_query_tokens == 0:
            object.__setattr__(self, "num_query_tokens", self.action_horizon)

    @property
    def model_type(self):
        # We don't add a new core ModelType to avoid touching shared code.
        # Training code should use config.pi05_ki_joint_query to detect this variant.
        from openpi.models.model import ModelType
        return ModelType.PI05_SUBTASK  # closest existing type


def create_pi05_ki_joint_query(config: Pi05KIJointQueryConfig, **kwargs):
    """Factory: create a PI05KIJointQueryPytorch model from a config.

    Args:
        config: Pi05KIJointQueryConfig instance
        **kwargs: additional kwargs passed to PI05KIJointQueryPytorch.__init__

    Returns:
        PI05KIJointQueryPytorch instance
    """
    from openpi.models_pytorch.pi05_ki_joint_query import PI05KIJointQueryPytorch

    model = PI05KIJointQueryPytorch(
        config,
        alpha=config.alpha,
        action_expert_name="subtask",
        **kwargs,
    )
    return model
