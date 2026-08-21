"""Config for pi0.5-KI Variant A: FAST discrete action tokens + cross-entropy."""

from __future__ import annotations

import dataclasses

from openpi.models.pi05_ki_joint_query_config import Pi05KIJointQueryConfig


@dataclasses.dataclass(frozen=True)
class Pi05KIJointFastConfig(Pi05KIJointQueryConfig):
    """Paper-accurate Knowledge Insulation variant.

    Inherits every shared field from the query-MSE variant so an A/B comparison
    changes only the backbone action objective.

    Differences from :class:`Pi05KIJointQueryConfig`:
      - the backbone action target is discrete FAST action tokens supervised by
        next-token cross-entropy, weighted by ``beta_action`` instead of
        ``beta_query``;
      - ``action_token_max_len`` sizes the fixed action-token segment;
      - ``num_query_tokens`` / ``query_emb_dim`` are inherited but UNUSED, since
        this variant creates no learned query parameters.
    """

    # ===== Loss weights =====
    # Backbone action cross-entropy weight. Kept separate from beta_query so the
    # two variants can be tuned independently; the units differ (nats vs MSE).
    beta_action: float = 1.0

    # ===== Action-token segment =====
    # This generic default is intentionally conservative. Formal configurations
    # must size it from their observed data contract (the V100 formal FAST config
    # uses 96 for an observed maximum of 73). Any overflow fails instead of
    # silently truncating and corrupting the target.
    action_token_max_len: int = 64

    # ===== Architecture flags =====
    # Keep the shared flag True so existing pi05-KI code paths still recognize
    # this as a joint KI model, and add a specific flag for the discrete variant.
    pi05_ki_joint_fast: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.action_token_max_len < 4:
            raise ValueError(
                f"action_token_max_len must be >= 4 (BOS + at least one action token + EOS + "
                f"one shift row), got {self.action_token_max_len}"
            )
        if not self.truncate_expert_kv:
            # The FAST action tokens are teacher-forced ground truth, so the
            # expert must not attend to them. Reject at config time rather than
            # letting the model raise after a distributed launch.
            raise ValueError(
                "Pi05KIJointFastConfig requires truncate_expert_kv=True: the FAST action "
                "tokens are ground truth, so exposing them to the flow expert leaks the "
                "target during training while they are absent at inference."
            )
