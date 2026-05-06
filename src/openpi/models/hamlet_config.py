import dataclasses

from openpi.models.pi0_config import Pi0Config


@dataclasses.dataclass(frozen=True)
class HamletConfig(Pi0Config):
    """Pi0.5-compatible config for the HAMLET memory baseline."""

    pi05: bool = True
    hamlet_num_moment_tokens: int = 4
    hamlet_history_length: int = 4
    hamlet_num_layers: int = 2
    hamlet_num_heads: int = 8
    hamlet_dropout: float = 0.0
    hamlet_gate_init: float = 0.0
    hamlet_project_to_action_expert: bool = True
