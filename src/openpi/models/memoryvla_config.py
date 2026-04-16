import dataclasses
from typing import Literal

from openpi.models.pi0_config import Pi0Config


@dataclasses.dataclass(frozen=True)
class MemoryVLAConfig(Pi0Config):
    """Pi0.5-compatible config for the MemoryVLA-inspired external memory baseline."""

    pi05: bool = True
    memoryvla_bank_capacity: int = 16
    memoryvla_feature_dim: int = 2048
    memoryvla_num_heads: int = 8
    memoryvla_num_layers: int = 2
    memoryvla_dropout: float = 0.0
    memoryvla_similarity_threshold: float = 0.7
    memoryvla_gate_init: float = 0.0
    memoryvla_use_cognitive_stream: bool = False
    memoryvla_cognitive_pool: Literal["eos", "mean"] = "eos"
    memoryvla_num_summary_tokens: int = 1
