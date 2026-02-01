"""VLM2: Vision-Language Model with Memory for Spatial Reasoning.

This module implements the View-Consistent 3D-Aware Representation and
Dual-Memory Module from the VLM2 paper, integrated with Pi-0.5 architecture.

Paper: "Vision-Language Memory for Spatial Reasoning" (arXiv:2511.20644)
"""

from openpi.models_pytorch.vlm2.view_consistent_3d import (
    Adaptive3DPositionInjection,
    ViewpointAwareGeometryAlignment,
    SemanticGeometricFusion,
    ViewConsistent3DRepresentation,
)
from openpi.models_pytorch.vlm2.dual_memory import (
    WorkingMemory,
    EpisodicMemory,
    DualMemoryModule,
)
from openpi.models_pytorch.vlm2.vlm2_model import VLM2WithPi05

__all__ = [
    # View-Consistent 3D-Aware Representation
    "Adaptive3DPositionInjection",
    "ViewpointAwareGeometryAlignment",
    "SemanticGeometricFusion",
    "ViewConsistent3DRepresentation",
    # Dual-Memory Module
    "WorkingMemory",
    "EpisodicMemory",
    "DualMemoryModule",
    # Main Model
    "VLM2WithPi05",
]
