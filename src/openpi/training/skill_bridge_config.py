"""Skill bridge configuration dataclass.

Phase 1 minimal config: enable/disable toggle + min boundary step thresholds.

This module is intentionally lightweight (stdlib only) so it can be imported
without pulling in the full data_config dependency tree.  Integration into
``DataConfig`` is a 2-line addition:

    from openpi.training.skill_bridge_config import SkillBridgeConfig
    # ... inside DataConfig ...
    skill_bridge: SkillBridgeConfig = dataclasses.field(default_factory=SkillBridgeConfig)
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class SkillBridgeConfig:
    """Configuration for skill bridge baseline.

    When enabled, action chunks that cross exactly one contiguous skill boundary
    use a combined subtask_text of "{current_skill} then {successor_skill}".
    All other cases (no crossing, multiple crossings, gaps, overlaps, padded
    tails) keep the original single-skill subtask_text.

    Attributes:
        enabled: If True, enable the skill bridge baseline.  Default False
            guarantees zero behavioral change.
        min_pre_boundary_steps: Minimum steps required before the boundary
            within the chunk for the bridge to be considered valid.
            Prevents bridges where the boundary is at the very start of the
            chunk (not enough current-skill context).
        min_post_boundary_steps: Minimum steps required after the boundary
            within the chunk for the bridge to be considered valid.
            Prevents bridges where the boundary is at the very end of the
            chunk (not enough successor-skill action context).
    """

    enabled: bool = False
    min_pre_boundary_steps: int = 1
    min_post_boundary_steps: int = 1
