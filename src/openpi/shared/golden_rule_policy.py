from collections import deque
import copy
import logging
from typing import Any, Optional

import numpy as np
from openpi_client.base_policy import BasePolicy
import torch

from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper

logger = logging.getLogger("policy")


class GoldenRulePolicyWrapper(B1KPolicyWrapper):
    """Policy wrapper that drives execution using a ground-truth skill plan.

    In ``fine_grained_level=2`` mode the wrapper queries the injected
    ``plan_loader`` for the current skill prompt instead of running a VLM
    reasoner.  Skill completion is detected via BEHAVIOR-1K lazy imports so
    that OmniGibson is not initialised at import time.
    """

    def __init__(
        self,
        policy: BasePolicy,
        task_name: str = "turning_on_radio",
        plan_loader: Any | None = None,
        control_mode: str = "receeding_horizon",
        max_len: int = 32,
        action_horizon: int = 5,
        skill_timeout_steps: int = 300,
        fine_grained_level: int = 2,
        temporal_ensemble_max: int = 3,
    ) -> None:
        # We always force fine_grained_level to the value requested by the
        # caller; the base class only instantiates a reasoner when > 0.
        super().__init__(
            policy=policy,
            task_name=task_name,
            control_mode=control_mode,
            max_len=max_len,
            action_horizon=action_horizon,
            temporal_ensemble_max=temporal_ensemble_max,
            fine_grained_level=fine_grained_level,
        )
        self.plan_loader = plan_loader
        self.skill_timeout_steps = skill_timeout_steps
        self._skill_step_counter: int = 0
        self._current_skill_desc: Optional[str] = None

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Reset rollout state *and* the underlying plan loader."""
        super().reset()
        self._skill_step_counter = 0
        self._current_skill_desc = None
        if self.plan_loader is not None:
            self.plan_loader.reset()

    # ------------------------------------------------------------------ #
    #  Prompt resolution  (fine_grained_level == 2  →  plan_loader)
    # ------------------------------------------------------------------ #
    def _resolve_policy_prompt(
        self, *, egocentric_camera: np.ndarray | None = None
    ) -> tuple[str, dict[str, object]]:
        """Return the prompt that should be fed to the policy.

        When ``fine_grained_level == 2`` we bypass the VLM reasoner and
        instead ask ``plan_loader`` for the current skill description.
        """
        prompt_debug: dict[str, object] = {
            "task_name": self.task_name,
            "task_prompt": self.task_prompt,
            "fine_grained_level": self.fine_grained_level,
            "plan_loader_exhausted": False,
            "fallback_to_task_prompt": False,
            "fallback_reason": None,
            "final_prompt": self.task_prompt,
        }

        if self.fine_grained_level != 2 or self.plan_loader is None:
            # Defer to the base-class logic (task-level or VLM reasoner)
            prompt_debug["fallback_reason"] = (
                "fine_grained_level != 2 or no plan_loader"
            )
            return super()._resolve_policy_prompt(egocentric_camera=egocentric_camera)

        if self.plan_loader.is_exhausted():
            prompt_debug["plan_loader_exhausted"] = True
            prompt_debug["fallback_to_task_prompt"] = True
            prompt_debug["fallback_reason"] = "plan_loader_exhausted"
            return self.task_prompt, prompt_debug

        skill = self.plan_loader.get_current_skill()
        if skill is None:
            prompt_debug["fallback_to_task_prompt"] = True
            prompt_debug["fallback_reason"] = "no_current_skill"
            return self.task_prompt, prompt_debug

        skill_desc = skill.get("skill_description", "") if isinstance(skill, dict) else str(skill)
        self._current_skill_desc = skill_desc
        skill_prompt = self.plan_loader.get_skill_prompt(skill_desc)

        prompt_debug["final_prompt"] = skill_prompt
        prompt_debug["skill_description"] = skill_desc
        return skill_prompt, prompt_debug

    # ------------------------------------------------------------------ #
    #  Skill completion  (lazy import from BEHAVIOR-1K)
    # ------------------------------------------------------------------ #
    def check_skill_completion(self, env: Any, info: dict[str, Any]) -> bool:
        """Return *True* when the current skill is finished or has timed out.

        The heavy BEHAVIOR-1K / OmniGibson imports are performed lazily inside
        this method so that merely importing ``golden_rule_policy`` does not
        trigger an expensive environment initialisation.
        """
        if self.plan_loader is None or self.plan_loader.is_exhausted():
            return True

        skill = self.plan_loader.get_current_skill()
        if skill is None:
            return True

        skill_desc = skill.get("skill_description", "") if isinstance(skill, dict) else str(skill)

        # Lazy import – never at module top-level.
        try:
            from omnigibson.learning.utils.skill_completion import check_skill_completed  # type: ignore[import-untyped]
        except Exception as exc:
            logger.warning("Could not lazy-import skill_completion checker: %s", exc)
            # If the checker is unavailable we fall back to a simple timeout.
            return self._skill_step_counter >= self.skill_timeout_steps

        try:
            result = check_skill_completed(env, skill_desc, info)
            completed = bool(result.get("completed", False))
        except Exception as exc:
            logger.warning("Skill completion check failed for %r: %s", skill_desc, exc)
            completed = False

        # Timeout guard – always treat timeout as completion so that the
        # evaluator does not hang on a single skill.
        if not completed and self._skill_step_counter >= self.skill_timeout_steps:
            logger.info(
                "Skill %r timed out after %d steps (timeout=%d)",
                skill_desc,
                self._skill_step_counter,
                self.skill_timeout_steps,
            )
            completed = True

        return completed

    # ------------------------------------------------------------------ #
    #  Act  (with automatic plan advancement)
    # ------------------------------------------------------------------ #
    def act(self, input_obs: dict[str, Any]) -> torch.Tensor:
        """Produce an action, advancing the plan when a skill finishes.

        Before running the normal ``B1KPolicyWrapper.act`` logic we check
        whether the current skill has completed (or timed out).  If so we
        advance the plan loader and **clear the action queue** so that the
        policy is forced to replan with the new prompt on the next step.
        """
        # The evaluator is expected to call ``check_skill_completion``
        # separately and pass the result through ``input_obs`` when available.
        # For convenience we also perform the check here if the obs carries
        # the required ``env`` / ``info`` keys.
        env = input_obs.get("env")
        info = input_obs.get("info", {})
        if env is not None and self.check_skill_completion(env, info):
            self._advance_plan()

        # Run the base action logic (receeding_horizon / temporal_ensemble …)
        return super().act(input_obs)

    def _advance_plan(self) -> None:
        """Advance to the next skill and force a replan."""
        if self.plan_loader is None or self.plan_loader.is_exhausted():
            return

        advanced = self.plan_loader.advance()
        self._skill_step_counter = 0
        # Force replan: clear the action queue so the next call to act()
        # will run the policy with the new prompt instead of dequeuing stale
        # actions.
        self.action_queue.clear()
        logger.info(
            "Advanced plan (advanced=%s); action queue cleared. New skill: %s",
            advanced,
            self.current_skill_desc,
        )

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #
    @property
    def current_skill_desc(self) -> Optional[str]:
        """Description of the skill currently being executed."""
        if self.plan_loader is None or self.plan_loader.is_exhausted():
            return None
        skill = self.plan_loader.get_current_skill()
        if skill is None:
            return None
        return skill.get("skill_description", "") if isinstance(skill, dict) else str(skill)
