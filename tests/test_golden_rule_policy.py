from collections import deque
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from openpi.shared.golden_rule_policy import GoldenRulePolicyWrapper


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
class _DummyPolicy:
    """Minimal stand-in for a BasePolicy."""

    def __init__(self) -> None:
        self.seen_prompts: list[str] = []
        self._model = None  # type: ignore[var-annotated]

    def infer(self, obs: dict) -> dict:
        self.seen_prompts.append(obs["prompt"])
        # Return a dummy action sequence (max_len, 23)
        return {"actions": np.zeros((10, 23), dtype=np.float64)}


class _MockPlanLoader:
    """In-memory plan loader for testing."""

    def __init__(self, skills: list[dict[str, str]]) -> None:
        self._skills = skills
        self._index = 0
        self.advance_calls: int = 0
        self.reset_calls: int = 0

    def get_current_skill(self) -> dict[str, str] | None:
        if self._index >= len(self._skills):
            return None
        return self._skills[self._index]

    def advance(self) -> bool:
        if self._index < len(self._skills):
            self._index += 1
            self.advance_calls += 1
            return True
        return False

    def get_skill_prompt(self, skill_desc: str) -> str:
        return f"prompt_for_{skill_desc}"

    def is_exhausted(self) -> bool:
        return self._index >= len(self._skills)

    def reset(self) -> None:
        self._index = 0
        self.reset_calls += 1


def _make_obs_with_env(skill_complete: bool = False) -> dict:
    """Build a minimal observation dict that ``act`` can process.

    We monkey-patch ``process_obs`` in most tests, but when we want to
    exercise the full ``act`` path we need the real RGB keys.
    """
    return {
        "robot_r1::proprio": np.zeros(16, dtype=np.float64),
        "robot_r1::robot_r1:zed_link:Camera:0::rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        "env": MagicMock(),
        "info": {"skill_complete": skill_complete},
    }


# --------------------------------------------------------------------------- #
#  Tests
# --------------------------------------------------------------------------- #
def test_resolve_policy_prompt_uses_plan_loader_skill() -> None:
    """When fine_grained_level==2 the prompt comes from the plan_loader."""
    plan = _MockPlanLoader([{"skill_description": "open cabinet"}])
    policy = _DummyPolicy()
    wrapper = GoldenRulePolicyWrapper(
        policy,
        task_name="turning_on_radio",
        plan_loader=plan,
        fine_grained_level=2,
    )

    prompt, debug = wrapper._resolve_policy_prompt()

    assert prompt == "prompt_for_open cabinet"
    assert debug["final_prompt"] == "prompt_for_open cabinet"
    assert debug["skill_description"] == "open cabinet"
    assert debug["fallback_to_task_prompt"] is False


def test_resolve_policy_prompt_fallback_when_exhausted() -> None:
    """If the plan_loader is exhausted we fall back to the task prompt."""
    plan = _MockPlanLoader([])
    policy = _DummyPolicy()
    wrapper = GoldenRulePolicyWrapper(
        policy,
        task_name="turning_on_radio",
        plan_loader=plan,
        fine_grained_level=2,
    )

    prompt, debug = wrapper._resolve_policy_prompt()

    assert prompt == wrapper.task_prompt
    assert debug["plan_loader_exhausted"] is True
    assert debug["fallback_to_task_prompt"] is True


def test_resolve_policy_prompt_defer_to_base_when_level_not_2() -> None:
    """fine_grained_level != 2 should delegate to the base class."""
    plan = _MockPlanLoader([{"skill_description": "open cabinet"}])
    policy = _DummyPolicy()
    wrapper = GoldenRulePolicyWrapper(
        policy,
        task_name="turning_on_radio",
        plan_loader=plan,
        fine_grained_level=0,
    )

    prompt, debug = wrapper._resolve_policy_prompt()

    # Base class returns task_prompt when fine_grained_level == 0
    assert prompt == wrapper.task_prompt
    assert debug["fallback_reason"] is not None


def test_reset_clears_plan_loader_and_counters() -> None:
    plan = _MockPlanLoader([{"skill_description": "a"}, {"skill_description": "b"}])
    policy = _DummyPolicy()
    wrapper = GoldenRulePolicyWrapper(
        policy,
        task_name="turning_on_radio",
        plan_loader=plan,
        fine_grained_level=2,
    )
    wrapper._skill_step_counter = 42
    wrapper.action_queue.append(np.zeros(23))

    wrapper.reset()

    assert plan.reset_calls == 1
    assert wrapper._skill_step_counter == 0
    assert len(wrapper.action_queue) == 0


def test_check_skill_completion_lazy_import_timeout(monkeypatch) -> None:
    """If the lazy import fails we fall back to timeout logic."""
    plan = _MockPlanLoader([{"skill_description": "open cabinet"}])
    policy = _DummyPolicy()
    wrapper = GoldenRulePolicyWrapper(
        policy,
        task_name="turning_on_radio",
        plan_loader=plan,
        fine_grained_level=2,
        skill_timeout_steps=5,
    )
    wrapper._skill_step_counter = 10  # past timeout

    # Force the lazy import to fail
    def _bad_import(*args, **kwargs):
        raise ImportError("no omnigibson")

    monkeypatch.setattr("builtins.__import__", _bad_import)

    # Because ImportError is caught, we should get True due to timeout
    assert wrapper.check_skill_completion(MagicMock(), {}) is True


def test_check_skill_completion_uses_lazy_checker(monkeypatch) -> None:
    """When the lazy import succeeds we delegate to ``check_skill_completed``."""
    plan = _MockPlanLoader([{"skill_description": "open cabinet"}])
    policy = _DummyPolicy()
    wrapper = GoldenRulePolicyWrapper(
        policy,
        task_name="turning_on_radio",
        plan_loader=plan,
        fine_grained_level=2,
        skill_timeout_steps=300,
    )
    wrapper._skill_step_counter = 0

    mock_checker = MagicMock(return_value={"completed": True})

    def _fake_import(name, *args, **kwargs):
        if name == "omnigibson.learning.utils.skill_completion":
            mod = type(sys)("omnigibson.learning.utils.skill_completion")
            mod.check_skill_completed = mock_checker
            sys.modules[name] = mod
            return mod
        return __builtins__["__import__"](name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)

    env = MagicMock()
    info = {"frame": 7}
    result = wrapper.check_skill_completion(env, info)

    assert result is True
    mock_checker.assert_called_once_with(env, "open cabinet", info)


def test_act_advances_plan_when_skill_done(monkeypatch) -> None:
    """If check_skill_completion returns True, act() should advance the plan."""
    plan = _MockPlanLoader(
        [
            {"skill_description": "skill_a"},
            {"skill_description": "skill_b"},
        ]
    )
    policy = _DummyPolicy()
    wrapper = GoldenRulePolicyWrapper(
        policy,
        task_name="turning_on_radio",
        plan_loader=plan,
        fine_grained_level=2,
        control_mode="receeding_horizon",
    )

    # Monkey-patch process_obs so we don't need real camera images
    monkeypatch.setattr(
        wrapper,
        "process_obs",
        lambda obs: {
            "observation": np.zeros((1, 3, 224, 224, 3), dtype=np.uint8),
            "proprio": np.zeros((1, 16), dtype=np.float64),
        },
    )

    # Monkey-patch check_skill_completion to always say "done"
    monkeypatch.setattr(wrapper, "check_skill_completion", lambda env, info: True)

    obs = _make_obs_with_env()
    wrapper.act(obs)

    assert plan.advance_calls == 1
    assert wrapper.current_skill_desc == "skill_b"
    # Action queue is cleared by _advance_plan, but super().act() immediately
    # replans and refills it (receeding_horizon mode).  The important thing is
    # that the *old* queue was discarded and the policy ran with the new skill.
    assert policy.seen_prompts[-1] == "prompt_for_skill_b"


def test_act_does_not_advance_when_skill_incomplete(monkeypatch) -> None:
    """If the skill is not done the plan should stay on the current skill."""
    plan = _MockPlanLoader([{"skill_description": "skill_a"}])
    policy = _DummyPolicy()
    wrapper = GoldenRulePolicyWrapper(
        policy,
        task_name="turning_on_radio",
        plan_loader=plan,
        fine_grained_level=2,
        control_mode="receeding_horizon",
    )

    monkeypatch.setattr(
        wrapper,
        "process_obs",
        lambda obs: {
            "observation": np.zeros((1, 3, 224, 224, 3), dtype=np.uint8),
            "proprio": np.zeros((1, 16), dtype=np.float64),
        },
    )
    monkeypatch.setattr(wrapper, "check_skill_completion", lambda env, info: False)

    obs = _make_obs_with_env()
    wrapper.act(obs)

    assert plan.advance_calls == 0
    assert wrapper.current_skill_desc == "skill_a"


def test_current_skill_desc_property() -> None:
    plan = _MockPlanLoader([{"skill_description": "first"}, {"skill_description": "second"}])
    policy = _DummyPolicy()
    wrapper = GoldenRulePolicyWrapper(
        policy,
        task_name="turning_on_radio",
        plan_loader=plan,
        fine_grained_level=2,
    )

    assert wrapper.current_skill_desc == "first"
    plan.advance()
    assert wrapper.current_skill_desc == "second"
    plan.advance()
    assert wrapper.current_skill_desc is None


def test_act_returns_tensor(monkeypatch) -> None:
    """Smoke test: act() must return a torch.Tensor of the right shape."""
    plan = _MockPlanLoader([{"skill_description": "skill_a"}])
    policy = _DummyPolicy()
    wrapper = GoldenRulePolicyWrapper(
        policy,
        task_name="turning_on_radio",
        plan_loader=plan,
        fine_grained_level=2,
        control_mode="receeding_horizon",
    )

    monkeypatch.setattr(
        wrapper,
        "process_obs",
        lambda obs: {
            "observation": np.zeros((1, 3, 224, 224, 3), dtype=np.uint8),
            "proprio": np.zeros((1, 16), dtype=np.float64),
        },
    )
    monkeypatch.setattr(wrapper, "check_skill_completion", lambda env, info: False)

    obs = _make_obs_with_env()
    action = wrapper.act(obs)

    assert isinstance(action, torch.Tensor)
    assert action.shape == (1, 23)


def test_advance_plan_clears_action_queue() -> None:
    plan = _MockPlanLoader([{"skill_description": "a"}, {"skill_description": "b"}])
    policy = _DummyPolicy()
    wrapper = GoldenRulePolicyWrapper(
        policy,
        task_name="turning_on_radio",
        plan_loader=plan,
        fine_grained_level=2,
    )
    wrapper.action_queue = deque([np.zeros(23), np.zeros(23)])

    wrapper._advance_plan()

    assert len(wrapper.action_queue) == 0
    assert plan.advance_calls == 1
    assert wrapper._skill_step_counter == 0
