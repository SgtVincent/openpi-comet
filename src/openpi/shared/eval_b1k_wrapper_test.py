import numpy as np

from openpi.shared.client import Client
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
from openpi.shared.eval_b1k_wrapper import DESPTH_RESIZE_SIZE


class _DummyModel:
    def __init__(self) -> None:
        self.active_sessions: list[int] = []
        self.reset_sessions: list[int] = []

    def set_active_session(self, session_id: int) -> None:
        self.active_sessions.append(session_id)

    def reset_streaming_state(self, session_id: int) -> None:
        self.reset_sessions.append(session_id)


class _DummyPolicy:
    def __init__(self, model: _DummyModel) -> None:
        # Mimic openpi.policies.policy.Policy private attribute access used by the wrapper.
        self._model = model
        self.seen_prompts: list[str] = []

    def infer(self, obs: dict) -> dict:
        self.seen_prompts.append(obs["prompt"])
        return {"actions": np.zeros((5, 23), dtype=np.float32)}


class _SequencePolicy(_DummyPolicy):
    def infer(self, obs: dict) -> dict:
        self.seen_prompts.append(obs["prompt"])
        actions = np.stack([np.full((23,), fill_value=i, dtype=np.float32) for i in range(5)], axis=0)
        return {"actions": actions}


class _DummyReasoner:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def generate_subtask(self, *, high_level_task: str, multi_modals: list[object]) -> str:
        self.calls.append({"high_level_task": high_level_task, "multi_modals": multi_modals})
        return self.response

    def reset(self) -> None:
        return None


class _FailingReasoner:
    def __init__(self, message: str) -> None:
        self.message = message

    def generate_subtask(self, *, high_level_task: str, multi_modals: list[object]) -> str:
        raise RuntimeError(self.message)

    def reset(self) -> None:
        return None


def test_b1k_wrapper_reset_sets_session_and_clears_streaming_memory() -> None:
    model = _DummyModel()
    wrapper = B1KPolicyWrapper(_DummyPolicy(model), task_name="turning_on_radio", fine_grained_level=0)
    wrapper.reset()

    assert model.active_sessions[-1] == wrapper._session_id  # noqa: SLF001
    assert model.reset_sessions[-1] == wrapper._session_id  # noqa: SLF001


def test_b1k_wrapper_spawn_session_has_distinct_session_id() -> None:
    model = _DummyModel()
    wrapper = B1KPolicyWrapper(_DummyPolicy(model), task_name="turning_on_radio", fine_grained_level=0)
    session = wrapper.spawn_session()

    assert session._session_id != wrapper._session_id  # noqa: SLF001

    session.reset()
    assert model.active_sessions[-1] == session._session_id  # noqa: SLF001
    assert model.reset_sessions[-1] == session._session_id  # noqa: SLF001


def test_b1k_wrapper_uses_normalized_allowed_skill_prompt(monkeypatch) -> None:
    model = _DummyModel()
    policy = _DummyPolicy(model)
    wrapper = B1KPolicyWrapper(policy, task_name="make_pizza", fine_grained_level=0)
    wrapper.fine_grained_level = 1
    wrapper.reasoner = _DummyReasoner("Please TURN TO the oven.")
    monkeypatch.setattr(
        wrapper,
        "process_obs",
        lambda obs: {
            "observation": np.zeros((1, 3, 4, 4, 3), dtype=np.uint8),
            "proprio": np.zeros((1, 16), dtype=np.float32),
        },
    )

    wrapper.act({})

    assert policy.seen_prompts[-1] == "turn to"
    assert wrapper.last_prompt_debug is not None
    assert wrapper.last_prompt_debug["selected_skill"] == "turn to"
    assert wrapper.last_prompt_debug["fallback_to_task_prompt"] is False
    assert wrapper.last_prompt_debug["match_type"] == "contains"


def test_b1k_wrapper_falls_back_to_task_prompt_for_oov_reasoner_output(monkeypatch) -> None:
    model = _DummyModel()
    policy = _DummyPolicy(model)
    wrapper = B1KPolicyWrapper(policy, task_name="turning_on_radio", fine_grained_level=0)
    wrapper.fine_grained_level = 1
    wrapper.reasoner = _DummyReasoner("inspect the room carefully before doing anything")
    monkeypatch.setattr(
        wrapper,
        "process_obs",
        lambda obs: {
            "observation": np.zeros((1, 3, 4, 4, 3), dtype=np.uint8),
            "proprio": np.zeros((1, 16), dtype=np.float32),
        },
    )

    wrapper.act({})

    assert policy.seen_prompts[-1] == wrapper.task_prompt
    assert wrapper.last_prompt_debug is not None
    assert wrapper.last_prompt_debug["fallback_to_task_prompt"] is True
    assert wrapper.last_prompt_debug["fallback_reason"] == "no_skill_match"
    assert wrapper.last_prompt_debug["reasoner_output"] == "inspect the room carefully before doing anything"


def test_b1k_wrapper_falls_back_to_task_prompt_when_reasoner_errors(monkeypatch) -> None:
    model = _DummyModel()
    policy = _DummyPolicy(model)
    wrapper = B1KPolicyWrapper(policy, task_name="make_pizza", fine_grained_level=0)
    wrapper.fine_grained_level = 1
    wrapper.reasoner = _FailingReasoner("upstream 530")
    monkeypatch.setattr(
        wrapper,
        "process_obs",
        lambda obs: {
            "observation": np.zeros((1, 3, 4, 4, 3), dtype=np.uint8),
            "proprio": np.zeros((1, 16), dtype=np.float32),
        },
    )

    wrapper.act({})

    assert policy.seen_prompts[-1] == wrapper.task_prompt
    assert wrapper.last_prompt_debug is not None
    assert wrapper.last_prompt_debug["fallback_to_task_prompt"] is True
    assert wrapper.last_prompt_debug["fallback_reason"] == "reasoner_error"
    assert wrapper.last_prompt_debug["reasoner_error"] == "RuntimeError: upstream 530"


def test_client_reset_initializes_reasoner_state() -> None:
    client = Client.__new__(Client)
    client.history_multi_modals = ["stale"]
    client.uuid = "old"
    client.plan_status = "in progress"
    client.subtask_history = ["move to"]

    Client.reset(client)

    assert len(client.history_multi_modals) == 0
    assert client.plan_status == ""
    assert client.subtask_history == []


def test_b1k_wrapper_receeding_horizon_marks_only_replans_as_fresh(monkeypatch) -> None:
    model = _DummyModel()
    policy = _SequencePolicy(model)
    wrapper = B1KPolicyWrapper(
        policy,
        task_name="turning_on_radio",
        control_mode="receeding_horizon",
        max_len=3,
        fine_grained_level=0,
    )
    monkeypatch.setattr(
        wrapper,
        "process_obs",
        lambda obs: {
            "observation": np.zeros((1, 3, 4, 4, 3), dtype=np.uint8),
            "proprio": np.zeros((1, 16), dtype=np.float32),
        },
    )

    first = wrapper.act({})
    first_fresh = wrapper.last_policy_inferred
    first_chunk = wrapper.last_action_chunk
    first_remaining = wrapper.cached_actions_remaining
    second = wrapper.act({})
    second_fresh = wrapper.last_policy_inferred
    third = wrapper.act({})
    third_fresh = wrapper.last_policy_inferred

    assert first_fresh is True
    assert second_fresh is False
    assert third_fresh is False
    assert first.shape == (1, 23)
    assert np.allclose(first.numpy(), np.zeros((1, 23), dtype=np.float32))
    assert np.allclose(second.numpy(), np.ones((1, 23), dtype=np.float32))
    assert np.allclose(third.numpy(), np.full((1, 23), 2.0, dtype=np.float32))
    assert first_chunk is not None
    assert first_chunk.shape == (3, 23)
    assert first_remaining == 2


def test_b1k_wrapper_process_obs_resizes_depth_without_cv2() -> None:
    model = _DummyModel()
    wrapper = B1KPolicyWrapper(_DummyPolicy(model), task_name="turning_on_radio", fine_grained_level=0)

    obs = {
        "robot_r1::proprio": np.zeros((16,), dtype=np.float32),
        "robot_r1::robot_r1:zed_link:Camera:0::rgb": np.zeros((8, 10, 4), dtype=np.uint8),
        "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": np.zeros((8, 10, 4), dtype=np.uint8),
        "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": np.zeros((8, 10, 4), dtype=np.uint8),
        "robot_r1::robot_r1:zed_link:Camera:0::depth_linear": np.arange(80, dtype=np.float32).reshape(8, 10),
    }

    processed = wrapper.process_obs(obs)

    assert processed["observation/egocentric_depth"].shape == (1, DESPTH_RESIZE_SIZE, DESPTH_RESIZE_SIZE)
    assert processed["observation/egocentric_depth"].dtype == np.float32
