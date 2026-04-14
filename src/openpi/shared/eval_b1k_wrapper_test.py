import numpy as np

from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper


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

    def infer(self, obs: dict) -> dict:
        # Not used in this test file (we only validate reset/spawn_session wiring).
        return {"actions": np.zeros((5, 23), dtype=np.float32)}


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

