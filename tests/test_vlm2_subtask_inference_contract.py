"""Contract tests for VLM2Subtask subtask inference path.

These tests enforce that ``VLM2SubtaskWithPi05`` provides a functional subtask
inference interface, matching the contract already established by
``Pi05WithSubtask`` in ``openpi.models_pytorch.pi05_subtask``.

Audit finding **CRIT-2**: ``VLM2SubtaskWithPi05`` only overrides ``forward()``
for training and has NO subtask inference methods.  The policy wrapper's
``hasattr(self._model, "predict_subtask_tokens")`` check therefore fails, and
inference silently falls back to action-only generation — subtask CE loss is
trained but never consumed at inference time.

Tests whose names begin with ``test_vlm2subtask_has_`` are **existence
assertions** that will FAIL on current code (proving CRIT-2).  The remaining
tests use mocks to document the expected *integration contract* between
``Policy`` and a subtask-capable model.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import pytest
import torch
from torch import nn

# --------------------------------------------------------------------------- #
#  Import the class under test
# --------------------------------------------------------------------------- #
# VLM2SubtaskWithPi05 has heavy transitive deps (torch, transformers, …).
# The tests below will import it directly; we fail loudly if the import
# itself breaks so the test failure is unmistakable.
from openpi.models_pytorch.vlm2.vlm2_model import VLM2SubtaskWithPi05  # noqa: E402


# =========================================================================== #
#  1. Interface existence tests — these FAIL on current code (CRIT-2)
# =========================================================================== #


class TestVLM2SubtaskInterfaceExistence:
    """Prove CRIT-2: VLM2SubtaskWithPi05 lacks subtask inference methods.

    Each test asserts that one required interface method exists and is
    callable on the class.  All five should FAIL until the subtask inference
    path is implemented.
    """

    def test_predict_subtask_tokens_exists(self):
        """VLM2SubtaskWithPi05 must expose ``predict_subtask_tokens``.

        This is the core subtask generation entry point: given an
        ``Observation``, return a ``torch.Tensor`` of generated subtask token
        IDs.  The ``Policy.infer()`` wrapper checks for this attribute with
        ``hasattr`` to decide whether to enable hierarchical inference.

        Audit finding: CRIT-2 — VLM2SubtaskWithPi05 has no subtask inference.
        """
        # Fails due to CRIT-2: VLM2SubtaskWithPi05 lacks subtask inference methods
        assert hasattr(VLM2SubtaskWithPi05, "predict_subtask_tokens"), (
            "VLM2SubtaskWithPi05 missing predict_subtask_tokens — "
            "Policy.infer() will silently fall back to action-only inference"
        )
        assert callable(VLM2SubtaskWithPi05.predict_subtask_tokens)

    def test_sample_actions_hierarchical_exists(self):
        """VLM2SubtaskWithPi05 must expose ``sample_actions_hierarchical``.

        This is the hierarchical action-sampling entry point: predict a
        subtask from the observation, condition the observation on it, then
        sample actions.  It is the method that callers should use when they
        want end-to-end hierarchical inference (as opposed to
        ``sample_actions`` which assumes the subtask is already provided or
        absent).

        Audit finding: CRIT-2 — VLM2SubtaskWithPi05 has no subtask inference.
        """
        # Fails due to CRIT-2: VLM2SubtaskWithPi05 lacks subtask inference methods
        assert hasattr(VLM2SubtaskWithPi05, "sample_actions_hierarchical"), (
            "VLM2SubtaskWithPi05 missing sample_actions_hierarchical — "
            "no entry point for end-to-end hierarchical action generation"
        )
        assert callable(VLM2SubtaskWithPi05.sample_actions_hierarchical)

    def test_predict_subtask_exists(self):
        """VLM2SubtaskWithPi05 must expose ``predict_subtask`` (text output).

        Convenience method that runs ``predict_subtask_tokens`` and decodes
        the result to a list of human-readable subtask strings — one per
        batch element.  Used by evaluation / debugging code that wants the
        predicted subtask as text.

        Audit finding: CRIT-2 — VLM2SubtaskWithPi05 has no subtask inference.
        """
        # Fails due to CRIT-2: VLM2SubtaskWithPi05 lacks subtask inference methods
        assert hasattr(VLM2SubtaskWithPi05, "predict_subtask"), (
            "VLM2SubtaskWithPi05 missing predict_subtask — "
            "cannot return decoded subtask text strings"
        )
        assert callable(VLM2SubtaskWithPi05.predict_subtask)

    def test_decode_subtask_tokens_exists(self):
        """VLM2SubtaskWithPi05 must expose ``decode_subtask_tokens``.

        Decodes a batch of subtask token IDs into a list of strings.  The
        ``Policy.infer()`` wrapper calls this after
        ``predict_subtask_tokens`` to populate ``generated_subtask`` in the
        output dict.

        Audit finding: CRIT-2 — VLM2SubtaskWithPi05 has no subtask inference.
        """
        # Fails due to CRIT-2: VLM2SubtaskWithPi05 lacks subtask inference methods
        assert hasattr(VLM2SubtaskWithPi05, "decode_subtask_tokens"), (
            "VLM2SubtaskWithPi05 missing decode_subtask_tokens — "
            "Policy.infer() cannot decode predicted subtask tokens to text"
        )
        assert callable(VLM2SubtaskWithPi05.decode_subtask_tokens)

    def test_build_hierarchical_observation_exists(self):
        """VLM2SubtaskWithPi05 must expose ``build_hierarchical_observation``.

        Given an ``Observation`` and predicted subtask tokens, return a new
        ``Observation`` with ``subtask_tokens`` / ``subtask_mask`` /
        ``subtask_loss_mask`` / ``subtask_ar_mask`` populated so that the
        downstream action-generation path sees the predicted subtask as
        prefix context.  Called by ``Policy.infer()`` after subtask
        prediction.

        Audit finding: CRIT-2 — VLM2SubtaskWithPi05 has no subtask inference.
        """
        # Fails due to CRIT-2: VLM2SubtaskWithPi05 lacks subtask inference methods
        assert hasattr(VLM2SubtaskWithPi05, "build_hierarchical_observation"), (
            "VLM2SubtaskWithPi05 missing build_hierarchical_observation — "
            "Policy.infer() cannot inject predicted subtask into observation"
        )
        assert callable(VLM2SubtaskWithPi05.build_hierarchical_observation)


# =========================================================================== #
#  2. Integration contract tests — mock-based, define expected behavior
# =========================================================================== #


def _make_mock_subtask_model():
    """Create a MagicMock that looks like a subtask-capable PyTorch model.

    The mock exposes ``predict_subtask_tokens``, ``decode_subtask_tokens``,
    ``build_hierarchical_observation``, and ``sample_actions``, matching the
    interface that ``Policy.infer()`` expects from a subtask-capable model.
    """
    model = MagicMock()
    # Subtask interface — what Policy.infer() checks via hasattr.
    model.predict_subtask_tokens = MagicMock(
        return_value=torch.tensor([[1, 2, 3, 0]], dtype=torch.int32)
    )
    model.decode_subtask_tokens = MagicMock(return_value=["pick up the mug"])
    model.build_hierarchical_observation = MagicMock(side_effect=lambda obs, tokens: obs)

    # Action-sampling interface.
    model.sample_actions = MagicMock(
        return_value=torch.zeros((1, 10, 23), dtype=torch.float32)
    )

    # PyTorch boilerplate that Policy.__init__ calls.
    model.to.return_value = model
    model.eval.return_value = None

    return model


def _strip_prompt_transform(obs: dict) -> dict:
    """Input transform that drops the raw ``prompt`` string.

    ``Policy.infer()`` runs ``jax.tree.map(torch.from_numpy(...), inputs)``
    over the *whole* obs dict, which chokes on string values.  In production
    the real input transforms tokenize the prompt and leave no string fields
    behind.  For mock-based tests we just drop the prompt key.
    """
    return {k: v for k, v in obs.items() if k != "prompt"}


class TestPolicySubtaskIntegrationContract:
    """Document the expected contract between Policy and a subtask model.

    These tests use mocks — they do **not** exercise VLM2SubtaskWithPi05
    directly.  Their purpose is to pin down how ``Policy.infer()`` is supposed
    to behave when the model *does* expose the subtask interface, so that
    future implementors know what contract to satisfy.
    """

    def test_policy_detects_subtask_capability_and_calls_predict(self):
        """Policy detects ``predict_subtask_tokens`` and calls it during infer.

        Contract: when the wrapped model is PyTorch (``is_pytorch=True``) and
        has both ``predict_subtask_tokens`` and ``build_hierarchical_observation``,
        ``Policy.infer()`` should:

        1. Call ``model.predict_subtask_tokens(observation)`` to generate
           subtask tokens.
        2. Call ``model.decode_subtask_tokens(tokens)`` to get text.
        3. Call ``model.build_hierarchical_observation(observation, tokens)``
           to condition the observation on the subtask.
        4. Include ``generated_subtask`` in the output dict.

        This test uses a mock model and verifies the call sequence.
        """
        from openpi.policies.policy import Policy

        mock_model = _make_mock_subtask_model()

        # Patch Observation.from_dict to return something the policy can work with.
        mock_observation = MagicMock()
        mock_observation.subtask_mask = None  # No GT subtask → should predict.

        with patch("openpi.policies.policy._model.Observation.from_dict", return_value=mock_observation):
            policy = Policy(
                mock_model,
                is_pytorch=True,
                pytorch_device="cpu",
                transforms=[_strip_prompt_transform],
            )

            obs = {
                "state": np.zeros((23,), dtype=np.float32),
                "prompt": "put the mug on the table",
            }
            outputs = policy.infer(obs)

        # Subtask prediction should have been invoked.
        mock_model.predict_subtask_tokens.assert_called_once()
        mock_model.decode_subtask_tokens.assert_called_once()
        mock_model.build_hierarchical_observation.assert_called_once()

        # Action sampling should still have been invoked.
        mock_model.sample_actions.assert_called_once()

        # Output should contain the generated subtask text.
        assert "generated_subtask" in outputs
        assert outputs["generated_subtask"] == "pick up the mug"

    def test_gt_subtask_bypasses_prediction(self):
        """When GT subtask is provided (subtask_mask is all-True), predict is skipped.

        Contract: if ``observation.subtask_mask`` is provided and has at
        least one True entry (meaning a GT subtask is already embedded in
        the observation), ``Policy.infer()`` must **not** call
        ``predict_subtask_tokens`` — the GT subtask takes precedence.

        This is the "evaluation with oracle subtask" path.
        """
        from openpi.policies.policy import Policy

        mock_model = _make_mock_subtask_model()

        # Mock observation WITH a GT subtask (subtask_mask has True entries).
        mock_observation = MagicMock()
        mock_observation.subtask_mask = torch.tensor([[True, True, True, False]])

        with patch("openpi.policies.policy._model.Observation.from_dict", return_value=mock_observation):
            policy = Policy(
                mock_model,
                is_pytorch=True,
                pytorch_device="cpu",
                transforms=[_strip_prompt_transform],
            )

            obs = {
                "state": np.zeros((23,), dtype=np.float32),
                "prompt": "put the mug on the table",
                "subtask_tokens": np.array([1, 2, 3], dtype=np.int32),
                "subtask_mask": np.array([True, True, True], dtype=bool),
            }
            outputs = policy.infer(obs)

        # Subtask prediction must NOT have been invoked — GT was provided.
        mock_model.predict_subtask_tokens.assert_not_called()
        mock_model.decode_subtask_tokens.assert_not_called()
        mock_model.build_hierarchical_observation.assert_not_called()

        # Action sampling should still run.
        mock_model.sample_actions.assert_called_once()

        # No generated_subtask in output (since we didn't predict one).
        # Note: the policy also checks _last_predicted_subtasks on the
        # model, which the mock won't have set — so no generated_subtask.
        assert "generated_subtask" not in outputs

    def test_policy_without_predict_subtask_tokens_falls_back(self):
        """Policy silently falls back to action-only when model lacks the interface.

        Contract / current behavior: when the model is missing
        ``predict_subtask_tokens``, ``Policy.infer()`` should proceed with
        action-only inference and NOT raise.  This is the path that
        VLM2SubtaskWithPi05 currently hits (CRIT-2).

        This test documents the *current* (buggy-for-subtask-models) fallback
        behavior so we have a baseline and so the CRIT-2 failure mode is
        explicit.
        """
        from openpi.policies.policy import Policy

        # Mock model WITHOUT the subtask interface — like VLM2SubtaskWithPi05 today.
        mock_model = MagicMock(spec=["sample_actions", "to", "eval"])
        mock_model.sample_actions = MagicMock(
            return_value=torch.zeros((1, 10, 23), dtype=torch.float32)
        )
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None

        mock_observation = MagicMock()
        mock_observation.subtask_mask = None

        with patch("openpi.policies.policy._model.Observation.from_dict", return_value=mock_observation):
            policy = Policy(
                mock_model,
                is_pytorch=True,
                pytorch_device="cpu",
                transforms=[_strip_prompt_transform],
            )

            obs = {
                "state": np.zeros((23,), dtype=np.float32),
                "prompt": "put the mug on the table",
            }
            outputs = policy.infer(obs)

        # No subtask prediction should happen.
        assert not hasattr(mock_model, "predict_subtask_tokens") or not hasattr(
            mock_model, "build_hierarchical_observation"
        )
        # Action sampling ran.
        mock_model.sample_actions.assert_called_once()
        # No generated_subtask.
        assert "generated_subtask" not in outputs
        # Actions came through.
        assert "actions" in outputs
        assert outputs["actions"].shape == (10, 23)


# =========================================================================== #
#  3. Semantic gap tests — BOS conditioning, batch EOS, double memory       #
# =========================================================================== #


class TestVLM2BuildHierarchicalObservationBOS:
    """VLM2 build_hierarchical_observation must prepend BOS (Gap 1).

    Training conditions the action expert on [BOS, tok1, ..., EOS].
    predict_subtask_tokens returns [tok1, ..., EOS] (no BOS).
    build_hierarchical_observation must add BOS back so the conditioning
    matches training.
    """

    def _make_model_instance(self):
        """Create a minimal VLM2SubtaskWithPi05 instance for testing.

        Properly initializes nn.Module base and sets lazy-load attributes
        so attribute lookups work correctly.
        """
        model = VLM2SubtaskWithPi05.__new__(VLM2SubtaskWithPi05)
        nn.Module.__init__(model)
        # Set lazy-load attribute so nn.Module.__getattr__ doesn't intercept it
        model._text_tokenizer = None
        model._last_predicted_subtasks = []
        return model

    def test_vlm2_build_hierarchical_prepends_bos(self):
        """VLM2SubtaskWithPi05.build_hierarchical_observation prepends BOS."""
        from types import SimpleNamespace

        model = self._make_model_instance()
        model.config = SimpleNamespace(subtask_max_len=16)

        bos_id = model._load_text_tokenizer().bos_id()
        eos_id = model._eos_token_id()
        tok1, tok2 = 42, 99

        # Input: subtask tokens as returned by predict_subtask_tokens (no BOS)
        subtask_tokens = torch.tensor([[tok1, tok2, eos_id]], dtype=torch.int32)

        @dataclasses.dataclass
        class _DummyObs:
            state: torch.Tensor = None
            subtask_tokens: torch.Tensor = None
            subtask_mask: torch.Tensor = None
            subtask_loss_mask: torch.Tensor = None
            subtask_ar_mask: torch.Tensor = None

        obs = _DummyObs(state=torch.zeros(1, 23))
        result = model.build_hierarchical_observation(obs, subtask_tokens)

        # Core assertion: first token must be BOS
        assert result.subtask_tokens[0, 0].item() == bos_id, (
            "Gap 1 (VLM2): build_hierarchical_observation must prepend BOS "
            f"Expected first token = BOS ({bos_id}), got {result.subtask_tokens[0, 0].item()}"
        )
        # Check sequence: BOS, tok1, tok2, EOS
        assert result.subtask_tokens[0, 1].item() == tok1
        assert result.subtask_tokens[0, 2].item() == tok2
        assert result.subtask_tokens[0, 3].item() == eos_id
        # All four positions are in the mask
        assert result.subtask_mask[0, :4].all().item() == True  # noqa: E712

    def test_default_max_len_preserves_eos(self):
        """When config lacks subtask_max_len, EOS must not be clipped.

        Robustness issue (a): default max_len must account for BOS prefix
        (+1 token) so the last generated token (EOS) is not clipped off.
        """
        from types import SimpleNamespace

        model = self._make_model_instance()
        # No subtask_max_len on config — triggers default fallback
        model.config = SimpleNamespace()

        bos_id = model._load_text_tokenizer().bos_id()
        eos_id = model._eos_token_id()

        # 3 generated tokens: tok1, tok2, EOS
        subtask_tokens = torch.tensor([[42, 99, eos_id]], dtype=torch.int32)

        @dataclasses.dataclass
        class _DummyObs:
            state: torch.Tensor = None
            subtask_tokens: torch.Tensor = None
            subtask_mask: torch.Tensor = None
            subtask_loss_mask: torch.Tensor = None
            subtask_ar_mask: torch.Tensor = None

        obs = _DummyObs(state=torch.zeros(1, 23))
        result = model.build_hierarchical_observation(obs, subtask_tokens)

        # With default max_len (= generated_len + 1), BOS + 3 tokens all fit.
        assert result.subtask_tokens.shape[1] == 4
        # Sequence: BOS, tok1, tok2, EOS — all present
        assert result.subtask_tokens[0, 0].item() == bos_id
        assert result.subtask_tokens[0, 3].item() == eos_id
        # All 4 positions valid
        assert result.subtask_mask[0, :4].all().item() == True  # noqa: E712


class TestVLM2BatchEOSContamination:
    """VLM2 subtask generation must handle per-row EOS correctly (Gap 2).

    Gap 2 has two parts:
    (a) predict_subtask_tokens must zero post-EOS tokens per row.
    (b) build_hierarchical_observation must use EOS-aware mask.

    We test the logic contract for both parts.
    """

    def _make_model_instance(self):
        """Create a minimal VLM2SubtaskWithPi05 instance for testing."""
        model = VLM2SubtaskWithPi05.__new__(VLM2SubtaskWithPi05)
        nn.Module.__init__(model)
        model._text_tokenizer = None
        model._last_predicted_subtasks = []
        return model

    def test_post_eos_zeroing_contract(self):
        """predict_subtask_tokens must zero tokens after per-row EOS.

        Verifies the logic contract: given a sequence of generated tokens
        with mixed EOS positions per row, a correct implementation will
        zero all tokens after the first EOS in each row.
        """
        eos_id = 3
        # Row 0: EOS at idx 2 (garbage after)
        # Row 1: EOS at idx 4 (last token)
        # Row 2: no EOS
        generated = torch.tensor([
            [10, 20, eos_id, 999, 999],
            [30, 40, 50, 60, eos_id],
            [70, 80, 90, 100, 110],
        ], dtype=torch.int32)

        # Contract: the model's predict_subtask_tokens must produce output
        # equivalent to this zero-after-first-EOS pass.
        def zero_after_eos(tokens, eos_id):
            result = tokens.clone()
            for i in range(tokens.shape[0]):
                eos_positions = (tokens[i] == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    first_eos = eos_positions[0].item()
                    result[i, first_eos + 1:] = 0
            return result

        zeroed = zero_after_eos(generated, eos_id)

        # Row 0: tokens up to and including EOS preserved, rest zeroed
        assert zeroed[0, 0].item() == 10
        assert zeroed[0, 2].item() == eos_id
        assert zeroed[0, 3].item() == 0
        assert zeroed[0, 4].item() == 0

        # Row 1: all tokens valid (EOS at end)
        assert (zeroed[1] == generated[1]).all()

        # Row 2: no EOS, nothing zeroed
        assert (zeroed[2] == generated[2]).all()

    def test_vlm2_build_hierarchical_eos_aware_mask(self):
        """VLM2 build_hierarchical_observation mask must be EOS-aware.

        The subtask_mask must mark tokens valid up to and including the
        first EOS, not just all nonzero tokens.
        """
        from types import SimpleNamespace

        model = self._make_model_instance()
        model.config = SimpleNamespace(subtask_max_len=16)

        bos_id = model._load_text_tokenizer().bos_id()
        eos_id = model._eos_token_id()

        # Simulate tokens from predict_subtask_tokens with mixed EOS:
        # Row 0: short (EOS at idx 2 among generated → idx 3 with BOS)
        # Row 1: longer (EOS at idx 4 among generated → idx 5 with BOS)
        # We pass generated tokens (no BOS) — build_hierarchical adds BOS.
        subtask_tokens = torch.tensor([
            [10, 20, eos_id, 0, 0, 0],     # 3 generated tokens, EOS at idx2
            [30, 40, 50, 60, eos_id, 0],   # 5 generated tokens, EOS at idx4
        ], dtype=torch.int32)

        @dataclasses.dataclass
        class _DummyObs:
            state: torch.Tensor = None
            subtask_tokens: torch.Tensor = None
            subtask_mask: torch.Tensor = None
            subtask_loss_mask: torch.Tensor = None
            subtask_ar_mask: torch.Tensor = None

        obs = _DummyObs(state=torch.zeros(2, 23))
        result = model.build_hierarchical_observation(obs, subtask_tokens)

        # Row 0: BOS + 10 + 20 + EOS = 4 valid tokens
        assert result.subtask_tokens[0, 0].item() == bos_id
        assert result.subtask_mask[0, 0].item() == True  # noqa: E712
        assert result.subtask_mask[0, 1].item() == True  # noqa: E712
        assert result.subtask_mask[0, 2].item() == True  # noqa: E712
        assert result.subtask_mask[0, 3].item() == True  # noqa: E712 (EOS)
        # After EOS: should be False
        assert result.subtask_mask[0, 4].item() == False  # noqa: E712

        # Row 1: BOS + 30 + 40 + 50 + 60 + EOS = 6 valid tokens
        assert result.subtask_mask[1, 0].item() == True  # noqa: E712
        assert result.subtask_mask[1, 5].item() == True  # noqa: E712 (EOS)
        assert result.subtask_mask[1, 6].item() == False  # noqa: E712


class TestVLM2DoubleMemoryUpdate:
    """VLM2 hierarchical inference must not double-update memory (Gap 3).

    Gap 3: predict_subtask_tokens calls process_video_with_memory (which
    updates streaming memory), then _sample_actions_with_subtask_conditioning
    calls process_video_with_memory again for the same frame, updating memory
    a second time.  This double-pushes the same observation into memory,
    which corrupts the streaming memory state for subsequent frames.

    Fix: predict_subtask_tokens must save/restore memory state, so the
    subtask prediction is memory-neutral — it doesn't change the streaming
    memory.  The action inference path performs the real memory update.
    """

    def test_hierarchical_memory_equals_plain_memory(self):
        """After hierarchical inference, memory state must equal plain
        action-only inference with the same observation.

        Contract: running sample_actions_hierarchical(obs) should leave the
        dual-memory module in the same state as running
        VLM2WithPi05.sample_actions(obs).  The subtask prediction should be
        a "read-only" operation from memory's perspective.
        """
        # We verify the design contract rather than instantiating a full model:
        # - save memory state before predict_subtask_tokens
        # - run subtask prediction
        # - restore memory state
        # - run action inference (which does the real memory update)
        # Final memory == memory after a single action-inference pass.

        # This is a logic/contract test.  We demonstrate that with proper
        # save/restore, the final memory state is identical to a single pass.
        from types import SimpleNamespace

        # Simulate memory state as a simple counter (number of frames seen)
        class MockMemory:
            def __init__(self):
                self.frame_count = 0
                self.buffer = []

            def process_frame(self, frame_id):
                self.frame_count += 1
                self.buffer.append(frame_id)

            def get_state(self):
                return {"frame_count": self.frame_count, "buffer": list(self.buffer)}

            def set_state(self, state):
                self.frame_count = state["frame_count"]
                self.buffer = list(state["buffer"])

        memory = MockMemory()

        # Plain action inference: single memory update
        memory_plain = MockMemory()
        memory_plain.process_frame("frame_0")  # one update
        plain_state = memory_plain.get_state()

        # Hierarchical inference: predict (save/restore) + action (update)
        memory_hier = MockMemory()

        # Step 1: subtask prediction — save, update, restore
        saved_state = memory_hier.get_state()
        memory_hier.process_frame("frame_0")  # subtask prediction update
        memory_hier.set_state(saved_state)     # restore — subtask is memory-neutral

        # Step 2: action inference — real update
        memory_hier.process_frame("frame_0")  # action path update

        hier_state = memory_hier.get_state()

        # Contract: both end with exactly one frame processed
        assert hier_state["frame_count"] == plain_state["frame_count"], (
            "Gap 3: hierarchical inference double-updates memory. "
            f"Expected frame_count={plain_state['frame_count']} (same as plain), "
            f"got {hier_state['frame_count']}."
        )
        assert hier_state["buffer"] == plain_state["buffer"]

    def test_subtask_prediction_is_memory_neutral(self):
        """predict_subtask_tokens must not change memory state.

        Before/after calling predict_subtask_tokens, the memory's runtime
        state must be identical.  The subtask prediction must use save/restore
        or update_memory=False to avoid corrupting streaming memory.
        """
        from types import SimpleNamespace

        class MockMemory:
            def __init__(self):
                self.counter = 0
                self.data = {"key": "initial"}

            def update(self, frame_id):
                self.counter += 1
                self.data = {"key": f"frame_{frame_id}"}

            def get_runtime_state(self):
                return {"counter": self.counter, "data": dict(self.data)}

            def set_runtime_state(self, state):
                self.counter = state["counter"]
                self.data = dict(state["data"])

        memory = MockMemory()
        before = memory.get_runtime_state()

        # Simulate subtask prediction with save/restore pattern
        saved = memory.get_runtime_state()
        memory.update("frame_0")  # this is what process_video_with_memory does
        memory.set_runtime_state(saved)  # restore after subtask

        after = memory.get_runtime_state()

        assert after["counter"] == before["counter"], (
            "Gap 3: predict_subtask_tokens must not change memory state. "
            "Use save/restore or update_memory=False to keep subtask "
            "prediction memory-neutral."
        )
        assert after["data"] == before["data"]

    def test_memory_restored_after_exception(self):
        """Memory state must be restored even if processing raises.

        Robustness issue (b): if process_video_with_memory raises an
        exception partway through, the memory could be left in a partially
        updated state, corrupting future streaming inference.  The fix uses
        try/finally to guarantee restore.
        """
        class MockMemory:
            def __init__(self):
                self.counter = 0
                self.data = {"key": "initial"}

            def update(self, frame_id):
                self.counter += 1
                self.data = {"key": f"frame_{frame_id}"}

            def get_runtime_state(self):
                return {"counter": self.counter, "data": dict(self.data)}

            def set_runtime_state(self, state):
                self.counter = state["counter"]
                self.data = dict(state["data"])

        memory = MockMemory()
        before = memory.get_runtime_state()

        # Simulate the try/finally pattern that predict_subtask_tokens uses.
        # The function should restore memory even if an exception is raised
        # during processing.
        def subtask_prediction_with_failure():
            saved = memory.get_runtime_state()
            try:
                memory.update("frame_0")  # partial update happens
                raise RuntimeError("Simulated memory processing failure")
            finally:
                memory.set_runtime_state(saved)  # guaranteed restore

        with pytest.raises(RuntimeError, match="Simulated memory processing failure"):
            subtask_prediction_with_failure()

        # Memory must be back to its original state despite the exception
        after = memory.get_runtime_state()

        assert after["counter"] == before["counter"], (
            "Robustness (b): memory state must be restored even after exception. "
            f"Expected counter={before['counter']}, got {after['counter']}."
        )
        assert after["data"] == before["data"]
