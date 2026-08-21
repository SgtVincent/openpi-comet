"""Variant A (FAST discrete action tokens + CE) contract tests.

These pin the invariants that make Variant A a valid A/B counterpart to the
shipping query-MSE variant, and that keep its ground-truth action tokens from
leaking into the flow expert.

Network note: the FAST processor itself lives on HuggingFace, which is blocked
without the corporate proxy, so tokenizer-level tests are skipped when it
cannot be constructed rather than failing the suite.
"""

from __future__ import annotations

import inspect
from types import MethodType
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

BASE_LEN = 4
SUBTASK_MAX = 8
ACTION_MAX = 10
ACTION_TOKEN_FIELDS = (
    "action_tokens",
    "action_token_mask",
    "action_token_loss_mask",
    "action_token_ar_mask",
)


def _make_action_token_observation():
    from openpi.models.model import Observation

    return Observation(
        images={"base_0_rgb": torch.zeros(1, 2, 2, 3)},
        image_masks={"base_0_rgb": torch.ones(1, dtype=torch.bool)},
        state=torch.zeros(1, 7),
        action_tokens=torch.arange(4, dtype=torch.int32).reshape(1, 4),
        action_token_mask=torch.ones(1, 4, dtype=torch.bool),
        action_token_loss_mask=torch.tensor([[False, True, True, True]]),
        action_token_ar_mask=torch.ones(1, 4, dtype=torch.int32),
    )


def _build_sequence(n_subtask: int, n_action: int):
    """Assemble the Variant A backbone layout.

    Mirrors production assembly: embed_prefix emits att=0 for images and prompt
    (one bidirectional block), _embed_conditioning_subtask emits ones_like over
    every physical subtask slot, and _embed_action_tokens likewise emits
    ones_like so the action segment is causal and opens its own block.
    """
    base_att = [0] * BASE_LEN
    base_pad = [1] * BASE_LEN
    sub_att = [1] * SUBTASK_MAX
    sub_pad = [1] * n_subtask + [0] * (SUBTASK_MAX - n_subtask)
    act_att = [1] * ACTION_MAX
    act_pad = [1] * n_action + [0] * (ACTION_MAX - n_action)

    att = torch.tensor([base_att + sub_att + act_att], dtype=torch.bool)
    pad = torch.tensor([base_pad + sub_pad + act_pad], dtype=torch.bool)
    return pad, att, BASE_LEN + SUBTASK_MAX


def _ce_rows(n_valid: int, offset: int, seg_len: int) -> set[int]:
    """Absolute indices of logit rows that actually contribute to a CE term."""
    loss_mask = [False] + [True] * (n_valid - 1) + [False] * (seg_len - n_valid)
    return {offset + t for t, keep in enumerate(loss_mask[1:]) if keep}


class TestVariantAAttentionLayout:
    @pytest.mark.parametrize(("n_subtask", "n_action"), [(3, 6), (8, 10), (5, 1), (1, 4)])
    def test_prefix_never_attends_action_tokens(self, n_subtask, n_action):
        """The teacher-forced action tokens must be invisible to the prefix.

        This is the correctness core of Variant A: the tokens are ground truth,
        so any backward edge into the prefix would let a supervised position see
        its own answer.
        """
        pad, att, prefix_len = _build_sequence(n_subtask, n_action)
        att_2d = make_att_2d_masks(pad, att)[0]
        valid = pad[0]

        leak = att_2d[:prefix_len, prefix_len:] & valid[:prefix_len, None]
        assert not leak.any(), "prefix positions must not attend the action-token segment"

    @pytest.mark.parametrize(("n_subtask", "n_action"), [(3, 6), (8, 10), (1, 4)])
    def test_action_segment_is_strictly_causal(self, n_subtask, n_action):
        """Matches the paper: FAST tokens attend autoregressively on previous ones.

        This is the deliberate difference from Variant B, whose query block is
        bidirectional because it is parallel-decoded and has no autoregressive
        structure to respect.
        """
        pad, att, prefix_len = _build_sequence(n_subtask, n_action)
        att_2d = make_att_2d_masks(pad, att)[0]

        block = att_2d[prefix_len : prefix_len + n_action, prefix_len : prefix_len + n_action]
        assert torch.equal(block, torch.tril(torch.ones_like(block))), (
            "action tokens must be causal within their segment"
        )

    @pytest.mark.parametrize(("n_subtask", "n_action"), [(3, 6), (8, 10)])
    def test_action_rows_see_entire_valid_prefix(self, n_subtask, n_action):
        pad, att, prefix_len = _build_sequence(n_subtask, n_action)
        att_2d = make_att_2d_masks(pad, att)[0]
        valid_prefix = pad[0][:prefix_len]

        seen = att_2d[prefix_len : prefix_len + n_action, :prefix_len][:, valid_prefix]
        assert seen.all(), "action tokens must condition on the whole valid prefix"

    @pytest.mark.parametrize(("n_subtask", "n_action"), [(3, 6), (8, 10), (5, 1)])
    def test_subtask_ce_rows_untouched_by_action_segment(self, n_subtask, n_action):
        """Appending the action segment must not perturb the subtask CE."""
        pad, att, prefix_len = _build_sequence(n_subtask, n_action)
        att_2d = make_att_2d_masks(pad, att)[0]

        for row in sorted(_ce_rows(n_subtask, BASE_LEN, SUBTASK_MAX)):
            assert not att_2d[row, prefix_len:].any(), (
                f"supervised subtask row {row} must not attend the action segment"
            )

    def test_prefix_block_identical_with_and_without_action_segment(self):
        """The expert's truncated prefix KV is unaffected by the action segment."""
        pad, att, prefix_len = _build_sequence(SUBTASK_MAX, ACTION_MAX)
        joint = make_att_2d_masks(pad, att)[0]
        prefix_only = make_att_2d_masks(pad[:, :prefix_len], att[:, :prefix_len])[0]
        assert torch.equal(joint[:prefix_len, :prefix_len], prefix_only)


class TestVariantADeviceRouting:
    def test_action_token_fields_move_together_and_preserve_none(self):
        """CPU-safe regression for the step-0 multi-GPU device mismatch."""
        from scripts import train_accelerate

        observation = _make_action_token_observation()
        moved = train_accelerate._move_observation_to_device(observation, torch.device("meta"))

        for field in ACTION_TOKEN_FIELDS:
            assert getattr(moved, field).device.type == "meta", field

        without_action_targets = observation.replace(**{field: None for field in ACTION_TOKEN_FIELDS})
        moved_without_targets = train_accelerate._move_observation_to_device(
            without_action_targets, torch.device("meta")
        )
        for field in ACTION_TOKEN_FIELDS:
            assert getattr(moved_without_targets, field) is None, field

    def test_train_and_validation_share_action_token_transfer_helper(self):
        """Both explicit Observation.replace paths must use the tested helper."""
        from scripts import train_accelerate

        for function in (train_accelerate.train_loop, train_accelerate.run_validation):
            source = inspect.getsource(function)
            assert source.count("_move_observation_to_device(observation, accelerator.device)") == 1, (
                function.__name__
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_cuda_embedding_receives_cuda_action_indices(self):
        """Exercise the original CPU-index/GPU-weight failure boundary on CUDA."""
        from openpi.models_pytorch.pi05_ki_joint_fast import PI05KIJointFastPytorch
        from scripts import train_accelerate

        class _FakePaliGemmaWithExpert:
            def __init__(self):
                self.embedding = torch.nn.Embedding(8, 4, device="cuda")
                self.seen_tokens = None

            def embed_language_tokens(self, tokens):
                self.seen_tokens = tokens
                return self.embedding(tokens)

        fake_paligemma = _FakePaliGemmaWithExpert()
        fake_model = type("FakeModel", (), {"paligemma_with_expert": fake_paligemma})()
        observation = train_accelerate._move_observation_to_device(
            _make_action_token_observation(), torch.device("cuda")
        )

        embeddings, pad_masks, attention_masks = PI05KIJointFastPytorch._embed_action_tokens(
            fake_model,
            action_tokens=observation.action_tokens,
            action_token_mask=observation.action_token_mask,
            target_dtype=torch.float32,
        )

        assert fake_paligemma.seen_tokens.device.type == "cuda"
        assert embeddings.device.type == "cuda"
        assert pad_masks.device.type == "cuda"
        assert attention_masks.device.type == "cuda"


class TestVariantAConfig:
    def test_config_registered_and_not_silently_defaulted(self):
        """get_config() falls back to a default instead of raising on typos."""
        from openpi.training import config as _config

        name = "pi05_ki_joint_fast_b1k-full_task-ki_on_bf16"
        resolved = _config.get_config(name)
        assert resolved.name == name, (
            f"config silently fell back to {resolved.name!r}; Variant A is not registered"
        )
        assert resolved.pytorch_model_name == "pi05_ki_joint_fast"

    def test_shared_hyperparameters_match_variant_b(self):
        """Only the backbone action objective may differ between the arms."""
        from openpi.training import config as _config

        a = _config.get_config("pi05_ki_joint_fast_b1k-full_task-ki_on_bf16")
        b = _config.get_config("pi05_ki_joint_query_b1k-full_task-ki_on_bf16")

        for field in (
            "action_horizon",
            "subtask_max_len",
            "knowledge_insulation",
            "truncate_expert_kv",
            "beta_text",
            "flow_loss_weight",
            "action_dim",
            "max_token_len",
        ):
            assert getattr(a.model, field) == getattr(b.model, field), f"{field} drifted"

        for field in ("num_train_steps", "batch_size", "save_interval"):
            assert getattr(a, field) == getattr(b, field), f"{field} drifted"

    def test_truncate_expert_kv_false_is_rejected(self):
        """Variant B tolerates it as an ablation; Variant A must not.

        The action tokens are teacher-forced ground truth, so exposing them to
        the flow expert leaks the target during training while they are absent
        at inference.
        """
        from openpi.models.pi05_ki_joint_fast_config import Pi05KIJointFastConfig

        with pytest.raises(ValueError, match="truncate_expert_kv=True"):
            Pi05KIJointFastConfig(action_horizon=32, truncate_expert_kv=False)

    def test_action_token_max_len_lower_bound(self):
        from openpi.models.pi05_ki_joint_fast_config import Pi05KIJointFastConfig

        with pytest.raises(ValueError, match="action_token_max_len must be >= 4"):
            Pi05KIJointFastConfig(action_horizon=32, action_token_max_len=2)


class TestVariantAModelWiring:
    def test_variant_a_declares_no_query_parameters(self):
        """Variant A must not carry Variant B's query-MSE parameters."""
        from openpi.models_pytorch.pi05_ki_joint_fast import PI05KIJointFastPytorch
        from openpi.models_pytorch.pi05_ki_joint_query import PI05KIJointQueryPytorch

        assert PI05KIJointQueryPytorch._uses_learned_query_tokens is True
        assert PI05KIJointFastPytorch._uses_learned_query_tokens is False

    def test_variant_b_code_paths_are_blocked(self):
        """Calling a query-MSE hook on Variant A must fail loudly."""
        from openpi.models_pytorch.pi05_ki_joint_fast import PI05KIJointFastPytorch

        for name in ("_embed_query_tokens", "_compute_query_mse_loss"):
            with pytest.raises(NotImplementedError, match="Variant B"):
                getattr(PI05KIJointFastPytorch, name)(object())


class TestVariantAEvalMetrics:
    @staticmethod
    def _make_fast_eval_model():
        from openpi.models_pytorch.pi05_ki_joint_fast import PI05KIJointFastPytorch

        class _FakeLanguageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(16, 6)
                self.config = SimpleNamespace(_attn_implementation=None)

        class _FakePaliGemma(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = _FakeLanguageModel()

        class _FakePaliGemmaWithExpert(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.paligemma = _FakePaliGemma()

            def embed_language_tokens(self, tokens):
                return self.paligemma.language_model.embed_tokens(tokens)

            def forward(self, *, inputs_embeds, **kwargs):
                del kwargs
                return (inputs_embeds[0], None), None

        model = PI05KIJointFastPytorch.__new__(PI05KIJointFastPytorch)
        torch.nn.Module.__init__(model)
        model.beta_text = 1.5
        model.beta_action = 2.0
        model.paligemma_with_expert = _FakePaliGemmaWithExpert()

        def _preprocess_observation(self, observation, *, train):
            del self, observation
            assert train is False
            return None, None, None, None, None

        def _embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
            del images, img_masks, lang_tokens, lang_masks
            base = torch.zeros(1, 2, 6)
            pad = torch.ones(1, 2, dtype=torch.bool)
            att = torch.zeros(1, 2, dtype=torch.bool)
            return base, pad, att

        model._preprocess_observation = MethodType(_preprocess_observation, model)
        model.embed_prefix = MethodType(_embed_prefix, model)
        model._prepare_attention_masks_4d = MethodType(lambda self, mask: mask, model)
        model.compute_expert_loss = MethodType(
            lambda self, observation, actions: {
                "expert_loss": torch.tensor(3.0, requires_grad=True),
                "flow_loss": torch.tensor(0.25, requires_grad=True),
            },
            model,
        )
        return model

    def test_fast_train_reports_only_action_ce_objective(self):
        """FAST train metrics must use the same arm-correct naming as validation."""
        from openpi.models_pytorch.pi05_ki_joint_fast import PI05KIJointFastPytorch

        torch.manual_seed(0)
        model = self._make_fast_eval_model()

        def _preprocess_observation(self, observation, *, train):
            del self, observation
            assert train is True
            return None, None, None, None, None

        model._preprocess_observation = MethodType(_preprocess_observation, model)
        observation = SimpleNamespace(
            action_tokens=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
            action_token_mask=torch.ones(1, 4, dtype=torch.bool),
            action_token_loss_mask=torch.tensor([[False, True, True, True]]),
        )

        metrics = PI05KIJointFastPytorch.compute_backbone_losses(
            model, observation, torch.zeros(1, 2, 2)
        )

        assert set(metrics) == {"backbone_loss", "ce_loss", "action_ce_loss"}
        assert "query_mse_loss" not in metrics
        assert metrics["backbone_loss"].requires_grad
        assert not metrics["action_ce_loss"].requires_grad
        expected_backbone = model.beta_text * metrics["ce_loss"] + model.beta_action * metrics["action_ce_loss"]
        assert torch.allclose(metrics["backbone_loss"].detach(), expected_backbone)

    def test_fast_eval_uses_action_tokens_and_reports_arm_correct_metrics(self):
        """FAST validation must never dispatch through learned-query hooks."""
        from openpi.models_pytorch.pi05_ki_joint_fast import PI05KIJointFastPytorch

        torch.manual_seed(0)
        model = self._make_fast_eval_model()
        learned_query_hook = mock.Mock(side_effect=AssertionError("learned-query path reached"))
        model._embed_query_tokens = learned_query_hook
        observation = SimpleNamespace(
            action_tokens=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
            action_token_mask=torch.ones(1, 4, dtype=torch.bool),
            action_token_loss_mask=torch.tensor([[False, True, True, True]]),
        )
        actions = torch.zeros(1, 2, 2)

        metrics = PI05KIJointFastPytorch.compute_eval_metrics(model, observation, actions)

        assert set(metrics) == {
            "total_loss",
            "backbone_loss",
            "expert_loss",
            "ce_loss",
            "action_ce_loss",
            "flow_loss",
            "subtask_accuracy",
            "action_token_accuracy",
            "flow_mse",
        }
        assert "query_mse_loss" not in metrics
        assert "query_l1" not in metrics
        learned_query_hook.assert_not_called()

        for name, value in metrics.items():
            assert value.numel() == 1, name
            assert torch.isfinite(value), name
            assert not value.requires_grad, name

        expected_backbone = model.beta_text * metrics["ce_loss"] + model.beta_action * metrics["action_ce_loss"]
        assert torch.allclose(metrics["backbone_loss"], expected_backbone)
        assert torch.allclose(
            metrics["total_loss"], metrics["backbone_loss"] + metrics["expert_loss"]
        )
        assert torch.equal(metrics["flow_mse"], metrics["flow_loss"])
        assert 0.0 <= metrics["action_token_accuracy"].item() <= 1.0

    def test_variant_b_eval_metric_contract_is_unchanged(self):
        from openpi.models_pytorch.pi05_ki_joint_query import PI05KIJointQueryPytorch

        model = PI05KIJointQueryPytorch.__new__(PI05KIJointQueryPytorch)
        torch.nn.Module.__init__(model)
        model._compute_backbone_eval_metrics = MethodType(
            lambda self, observation, actions: (
                torch.tensor(2.0, requires_grad=True),
                torch.tensor(0.5),
                torch.tensor(1.5),
                torch.tensor(0.75),
                torch.tensor(0.25),
            ),
            model,
        )
        model.compute_expert_loss = MethodType(
            lambda self, observation, actions: {
                "expert_loss": torch.tensor(3.0, requires_grad=True),
                "flow_loss": torch.tensor(0.3, requires_grad=True),
            },
            model,
        )

        metrics = PI05KIJointQueryPytorch.compute_eval_metrics(model, object(), object())

        assert set(metrics) == {
            "total_loss",
            "backbone_loss",
            "expert_loss",
            "ce_loss",
            "query_mse_loss",
            "flow_loss",
            "subtask_accuracy",
            "query_l1",
            "flow_mse",
        }
        assert "action_ce_loss" not in metrics
        assert "action_token_accuracy" not in metrics
        assert metrics["query_mse_loss"].item() == pytest.approx(1.5)
        assert metrics["query_l1"].item() == pytest.approx(0.25)
        assert metrics["total_loss"].item() == pytest.approx(5.0)


class TestArmAwareTrainLogging:
    def test_structured_metrics_keep_fast_and_query_objectives_separate(self):
        from scripts import train_accelerate

        fast_metrics = {"ce_loss": 0.5, "action_ce_loss": 8.0}
        train_accelerate._add_pi05_ki_structured_backbone_metrics(fast_metrics, 9.0)
        assert fast_metrics["loss_action_ce"] == pytest.approx(8.0)
        assert "query_mse_loss" not in fast_metrics
        assert "loss_query_mse" not in fast_metrics

        query_metrics = {"ce_loss": 0.5, "query_mse_loss": 0.25}
        train_accelerate._add_pi05_ki_structured_backbone_metrics(query_metrics, 1.0)
        assert query_metrics["loss_query_mse"] == pytest.approx(0.25)
        assert "action_ce_loss" not in query_metrics
        assert "loss_action_ce" not in query_metrics

    def test_wandb_mapping_uses_only_the_active_arm_objective(self):
        from scripts import train_accelerate

        fast_payload = {}
        train_accelerate._update_pi05_ki_wandb_loss_metrics(
            fast_payload,
            [{"loss_backbone": 9.0, "loss_ce": 0.5, "loss_action_ce": 8.0}],
        )
        assert fast_payload["loss/action_ce"] == pytest.approx(8.0)
        assert "loss/query_mse" not in fast_payload

        query_payload = {}
        train_accelerate._update_pi05_ki_wandb_loss_metrics(
            query_payload,
            [{"loss_backbone": 1.0, "loss_ce": 0.5, "loss_query_mse": 0.25}],
        )
        assert query_payload["loss/query_mse"] == pytest.approx(0.25)
        assert "loss/action_ce" not in query_payload


class TestFastActionTokenization:
    """Tokenizer-level contract. Skipped when the FAST processor is unreachable."""

    @staticmethod
    def _tokenizer():
        from openpi.models.tokenizer import FASTTokenizer

        try:
            return FASTTokenizer(max_len=512)
        except Exception as exc:  # pragma: no cover - network dependent
            pytest.skip(f"FAST processor unavailable (needs the corporate proxy): {exc}")

    @staticmethod
    def _tokenizer_with_action_length(action_length: int):
        """Build a deterministic tokenizer without external processor access."""
        from openpi.models.tokenizer import FASTTokenizer

        class _FakePaliGemmaTokenizer:
            @staticmethod
            def bos_id():
                return 1

            @staticmethod
            def eos_id():
                return 2

            @staticmethod
            def vocab_size():
                return 257_152

        tokenizer = FASTTokenizer.__new__(FASTTokenizer)
        tokenizer._fast_tokenizer = lambda batch: [np.arange(action_length, dtype=np.int32)]
        tokenizer._paligemma_tokenizer = _FakePaliGemmaTokenizer()
        tokenizer._fast_skip_tokens = 128
        return tokenizer

    @staticmethod
    def _smooth_chunk(horizon: int = 32, action_dim: int = 23) -> np.ndarray:
        """A temporally smooth chunk, i.e. what real robot actions look like.

        Do NOT use i.i.d. noise here: FAST compresses along time, so white noise
        is the worst case and yields ~300-500 tokens, whereas measured real B1K
        chunks land at p50=20 / p99=47. Testing with noise would silently
        exercise the truncation path instead of the normal contract.
        """
        t = np.linspace(0.0, 1.0, horizon)[:, None]
        phase = np.arange(action_dim)[None, :]
        return np.clip(0.5 * np.sin(2 * np.pi * t + phase), -1, 1).astype(np.float32)

    def test_contract_matches_tokenize_subtask(self):
        tk = self._tokenizer()
        chunk = self._smooth_chunk()

        tokens, mask, ar_mask, loss_mask = tk.tokenize_action_chunk(chunk, max_len=256)
        assert tokens.shape == mask.shape == ar_mask.shape == loss_mask.shape == (256,)
        assert tokens.dtype == np.int32
        assert mask.dtype == np.bool_

        n = int(mask.sum())
        assert n >= 3, "expected BOS + at least one action token + EOS"
        assert tokens[0] == tk._paligemma_tokenizer.bos_id()
        assert tokens[n - 1] == tk._paligemma_tokenizer.eos_id()

        # Row t predicts token t+1, so BOS is unsupervised and there are n-1
        # supervised rows -- identical to tokenize_subtask.
        assert not loss_mask[0]
        assert int(loss_mask.sum()) == n - 1

        # Padding must be inert on every channel.
        assert not mask[n:].any()
        assert (tokens[n:] == 0).all()
        assert (ar_mask[n:] == 0).all()
        assert not loss_mask[n:].any()

    def test_formal_capacity_preserves_observed_73_token_target(self):
        """The observed maximum includes BOS/EOS and must remain complete."""
        tk = self._tokenizer_with_action_length(71)
        chunk = np.zeros((32, 23), dtype=np.float32)

        tokens, mask, ar_mask, loss_mask = tk.tokenize_action_chunk(chunk, max_len=96)

        assert tokens.shape == mask.shape == ar_mask.shape == loss_mask.shape == (96,)
        assert int(mask.sum()) == 73
        assert tokens[0] == tk._paligemma_tokenizer.bos_id()
        assert tokens[72] == tk._paligemma_tokenizer.eos_id()
        assert int(loss_mask.sum()) == 72
        assert not mask[73:].any()

    def test_over_limit_action_target_fails_instead_of_truncating(self):
        """An unexpected longer chunk must never return a partial target."""
        tk = self._tokenizer_with_action_length(95)  # 97 including BOS/EOS.
        chunk = np.zeros((32, 23), dtype=np.float32)

        with pytest.raises(
            ValueError,
            match=r"produced 97 tokens.*action_token_max_len=96.*Refusing to truncate",
        ):
            tk.tokenize_action_chunk(chunk, max_len=96)

    def test_rejects_wrong_rank(self):
        tk = self._tokenizer()
        with pytest.raises(ValueError, match=r"\[action_horizon, action_dim\]"):
            tk.tokenize_action_chunk(np.zeros((4, 4, 4), dtype=np.float32))

    def test_action_ids_avoid_the_configured_image_token(self):
        """The mapping reaches below the <loc> range, so pin the real constraint.

        FAST's vocabulary is 2048 while PaliGemma's <loc> block is only 1024
        slots, so ``vocab_size - 1 - skip - t`` reaches into rare text ids. That
        is the upstream pi0-FAST convention and is harmless here, but it must
        never collide with the id the model treats as the image token.
        """
        tk = self._tokenizer()
        rng = np.random.default_rng(1)
        image_token_index = 257152  # gemma_pytorch sets this on the HF config.

        for _ in range(4):
            chunk = np.clip(rng.normal(0, 0.3, (32, 23)), -1, 1).astype(np.float32)
            tokens, mask, _, _ = tk.tokenize_action_chunk(chunk, max_len=1024)
            ids = tokens[mask]
            assert (ids < image_token_index).all(), "action id collided with the image token"
            assert (ids >= 0).all()
