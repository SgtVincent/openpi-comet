"""Tests for the batched fp32 ZeRO gradient-norm replacement.

Validates that ``_patch_deepspeed_grad_norm`` preserves the semantics of
DeepSpeed's ``get_grad_norm_direct`` while dropping the per-tensor float64
upcast:

1. the env gate defaults to off
2. the patched norm matches the float64 original within float32 tolerance
3. parameter filtering (model-parallel rank, pipeline-replicated) is preserved
4. the cross-rank SUM reduction is still issued exactly once
5. non-finite gradients still produce the ``-1.0`` sentinel
6. the empty-gradient case returns zero rather than raising
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest.mock as mock

import pytest
import torch


def _import_train_accelerate():
    repo_root = Path(__file__).resolve().parent.parent
    for sub in ("src", "scripts", ""):
        path = str(repo_root / sub) if sub else str(repo_root)
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location(
        "train_accelerate", repo_root / "scripts" / "train_accelerate.py"
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    return module


_ta = _import_train_accelerate()
HAS_TA = _ta is not None


def _reference_norm(grads: list[torch.Tensor], norm_type: float = 2.0) -> torch.Tensor:
    """DeepSpeed's original float64 formulation, for comparison."""
    norms = [torch.linalg.vector_norm(g.double().detach(), ord=norm_type) for g in grads]
    return torch.stack(norms).square().sum().float().pow(1.0 / norm_type)


class _FakeZeroOptimizer:
    """Minimal stand-in exposing only what get_grad_norm_direct touches."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.model_parallel_rank = 0
        self.dp_process_group = None
        self.model_parallel_all_reduce_calls = 0

    def _model_parallel_all_reduce(self, tensor, op):  # noqa: ARG002
        self.model_parallel_all_reduce_calls += 1
        return tensor


@pytest.mark.skipif(not HAS_TA, reason="train_accelerate not importable")
class TestFastGradNormGate:
    def test_defaults_to_disabled(self, monkeypatch):
        monkeypatch.delenv("OPENPI_DS_FAST_GRAD_NORM", raising=False)
        assert _ta._fast_grad_norm_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes"])
    def test_enabled_values(self, monkeypatch, value):
        monkeypatch.setenv("OPENPI_DS_FAST_GRAD_NORM", value)
        assert _ta._fast_grad_norm_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", ""])
    def test_disabled_values(self, monkeypatch, value):
        monkeypatch.setenv("OPENPI_DS_FAST_GRAD_NORM", value)
        assert _ta._fast_grad_norm_enabled() is False


@pytest.mark.skipif(not HAS_TA, reason="train_accelerate not importable")
class TestPatchedGradNorm:
    """Exercise the patched implementation against the float64 original."""

    def _patched_fn(self):
        pytest.importorskip("deepspeed")
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

        original = DeepSpeedZeroOptimizer.get_grad_norm_direct
        # Force a fresh install so repeated test runs are deterministic.
        if getattr(DeepSpeedZeroOptimizer, "_openpi_fast_grad_norm_patched", False):
            DeepSpeedZeroOptimizer._openpi_fast_grad_norm_patched = False
        _ta._patch_deepspeed_grad_norm()
        patched = DeepSpeedZeroOptimizer.get_grad_norm_direct
        assert patched is not original, "patch was not installed"
        return patched

    def test_matches_float64_reference(self):
        """fp32 batched norm agrees with the float64 original to ~1e-6."""
        fn = self._patched_fn()
        torch.manual_seed(0)
        grads = [torch.randn(2048, dtype=torch.bfloat16) for _ in range(64)]
        params = [torch.nn.Parameter(torch.empty(0)) for _ in grads]
        opt = _FakeZeroOptimizer()

        with mock.patch("deepspeed.comm.all_reduce", side_effect=lambda t, **k: t):
            got = fn(opt, grads, params, 2)

        expected = _reference_norm(grads)
        rel = abs(got.double() - expected.double()).item() / expected.double().item()
        assert got.dtype == torch.float32
        assert rel < 1e-5, f"relative deviation {rel:.3e} exceeds float32 tolerance"

    def test_mixed_dtypes_are_batched_per_dtype(self):
        """Gradients of differing dtypes must still aggregate correctly."""
        fn = self._patched_fn()
        torch.manual_seed(1)
        grads = [
            torch.randn(512, dtype=torch.bfloat16),
            torch.randn(512, dtype=torch.float32),
            torch.randn(512, dtype=torch.bfloat16),
        ]
        params = [torch.nn.Parameter(torch.empty(0)) for _ in grads]
        opt = _FakeZeroOptimizer()

        with mock.patch("deepspeed.comm.all_reduce", side_effect=lambda t, **k: t):
            got = fn(opt, grads, params, 2)

        expected = _reference_norm(grads)
        rel = abs(got.double() - expected.double()).item() / expected.double().item()
        assert rel < 1e-5

    def test_issues_exactly_one_dp_all_reduce(self):
        """Reduction semantics must be unchanged: one SUM over the DP group."""
        fn = self._patched_fn()
        grads = [torch.randn(128, dtype=torch.bfloat16) for _ in range(4)]
        params = [torch.nn.Parameter(torch.empty(0)) for _ in grads]
        opt = _FakeZeroOptimizer()

        calls = []

        def _record(tensor, **kwargs):
            calls.append(kwargs.get("op"))
            return tensor

        with mock.patch("deepspeed.comm.all_reduce", side_effect=_record):
            fn(opt, grads, params, 2)

        import deepspeed.comm as dist

        assert len(calls) == 1, f"expected 1 dp all_reduce, got {len(calls)}"
        assert calls[0] == dist.ReduceOp.SUM
        assert opt.model_parallel_all_reduce_calls == 1

    def test_skips_non_zero_model_parallel_rank(self):
        """Non-model-parallel params are excluded when rank != 0."""
        fn = self._patched_fn()
        grads = [torch.full((16,), 3.0, dtype=torch.float32)]
        params = [torch.nn.Parameter(torch.empty(0))]
        opt = _FakeZeroOptimizer()
        opt.model_parallel_rank = 1

        with mock.patch("deepspeed.comm.all_reduce", side_effect=lambda t, **k: t):
            got = fn(opt, grads, params, 2)

        assert got.item() == pytest.approx(0.0), "excluded grads must not contribute"

    def test_empty_gradients_return_zero(self):
        fn = self._patched_fn()
        opt = _FakeZeroOptimizer()
        with mock.patch("deepspeed.comm.all_reduce", side_effect=lambda t, **k: t):
            got = fn(opt, [], [], 2)
        assert got.dtype == torch.float32
        assert got.item() == pytest.approx(0.0)

    def test_non_finite_maps_to_sentinel(self):
        """A NaN gradient must still yield DeepSpeed's -1.0 sentinel."""
        fn = self._patched_fn()
        grads = [torch.tensor([float("nan"), 1.0], dtype=torch.float32)]
        params = [torch.nn.Parameter(torch.empty(0))]
        opt = _FakeZeroOptimizer()

        with mock.patch("deepspeed.comm.all_reduce", side_effect=lambda t, **k: t):
            got = fn(opt, grads, params, 2)

        assert got.item() == pytest.approx(-1.0)

    def test_falls_back_on_internal_error(self):
        """Any failure inside the fast path defers to DeepSpeed's version."""
        fn = self._patched_fn()
        grads = [torch.randn(32, dtype=torch.bfloat16)]
        params = [torch.nn.Parameter(torch.empty(0))]
        opt = _FakeZeroOptimizer()

        with mock.patch("deepspeed.comm.all_reduce", side_effect=lambda t, **k: t), \
             mock.patch.object(torch, "_foreach_norm", side_effect=RuntimeError("boom")):
            got = fn(opt, grads, params, 2)

        expected = _reference_norm(grads)
        assert got.double().item() == pytest.approx(expected.double().item(), rel=1e-6)
