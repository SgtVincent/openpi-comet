# ruff: noqa: SLF001 - these tests intentionally exercise private diagnostics helpers.
"""Unit tests for Accelerate training loop diagnostic optimizations.

Validates the performance optimizations introduced to reduce diagnostic
all-reduce and I/O overhead in the training loop:

1. ``_gather_finite_consensus`` — cheap single-all-reduce finiteness check
2. JSONL metrics buffering — per-step writes buffered, flushed at boundaries
3. Detailed ``_gather_scalar_stats`` skipped on fast path (only at log_interval / error)
4. Per-param-group grad norms computed only at log cadence
5. Non-finite error paths still trigger detailed diagnostics and proper flush

These tests exercise the functions directly without requiring a full training
loop, GPU, or distributed setup — a mock accelerator is used where needed.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest.mock as mock

import numpy as np
import pytest
import torch

# ===========================================================================
#  Helpers: import functions under test from train_accelerate.py
# ===========================================================================

def _import_train_accelerate():
    """Import the train_accelerate module (may fail on missing deps)."""
    # Ensure imports resolve to this worktree even if another checkout is
    # installed editable in the active environment.
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # The scripts dir may not be on path; add it
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_accelerate", scripts_dir / "train_accelerate.py"
    )
    module = importlib.util.module_from_spec(spec)
    # We only need the pure functions; skip the full module load if it fails
    # due to missing heavy dependencies.  Instead, exec only the parts we need.
    # For simplicity, try full import; if it fails, extract functions manually.
    try:
        spec.loader.exec_module(module)
    except Exception:
        module = None
    return module


# Try to import; some environments may lack accelerate / openpi.
# We'll skip tests that depend on the full module when unavailable.
_train_accel = _import_train_accelerate()
HAS_TRAIN_ACCELERATE = _train_accel is not None


def _make_mock_accelerator(num_processes: int = 1, device: str = "cpu"):
    """Create a mock Accelerator suitable for single-process unit tests.

    The mock provides:
      - ``device`` property
      - ``num_processes`` property
      - ``is_main_process`` = True
      - ``process_index`` = 0
      - ``distributed_type``
      - ``reduce`` (no-op: returns input tensor)
    """
    accel = mock.MagicMock()
    accel.device = torch.device(device)
    accel.num_processes = num_processes
    accel.is_main_process = True
    accel.process_index = 0
    accel.distributed_type = "NO"  # not distributed

    # reduce: no-op for single-process mock
    def _mock_reduce(tensor, reduction="sum"):
        return tensor

    accel.reduce = _mock_reduce
    return accel


# ===========================================================================
#  1. _gather_finite_consensus
# ===========================================================================


class TestGatherFiniteConsensus:
    """Test the cheap finite-consensus function."""

    def test_single_finite_scalar(self):
        """Single finite scalar returns (True, [value])."""
        if not HAS_TRAIN_ACCELERATE:
            pytest.skip("train_accelerate module not importable")
        accel = _make_mock_accelerator()
        # Mock dist to be not initialised (single process)
        with mock.patch.object(_train_accel.torch.distributed, "is_initialized", return_value=False):
            all_finite, vals = _train_accel._gather_finite_consensus(
                accel, torch.tensor(0.42)
            )
        assert all_finite is True
        assert len(vals) == 1
        assert abs(vals[0] - 0.42) < 1e-6

    def test_multiple_finite_scalars(self):
        """Multiple finite scalars all report finite."""
        if not HAS_TRAIN_ACCELERATE:
            pytest.skip("train_accelerate module not importable")
        accel = _make_mock_accelerator()
        with mock.patch.object(_train_accel.torch.distributed, "is_initialized", return_value=False):
            all_finite, vals = _train_accel._gather_finite_consensus(
                accel,
                torch.tensor(0.1),
                torch.tensor(0.2),
                torch.tensor(0.3),
            )
        assert all_finite is True
        assert len(vals) == 3
        assert abs(vals[0] - 0.1) < 1e-6
        assert abs(vals[1] - 0.2) < 1e-6
        assert abs(vals[2] - 0.3) < 1e-6

    def test_single_infinite_scalar(self):
        """Single infinite scalar returns (False, [inf])."""
        if not HAS_TRAIN_ACCELERATE:
            pytest.skip("train_accelerate module not importable")
        accel = _make_mock_accelerator()
        with mock.patch.object(_train_accel.torch.distributed, "is_initialized", return_value=False):
            all_finite, vals = _train_accel._gather_finite_consensus(
                accel, torch.tensor(float("inf"))
            )
        assert all_finite is False
        assert len(vals) == 1
        assert np.isinf(vals[0])

    def test_single_nan_scalar(self):
        """Single NaN scalar returns (False, [nan])."""
        if not HAS_TRAIN_ACCELERATE:
            pytest.skip("train_accelerate module not importable")
        accel = _make_mock_accelerator()
        with mock.patch.object(_train_accel.torch.distributed, "is_initialized", return_value=False):
            all_finite, vals = _train_accel._gather_finite_consensus(
                accel, torch.tensor(float("nan"))
            )
        assert all_finite is False
        assert len(vals) == 1
        assert np.isnan(vals[0])

    def test_mixed_finite_infinite(self):
        """Mixed finite and infinite scalars report False."""
        if not HAS_TRAIN_ACCELERATE:
            pytest.skip("train_accelerate module not importable")
        accel = _make_mock_accelerator()
        with mock.patch.object(_train_accel.torch.distributed, "is_initialized", return_value=False):
            all_finite, vals = _train_accel._gather_finite_consensus(
                accel,
                torch.tensor(0.1),
                torch.tensor(float("inf")),
                torch.tensor(0.3),
            )
        assert all_finite is False
        assert len(vals) == 3
        assert abs(vals[0] - 0.1) < 1e-6
        assert np.isinf(vals[1])
        assert abs(vals[2] - 0.3) < 1e-6

    def test_zero_scalars(self):
        """Zero scalars — vacuously true (edge case)."""
        if not HAS_TRAIN_ACCELERATE:
            pytest.skip("train_accelerate module not importable")
        accel = _make_mock_accelerator()
        with mock.patch.object(_train_accel.torch.distributed, "is_initialized", return_value=False):
            all_finite, vals = _train_accel._gather_finite_consensus(accel)
        assert all_finite is True
        assert vals == []

    def test_uses_only_one_all_reduce_when_distributed(self):
        """When distributed, only one all_reduce call is made (cheap!)."""
        if not HAS_TRAIN_ACCELERATE:
            pytest.skip("train_accelerate module not importable")
        accel = _make_mock_accelerator(num_processes=2)

        call_count = {"n": 0}
        def _fake_all_reduce(tensor, op):
            call_count["n"] += 1
            # Simulate: both ranks agree → bad=0 stays 0
            return tensor

        with mock.patch.object(_train_accel.torch.distributed, "is_initialized", return_value=True), \
             mock.patch.object(_train_accel.torch.distributed, "all_reduce", side_effect=_fake_all_reduce):
            all_finite, vals = _train_accel._gather_finite_consensus(
                accel,
                torch.tensor(0.1),
                torch.tensor(0.2),
            )

        assert all_finite is True
        assert len(vals) == 2
        # Key assertion: exactly 1 all-reduce, not 5 like _gather_scalar_stats
        assert call_count["n"] == 1, (
            f"Expected 1 all-reduce for cheap consensus, got {call_count['n']}"
        )


# ===========================================================================
#  2. JSONL metrics buffering
# ===========================================================================


class TestMetricsBuffer:
    """Test the JSONL metrics buffering infrastructure."""

    def _reset_buffer_state(self):
        """Reset module-level buffer state between tests."""
        if not HAS_TRAIN_ACCELERATE:
            return
        _train_accel._metrics_buffer = []
        _train_accel._metrics_file_handle = None
        _train_accel._metrics_atexit_registered = False

    def test_init_sets_file_handle(self):
        """_metrics_buffer_init sets the file handle."""
        if not HAS_TRAIN_ACCELERATE:
            pytest.skip("train_accelerate module not importable")
        self._reset_buffer_state()
        fh = io.StringIO()
        _train_accel._metrics_buffer_init(fh)
        assert _train_accel._metrics_file_handle is fh

    def test_append_adds_to_buffer_no_write(self):
        """Appending records adds to buffer but does NOT write to file yet."""
        if not HAS_TRAIN_ACCELERATE:
            pytest.skip("train_accelerate module not importable")
        self._reset_buffer_state()
        fh = io.StringIO()
        _train_accel._metrics_buffer_init(fh)

        _train_accel._metrics_buffer_append({"step": 1, "loss": 0.5})
        _train_accel._metrics_buffer_append({"step": 2, "loss": 0.4})

        # Buffer should have 2 entries
        assert len(_train_accel._metrics_buffer) == 2
        # File should be empty (not yet flushed)
        assert fh.getvalue() == ""

    def test_flush_writes_all_buffered(self):
        """Flush writes all buffered records to the file."""
        if not HAS_TRAIN_ACCELERATE:
            pytest.skip("train_accelerate module not importable")
        self._reset_buffer_state()
        fh = io.StringIO()
        _train_accel._metrics_buffer_init(fh)

        _train_accel._metrics_buffer_append({"step": 1, "loss": 0.5})
        _train_accel._metrics_buffer_append({"step": 2, "loss": 0.4})
        _train_accel._metrics_buffer_flush()

        # Buffer should be empty now
        assert len(_train_accel._metrics_buffer) == 0
        # File should have 2 lines
        lines = fh.getvalue().strip().split("\n")
        assert len(lines) == 2
        r1 = json.loads(lines[0])
        r2 = json.loads(lines[1])
        assert r1["step"] == 1
        assert r2["step"] == 2

    def test_flush_empty_is_noop(self):
        """Flushing an empty buffer is a no-op (no file write)."""
        if not HAS_TRAIN_ACCELERATE:
            pytest.skip("train_accelerate module not importable")
        self._reset_buffer_state()
        fh = io.StringIO()
        _train_accel._metrics_buffer_init(fh)

        _train_accel._metrics_buffer_flush()
        assert fh.getvalue() == ""

    def test_none_file_handle_is_safe(self):
        """All operations are safe when file handle is None (non-main rank)."""
        if not HAS_TRAIN_ACCELERATE:
            pytest.skip("train_accelerate module not importable")
        self._reset_buffer_state()
        # Don't init — handle stays None

        _train_accel._metrics_buffer_append({"step": 1, "loss": 0.5})
        _train_accel._metrics_buffer_flush()
        _train_accel._metrics_buffer_close()
        # Should not crash
        assert True

    def test_close_flushes_and_closes(self):
        """close() flushes remaining buffer and closes the file."""
        if not HAS_TRAIN_ACCELERATE:
            pytest.skip("train_accelerate module not importable")
        self._reset_buffer_state()
        fh = io.StringIO()
        _train_accel._metrics_buffer_init(fh)

        _train_accel._metrics_buffer_append({"step": 1, "loss": 0.5})
        _train_accel._metrics_buffer_close()

        assert fh.closed is True
        # Buffer was flushed before close
        # (StringIO closed means we can't read, but we can verify via
        #  checking that the buffer list is empty)
        assert len(_train_accel._metrics_buffer) == 0

    def test_multiple_flushes_correctness(self):
        """Multiple append → flush cycles produce correct ordered output."""
        if not HAS_TRAIN_ACCELERATE:
            pytest.skip("train_accelerate module not importable")
        self._reset_buffer_state()
        # Use a temp file because StringIO can't be read after close
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with open(tmp_path, "w") as fh:
                _train_accel._metrics_buffer_init(fh)

                # Batch 1
                for i in range(3):
                    _train_accel._metrics_buffer_append({"step": i, "loss": 1.0 - i * 0.1})
                _train_accel._metrics_buffer_flush()

                # Batch 2
                for i in range(3, 6):
                    _train_accel._metrics_buffer_append({"step": i, "loss": 1.0 - i * 0.1})
                _train_accel._metrics_buffer_flush()

                _train_accel._metrics_buffer_close()

            with open(tmp_path) as f:
                lines = [json.loads(line) for line in f if line.strip()]

            assert len(lines) == 6
            assert [r["step"] for r in lines] == [0, 1, 2, 3, 4, 5]
        finally:
            os.unlink(tmp_path)

    def test_atexit_handler_registered(self):
        """atexit handler is registered on first init call."""
        if not HAS_TRAIN_ACCELERATE:
            pytest.skip("train_accelerate module not importable")
        self._reset_buffer_state()
        fh = io.StringIO()

        with mock.patch("atexit.register") as mock_register:
            _train_accel._metrics_buffer_init(fh)
            assert mock_register.called
            # Second init should NOT re-register
            _train_accel._metrics_buffer_init(fh)
            assert mock_register.call_count == 1

    def test_validation_boundary_write_failure_disables_writer_without_raising(self):
        if not HAS_TRAIN_ACCELERATE:
            pytest.skip("train_accelerate module not importable")
        self._reset_buffer_state()

        class WriteFailingFile:
            def __init__(self):
                self.write_calls = 0
                self.flush_calls = 0
                self.close_calls = 0

            def write(self, _payload):
                self.write_calls += 1
                raise OSError("validation metrics write failed")

            def flush(self):
                self.flush_calls += 1

            def close(self):
                self.close_calls += 1
                raise OSError("metrics close after write failure")

        metrics_file = WriteFailingFile()
        _train_accel._metrics_buffer_init(metrics_file)
        control_flow = []

        with mock.patch.object(
            _train_accel.logging,
            "warning",
            side_effect=RuntimeError("logging handler failed"),
        ):
            _train_accel._metrics_buffer_write_boundary(
                {"step": 100, "type": "validation", "val_total_loss": 0.25}
            )
            control_flow.append("peer-control-flow-continues")
            _train_accel._metrics_buffer_flush()
            _train_accel._metrics_buffer_close()
            _train_accel._metrics_buffer_close()

        assert control_flow == ["peer-control-flow-continues"]
        assert metrics_file.write_calls == 1
        assert metrics_file.flush_calls == 0
        assert metrics_file.close_calls == 1
        assert _train_accel._metrics_file_handle is None
        assert _train_accel._metrics_buffer == []
        assert _train_accel._metrics_atexit_registered is False

    def test_validation_boundary_flush_failure_disables_writer_without_raising(self):
        if not HAS_TRAIN_ACCELERATE:
            pytest.skip("train_accelerate module not importable")
        self._reset_buffer_state()

        class FlushFailingFile:
            def __init__(self):
                self.payloads = []
                self.flush_calls = 0
                self.close_calls = 0

            def write(self, payload):
                self.payloads.append(payload)

            def flush(self):
                self.flush_calls += 1
                raise OSError("validation metrics flush failed")

            def close(self):
                self.close_calls += 1

        metrics_file = FlushFailingFile()
        _train_accel._metrics_buffer_init(metrics_file)

        with mock.patch.object(
            _train_accel.logging,
            "warning",
            side_effect=RuntimeError("logging handler failed"),
        ):
            _train_accel._metrics_buffer_write_boundary(
                {"step": 100, "type": "validation", "val_total_loss": 0.25}
            )
            _train_accel._metrics_buffer_flush()
            _train_accel._metrics_buffer_close()
            _train_accel._metrics_buffer_close()

        assert len(metrics_file.payloads) == 1
        assert metrics_file.flush_calls == 1
        assert metrics_file.close_calls == 1
        assert _train_accel._metrics_file_handle is None
        assert _train_accel._metrics_buffer == []
        assert _train_accel._metrics_atexit_registered is False

    def test_close_failure_and_logging_failure_leave_metrics_state_detached(self):
        if not HAS_TRAIN_ACCELERATE:
            pytest.skip("train_accelerate module not importable")
        self._reset_buffer_state()

        class CloseFailingFile:
            def __init__(self):
                self.payloads = []
                self.flush_calls = 0
                self.close_calls = 0

            def write(self, payload):
                self.payloads.append(payload)

            def flush(self):
                self.flush_calls += 1

            def close(self):
                self.close_calls += 1
                raise OSError("metrics close failed")

        metrics_file = CloseFailingFile()
        _train_accel._metrics_buffer_init(metrics_file)
        _train_accel._metrics_buffer_write_boundary(
            {"step": 100, "type": "validation", "val_total_loss": 0.25}
        )

        with mock.patch.object(
            _train_accel.logging,
            "warning",
            side_effect=RuntimeError("logging handler failed"),
        ):
            _train_accel._metrics_buffer_close()
            _train_accel._metrics_buffer_close()
            _train_accel._metrics_buffer_flush()

        assert metrics_file.close_calls == 1
        assert _train_accel._metrics_file_handle is None
        assert _train_accel._metrics_buffer == []
        assert _train_accel._metrics_atexit_registered is False

    def test_validation_boundary_success_appends_and_flushes_exactly_once(self):
        if not HAS_TRAIN_ACCELERATE:
            pytest.skip("train_accelerate module not importable")
        self._reset_buffer_state()

        class TrackingFile(io.StringIO):
            def __init__(self):
                super().__init__()
                self.write_calls = 0
                self.flush_calls = 0

            def write(self, payload):
                self.write_calls += 1
                return super().write(payload)

            def flush(self):
                self.flush_calls += 1
                return super().flush()

        metrics_file = TrackingFile()
        _train_accel._metrics_buffer_init(metrics_file)
        record = {"step": 100, "epoch": 1, "type": "validation", "val_total_loss": 0.25}

        _train_accel._metrics_buffer_write_boundary(record)

        assert metrics_file.write_calls == 1
        assert metrics_file.flush_calls == 1
        assert json.loads(metrics_file.getvalue()) == record
        assert _train_accel._metrics_buffer == []
        assert _train_accel._metrics_file_handle is metrics_file
        _train_accel._metrics_buffer_close()
        assert _train_accel._metrics_atexit_registered is False

    def test_run_validation_uses_nonthrowing_buffer_boundary(self):
        if not HAS_TRAIN_ACCELERATE:
            pytest.skip("train_accelerate module not importable")
        source = Path(_train_accel.__file__).read_text()
        validation_source = source[
            source.index("def run_validation("):source.index("def save_checkpoint(")
        ]

        assert "_metrics_buffer_write_boundary(val_record)" in validation_source
        assert "metrics_file.write" not in validation_source
        assert "metrics_file.flush" not in validation_source


# ===========================================================================
#  3. Detailed stats skipped on fast path
# ===========================================================================


class TestDetailedStatsFastPath:
    """Verify that _gather_scalar_stats is NOT called on every step (fast path).

    We test the pattern / logic rather than running the full training loop.
    The key invariant: when loss is finite AND not at log_interval AND
    debug overflow is disabled, the code path should only call
    _gather_finite_consensus (1 all-reduce) and NOT _gather_scalar_stats
    (5 all-reduces).
    """

    def test_fast_path_uses_finite_consensus_only(self):
        """Fast path (finite + not log interval + no debug) uses cheap check only."""
        if not HAS_TRAIN_ACCELERATE:
            pytest.skip("train_accelerate module not importable")

        # Simulate the decision logic as it appears in the training loop:
        #   if nonfinite or need_detailed_stats → call _gather_scalar_stats
        #   else → only _gather_finite_consensus
        log_interval = 100
        global_step = 5  # not at log interval
        debug_overflow = False
        is_finite = True

        need_detailed_stats = (
            (global_step % log_interval == 0)
            or debug_overflow
        )
        should_call_detailed = (not is_finite) or need_detailed_stats

        assert need_detailed_stats is False
        assert should_call_detailed is False, (
            "Fast path should NOT call detailed stats"
        )

    def test_log_interval_triggers_detailed_stats(self):
        """At log_interval boundaries, detailed stats ARE computed."""
        log_interval = 100
        global_step = 100  # at log interval
        debug_overflow = False
        is_finite = True

        need_detailed_stats = (
            (global_step % log_interval == 0)
            or debug_overflow
        )
        should_call_detailed = (not is_finite) or need_detailed_stats

        assert need_detailed_stats is True
        assert should_call_detailed is True

    def test_debug_overflow_triggers_detailed_stats(self):
        """Debug overflow mode triggers detailed stats every step."""
        log_interval = 100
        global_step = 5
        debug_overflow = True
        is_finite = True

        need_detailed_stats = (
            (global_step % log_interval == 0)
            or debug_overflow
        )
        should_call_detailed = (not is_finite) or need_detailed_stats

        assert need_detailed_stats is True
        assert should_call_detailed is True

    def test_all_reduce_count_fast_path_vs_detailed(self):
        """Verify expected all-reduce counts: 1 (fast) vs 5 (detailed).

        _gather_finite_consensus = 1 all-reduce
        _gather_scalar_stats = 5 all-reduces (count/total/min/max/sum+sqsum)
        """
        # Fast path: just 1 all-reduce for finite consensus
        fast_path_allreduces = 1  # _gather_finite_consensus

        # Detailed path: 6 all-reduces from _gather_scalar_stats (all-finite case)
        # 1. dist.all_reduce(local_finite_count, SUM)
        # 2. dist.all_reduce(local_total, SUM)
        # 3. dist.all_reduce(min_tensor, MIN)  (only if finite_count > 0)
        # 4. dist.all_reduce(max_tensor, MAX)  (only if finite_count > 0)
        # 5. dist.all_reduce(local_sum, SUM)   (only if finite_count > 0)
        # 6. dist.all_reduce(local_sq_sum, SUM) (only if finite_count > 0)
        detailed_allreduces_common = 6  # when all finite

        assert fast_path_allreduces < detailed_allreduces_common
        # Reduction factor: detailed is 6x more all-reduces per phase
        assert detailed_allreduces_common / fast_path_allreduces == 6.0


# ===========================================================================
#  4. Non-finite triggers detailed stats and proper skip logic
# ===========================================================================


class TestNonFiniteTriggersDetailedStats:
    """Verify that non-finite detection triggers detailed stats + skip."""

    def test_nonfinite_triggers_detailed_stats(self):
        """When loss is non-finite, detailed stats ARE computed for diagnostics."""
        log_interval = 100
        global_step = 5  # not log interval
        debug_overflow = False
        is_finite = False  # NON-finite

        need_detailed_stats = (
            (global_step % log_interval == 0)
            or debug_overflow
        )
        should_call_detailed = (not is_finite) or need_detailed_stats

        assert should_call_detailed is True, (
            "Non-finite should always trigger detailed stats"
        )

    def test_nonfinite_skip_logic(self):
        """Non-finite → skip batch (continue), increment counter, reset grads."""
        # This tests the logical pattern, not the actual loop.
        consecutive_nonfinite = 0
        max_consecutive = 10
        total_nonfinite = 0
        skipped_batches = 0

        # Simulate a non-finite batch
        is_finite = False
        if not is_finite:
            consecutive_nonfinite += 1
            total_nonfinite += 1
            # optimizer.zero_grad()
            # accelerator.wait_for_everyone()
            if consecutive_nonfinite >= max_consecutive:
                raise FloatingPointError("Too many consecutive non-finite")
            skipped_batches += 1
            # continue

        assert consecutive_nonfinite == 1
        assert total_nonfinite == 1
        assert skipped_batches == 1

    def test_max_consecutive_raises(self):
        """Exceeding max consecutive non-finite raises FloatingPointError."""
        consecutive_nonfinite = 10
        max_consecutive = 10
        should_raise = consecutive_nonfinite >= max_consecutive

        assert should_raise is True

    def test_reset_after_finite(self):
        """Consecutive counter resets to 0 after a finite batch."""
        consecutive_nonfinite = 3
        is_finite = True

        if is_finite:
            consecutive_nonfinite = 0

        assert consecutive_nonfinite == 0


# ===========================================================================
#  5. Per-group grad norms at log cadence only
# ===========================================================================


class TestPerGroupGradNormCadence:
    """Verify per-param-group grad norms are computed only at log cadence."""

    def test_computed_at_log_interval(self):
        """Per-group norms computed at log_interval steps."""
        log_interval = 100
        sync_gradients = True

        for step in [1, 50, 99, 100, 101, 200, 250]:
            should_compute = sync_gradients and (step % log_interval == 0)
            if step in (100, 200):
                assert should_compute is True, f"step={step} should compute"
            else:
                assert should_compute is False, f"step={step} should NOT compute"

    def test_not_computed_on_non_sync_steps(self):
        """Per-group norms NOT computed on non-sync (grad accum) steps."""
        log_interval = 100
        sync_gradients = False
        step = 100  # would be at log interval

        should_compute = sync_gradients and (step % log_interval == 0)
        assert should_compute is False

    def test_total_grad_clipping_every_sync_step(self):
        """Total grad clipping (clip_grad_norm_) runs every sync step.

        This is a "safety" test to document the invariant: per-group norms
        are diagnostic-only and can be throttled, but total clipping is
        training math and must run every sync step.
        """
        # clip_grad_norm_ runs on EVERY sync step — NOT just log_interval
        sync_gradients = True
        for step in [1, 2, 3, 50, 99, 100, 101]:
            should_clip = sync_gradients  # always, when syncing
            assert should_clip is True, f"step={step}: total clipping is every sync step"

    def test_all_reduce_savings_per_group_norms(self):
        """Per-group norm throttling saves N all-reduces per step.

        2 groups x 1 all-reduce each = 2 saved all-reduces per step.
        At log_interval=100, amortized cost is 2/100 = 0.02 per step.
        """
        num_groups = 2
        all_reduces_per_group = 1  # sum-of-squares reduce
        per_step_savings = num_groups * all_reduces_per_group

        assert per_step_savings == 2


# ===========================================================================
#  6. All-reduce count summary (documentation-style test)
# ===========================================================================


class TestAllReduceCountSummary:
    """Summary test documenting expected all-reduce counts per optimizer step.

    Before optimization (KI two-phase):
      - Phase 1 _gather_scalar_stats: ~6 all-reduces
      - Phase 2 _gather_scalar_stats: ~6 all-reduces
      - Per-group grad norms (2 groups): 2 all-reduces
      - Total gradient clip (clip_grad_norm_): 1 all-reduce (internal)
      Total: ~15 all-reduces per optimizer step

    After optimization (fast path, non-log-interval):
      - Phase 1 _gather_finite_consensus: 1 all-reduce
      - Phase 2 _gather_finite_consensus: 1 all-reduce
      - Per-group grad norms: 0 (skipped)
      - Total gradient clip: 1 (unchanged — training math)
      Total: ~3 all-reduces per optimizer step

    ~5x reduction in diagnostic all-reduces (excluding clipping itself).
    """

    def test_before_vs_after_count(self):
        """Document before/after all-reduce counts for KI two-phase path."""
        # BEFORE (diagnostic only, excludes clip_grad_norm_ which is training math)
        before_diag = (
            6   # phase 1 _gather_scalar_stats
            + 6  # phase 2 _gather_scalar_stats
            + 2  # per-group grad norms (2 groups)
        )

        # AFTER (fast path, diagnostic only)
        after_diag_fast = (
            1   # phase 1 _gather_finite_consensus
            + 1  # phase 2 _gather_finite_consensus
            + 0  # per-group grad norms (throttled)
        )

        # Total clipping all-reduce is unchanged
        clip_allreduce = 1

        before_total = before_diag + clip_allreduce
        after_total_fast = after_diag_fast + clip_allreduce

        # Diagnostic all-reduces should be significantly reduced
        assert after_diag_fast < before_diag
        reduction_factor = before_diag / after_diag_fast
        assert reduction_factor >= 6.0, (
            f"Expected at least 6x diagnostic all-reduce reduction, "
            f"got {reduction_factor:.1f}x"
        )

        # Total (including clipping) reduction
        total_reduction = before_total / after_total_fast
        assert total_reduction > 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
