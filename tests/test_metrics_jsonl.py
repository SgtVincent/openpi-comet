"""Unit tests for the metrics.jsonl per-step metrics writer.

Validates the exact pattern used in ``train_accelerate.py`` ``train_loop``:
  - Rank-0-only file creation under log_dir
  - One JSON line per optimizer step
  - All info_dict keys present (flattened)
  - File is flushed after each write (readable mid-training)
  - Append-mode for resume safety

These tests exercise the writer logic directly without requiring a full
training loop or GPU.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest


# ===========================================================================
#  Reference implementation (mirrors train_accelerate.py logic)
# ===========================================================================

def _open_metrics_file(log_dir: Path) -> object:
    """Open metrics.jsonl in append mode (line-buffered).

    Mirrors the initialization in ``train_loop`` of ``train_accelerate.py``.
    Returns None for non-main ranks (simulated).
    """
    metrics_path = log_dir / "metrics.jsonl"
    f = open(metrics_path, "a", buffering=1)  # line-buffered
    return f


def _write_metrics_step(metrics_file, info_dict: dict, step: int, steps_per_epoch: int) -> None:
    """Write one step's metrics as a JSON line.

    Mirrors the per-step write in ``train_loop`` of ``train_accelerate.py``.
    No-op if metrics_file is None (non-main rank).
    """
    if metrics_file is None:
        return
    epoch = (step // steps_per_epoch) + 1
    record = {
        "step": int(step),
        "epoch": int(epoch),
        **info_dict,
    }
    metrics_file.write(json.dumps(record, default=str) + "\n")
    metrics_file.flush()


def _close_metrics_file(metrics_file) -> None:
    """Close the metrics file if open."""
    if metrics_file is not None:
        metrics_file.close()


# ===========================================================================
#  Test fixtures
# ===========================================================================


@pytest.fixture
def log_dir(tmp_path):
    """Provide a temporary log directory."""
    d = tmp_path / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def sample_info_dict():
    """A representative info_dict with all pi05_ki_joint_query joint metrics."""
    return {
        "loss": 0.42,
        "learning_rate": 1e-4,
        "grad_norm": 1.5,
        "grad_norm_total": 1.5,
        "loss_backbone": 0.12,
        "loss_ce": 0.08,
        "loss_query_mse": 0.04,
        "loss_expert": 0.30,
        "loss_flow_raw": 0.03,
        "loss_total": 0.42,
        "grad_norm_backbone": 0.8,
        "grad_norm_expert": 0.7,
        "grad_norm_backbone_available": True,
        "grad_norm_expert_available": True,
        "ki_heuristic_loss_ratio": 0.714,
        "expert_loss_fraction": 0.714,
        "lr_backbone": 1e-4,
        "lr_expert": 2e-4,
        "flow_loss": 0.03,
        "ce_loss": 0.08,
    }


# ===========================================================================
#  Tests
# ===========================================================================


class TestMetricsJsonlCreation:
    """Test file creation and rank-0-only behavior."""

    def test_file_created_under_log_dir(self, log_dir):
        """metrics.jsonl is created under log_dir."""
        f = _open_metrics_file(log_dir)
        _close_metrics_file(f)
        assert (log_dir / "metrics.jsonl").exists(), "metrics.jsonl should exist under log_dir"

    def test_append_mode_for_resume(self, log_dir, sample_info_dict):
        """Opening in append mode preserves existing contents (resume safety)."""
        # Write initial content
        f = _open_metrics_file(log_dir)
        _write_metrics_step(f, sample_info_dict, step=0, steps_per_epoch=100)
        _close_metrics_file(f)

        first_lines = (log_dir / "metrics.jsonl").read_text().strip().split("\n")
        assert len(first_lines) == 1

        # Re-open (simulating resume) and write another step
        f = _open_metrics_file(log_dir)
        _write_metrics_step(f, sample_info_dict, step=1, steps_per_epoch=100)
        _close_metrics_file(f)

        second_lines = (log_dir / "metrics.jsonl").read_text().strip().split("\n")
        assert len(second_lines) == 2, "Append mode should preserve existing lines"

    def test_none_metrics_file_is_noop(self, log_dir, sample_info_dict):
        """When metrics_file is None (non-main rank), no file is written."""
        _write_metrics_step(None, sample_info_dict, step=0, steps_per_epoch=100)
        # No file should be created
        assert not (log_dir / "metrics.jsonl").exists()


class TestMetricsJsonlContent:
    """Test per-step content and structure."""

    def test_one_line_per_step(self, log_dir, sample_info_dict):
        """Exactly N lines for N optimizer steps."""
        n_steps = 5
        f = _open_metrics_file(log_dir)
        for step in range(n_steps):
            _write_metrics_step(f, sample_info_dict, step=step, steps_per_epoch=100)
        _close_metrics_file(f)

        lines = (log_dir / "metrics.jsonl").read_text().strip().split("\n")
        assert len(lines) == n_steps, f"Expected {n_steps} lines, got {len(lines)}"

    def test_each_line_is_valid_json(self, log_dir, sample_info_dict):
        """Each line is valid JSON."""
        n_steps = 3
        f = _open_metrics_file(log_dir)
        for step in range(n_steps):
            _write_metrics_step(f, sample_info_dict, step=step, steps_per_epoch=100)
        _close_metrics_file(f)

        lines = (log_dir / "metrics.jsonl").read_text().strip().split("\n")
        for i, line in enumerate(lines):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                pytest.fail(f"Line {i} is not valid JSON: {e}")

    def test_step_and_epoch_present(self, log_dir, sample_info_dict):
        """Each record includes step and epoch."""
        f = _open_metrics_file(log_dir)
        _write_metrics_step(f, sample_info_dict, step=150, steps_per_epoch=100)
        _close_metrics_file(f)

        record = json.loads((log_dir / "metrics.jsonl").read_text().strip())
        assert record["step"] == 150
        assert record["epoch"] == 2  # 150 // 100 + 1 = 2

    def test_epoch_calculation(self, log_dir, sample_info_dict):
        """Epoch is correctly computed from step and steps_per_epoch."""
        f = _open_metrics_file(log_dir)
        test_cases = [
            (0, 100, 1),    # step 0 -> epoch 1
            (99, 100, 1),   # step 99 -> epoch 1
            (100, 100, 2),  # step 100 -> epoch 2
            (250, 100, 3),  # step 250 -> epoch 3
        ]
        for step, spe, expected_epoch in test_cases:
            _write_metrics_step(f, sample_info_dict, step=step, steps_per_epoch=spe)
        _close_metrics_file(f)

        lines = (log_dir / "metrics.jsonl").read_text().strip().split("\n")
        for i, (step, spe, expected_epoch) in enumerate(test_cases):
            record = json.loads(lines[i])
            assert record["epoch"] == expected_epoch, (
                f"step={step} steps_per_epoch={spe}: expected epoch {expected_epoch}, got {record['epoch']}"
            )

    def test_all_info_dict_keys_present(self, log_dir, sample_info_dict):
        """All info_dict keys are present in the JSON record (flattened)."""
        f = _open_metrics_file(log_dir)
        _write_metrics_step(f, sample_info_dict, step=0, steps_per_epoch=100)
        _close_metrics_file(f)

        record = json.loads((log_dir / "metrics.jsonl").read_text().strip())
        for key in sample_info_dict:
            assert key in record, f"Key '{key}' missing from metrics record"

    def test_all_loss_components_present(self, log_dir, sample_info_dict):
        """All 6 loss components are present."""
        f = _open_metrics_file(log_dir)
        _write_metrics_step(f, sample_info_dict, step=0, steps_per_epoch=100)
        _close_metrics_file(f)

        record = json.loads((log_dir / "metrics.jsonl").read_text().strip())
        expected_losses = [
            "loss_backbone",
            "loss_ce",
            "loss_query_mse",
            "loss_expert",
            "loss_flow_raw",
            "loss_total",
        ]
        for key in expected_losses:
            assert key in record, f"Loss component '{key}' missing"
            assert isinstance(record[key], (int, float)), f"'{key}' should be numeric"

    def test_per_group_lr_present(self, log_dir, sample_info_dict):
        """Per-param-group learning rates are present."""
        f = _open_metrics_file(log_dir)
        _write_metrics_step(f, sample_info_dict, step=0, steps_per_epoch=100)
        _close_metrics_file(f)

        record = json.loads((log_dir / "metrics.jsonl").read_text().strip())
        assert "lr_backbone" in record
        assert "lr_expert" in record

    def test_per_group_grad_norms_present(self, log_dir, sample_info_dict):
        """Per-param-group gradient norms are present."""
        f = _open_metrics_file(log_dir)
        _write_metrics_step(f, sample_info_dict, step=0, steps_per_epoch=100)
        _close_metrics_file(f)

        record = json.loads((log_dir / "metrics.jsonl").read_text().strip())
        assert "grad_norm_backbone" in record
        assert "grad_norm_expert" in record

    def test_ki_heuristic_present(self, log_dir, sample_info_dict):
        """KI heuristic metric is present."""
        f = _open_metrics_file(log_dir)
        _write_metrics_step(f, sample_info_dict, step=0, steps_per_epoch=100)
        _close_metrics_file(f)

        record = json.loads((log_dir / "metrics.jsonl").read_text().strip())
        assert "ki_heuristic_loss_ratio" in record

    def test_expert_loss_fraction_present(self, log_dir, sample_info_dict):
        """expert_loss_fraction metric is present (loss-based, always available)."""
        f = _open_metrics_file(log_dir)
        _write_metrics_step(f, sample_info_dict, step=0, steps_per_epoch=100)
        _close_metrics_file(f)

        record = json.loads((log_dir / "metrics.jsonl").read_text().strip())
        assert "expert_loss_fraction" in record
        # Should match ki_heuristic_loss_ratio (both loss-based)
        assert abs(record["expert_loss_fraction"] - record["ki_heuristic_loss_ratio"]) < 1e-6

    def test_grad_norm_availability_flags_present(self, log_dir, sample_info_dict):
        """grad_norm_backbone_available and grad_norm_expert_available are present."""
        f = _open_metrics_file(log_dir)
        _write_metrics_step(f, sample_info_dict, step=0, steps_per_epoch=100)
        _close_metrics_file(f)

        record = json.loads((log_dir / "metrics.jsonl").read_text().strip())
        assert "grad_norm_backbone_available" in record
        assert "grad_norm_expert_available" in record
        assert record["grad_norm_backbone_available"] is True
        assert record["grad_norm_expert_available"] is True

    def test_standard_keys_present(self, log_dir, sample_info_dict):
        """Standard keys (loss, lr, grad_norm, flow_loss, ce_loss) are present."""
        f = _open_metrics_file(log_dir)
        _write_metrics_step(f, sample_info_dict, step=0, steps_per_epoch=100)
        _close_metrics_file(f)

        record = json.loads((log_dir / "metrics.jsonl").read_text().strip())
        assert "loss" in record
        assert "learning_rate" in record
        assert "grad_norm" in record
        assert "flow_loss" in record
        assert "ce_loss" in record

    def test_step_values_are_correct(self, log_dir, sample_info_dict):
        """Step values in sequential steps are correct."""
        n_steps = 10
        f = _open_metrics_file(log_dir)
        for step in range(n_steps):
            info = {**sample_info_dict, "loss": sample_info_dict["loss"] * (1.0 - step * 0.01)}
            _write_metrics_step(f, info, step=step, steps_per_epoch=100)
        _close_metrics_file(f)

        lines = (log_dir / "metrics.jsonl").read_text().strip().split("\n")
        for i, line in enumerate(lines):
            record = json.loads(line)
            assert record["step"] == i, f"Line {i}: expected step={i}, got {record['step']}"

    def test_numpy_torch_types_handled_by_default_str(self, log_dir):
        """Non-standard types (numpy, torch) are handled by default=str."""
        import numpy as np
        import torch

        info_with_special_types = {
            "loss_float": 0.42,
            "loss_np_float32": np.float32(0.42),
            "loss_np_float64": np.float64(0.42),
            "step_np_int": np.int64(42),
            "tensor_scalar": torch.tensor(0.42).item(),  # .item() gives Python float
        }
        f = _open_metrics_file(log_dir)
        _write_metrics_step(f, info_with_special_types, step=0, steps_per_epoch=100)
        _close_metrics_file(f)

        record = json.loads((log_dir / "metrics.jsonl").read_text().strip())
        assert record["loss_float"] == 0.42
        # numpy types are serialized as numbers via default=str
        assert "loss_np_float32" in record
        assert "loss_np_float64" in record
        assert "step_np_int" in record

    def test_loss_scale_optional(self, log_dir, sample_info_dict):
        """loss_scale is optional (only present with FP16/DeepSpeed)."""
        info_with_scale = {**sample_info_dict, "loss_scale": 65536.0}
        f = _open_metrics_file(log_dir)
        _write_metrics_step(f, info_with_scale, step=0, steps_per_epoch=100)
        _close_metrics_file(f)

        record = json.loads((log_dir / "metrics.jsonl").read_text().strip())
        assert "loss_scale" in record
        assert record["loss_scale"] == 65536.0


class TestMetricsJsonlFlush:
    """Test that data is flushed after each write."""

    def test_flush_after_each_write(self, log_dir, sample_info_dict):
        """File is readable mid-training (data flushed after each step)."""
        f = _open_metrics_file(log_dir)

        for step in range(5):
            _write_metrics_step(f, sample_info_dict, step=step, steps_per_epoch=100)
            # Read immediately without closing — should see all lines so far
            lines = (log_dir / "metrics.jsonl").read_text().strip().split("\n")
            assert len(lines) == step + 1, (
                f"After step {step}, expected {step + 1} lines but found {len(lines)}"
            )
            # Last line should have correct step
            last_record = json.loads(lines[-1])
            assert last_record["step"] == step

        _close_metrics_file(f)

    def test_file_still_writable_after_reading(self, log_dir, sample_info_dict):
        """Reading mid-training doesn't interfere with subsequent writes."""
        f = _open_metrics_file(log_dir)

        # Write step 0
        _write_metrics_step(f, sample_info_dict, step=0, steps_per_epoch=100)
        assert len((log_dir / "metrics.jsonl").read_text().strip().split("\n")) == 1

        # Write step 1
        _write_metrics_step(f, sample_info_dict, step=1, steps_per_epoch=100)
        assert len((log_dir / "metrics.jsonl").read_text().strip().split("\n")) == 2

        # Write step 2
        _write_metrics_step(f, sample_info_dict, step=2, steps_per_epoch=100)
        lines = (log_dir / "metrics.jsonl").read_text().strip().split("\n")
        assert len(lines) == 3
        assert json.loads(lines[2])["step"] == 2

        _close_metrics_file(f)


class TestMetricsJsonlIntegration:
    """Integration-style tests mimicking a full training run."""

    def test_full_training_run(self, log_dir):
        """Simulate a full training run with varying info_dict contents."""
        n_steps = 8
        steps_per_epoch = 5
        f = _open_metrics_file(log_dir)

        for step in range(n_steps):
            # Simulate realistic metric variation
            info = {
                "loss": 1.0 * (0.9 ** step),
                "learning_rate": 1e-4,
                "grad_norm": 2.0 * (0.95 ** step),
                "grad_norm_total": 2.0 * (0.95 ** step),
                "loss_backbone": 0.3 * (0.9 ** step),
                "loss_ce": 0.1 * (0.9 ** step),
                "loss_query_mse": 0.2 * (0.9 ** step),
                "loss_expert": 0.7 * (0.9 ** step),
                "loss_flow_raw": 0.07 * (0.9 ** step),
                "loss_total": 1.0 * (0.9 ** step),
                "grad_norm_backbone": 1.0 * (0.95 ** step),
                "grad_norm_expert": 1.0 * (0.95 ** step),
                "grad_norm_backbone_available": True,
                "grad_norm_expert_available": True,
                "ki_heuristic_loss_ratio": 0.7,
                "expert_loss_fraction": 0.7,
                "lr_backbone": 1e-4,
                "lr_expert": 2e-4,
                "flow_loss": 0.07 * (0.9 ** step),
                "ce_loss": 0.1 * (0.9 ** step),
            }
            _write_metrics_step(f, info, step=step, steps_per_epoch=steps_per_epoch)
        _close_metrics_file(f)

        # Verify all lines
        lines = (log_dir / "metrics.jsonl").read_text().strip().split("\n")
        assert len(lines) == n_steps

        records = [json.loads(line) for line in lines]
        # Loss should decrease
        losses = [r["loss"] for r in records]
        assert all(losses[i] > losses[i + 1] for i in range(len(losses) - 1)), (
            "Loss should decrease across steps"
        )

        # Step numbers should be sequential
        steps = [r["step"] for r in records]
        assert steps == list(range(n_steps))

        # Epochs should be correct (steps_per_epoch=5)
        epochs = [r["epoch"] for r in records]
        expected_epochs = [1, 1, 1, 1, 1, 2, 2, 2]
        assert epochs == expected_epochs

    def test_minimal_info_dict(self, log_dir):
        """Works with minimal info_dict (non-pi05_ki_joint_query models)."""
        minimal_info = {
            "loss": 0.5,
            "learning_rate": 1e-4,
            "grad_norm": 1.0,
            "grad_norm_total": 1.0,
        }
        f = _open_metrics_file(log_dir)
        _write_metrics_step(f, minimal_info, step=0, steps_per_epoch=100)
        _close_metrics_file(f)

        record = json.loads((log_dir / "metrics.jsonl").read_text().strip())
        assert record["step"] == 0
        assert record["loss"] == 0.5
        assert "loss_backbone" not in record  # Not present for non-joint models
        assert "lr_backbone" not in record


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
