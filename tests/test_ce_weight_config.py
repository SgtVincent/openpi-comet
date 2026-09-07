"""Tests for the configurable planner-CE weight.

The point of the field is not to change anything today: it is that
`combined_loss = ce_loss + alpha * flow_loss` with `alpha = 10.0` weights the
action term ten times the planner term, and until now the planner side had no
knob at all -- the only way to rebalance was to move the action weight. So the
default must be provably inert, and the effective values must be visible in the
run's own output rather than inferred from whatever the defaults are that day.
"""

from __future__ import annotations

import dataclasses
import pathlib

import pytest
import torch

from openpi.models.pi05_subtask_config import Pi05SubtaskConfig

REPO = pathlib.Path(__file__).resolve().parents[1]
MODEL_SRC = REPO / "src" / "openpi" / "models_pytorch" / "pi05_subtask.py"
TRAIN_SRC = REPO / "scripts" / "train_accelerate.py"


def test_defaults_are_unchanged():
    cfg = Pi05SubtaskConfig()
    assert cfg.ce_weight == 1.0, "default must keep the historical implicit 1.0"
    assert cfg.alpha == 10.0


def test_both_weights_are_configurable():
    cfg = dataclasses.replace(Pi05SubtaskConfig(), ce_weight=0.25, alpha=2.0)
    assert (cfg.ce_weight, cfg.alpha) == (0.25, 2.0)


def test_ce_weight_rejects_nothing_silently_useful_at_least_it_is_a_float():
    """`ce_weight=True` would evaluate as 1.0 and look like the default.

    Not currently guarded in the config dataclass; asserted here so the gap is
    recorded rather than discovered later. Change to pytest.raises if a validator
    is added.
    """
    cfg = dataclasses.replace(Pi05SubtaskConfig(), ce_weight=True)
    assert cfg.ce_weight is True
    assert float(cfg.ce_weight) == 1.0, (
        "bool weight silently equals the default; if this ever matters, validate it"
    )


def test_model_source_scales_ce_by_the_configured_weight():
    """Source-level assertion, because a real forward needs 2B+ of weights.

    Pins that the combined loss reads the field instead of re-hardcoding 1.0.
    """
    src = MODEL_SRC.read_text()
    assert "self.ce_weight * ce_loss + self.alpha * flow_loss" in src
    assert "combined_loss = ce_loss + self.alpha * flow_loss" not in src, (
        "the old unweighted form is still present"
    )
    assert "ce_weight: float = 1.0" in src, "constructor default must stay 1.0"


def test_launcher_passes_ce_weight_and_logs_the_effective_values():
    src = TRAIN_SRC.read_text()
    assert 'ce_weight = getattr(model_cfg, "ce_weight", 1.0)' in src
    assert "ce_weight=ce_weight" in src, "value is read but never passed to the model"
    assert "loss weights in effect" in src, (
        "effective weights must be logged so a run's weights are evidenced by its output"
    )


@pytest.mark.parametrize("ce_loss,flow_loss", [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
                                               (0.37, 2.5), (1e-8, 1e8)])
def test_default_weight_is_bit_identical_to_the_previous_formula(ce_loss, flow_loss):
    """float32 bit-exact, not approximate: `1.0 * x` must not perturb the sum."""
    ce = torch.tensor(ce_loss, dtype=torch.float32)
    flow = torch.tensor(flow_loss, dtype=torch.float32)
    alpha = 10.0
    legacy = ce + alpha * flow
    new = 1.0 * ce + alpha * flow
    assert torch.equal(legacy, new), f"{legacy.item()!r} != {new.item()!r}"


def test_nondefault_weight_actually_changes_the_loss():
    """Guards the guard: a knob that cannot move is not a knob."""
    ce = torch.tensor(0.5, dtype=torch.float32)
    flow = torch.tensor(0.5, dtype=torch.float32)
    assert not torch.equal(1.0 * ce + 10.0 * flow, 0.5 * ce + 10.0 * flow)
