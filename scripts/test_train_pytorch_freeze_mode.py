import pytest
import torch

from scripts import train_pytorch


class _NamedParameterModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.paligemma_with_expert = torch.nn.Module()
        self.paligemma_with_expert.paligemma = torch.nn.Linear(2, 2)
        self.paligemma_with_expert.gemma_expert = torch.nn.Linear(2, 2)
        self.action_in_proj = torch.nn.Linear(2, 2)
        self.action_out_proj = torch.nn.Linear(2, 2)
        self.time_mlp_in = torch.nn.Linear(2, 2)
        self.time_mlp_out = torch.nn.Linear(2, 2)
        self.unrelated = torch.nn.Linear(2, 2)


def test_pi0_action_expert_only_freezes_paligemma_prefix() -> None:
    model = _NamedParameterModel()

    summary = train_pytorch.apply_pytorch_freeze_mode(
        model,
        mode="pi0_action_expert_only",
        pytorch_model_name="pi0",
    )

    assert summary["applied"] is True
    assert summary["trainable_param_count"] > 0
    named = dict(model.named_parameters())
    assert named["paligemma_with_expert.paligemma.weight"].requires_grad is False
    assert named["paligemma_with_expert.paligemma.bias"].requires_grad is False
    assert named["paligemma_with_expert.gemma_expert.weight"].requires_grad is True
    assert named["action_in_proj.weight"].requires_grad is True
    assert named["action_out_proj.weight"].requires_grad is True
    assert named["time_mlp_in.weight"].requires_grad is True
    assert named["time_mlp_out.weight"].requires_grad is True
    assert named["unrelated.weight"].requires_grad is False


def test_pi0_action_expert_only_rejects_non_pi0_model_name() -> None:
    model = _NamedParameterModel()

    with pytest.raises(ValueError, match="only valid for pytorch_model_name='pi0'"):
        train_pytorch.apply_pytorch_freeze_mode(
            model,
            mode="pi0_action_expert_only",
            pytorch_model_name="subtask",
        )


def test_unknown_freeze_mode_rejected() -> None:
    model = _NamedParameterModel()

    with pytest.raises(ValueError, match="Unsupported OPENPI_PYTORCH_FREEZE_MODE"):
        train_pytorch.apply_pytorch_freeze_mode(
            model,
            mode="freeze_everything",
            pytorch_model_name="pi0",
        )
