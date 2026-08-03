"""Regression tests for PyTorch checkpoint loading compatibility."""

import pytest
import safetensors.torch
import torch
from torch import nn

from openpi.models_pytorch.checkpoint_utils import load_pytorch_weights


class _TinyPaliGemma(nn.Module):
    """Minimal PaliGemma-style module with a tied embedding/output weight."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight


def _cloned_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Mirror Accelerator/ZeRO materialization of tied aliases as separate tensors."""
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def test_pi05_ki_loader_accepts_save_file_tied_aliases(tmp_path):
    source = _TinyPaliGemma()
    checkpoint_state = _cloned_state_dict(source)
    checkpoint_path = tmp_path / "model.safetensors"
    safetensors.torch.save_file(checkpoint_state, checkpoint_path)

    embed_key = "model.language_model.embed_tokens.weight"
    assert set(checkpoint_state) == {"lm_head.weight", embed_key}
    assert torch.equal(checkpoint_state["lm_head.weight"], checkpoint_state[embed_key])

    # load_model removes one model alias before checking the file and therefore
    # rejects a save_file checkpoint containing both equal tied keys.
    with pytest.raises(RuntimeError, match=r"model\.language_model\.embed_tokens\.weight"):
        safetensors.torch.load_model(_TinyPaliGemma(), checkpoint_path)

    target = _TinyPaliGemma()
    with torch.no_grad():
        target.lm_head.weight.zero_()
    load_pytorch_weights(target, checkpoint_path, pytorch_model_name="pi05_ki_joint_query")

    assert target.lm_head.weight is target.model.language_model.embed_tokens.weight
    assert set(target.state_dict()) == set(checkpoint_state)
    for name, tensor in target.state_dict().items():
        assert torch.equal(tensor, checkpoint_state[name]), name


def test_pi05_ki_loader_remains_strict(tmp_path):
    state_dict = _cloned_state_dict(_TinyPaliGemma())
    state_dict["unexpected.weight"] = torch.ones(1)
    checkpoint_path = tmp_path / "model.safetensors"
    safetensors.torch.save_file(state_dict, checkpoint_path)

    with pytest.raises(RuntimeError, match=r"Unexpected key\(s\).*unexpected\.weight"):
        load_pytorch_weights(_TinyPaliGemma(), checkpoint_path, pytorch_model_name="pi05_ki_joint_query")


def test_other_models_keep_safetensors_shared_tensor_loader(tmp_path):
    source = nn.Linear(4, 3)
    checkpoint_path = tmp_path / "model.safetensors"
    safetensors.torch.save_model(source, checkpoint_path)

    target = nn.Linear(4, 3)
    load_pytorch_weights(target, checkpoint_path, pytorch_model_name="pi0")

    for name, tensor in target.state_dict().items():
        assert torch.equal(tensor, source.state_dict()[name]), name
