"""PyTorch checkpoint loading helpers."""

import os

import safetensors.torch
import torch


def load_pytorch_weights(
    model: torch.nn.Module,
    weight_path: str | os.PathLike[str],
    *,
    pytorch_model_name: str,
) -> None:
    """Load a checkpoint while preserving model-specific tied-weight contracts."""
    if pytorch_model_name == "pi05_ki_joint_query":
        # Accelerate/ZeRO materializes tied PaliGemma weights as separate tensors before
        # save_file. safetensors.load_model then misclassifies the duplicate on-disk alias
        # as unexpected even though the strict PyTorch state_dict contract is exact.
        state_dict = safetensors.torch.load_file(weight_path, device="cpu")
        model.load_state_dict(state_dict, strict=True)
        return

    safetensors.torch.load_model(model, weight_path)
