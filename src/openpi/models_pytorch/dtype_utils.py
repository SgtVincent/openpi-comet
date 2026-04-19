"""Helpers for aligning activations to model compute dtypes."""

from __future__ import annotations

from typing import Any

import torch


_STANDARD_COMPUTE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def get_standard_compute_dtype(reference_tensor: Any, *, context: str) -> torch.dtype:
    """Validate and return a supported compute dtype from a reference tensor."""
    target_dtype = reference_tensor.dtype
    if target_dtype not in _STANDARD_COMPUTE_DTYPES:
        raise TypeError(f"Unexpected {context} compute dtype: {target_dtype}")
    return target_dtype


def align_tensors_to_reference_dtype(
    reference_tensor: Any,
    *tensors: torch.Tensor,
    context: str,
) -> tuple[torch.Tensor, ...]:
    """Cast tensors to the validated compute dtype of a reference tensor."""
    target_dtype = get_standard_compute_dtype(reference_tensor, context=context)
    return tuple(tensor if tensor.dtype == target_dtype else tensor.to(dtype=target_dtype) for tensor in tensors)


def align_inputs_to_reference_dtype_with_restore(
    reference_tensor: Any,
    primary_input: torch.Tensor,
    *other_inputs: torch.Tensor,
    context: str,
) -> tuple[torch.dtype, tuple[torch.Tensor, ...]]:
    """Align inputs to a reference dtype and return the primary input's original dtype for output restore."""
    original_dtype = primary_input.dtype
    aligned_inputs = align_tensors_to_reference_dtype(
        reference_tensor,
        primary_input,
        *other_inputs,
        context=context,
    )
    return original_dtype, aligned_inputs
