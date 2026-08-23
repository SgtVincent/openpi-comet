#!/usr/bin/env python3
"""Per-node CUDA / driver capability preflight.

Motivation (concrete, not hypothetical): a single host in the training pool ran
NVIDIA driver 470.129.06 under a CUDA-12 PyTorch image.  ``cuInit(0)`` returned
error code 803 (``cudaErrorSystemDriverMismatch``), DeepSpeed silently fell back
to a CPU accelerator, and the job hung for 30 minutes in a c10d TCPStore wait
because bootstrap-phase ranks are **not** NCCL-watchdog-protected.
``nvidia-smi`` listing 8 cards does **not** prove CUDA is usable.

This preflight must run on **every node** before ``accelerate launch`` so that
a heterogeneous-pool bad host fails the job in ~seconds instead of 30 minutes.

Usage::

    python cuda_preflight.py [--min-driver-version 525] [--min-gpus 8]

Exit code 0 means every visible GPU reports a usable CUDA context; non-zero
prints a single diagnostic line and exits 2.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import sys


def _cuinit_return_code() -> int | None:
    """Call ``cuInit(0)`` via ctypes and return its CUDA result code.

    Returns None if libcuda cannot be loaded at all (e.g. CPU-only node).
    """
    lib_name = ctypes.util.find_library("cuda")
    if lib_name is None:
        # Common fallback inside Merlin images.
        for candidate in ("libcuda.so.1", "libcuda.so"):
            try:
                ctypes.CDLL(candidate)
                lib_name = candidate
                break
            except OSError:
                continue
    if lib_name is None:
        return None
    try:
        libcuda = ctypes.CDLL(lib_name)
    except OSError:
        return None
    try:
        return int(libcuda.cuInit(0))
    except (AttributeError, OSError):
        return None


def _cuda_device_count() -> int | None:
    """Return the CUDA device count from the runtime, or None if unavailable."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def _driver_version_tuple() -> tuple[int, ...] | None:
    """Return ``(major, minor, patch)`` from ``nvidia-smi`` or None."""
    import shutil
    import subprocess

    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        result = subprocess.run(
            [smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    # Use the first GPU's driver; all GPUs in a node share the kernel driver.
    first_line = result.stdout.strip().splitlines()[0].strip()
    parts: list[int] = []
    for token in first_line.split("."):
        try:
            parts.append(int(token))
        except ValueError:
            break
    return tuple(parts) if parts else None


def run_preflight(*, min_gpus: int = 8, min_driver_major: int = 525) -> list[str]:
    """Run all checks. Returns a list of error strings (empty == pass)."""
    errors: list[str] = []

    # 1. torch CUDA availability.
    try:
        import torch
    except ImportError as exc:
        errors.append(f"torch import failed: {exc}")
        return errors  # nothing else is meaningful

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        errors.append("torch.cuda.is_available() returned False")

    # 2. cuInit return code.  0 == CUDA_SUCCESS.
    rc = _cuinit_return_code()
    if rc is None:
        errors.append("libcuda could not be loaded (cuInit not callable)")
    elif rc != 0:
        errors.append(f"cuInit(0) returned {rc} (driver/CUDA mismatch)")

    # 3. Device count.
    count = _cuda_device_count()
    if count is None:
        errors.append("could not determine CUDA device count (torch import issue)")
    elif count < min_gpus:
        errors.append(f"visible CUDA devices={count}, need at least {min_gpus}")

    # 4. Driver version floor.
    if min_driver_major > 0:
        version = _driver_version_tuple()
        if version is None:
            errors.append("could not read NVIDIA driver version from nvidia-smi")
        elif version[0] < min_driver_major:
            errors.append(
                f"NVIDIA driver {'.'.join(str(v) for v in version)} "
                f"is older than required major version {min_driver_major}"
            )

    # 5. If CUDA is reported available, actually create a tiny tensor on GPU 0
    #    to prove context creation works end-to-end.
    if cuda_available and count and count > 0:
        try:
            probe = torch.zeros(1, device="cuda:0")
            del probe
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"GPU-0 tensor allocation failed: {exc}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="CUDA/driver capability preflight")
    parser.add_argument("--min-gpus", type=int, default=8)
    parser.add_argument(
        "--min-driver-major",
        type=int,
        default=525,
        help="Minimum NVIDIA driver major version (CUDA 12 needs >=525).",
    )
    args = parser.parse_args()

    errors = run_preflight(min_gpus=args.min_gpus, min_driver_major=args.min_driver_major)

    node_hostname = os.environ.get("ARNOLD_ID", "?")
    node_name = os.environ.get("METRICS_LEVEL", os.environ.get("HOSTNAME", "unknown"))

    if errors:
        print(
            f"CUDA_PREFLIGHT_FAIL node_rank={node_hostname} host={node_name}: "
            + "; ".join(errors),
            file=sys.stderr,
        )
        return 2

    try:
        import torch

        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
        cuda_version = torch.version.cuda or "unknown"
    except Exception:  # noqa: BLE001
        device_name = "unknown"
        cuda_version = "unknown"

    print(
        f"CUDA_PREFLIGHT_OK node_rank={node_hostname} host={node_name} "
        f"devices={args.min_gpus}+ cuda={cuda_version} gpu0={device_name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
