#!/usr/bin/env python3
"""Per-**device** CUDA capability preflight (companion to ``cuda_preflight.py``).

Why this exists as a separate script
------------------------------------
``scripts/cuda_preflight.py`` is the established per-*node* gate and is
depended upon by the live V100 and A100 formal launchers, so it is deliberately
left byte-identical here.  It has one blind spot that matters on a
heterogeneous pool: after confirming ``torch.cuda.is_available()``, a driver
floor and a device *count*, it materialises a probe tensor on **GPU 0 only**
(``torch.zeros(1, device="cuda:0")``).  A node whose GPU 5 is fenced, ECC-locked,
already exclusively bound by another process, or otherwise unable to create a
context therefore **passes** that gate and instead fails later, inside c10d /
NCCL bootstrap, where ranks are not watchdog-protected.  The historical failure
this guards against is the 470.129.06 host that returned ``cuInit`` 803, let
DeepSpeed silently fall back to a CPU accelerator, and hung the job for 30
minutes.

This script closes that gap by touching **every** visible device and by proving
the two things a bf16 ZeRO-2 job actually depends on, per device:

1. a real CUDA context (allocate, compute, synchronize) — not just enumeration;
2. native BF16 arithmetic (compute capability >= 8.0 and a live bf16 matmul),
   because a silent bf16 fallback is exactly the class of defect that survives
   a GPU-0-only probe.

It also reports per-device free/total memory so a node that is still carrying
leftover occupier or orphan processes is visible *before* training starts
rather than as a mid-init OOM.

Exit code 0 == every visible device passed. Non-zero prints one diagnostic line
per failing device and exits 2. Read-only: it allocates a few small tensors and
frees them; it never kills or signals anything.

Usage::

    python cuda_preflight_all_devices.py --expect-gpus 8 --require-bf16 \
        --min-free-mib 40000
"""

from __future__ import annotations

import argparse
import os
import sys

# A small but non-trivial matmul: big enough to require a real kernel launch and
# to surface a broken/downclocked device, small enough to stay well inside any
# residual memory left by a keepalive occupier.
_PROBE_DIM = 512


def _probe_device(index: int, *, require_bf16: bool, min_free_mib: int) -> list[str]:
    """Return a list of error strings for one device (empty == pass)."""
    import torch

    errors: list[str] = []
    try:
        torch.cuda.set_device(index)
    except Exception as exc:  # noqa: BLE001
        return [f"gpu{index}: set_device failed: {exc}"]

    # 1. Real context + fp32 compute.
    try:
        a = torch.randn(_PROBE_DIM, _PROBE_DIM, device=f"cuda:{index}", dtype=torch.float32)
        c = a @ a
        torch.cuda.synchronize(index)
        if not bool(torch.isfinite(c).all().item()):
            errors.append(f"gpu{index}: fp32 matmul produced non-finite values")
        del a, c
    except Exception as exc:  # noqa: BLE001
        errors.append(f"gpu{index}: fp32 context/matmul failed: {exc}")

    # 2. Native BF16. A silent bf16 fallback is the defect a GPU-0-only probe misses.
    if require_bf16:
        try:
            major, minor = torch.cuda.get_device_capability(index)
        except Exception as exc:  # noqa: BLE001
            major, minor = (-1, -1)
            errors.append(f"gpu{index}: could not read compute capability: {exc}")
        if major >= 0 and major < 8:
            errors.append(
                f"gpu{index}: compute capability {major}.{minor} lacks native BF16 "
                "(need >= 8.0); a BF16 ZeRO-2 contract cannot be honoured here"
            )
        try:
            if not torch.cuda.is_bf16_supported():
                errors.append(f"gpu{index}: torch.cuda.is_bf16_supported() is False")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"gpu{index}: is_bf16_supported() raised: {exc}")
        try:
            b = torch.randn(_PROBE_DIM, _PROBE_DIM, device=f"cuda:{index}", dtype=torch.bfloat16)
            d = b @ b
            torch.cuda.synchronize(index)
            if d.dtype is not torch.bfloat16:
                errors.append(f"gpu{index}: bf16 matmul returned dtype {d.dtype}, expected bfloat16")
            if not bool(torch.isfinite(d.float()).all().item()):
                errors.append(f"gpu{index}: bf16 matmul produced non-finite values")
            del b, d
        except Exception as exc:  # noqa: BLE001
            errors.append(f"gpu{index}: bf16 matmul failed: {exc}")

    # 3. Free memory headroom. Reported even when it passes, so a node still
    #    carrying occupiers/orphans is visible before training allocates.
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        free_mib = free_bytes // (1024 * 1024)
        total_mib = total_bytes // (1024 * 1024)
        if min_free_mib > 0 and free_mib < min_free_mib:
            errors.append(
                f"gpu{index}: only {free_mib} MiB free of {total_mib} MiB "
                f"(need >= {min_free_mib} MiB); another process may still hold this device"
            )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"gpu{index}: mem_get_info failed: {exc}")

    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return errors


def run(*, expect_gpus: int, require_bf16: bool, min_free_mib: int) -> tuple[list[str], list[str]]:
    """Return ``(errors, info_lines)``."""
    try:
        import torch
    except ImportError as exc:
        return ([f"torch import failed: {exc}"], [])

    if not torch.cuda.is_available():
        return (["torch.cuda.is_available() returned False"], [])

    count = torch.cuda.device_count()
    errors: list[str] = []
    if expect_gpus > 0 and count != expect_gpus:
        errors.append(f"visible CUDA devices={count}, expected exactly {expect_gpus}")

    info: list[str] = []
    for index in range(count):
        errors.extend(_probe_device(index, require_bf16=require_bf16, min_free_mib=min_free_mib))
        try:
            name = torch.cuda.get_device_name(index)
            major, minor = torch.cuda.get_device_capability(index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            info.append(
                f"gpu{index}={name} sm{major}{minor} "
                f"free={free_bytes // (1024 * 1024)}MiB/{total_bytes // (1024 * 1024)}MiB"
            )
        except Exception:  # noqa: BLE001
            info.append(f"gpu{index}=<unreadable>")
    return (errors, info)


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-device CUDA/BF16 capability preflight")
    parser.add_argument(
        "--expect-gpus",
        type=int,
        default=8,
        help="Exact number of visible devices required (0 disables the count check).",
    )
    parser.add_argument(
        "--require-bf16",
        action="store_true",
        help="Require native BF16 (compute capability >= 8.0) plus a live bf16 matmul.",
    )
    parser.add_argument(
        "--min-free-mib",
        type=int,
        default=0,
        help="Minimum free memory per device in MiB (0 disables). Catches leftover holders.",
    )
    args = parser.parse_args()

    errors, info = run(
        expect_gpus=args.expect_gpus,
        require_bf16=args.require_bf16,
        min_free_mib=args.min_free_mib,
    )

    node_rank = os.environ.get("ARNOLD_ID", "?")
    host = os.environ.get("HOSTNAME", "unknown")

    if errors:
        for message in errors:
            print(
                f"CUDA_ALL_DEVICES_PREFLIGHT_FAIL node_rank={node_rank} host={host}: {message}",
                file=sys.stderr,
            )
        return 2

    print(
        f"CUDA_ALL_DEVICES_PREFLIGHT_OK node_rank={node_rank} host={host} "
        f"bf16_required={args.require_bf16} " + " ".join(info)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
