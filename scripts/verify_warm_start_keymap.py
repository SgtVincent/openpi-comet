#!/usr/bin/env python3
"""Explicit warm-start key-mapping verifier (CPU-only, read-only).

Why this exists
---------------
``scripts/train_accelerate.py`` loads fine-tuning weights with::

    load_strict = config.pytorch_model_name not in ("pi05_ki_joint_fast", ...)
    safetensors.torch.load_model(model, model_path, strict=load_strict)

so for ``pi05_ki_joint_fast`` (Variant A) ``strict`` is **False**. That means a
warm start that silently matched *nothing* — or matched only the vision tower —
would log "Loaded PyTorch weights from ..." and train on essentially random
initialisation. The run would look healthy and the result would be a lie.

This script makes the mapping an explicit, checkable artifact **before** any GPU
is touched: it constructs the target model on the ``meta`` device (no 14.5 GB
allocation), reads the checkpoint's safetensors *header* (no tensor data), and
reports exact counts of

  * matched keys (name present in both, shapes equal)
  * missing keys (model expects, checkpoint lacks)  -> silently left at init
  * unexpected keys (checkpoint has, model lacks)   -> silently discarded
  * shape-mismatched keys (name in both, shape differs)

and then applies a fail-closed policy: a nonzero missing or shape-mismatch count
is an error unless explicitly allow-listed on the command line.

Usage::

    python scripts/verify_warm_start_keymap.py \
        --config pi05_ki_joint_fast_b1k-full_task-ki_on_h20_bf16 \
        --checkpoint /path/to/pi05-b1kpt50-cs32 \
        [--compare-config <other config>] [--allow-missing-prefix foo.] [--json out.json]

Exit 0 == mapping satisfies the policy. Exit 2 == violation. Nothing is written
except an optional JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _build_model_meta(config_name: str):
    """Instantiate the configured PyTorch model on the meta device.

    Returns ``(model, config)``. Falls back to a real CPU build if meta
    construction is unsupported by the model implementation.
    """
    import torch

    from openpi.training.train_config import get_config

    config = get_config(config_name)
    model_cfg = config.model

    # IMPORTANT: resolve the model class *outside* any device context.
    # Importing HuggingFace submodules under ``with torch.device("meta")``
    # breaks their module-level initialisation, and because Python caches the
    # partially-failed module in ``sys.modules`` a later retry inherits the
    # failure. Concretely this surfaced as a spurious
    # "cannot import name 'GemmaForCausalLM' from 'transformers'" that also
    # poisoned the CPU fallback. Import first, then choose the device.
    if config.pytorch_model_name == "pi05_ki_joint_fast":
        import openpi.models_pytorch.pi05_ki_joint_fast as mod

        cls = mod.PI05KIJointFastPytorch
    elif config.pytorch_model_name == "pi05_ki_joint_query":
        import openpi.models_pytorch.pi05_ki_joint_query as mod

        cls = mod.PI05KIJointQueryPytorch
    else:
        raise SystemExit(
            f"ERROR: unsupported pytorch_model_name {config.pytorch_model_name!r}; "
            "this verifier covers the two KI joint variants only"
        )

    try:
        with torch.device("meta"):
            return cls(model_cfg), config, "meta"
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] meta-device construction failed ({exc}); falling back to CPU build", file=sys.stderr)
        return cls(model_cfg), config, "cpu"


def _checkpoint_shapes(checkpoint_dir: str) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    """Read ``({key: shape}, tie_map)`` from the safetensors header.

    No tensor data is loaded. ``tie_map`` comes from the file's ``__metadata__``
    and maps an *absent* parameter name to the *present* tensor it is tied to.
    ``safetensors.torch.save_model`` deduplicates shared/tied tensors and records
    the aliasing there, and ``load_model`` uses it to repopulate the tied
    parameter. Without consulting it, a tied weight looks like a missing key and
    a perfectly good warm start looks broken.
    """
    import json
    import struct

    path = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.isfile(path):
        raise SystemExit(f"ERROR: checkpoint weights not found: {path}")

    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))

    metadata = header.get("__metadata__") or {}
    shapes = {
        key: tuple(value["shape"]) for key, value in header.items() if key != "__metadata__"
    }
    tie_map = {k: v for k, v in metadata.items() if isinstance(v, str) and v in shapes}
    return shapes, tie_map


def _classify(
    model_shapes: dict[str, tuple[int, ...]],
    ckpt_shapes: dict[str, tuple[int, ...]],
    tie_map: dict[str, str],
) -> dict[str, list[str]]:
    model_keys = set(model_shapes)
    ckpt_keys = set(ckpt_shapes)
    shared = model_keys & ckpt_keys
    matched = sorted(k for k in shared if model_shapes[k] == ckpt_shapes[k])
    mismatched = sorted(k for k in shared if model_shapes[k] != ckpt_shapes[k])

    raw_missing = model_keys - ckpt_keys
    tied: list[str] = []
    missing: list[str] = []
    for key in sorted(raw_missing):
        target = tie_map.get(key)
        if target is not None and ckpt_shapes[target] == model_shapes[key]:
            tied.append(key)
        else:
            missing.append(key)

    return {
        "matched": matched,
        "tied": tied,
        "missing": missing,
        "unexpected": sorted(ckpt_keys - model_keys),
        "shape_mismatch": mismatched,
    }


def _prefix_histogram(keys: list[str], depth: int = 2, limit: int = 12) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for key in keys:
        prefix = ".".join(key.split(".")[:depth])
        counts[prefix] = counts.get(prefix, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify warm-start key mapping (CPU, read-only)")
    parser.add_argument("--config", required=True, help="Registered TrainConfig name to build.")
    parser.add_argument("--checkpoint", required=True, help="Directory containing model.safetensors.")
    parser.add_argument(
        "--compare-config",
        default=None,
        help="Optional second config; reports how its parameter set differs (e.g. Variant B).",
    )
    parser.add_argument(
        "--allow-missing-prefix",
        action="append",
        default=[],
        help="Permit missing keys under this prefix (repeatable). Everything else fails closed.",
    )
    parser.add_argument(
        "--max-unexpected",
        type=int,
        default=-1,
        help="Fail if unexpected-key count exceeds this (-1 = do not gate on unexpected).",
    )
    parser.add_argument("--json", default=None, help="Optional path for a JSON report.")
    parser.add_argument(
        "--actually-load",
        action="store_true",
        help=(
            "Additionally perform the REAL safetensors load on a CPU-materialised model, "
            "exactly as train_accelerate.py does, and prove tied tensors were populated "
            "(not left at random init). Costs ~model-size RAM."
        ),
    )
    args = parser.parse_args()

    model, config, build_device = _build_model_meta(args.config)
    model_shapes = {name: tuple(t.shape) for name, t in model.state_dict().items()}
    n_params = sum(int(t.numel()) for t in model.state_dict().values())

    ckpt_shapes, tie_map = _checkpoint_shapes(args.checkpoint)
    n_ckpt = sum(int(_prod(s)) for s in ckpt_shapes.values())

    result = _classify(model_shapes, ckpt_shapes, tie_map)

    print("=" * 78)
    print(f"WARM START KEY MAPPING  config={args.config}")
    print(f"  model_name        : {config.pytorch_model_name}")
    print(f"  build_device      : {build_device}")
    print(f"  checkpoint        : {args.checkpoint}")
    print(f"  model tensors     : {len(model_shapes):>6}   params {n_params:,}")
    print(f"  ckpt  tensors     : {len(ckpt_shapes):>6}   params {n_ckpt:,}")
    print(f"  ckpt tie entries  : {len(tie_map):>6}   (shared/tied tensors deduplicated on save)")
    print("-" * 78)
    print(f"  MATCHED           : {len(result['matched']):>6}")
    print(f"  TIED (resolvable) : {len(result['tied']):>6}  (absent from file, aliased in __metadata__)")
    print(f"  MISSING           : {len(result['missing']):>6}  (model expects, ckpt lacks, NOT tied)")
    print(f"  UNEXPECTED        : {len(result['unexpected']):>6}  (ckpt has, model lacks)")
    print(f"  SHAPE_MISMATCH    : {len(result['shape_mismatch']):>6}")
    covered = result["matched"] + result["tied"]
    matched_params = sum(int(_prod(model_shapes[k])) for k in covered)
    coverage = 100.0 * matched_params / n_params if n_params else 0.0
    print(f"  PARAM COVERAGE    : {matched_params:,} / {n_params:,} = {coverage:.4f}%  (matched + tied)")
    print("=" * 78)

    if result["tied"]:
        print(f"\n--- TIED ({len(result['tied'])}) resolved via __metadata__ ---")
        for key in result["tied"]:
            print(f"    {key} {model_shapes[key]}")
            print(f"        <- aliased to {tie_map[key]} {ckpt_shapes[tie_map[key]]}")

    for label in ("missing", "unexpected", "shape_mismatch"):
        keys = result[label]
        if not keys:
            continue
        print(f"\n--- {label.upper()} ({len(keys)}) grouped by prefix ---")
        for prefix, count in _prefix_histogram(keys):
            print(f"    {count:>6}  {prefix}.*")
        print(f"  first {min(10, len(keys))} example(s):")
        for key in keys[:10]:
            if label == "shape_mismatch":
                print(f"    {key}: model={model_shapes[key]} ckpt={ckpt_shapes[key]}")
            elif label == "missing":
                print(f"    {key}: shape={model_shapes[key]}")
            else:
                print(f"    {key}: shape={ckpt_shapes[key]}")

    compare_summary = None
    if args.compare_config:
        other, other_cfg, _ = _build_model_meta(args.compare_config)
        other_shapes = {name: tuple(t.shape) for name, t in other.state_dict().items()}
        only_this = sorted(set(model_shapes) - set(other_shapes))
        only_other = sorted(set(other_shapes) - set(model_shapes))
        compare_summary = {
            "compare_config": args.compare_config,
            "compare_model_name": other_cfg.pytorch_model_name,
            "compare_tensors": len(other_shapes),
            "only_in_target": only_this,
            "only_in_compare": only_other,
        }
        print(f"\n--- STRUCTURAL DIFF vs {args.compare_config} ({other_cfg.pytorch_model_name}) ---")
        print(f"    target tensors  : {len(model_shapes)}")
        print(f"    compare tensors : {len(other_shapes)}")
        print(f"    only in target  : {len(only_this)}")
        for key in only_this[:10]:
            print(f"        + {key} {model_shapes[key]}")
        print(f"    only in compare : {len(only_other)}")
        for key in only_other[:10]:
            print(f"        - {key} {other_shapes[key]}")

    # ---- fail-closed policy -------------------------------------------------
    violations: list[str] = []
    disallowed_missing = [
        key
        for key in result["missing"]
        if not any(key.startswith(prefix) for prefix in args.allow_missing_prefix)
    ]
    if disallowed_missing:
        violations.append(
            f"{len(disallowed_missing)} missing key(s) not covered by --allow-missing-prefix "
            f"(e.g. {disallowed_missing[:3]})"
        )
    if result["shape_mismatch"]:
        violations.append(f"{len(result['shape_mismatch'])} shape-mismatched key(s)")
    if args.max_unexpected >= 0 and len(result["unexpected"]) > args.max_unexpected:
        violations.append(
            f"{len(result['unexpected'])} unexpected key(s) exceeds --max-unexpected={args.max_unexpected}"
        )
    if not result["matched"]:
        violations.append("zero matched keys: this would be a cold start disguised as a warm start")

    # ---- optional: prove the real load populates everything ----------------
    load_proof = None
    if args.actually_load:
        import torch

        import safetensors.torch

        print("\n--- REAL LOAD PROOF (CPU) ---")
        # Build on CPU (no device context) so parameters are materialised and
        # observable before/after the load.
        from openpi.training.train_config import get_config as _get_config

        _cfg = _get_config(args.config)
        if _cfg.pytorch_model_name == "pi05_ki_joint_fast":
            import openpi.models_pytorch.pi05_ki_joint_fast as _m

            real_model = _m.PI05KIJointFastPytorch(_cfg.model)
        else:
            import openpi.models_pytorch.pi05_ki_joint_query as _m

            real_model = _m.PI05KIJointQueryPytorch(_cfg.model)

        tied_keys = result["tied"]
        before = {
            k: float(real_model.state_dict()[k].detach().float().abs().sum().item())
            for k in tied_keys
        }
        model_path = os.path.join(args.checkpoint, "model.safetensors")
        # Mirror train_accelerate.py exactly: strict=False for the KI variants.
        missing_rt, unexpected_rt = safetensors.torch.load_model(real_model, model_path, strict=False)
        print(f"    loader returned missing={list(missing_rt)}")
        print(f"    loader returned unexpected={list(unexpected_rt)}")

        sd = real_model.state_dict()
        tie_checks = {}
        for key in tied_keys:
            target = tie_map[key]
            after = float(sd[key].detach().float().abs().sum().item())
            same_as_target = bool(torch.equal(sd[key], sd[target])) if target in sd else None
            tie_checks[key] = {
                "abs_sum_before": before[key],
                "abs_sum_after": after,
                "changed_by_load": before[key] != after,
                "equals_tie_target": same_as_target,
                "tie_target": target,
            }
            print(
                f"    {key}: abs_sum {before[key]:.4f} -> {after:.4f} "
                f"changed={before[key] != after} equals({target})={same_as_target}"
            )
        load_proof = {
            "loader_missing": list(missing_rt),
            "loader_unexpected": list(unexpected_rt),
            "tie_checks": tie_checks,
        }
        for key, info in tie_checks.items():
            if not info["changed_by_load"] or info["equals_tie_target"] is not True:
                violations.append(
                    f"tied key {key} was NOT correctly populated by the real load "
                    f"(changed={info['changed_by_load']}, equals_target={info['equals_tie_target']})"
                )
        if missing_rt:
            violations.append(
                f"real loader reported {len(missing_rt)} missing key(s): {list(missing_rt)[:5]}"
            )
        del real_model, sd

    report = {
        "config": args.config,
        "model_name": config.pytorch_model_name,
        "checkpoint": args.checkpoint,
        "build_device": build_device,
        "model_tensor_count": len(model_shapes),
        "ckpt_tensor_count": len(ckpt_shapes),
        "model_param_count": n_params,
        "ckpt_param_count": n_ckpt,
        "counts": {k: len(v) for k, v in result.items()},
        "tie_map": tie_map,
        "tied": result["tied"],
        "matched_param_count": matched_params,
        "param_coverage_pct": coverage,
        "missing": result["missing"],
        "unexpected": result["unexpected"],
        "shape_mismatch": {
            k: {"model": list(model_shapes[k]), "ckpt": list(ckpt_shapes[k])}
            for k in result["shape_mismatch"]
        },
        "compare": compare_summary,
        "load_proof": load_proof,
        "violations": violations,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\n[report] wrote {args.json}")

    if violations:
        print("\nWARM_START_KEYMAP_FAIL")
        for v in violations:
            print(f"  - {v}")
        return 2
    print(
        f"\nWARM_START_KEYMAP_OK matched={len(result['matched'])} tied={len(result['tied'])} "
        f"missing={len(result['missing'])} unexpected={len(result['unexpected'])} "
        f"shape_mismatch=0 coverage={coverage:.4f}%"
    )
    return 0


def _prod(shape) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
