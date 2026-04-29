#!/usr/bin/env python3
"""
Batch "load + one-step inference" smoke test for registered TrainConfig entries.

Why this exists:
- The lightweight unit smoke tests (e.g. test_pi05_memoryvla.py) intentionally do NOT
  load real checkpoints.
- This script exercises the full loading path:
    get_config -> create_trained_policy -> (load_pytorch + safetensors) -> Policy.infer

Typical usage (explicit pair):
  conda activate openpi-comet-nas
  python -u scripts/test_registered_model_loading_inference.py \
    --config pi05_memoryvla_b1k-make_pizza_lr1e-4_5ep_sft \
    --ckpt-dir /path/to/ckpt_dir \
    --device cuda:0

Batch usage (JSON mapping):
  python -u scripts/test_registered_model_loading_inference.py \
    --spec-json scripts/model_ckpt_map.json \
    --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np

# Ensure we import the repo-local `openpi` package (openpi-comet) instead of an
# unrelated `openpi` installation from another checkout/site-packages.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir():
    sys.path.insert(0, str(_SRC_DIR))
print(f"[stage] bootstrap: repo_root={_REPO_ROOT}", flush=True)
print(f"[stage] bootstrap: src_dir_inserted={_SRC_DIR}", flush=True)

from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402
print("[stage] bootstrap: openpi imports done", flush=True)


def _log_stage(message: str) -> None:
    print(f"[stage] {message}", flush=True)


def _resolve_checkpoint_dir(ckpt_dir: str) -> tuple[str, str | None]:
    """Return (resolved_dir, selected_step).

    Accepts either:
    - a direct step dir containing `model.safetensors` (PyTorch) or `params/_METADATA` (JAX), or
    - a parent dir containing numeric step subdirs (e.g. 149895/).
    """
    p = pathlib.Path(ckpt_dir)
    if (p / "model.safetensors").is_file() or (p / "params" / "_METADATA").is_file():
        return str(p), None

    if not p.is_dir():
        return str(p), None

    candidates: list[tuple[int, pathlib.Path]] = []
    for child in p.iterdir():
        if not child.is_dir():
            continue
        if not child.name.isdigit():
            continue
        if (child / "model.safetensors").is_file() or (child / "params" / "_METADATA").is_file():
            candidates.append((int(child.name), child))

    if not candidates:
        return str(p), None

    _, best = max(candidates, key=lambda t: t[0])
    return str(best), best.name


def _list_configs() -> list[str]:
    # Intentionally uses the internal registry for convenience in debugging.
    from openpi.training import train_config as _train_config  # local import to keep CLI fast

    cfg_dict = getattr(_train_config, "_CONFIGS_DICT", None)
    if not isinstance(cfg_dict, dict):
        return []
    return sorted(cfg_dict.keys())


def _make_dummy_b1k_obs(
    *,
    prompt: str,
    proprio_dim: int = 260,
    image_hw: int = 224,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Creates a minimal BEHAVIOR-1K-style observation in the same flat-key format
    used by the eval wrapper before policy inference.
    """
    rng = np.random.default_rng(seed)

    def _img() -> np.ndarray:
        return rng.integers(0, 256, size=(image_hw, image_hw, 3), dtype=np.uint8)

    # B1kInputs extracts indices up to ~256; keep it comfortably above that.
    proprio = rng.standard_normal(proprio_dim).astype(np.float32)

    return {
        "observation/egocentric_camera": _img(),
        "observation/wrist_image_left": _img(),
        "observation/wrist_image_right": _img(),
        "observation/state": proprio,
        "prompt": prompt,
    }


def _maybe_exercise_streaming_api(policy) -> None:
    # Exercise memory-model APIs if present; should be no-op for baseline models.
    model = getattr(policy, "_model", None)
    if model is None:
        return
    if hasattr(model, "set_active_session"):
        model.set_active_session(0)
    if hasattr(model, "reset_streaming_state"):
        model.reset_streaming_state()


def _load_and_infer_once(
    *,
    config_name: str,
    ckpt_dir: str,
    device: str,
    default_prompt: str | None,
    seed: int,
    skip_infer: bool,
) -> dict[str, Any]:
    _log_stage(f"resolve config: {config_name}")
    cfg = _config.get_config(config_name)
    # `get_config` falls back silently; detect and fail fast with a clearer message.
    if getattr(cfg, "name", None) != config_name:
        import openpi as _openpi_pkg

        raise ValueError(
            f"Config '{config_name}' not found (got '{getattr(cfg, 'name', None)}'). "
            f"Imported openpi from: {getattr(_openpi_pkg, '__file__', 'unknown')}"
        )
    _log_stage(f"config ok: {cfg.name} (pytorch_model_name={getattr(cfg, 'pytorch_model_name', None)})")

    ckpt_dir_resolved, selected_step = _resolve_checkpoint_dir(ckpt_dir)
    _log_stage(
        "checkpoint resolved: "
        f"input={ckpt_dir} resolved={ckpt_dir_resolved} step={selected_step if selected_step is not None else 'direct'}"
    )

    t0 = time.monotonic()
    _log_stage("create_trained_policy: start")
    policy = _policy_config.create_trained_policy(
        cfg,
        ckpt_dir_resolved,
        default_prompt=default_prompt,
        pytorch_device=device,
    )
    load_s = time.monotonic() - t0
    _log_stage(f"create_trained_policy: done ({load_s:.2f}s)")

    result: dict[str, Any] = {
        "config": config_name,
        "ckpt_dir": ckpt_dir,
        "ckpt_dir_resolved": ckpt_dir_resolved,
        "ckpt_step": selected_step,
        "pytorch_model_name": getattr(cfg, "pytorch_model_name", None),
        "is_pytorch": getattr(policy, "_is_pytorch_model", None),
        "device": device,
        "load_s": load_s,
    }

    if skip_infer:
        return result

    obs = _make_dummy_b1k_obs(prompt=default_prompt or "do something", seed=seed)
    _log_stage("dummy observation built")
    _maybe_exercise_streaming_api(policy)
    _log_stage("streaming API exercised")
    _log_stage("policy.infer: start")
    out = policy.infer(obs)
    _log_stage("policy.infer: done")

    actions = out.get("actions", None)
    timing = out.get("policy_timing", {})
    result.update(
        {
            "actions_shape": tuple(np.asarray(actions).shape) if actions is not None else None,
            "infer_ms": float(timing.get("infer_ms", float("nan"))),
            "generated_subtask": out.get("generated_subtask", None),
        }
    )
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--list-configs", action="store_true", help="List available TrainConfig names and exit.")
    p.add_argument(
        "--spec-json",
        type=str,
        default=None,
        help="JSON file mapping config_name -> checkpoint_dir.",
    )
    p.add_argument("--config", action="append", default=[], help="Config name (repeatable).")
    p.add_argument("--ckpt-dir", action="append", default=[], help="Checkpoint dir for each --config (repeatable).")
    p.add_argument("--device", type=str, default="cuda", help='PyTorch device, e.g. "cuda:0" or "cpu".')
    p.add_argument("--default-prompt", type=str, default=None, help="Optional default prompt override.")
    p.add_argument("--seed", type=int, default=0, help="Seed for dummy input generation.")
    p.add_argument("--skip-infer", action="store_true", help="Only test loading; do not run Policy.infer.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list_configs:
        for name in _list_configs():
            print(name, flush=True)
        return

    items: list[tuple[str, str]] = []
    if args.spec_json is not None:
        spec_path = pathlib.Path(args.spec_json)
        spec = json.loads(spec_path.read_text())
        if not isinstance(spec, dict):
            raise ValueError("--spec-json must be a JSON object mapping config->ckpt_dir")
        for k, v in spec.items():
            items.append((str(k), str(v)))

    if args.config or args.ckpt_dir:
        if len(args.config) != len(args.ckpt_dir):
            raise ValueError("Must provide the same number of --config and --ckpt-dir")
        items.extend(list(zip(args.config, args.ckpt_dir, strict=True)))

    if not items:
        raise ValueError("No items to test. Use --spec-json or --config/--ckpt-dir, or run --list-configs.")

    failures: list[tuple[str, str]] = []
    for config_name, ckpt_dir in items:
        print(f"==== {config_name} ====", flush=True)
        try:
            r = _load_and_infer_once(
                config_name=config_name,
                ckpt_dir=ckpt_dir,
                device=args.device,
                default_prompt=args.default_prompt,
                seed=args.seed,
                skip_infer=args.skip_infer,
            )
            print(json.dumps(r, indent=2, sort_keys=True), flush=True)
        except Exception as e:  # noqa: BLE001
            failures.append((config_name, ckpt_dir))
            print(f"[FAIL] {config_name}: {type(e).__name__}: {e}", flush=True)

    if failures:
        print("==== FAILURES ====", flush=True)
        for config_name, ckpt_dir in failures:
            print(f"{config_name} -> {ckpt_dir}", flush=True)
        raise SystemExit(1)

    print("All tests passed.", flush=True)


if __name__ == "__main__":
    main()
