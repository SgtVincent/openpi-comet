#!/usr/bin/env python3
"""
Test PI05_SUBTASK subtask prediction on real BEHAVIOR-1K (B1K) observations.

This script loads real samples from BehaviorLeRobotDataset (local parquet files),
converts them into the online eval observation schema (observation/* keys),
and runs the OpenPi policy inference to get:
  - generated_subtask (high-level text)
  - actions (low-level continuous action chunk)

Typical usage:
  PYTHONPATH=... python scripts/test_pi05_subtask_b1k.py \
    --config pi05_b1k_skill-pt12_pretrain_lr1e-4_2ep \
    --checkpoint /path/to/30997 \
    --dataset-root /mnt/bn/.../2025-challenge-demos \
    --task turning_on_radio \
    --sample-index 0 --num-samples 8 --stride 50
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from collections import Counter
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch


def _maybe_add_behavior_paths(behavior_dir: str | None) -> None:
    if not behavior_dir:
        return
    bdir = pathlib.Path(behavior_dir).resolve()
    # Minimal set commonly required by BEHAVIOR-1K code paths.
    for p in (bdir / "joylo", bdir / "OmniGibson", bdir / "bddl3"):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_checkpoint_dir(train_config: Any, ckpt_arg: str | None) -> pathlib.Path:
    if ckpt_arg:
        ckpt_dir = pathlib.Path(ckpt_arg)
    elif getattr(train_config, "pytorch_weight_path", None):
        ckpt_dir = pathlib.Path(train_config.pytorch_weight_path)
        if not ckpt_dir.is_absolute():
            ckpt_dir = REPO_ROOT / ckpt_dir
    else:
        raise ValueError("Must provide --checkpoint or train_config.pytorch_weight_path")

    ckpt_dir = ckpt_dir.resolve()
    if not (ckpt_dir / "model.safetensors").exists():
        raise FileNotFoundError(f"No model.safetensors under checkpoint dir: {ckpt_dir}")
    return ckpt_dir


def _load_task_prompt(task_name: str) -> str:
    mapping_path = REPO_ROOT / "scripts" / "task_mapping.json"
    try:
        data = json.loads(mapping_path.read_text(encoding="utf-8"))
        if task_name in data and isinstance(data[task_name], dict) and "prompt" in data[task_name]:
            return str(data[task_name]["prompt"])
    except Exception:
        pass
    # Fallback: turn snake_case into a readable phrase.
    return task_name.replace("_", " ")


def _sanitize_np(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _dataset_item_to_policy_obs(item: dict[str, Any], *, prompt_override: str | None) -> dict[str, Any]:
    # The dataset item schema varies slightly across loaders; handle common keys.
    # We standardize to the online-eval policy wrapper schema:
    #   observation/egocentric_camera, observation/wrist_image_left, observation/wrist_image_right, observation/state, prompt
    obs: dict[str, Any] = {}
    for k, v in item.items():
        obs[k] = _sanitize_np(v)

    key_map = {
        "observation.images.rgb.head": "observation/egocentric_camera",
        "observation.images.rgb.left_wrist": "observation/wrist_image_left",
        "observation.images.rgb.right_wrist": "observation/wrist_image_right",
        "observation.state": "observation/state",
    }
    for src, dst in key_map.items():
        if src in obs and dst not in obs:
            obs[dst] = obs[src]

    if prompt_override is not None:
        obs["prompt"] = prompt_override
    else:
        # Try to use prompt/task in the item, otherwise caller should provide prompt_override.
        if "prompt" not in obs and "task" in obs and isinstance(obs["task"], str):
            obs["prompt"] = obs["task"]
    return obs


def _load_dataset(
    *,
    repo_id: str,
    root: str,
    tasks: list[str] | None,
    episodes: list[int] | None,
    fine_grained_level: int,
    tolerance_s: float,
    modalities: list[str],
    train_rgb_type: str,
    return_seg_instance: bool,
    skill_list: list[str],
    local_only: bool,
    chunk_streaming_using_keyframe: bool,
    shuffle: bool,
) -> Any:
    # Import from BEHAVIOR package; requires behavior env / PYTHONPATH.
    from behavior.learning.datas.dataset import BehaviorLeRobotDataset

    dataset_kwargs = dict(
        repo_id=repo_id,
        root=root,
        tolerance_s=tolerance_s,
        tasks=tasks,
        modalities=modalities,
        local_only=local_only,
        check_files=False,
        check_timestamp_sync=False,
        delta_timestamps=None,
        episodes=episodes or [],
        # NOTE: chunk_streaming_using_keyframe=True with shuffle=False currently
        # hits an internal BEHAVIOR bug (uninitialized _active_chunks). For a
        # stable index-based test, default to disabling streaming.
        chunk_streaming_using_keyframe=chunk_streaming_using_keyframe,
        shuffle=shuffle,
        fine_grained_level=fine_grained_level,
        return_seg_instance=return_seg_instance,
        train_rgb_type=train_rgb_type,
        skill_list=skill_list,
    )
    return BehaviorLeRobotDataset(**dataset_kwargs)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test PI05_SUBTASK subtask prediction on real B1K observations.")
    p.add_argument("--config", type=str, default="pi05_b1k_skill-pt12_pretrain_lr1e-4_2ep")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--behavior-dir", type=str, default=None)
    p.add_argument(
        "--dataset-root",
        type=str,
        default="/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos/",
        help="Local dataset root containing a 'data/' directory with task-xxxx subfolders.",
    )
    p.add_argument("--task", type=str, default="turning_on_radio")
    p.add_argument("--prompt", type=str, default=None, help="Override prompt. If omitted, uses task_mapping.json.")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--stride", type=int, default=50, help="Stride between sampled indices in the dataset.")
    p.add_argument("--local-only", action="store_true", help="Use local parquet files only (recommended).")
    p.add_argument(
        "--chunk-streaming",
        action="store_true",
        help="Enable keyframe chunk streaming mode (not recommended for this script unless you also enable --shuffle).",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Enable dataset shuffle. Only meaningful for streaming mode.",
    )
    p.add_argument("--max-gen-steps", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--num-action-steps", type=int, default=10)
    p.add_argument("--out-jsonl", type=str, default=None, help="Optional path to write per-sample outputs as jsonl.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    behavior_dir = args.behavior_dir or os.environ.get("BEHAVIOR_DIR")
    if not behavior_dir:
        # Common local layout: openpi-comet and BEHAVIOR-1K live in the same parent dir.
        candidate = (REPO_ROOT.parent / "BEHAVIOR-1K").resolve()
        if candidate.exists():
            behavior_dir = str(candidate)
    _maybe_add_behavior_paths(behavior_dir)

    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config

    train_config = _config.get_config(args.config)
    ckpt_dir = _resolve_checkpoint_dir(train_config, args.checkpoint)
    device = _resolve_device(args.device)

    # Resolve dataset settings from config (fallback to user overrides).
    data_factory = train_config.data[0] if isinstance(train_config.data, (list, tuple)) else train_config.data
    base_cfg = getattr(data_factory, "base_config", None)
    if base_cfg is None:
        raise ValueError("Train config does not expose data_factory.base_config; cannot locate dataset.")
    dataset_root = args.dataset_root or getattr(base_cfg, "behavior_dataset_root", None)
    if not dataset_root:
        raise ValueError("Missing dataset root. Provide --dataset-root or set in config.")
    dataset_root_path = pathlib.Path(dataset_root)
    if not dataset_root_path.exists():
        raise FileNotFoundError(f"--dataset-root does not exist: {dataset_root_path}")
    if not dataset_root_path.is_dir():
        raise NotADirectoryError(f"--dataset-root is not a directory: {dataset_root_path}")
    if not (dataset_root_path / "data").exists():
        raise FileNotFoundError(
            f"--dataset-root must contain a 'data/' directory. Missing: {dataset_root_path / 'data'}"
        )

    prompt = args.prompt or _load_task_prompt(args.task)

    # Build policy with normalization from the training data config.
    data_config = data_factory.create(train_config.assets_dirs, train_config.model)
    policy = _policy_config.create_trained_policy(
        train_config,
        ckpt_dir,
        sample_kwargs={
            "num_steps": args.num_action_steps,
            "max_subtask_tokens": args.max_gen_steps,
            "temperature": args.temperature,
        },
        norm_stats=data_config.norm_stats,
        pytorch_device=device,
    )

    # Load dataset.
    dataset = _load_dataset(
        repo_id=data_factory.repo_id,
        root=str(dataset_root),
        tasks=[args.task],
        episodes=getattr(base_cfg, "episodes_index", None),
        fine_grained_level=getattr(base_cfg, "fine_grained_level", 0),
        tolerance_s=getattr(base_cfg, "tolerance_s", 1e-4),
        modalities=getattr(base_cfg, "modalities", ["rgb"]),
        train_rgb_type=getattr(base_cfg, "train_rgb_type", "regular"),
        return_seg_instance=getattr(base_cfg, "return_seg_instance", False),
        skill_list=getattr(base_cfg, "skill_list", ["all"]),
        local_only=bool(args.local_only),
        chunk_streaming_using_keyframe=bool(args.chunk_streaming),
        shuffle=bool(args.shuffle),
    )

    out_path = pathlib.Path(args.out_jsonl).resolve() if args.out_jsonl else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w", encoding="utf-8")
    else:
        out_f = None

    counter: Counter[str] = Counter()
    print(f"config={args.config}")
    print(f"checkpoint={ckpt_dir}")
    print(f"device={device}")
    print(f"dataset_root={dataset_root}")
    print(f"task={args.task}")
    print(f"prompt={prompt!r}")
    print(f"dataset_len={len(dataset)}")
    print(f"sample_index={args.sample_index} num_samples={args.num_samples} stride={args.stride}")

    try:
        for i in range(args.num_samples):
            idx = args.sample_index + i * args.stride
            if idx >= len(dataset):
                break
            item = dataset[idx]
            if i == 0:
                print("first_item_keys=", sorted(list(item.keys()))[:60])
            obs = _dataset_item_to_policy_obs(item, prompt_override=prompt)
            # Ensure we never pass ground-truth subtask text into hierarchical inference.
            obs.pop("subtask_text", None)
            required = [
                "observation/state",
                "observation/egocentric_camera",
                "observation/wrist_image_left",
                "observation/wrist_image_right",
                "prompt",
            ]
            missing = [k for k in required if k not in obs]
            if missing:
                raise KeyError(f"Policy obs missing required keys: {missing}. Available keys: {sorted(list(obs.keys()))[:80]}")
            result = policy.infer(obs)
            subtask = result.get("generated_subtask")
            subtask_str = subtask if isinstance(subtask, str) else None
            if subtask_str is not None:
                counter[subtask_str] += 1
            rec = {
                "idx": idx,
                "task": args.task,
                "prompt": prompt,
                "generated_subtask": subtask_str,
                "action_min": float(np.asarray(result["actions"]).min()),
                "action_max": float(np.asarray(result["actions"]).max()),
                "action_shape": list(np.asarray(result["actions"]).shape),
            }
            print(json.dumps(rec, ensure_ascii=False))
            if out_f is not None:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        if out_f is not None:
            out_f.close()

    print("---- unique_non_empty_subtasks ----")
    for text, n in counter.most_common(20):
        print(f"{n}\t{text}")


if __name__ == "__main__":
    main()
