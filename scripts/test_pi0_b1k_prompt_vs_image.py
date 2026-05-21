#!/usr/bin/env python3
"""Prompt-vs-image ablation test for PI0 on real B1K observations.

This script evaluates how sensitive a PI0 checkpoint is to:
1. task prompt changes, with the same real observation fixed
2. image/state changes, with the same prompt fixed

The script intentionally does NOT provide any subtask text input.
It consumes real BEHAVIOR-1K samples and writes:
- per-sample JSONL records
- a summary JSON with aggregate deltas and a coarse dominant signal verdict
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
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
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    if task_name in data and isinstance(data[task_name], dict):
        if "task" in data[task_name]:
            return str(data[task_name]["task"])
    return task_name.replace("_", " ")


def _sanitize_np(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _dataset_item_to_policy_obs(item: dict[str, Any], *, prompt: str) -> dict[str, Any]:
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

    obs["prompt"] = prompt
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
        raise KeyError(f"Missing policy observation keys: {missing}")
    return obs


def _load_dataset(
    *,
    repo_id: str,
    root: str,
    tasks: list[str],
    episodes: list[int] | None,
    fine_grained_level: int,
    tolerance_s: float,
    modalities: list[str],
    train_rgb_type: str,
    return_seg_instance: bool,
    skill_list: list[str],
    local_only: bool,
) -> Any:
    from behavior.learning.datas.dataset import BehaviorLeRobotDataset

    return BehaviorLeRobotDataset(
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
        chunk_streaming_using_keyframe=False,
        shuffle=False,
        fine_grained_level=fine_grained_level,
        return_seg_instance=return_seg_instance,
        train_rgb_type=train_rgb_type,
        skill_list=skill_list,
    )


def _infer_action(policy: Any, obs: dict[str, Any]) -> np.ndarray:
    out = policy.infer(obs)
    action = np.asarray(out["actions"], dtype=np.float64)
    return action


def _action_stats(action: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(action.shape),
        "min": float(action.min()),
        "max": float(action.max()),
        "mean": float(action.mean()),
        "std": float(action.std()),
    }


def _delta_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = a - b
    return {
        "l2": float(np.linalg.norm(diff.ravel())),
        "l1_mean": float(np.abs(diff).mean()),
        "linf": float(np.abs(diff).max()),
    }


def _median(vals: list[float]) -> float | None:
    return float(statistics.median(vals)) if vals else None


def _mean(vals: list[float]) -> float | None:
    return float(statistics.fmean(vals)) if vals else None


def _dominant_signal(prompt_vals: list[float], image_vals: list[float]) -> str:
    if not prompt_vals or not image_vals:
        return "unknown"
    mp = float(statistics.fmean(prompt_vals))
    mi = float(statistics.fmean(image_vals))
    thresh = 1e-6
    if mp < thresh and mi < thresh:
        return "weak_change"
    ratio = mp / max(mi, thresh)
    if ratio >= 1.5:
        return "text"
    if ratio <= (1.0 / 1.5):
        return "image"
    return "mixed"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prompt-vs-image ablation for PI0 on real B1K samples.")
    p.add_argument("--config", type=str, default="pi0_b1k_skill-pt50_pretrain_lr1e-4_1ep")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--behavior-dir", type=str, default=None)
    p.add_argument(
        "--dataset-root",
        type=str,
        default="/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos/",
    )
    p.add_argument("--task", type=str, default="turning_on_radio")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--stride", type=int, default=100)
    p.add_argument("--pair-offset", type=int, default=1, help="Partner sample offset for image ablation.")
    p.add_argument("--dummy-prompt", type=str, default="do something")
    p.add_argument("--empty-prompt", type=str, default="")
    p.add_argument("--compare-mode", choices=("prompt", "image", "both"), default="both")
    p.add_argument("--local-only", action="store_true")
    p.add_argument(
        "--out-jsonl",
        type=str,
        default="/tmp/test_pi0_b1k_prompt_vs_image.jsonl",
    )
    p.add_argument(
        "--summary-json",
        type=str,
        default="/tmp/test_pi0_b1k_prompt_vs_image_summary.json",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    behavior_dir = args.behavior_dir or os.environ.get("BEHAVIOR_DIR")
    if not behavior_dir:
        candidate = (REPO_ROOT.parent / "BEHAVIOR-1K").resolve()
        if candidate.exists():
            behavior_dir = str(candidate)
    _maybe_add_behavior_paths(behavior_dir)

    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config

    train_config = _config.get_config(args.config)
    ckpt_dir = _resolve_checkpoint_dir(train_config, args.checkpoint)
    device = _resolve_device(args.device)

    data_factory = train_config.data[0] if isinstance(train_config.data, (list, tuple)) else train_config.data
    base_cfg = data_factory.base_config
    dataset_root = pathlib.Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"--dataset-root does not exist: {dataset_root}")
    if not (dataset_root / "data").exists():
        raise FileNotFoundError(f"--dataset-root missing data/: {dataset_root / 'data'}")

    prompt = _load_task_prompt(args.task)

    data_config = data_factory.create(train_config.assets_dirs, train_config.model)
    policy = _policy_config.create_trained_policy(
        train_config,
        ckpt_dir,
        sample_kwargs={"num_steps": 10},
        norm_stats=data_config.norm_stats,
        pytorch_device=device,
    )

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
    )

    out_jsonl = pathlib.Path(args.out_jsonl).resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json = pathlib.Path(args.summary_json).resolve()
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    prompt_dummy_l2: list[float] = []
    prompt_empty_l2: list[float] = []
    image_swap_l2: list[float] = []
    prompt_dummy_l1_mean: list[float] = []
    prompt_empty_l1_mean: list[float] = []
    image_swap_l1_mean: list[float] = []
    prompt_records: list[dict[str, Any]] = []
    image_records: list[dict[str, Any]] = []

    print(f"config={args.config}")
    print(f"checkpoint={ckpt_dir}")
    print(f"device={device}")
    print(f"dataset_root={dataset_root}")
    print(f"task={args.task}")
    print(f"baseline_prompt={prompt!r}")
    print(f"dataset_len={len(dataset)}")

    with out_jsonl.open("w", encoding="utf-8") as f:
        for i in range(args.num_samples):
            base_idx = args.sample_index + i * args.stride
            if base_idx >= len(dataset):
                break
            pair_idx = base_idx + args.pair_offset
            if pair_idx >= len(dataset):
                pair_idx = max(0, base_idx - args.pair_offset)
            if pair_idx == base_idx and len(dataset) > 1:
                pair_idx = (base_idx + 1) % len(dataset)

            base_item = dataset[base_idx]
            pair_item = dataset[pair_idx]

            base_obs = _dataset_item_to_policy_obs(base_item, prompt=prompt)
            baseline_action = _infer_action(policy, base_obs)

            baseline_record = {
                "sample_idx": base_idx,
                "pair_idx": pair_idx,
                "has_subtask_text_input": False,
                "prompt_variant": "baseline",
                "prompt": prompt,
                "action_stats": _action_stats(baseline_action),
            }
            f.write(json.dumps(baseline_record, ensure_ascii=False) + "\n")

            if args.compare_mode in ("prompt", "both"):
                dummy_obs = dict(base_obs)
                dummy_obs["prompt"] = args.dummy_prompt
                dummy_action = _infer_action(policy, dummy_obs)
                dummy_delta = _delta_stats(baseline_action, dummy_action)
                prompt_dummy_l2.append(dummy_delta["l2"])
                prompt_dummy_l1_mean.append(dummy_delta["l1_mean"])
                rec = {
                    "sample_idx": base_idx,
                    "pair_idx": pair_idx,
                    "has_subtask_text_input": False,
                    "prompt_variant": "dummy",
                    "prompt": args.dummy_prompt,
                    "baseline_prompt": prompt,
                    "action_stats": _action_stats(dummy_action),
                    "delta_vs_baseline": dummy_delta,
                }
                prompt_records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                empty_obs = dict(base_obs)
                empty_obs["prompt"] = args.empty_prompt
                empty_action = _infer_action(policy, empty_obs)
                empty_delta = _delta_stats(baseline_action, empty_action)
                prompt_empty_l2.append(empty_delta["l2"])
                prompt_empty_l1_mean.append(empty_delta["l1_mean"])
                rec = {
                    "sample_idx": base_idx,
                    "pair_idx": pair_idx,
                    "has_subtask_text_input": False,
                    "prompt_variant": "empty",
                    "prompt": args.empty_prompt,
                    "baseline_prompt": prompt,
                    "action_stats": _action_stats(empty_action),
                    "delta_vs_baseline": empty_delta,
                }
                prompt_records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if args.compare_mode in ("image", "both"):
                swapped_obs = _dataset_item_to_policy_obs(pair_item, prompt=prompt)
                swapped_action = _infer_action(policy, swapped_obs)
                swap_delta = _delta_stats(baseline_action, swapped_action)
                image_swap_l2.append(swap_delta["l2"])
                image_swap_l1_mean.append(swap_delta["l1_mean"])
                rec = {
                    "sample_idx": base_idx,
                    "pair_idx": pair_idx,
                    "has_subtask_text_input": False,
                    "prompt_variant": "swapped_image",
                    "prompt": prompt,
                    "baseline_prompt": prompt,
                    "action_stats": _action_stats(swapped_action),
                    "delta_vs_baseline": swap_delta,
                }
                image_records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    prompt_signal_vals = prompt_dummy_l2 + prompt_empty_l2
    image_signal_vals = image_swap_l2
    summary = {
        "config": args.config,
        "checkpoint": str(ckpt_dir),
        "dataset_root": str(dataset_root),
        "task": args.task,
        "baseline_prompt": prompt,
        "compare_mode": args.compare_mode,
        "num_samples_requested": args.num_samples,
        "prompt_dummy_l2": {
            "mean": _mean(prompt_dummy_l2),
            "median": _median(prompt_dummy_l2),
            "max": max(prompt_dummy_l2) if prompt_dummy_l2 else None,
        },
        "prompt_empty_l2": {
            "mean": _mean(prompt_empty_l2),
            "median": _median(prompt_empty_l2),
            "max": max(prompt_empty_l2) if prompt_empty_l2 else None,
        },
        "image_swap_l2": {
            "mean": _mean(image_swap_l2),
            "median": _median(image_swap_l2),
            "max": max(image_swap_l2) if image_swap_l2 else None,
        },
        "prompt_dummy_l1_mean": {
            "mean": _mean(prompt_dummy_l1_mean),
            "median": _median(prompt_dummy_l1_mean),
            "max": max(prompt_dummy_l1_mean) if prompt_dummy_l1_mean else None,
        },
        "prompt_empty_l1_mean": {
            "mean": _mean(prompt_empty_l1_mean),
            "median": _median(prompt_empty_l1_mean),
            "max": max(prompt_empty_l1_mean) if prompt_empty_l1_mean else None,
        },
        "image_swap_l1_mean": {
            "mean": _mean(image_swap_l1_mean),
            "median": _median(image_swap_l1_mean),
            "max": max(image_swap_l1_mean) if image_swap_l1_mean else None,
        },
        "dominant_signal": _dominant_signal(prompt_signal_vals, image_signal_vals),
        "representative_records": {
            "max_prompt_delta": max(prompt_records, key=lambda r: r["delta_vs_baseline"]["l2"], default=None),
            "max_image_delta": max(image_records, key=lambda r: r["delta_vs_baseline"]["l2"], default=None),
        },
        "out_jsonl": str(out_jsonl),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("summary_json=", summary_json)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

