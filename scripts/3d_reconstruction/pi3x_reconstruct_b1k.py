#!/usr/bin/env python3
"""Run Pi3X reconstruction on a Behavior1K episode video.

This is a thin wrapper around third_party/Pi3/example_mm.py (Pi3X inference).
It:
- Locates the B1K episode mp4 under data_root/videos/task-XXXX/<camera_key>/episode_XXXXXXXX.mp4
- Calls Pi3's example_mm.py to produce a colored point cloud (.ply)
- Writes a metadata.json next to the output for reproducibility

Note:
- Pi3's default model download uses HuggingFace. If download is slow, enable proxy
  (see project rules: `proxy_on`) or pass --ckpt to a local .safetensors.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PI3_ROOT = PROJECT_ROOT / "src" / "openpi" / "third_party" / "Pi3"

CAMERA_KEY_MAP = {
    "head": "observation.images.rgb.head",
    "left_wrist": "observation.images.rgb.left_wrist",
    "right_wrist": "observation.images.rgb.right_wrist",
}


def find_episode_video(data_root: str, episode_id: int, camera: str) -> tuple[str | None, str | None]:
    episode_str = f"episode_{episode_id:08d}.mp4"
    videos_dir = Path(data_root) / "videos"
    camera_key = CAMERA_KEY_MAP[camera]

    for task_dir in sorted(videos_dir.iterdir()):
        if (not task_dir.is_dir()) or (not task_dir.name.startswith("task-")):
            continue
        video_path = task_dir / camera_key / episode_str
        if video_path.exists():
            task_id = task_dir.name.replace("task-", "")
            return task_id, str(video_path)
    return None, None


def _short_ep_id(episode_id: int) -> int:
    # B1K convention: episode_id like 130010 means task 13, episode index 0010.
    # For readability in paths, keep the last 4 digits.
    return int(episode_id) % 10000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos",
    )
    parser.add_argument("--episode_id", type=int, default=130010)
    parser.add_argument("--camera", type=str, default="head", choices=sorted(CAMERA_KEY_MAP.keys()))
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Pi3X frame sampling interval (Pi3/example_mm.py --interval). For long B1K episodes, use 20-60.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional path to Pi3X weights (.safetensors or .pt). If omitted, Pi3X downloads from HF.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory. If empty, uses ./outputs/pi3x_exp/<exp_name>.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not PI3_ROOT.exists():
        raise FileNotFoundError(f"Pi3 repo not found at: {PI3_ROOT}")
    example_path = PI3_ROOT / "example_mm.py"
    if not example_path.exists():
        raise FileNotFoundError(f"Pi3X example script not found: {example_path}")

    task_id, video_path = find_episode_video(args.data_root, args.episode_id, args.camera)
    if task_id is None or video_path is None:
        raise ValueError(f"Episode {args.episode_id} not found under {args.data_root}/videos for camera={args.camera}")

    exp_name = f"task{int(task_id):04d}_ep{_short_ep_id(args.episode_id):04d}_{args.camera}_pi3x_i{int(args.interval)}"
    out_dir = Path(args.output_dir) if args.output_dir else (PROJECT_ROOT / "outputs" / "pi3x_exp" / exp_name)
    if out_dir.exists() and (not args.overwrite) and any(out_dir.iterdir()):
        raise FileExistsError(f"Output dir not empty: {out_dir} (use --overwrite)")
    out_dir.mkdir(parents=True, exist_ok=True)

    ply_path = out_dir / "pi3x_reconstruction.ply"
    cmd = [
        sys.executable,
        str(example_path),
        "--data_path",
        str(video_path),
        "--save_path",
        str(ply_path),
        "--interval",
        str(int(args.interval)),
        "--device",
        str(args.device),
    ]
    if args.ckpt:
        cmd += ["--ckpt", str(args.ckpt)]

    print(f"[Pi3X] task-{task_id} episode_id={args.episode_id} camera={args.camera}")
    print(f"[Pi3X] video: {video_path}")
    print(f"[Pi3X] output: {out_dir}")
    print(f"[Pi3X] running: {' '.join(cmd)}")

    env = os.environ.copy()
    # Ensure Pi3 imports resolve, regardless of how this wrapper is called.
    env["PYTHONPATH"] = str(PI3_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    # Run in Pi3 root so relative paths and local package imports work.
    subprocess.run(cmd, cwd=str(PI3_ROOT), env=env, check=True)

    metadata = {
        "model": "Pi3X",
        "task_id": int(task_id),
        "episode_id": int(args.episode_id),
        "episode_short": _short_ep_id(args.episode_id),
        "camera": args.camera,
        "interval": int(args.interval),
        "device": args.device,
        "video_path": str(video_path),
        "ply_path": str(ply_path),
        "ckpt": args.ckpt,
        "pi3_root": str(PI3_ROOT),
        "command": cmd,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Pi3X] done: {ply_path}")
    print(f"[Pi3X] metadata: {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()

