#!/usr/bin/env python3
"""InfiniteVGGT streaming reconstruction on Behavior1K video."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INFINITEVGGT_ROOT = PROJECT_ROOT / "src" / "openpi" / "third_party" / "InfiniteVGGT"
sys.path.insert(0, str(INFINITEVGGT_ROOT))
sys.path.insert(0, str(INFINITEVGGT_ROOT / "src"))

from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.load_fn import load_and_preprocess_images


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


CAMERA_KEY_MAP = {
    "head": "observation.images.rgb.head",
    "left_wrist": "observation.images.rgb.left_wrist",
    "right_wrist": "observation.images.rgb.right_wrist",
}


def find_episode_video(data_root: str, episode_id: int, camera: str):
    episode_str = f"episode_{episode_id:08d}.mp4"
    videos_dir = Path(data_root) / "videos"
    camera_key = CAMERA_KEY_MAP[camera]
    for task_dir in sorted(videos_dir.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task-"):
            continue
        video_path = task_dir / camera_key / episode_str
        if video_path.exists():
            return task_dir.name.replace("task-", ""), str(video_path)
    return None, None


def ensure_empty_dir(path: str, overwrite: bool):
    if not os.path.exists(path):
        return
    if overwrite:
        return
    entries = [p for p in os.listdir(path) if p not in [".gitkeep"]]
    if entries:
        raise FileExistsError(f"Output dir not empty: {path} (use --overwrite)")


def extract_raw_frames_to_dir(video_path: str, frame_indices: list[int], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    paths = []
    for local_idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise ValueError(f"Failed to read frame {frame_idx} from {video_path}")
        out_path = os.path.join(out_dir, f"{local_idx:04d}.png")
        cv2.imwrite(out_path, frame)
        paths.append(out_path)

    cap.release()
    return paths


def build_windows(sampled_indices: list[int], window_size: int, window_step: int):
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    if window_step <= 0:
        raise ValueError("window_step must be > 0")
    if len(sampled_indices) < 2:
        raise ValueError("Need at least 2 sampled frames")

    if len(sampled_indices) <= window_size:
        return [sampled_indices]

    windows = []
    for start in range(0, len(sampled_indices) - window_size + 1, window_step):
        windows.append(sampled_indices[start:start + window_size])
    tail = sampled_indices[-window_size:]
    if not windows or windows[-1] != tail:
        windows.append(tail)
    return windows


def load_model(checkpoint_path: str, device: str, total_budget: int):
    if os.path.exists(checkpoint_path):
        model = StreamVGGT(total_budget=total_budget)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        del ckpt
        return model.to(device).eval()

    if "/" in checkpoint_path:
        model = StreamVGGT.from_pretrained(checkpoint_path)
        model.total_budget = total_budget
        return model.to(device).eval()

    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def run_stream_inference(model, image_paths: list[str], device: str):
    images = load_and_preprocess_images(image_paths).to(device)
    frames = [{"img": images[i].unsqueeze(0)} for i in range(images.shape[0])]

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            output = model.inference(frames, frame_writer=None, cache_results=True)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return images.detach().cpu(), output, elapsed, peak_gb


def save_window(window_dir: str, window_id: int, source_frame_indices: list[int], images_chw: torch.Tensor, output, elapsed: float, peak_gb: float):
    os.makedirs(window_dir, exist_ok=True)
    images_dir = os.path.join(window_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    images_uint8 = (images_chw.permute(0, 2, 3, 1).numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)

    ress = output.ress or []
    if len(ress) != images_uint8.shape[0]:
        raise ValueError(f"Unexpected output frames: images={images_uint8.shape[0]} ress={len(ress)}")

    for local_idx, (img, res) in enumerate(zip(images_uint8, ress)):
        cv2.imwrite(os.path.join(images_dir, f"{local_idx:04d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        point_map = res["pts3d_in_other_view"]
        point_conf = res["conf"]
        if isinstance(point_map, torch.Tensor):
            point_map = point_map.detach().cpu().numpy()
        if isinstance(point_conf, torch.Tensor):
            point_conf = point_conf.detach().cpu().numpy()
        np.savez_compressed(
            os.path.join(window_dir, f"frame_{local_idx:04d}.npz"),
            point_map=point_map.astype(np.float32),
            point_conf=point_conf.astype(np.float32),
            source_frame_index=np.array(source_frame_indices[local_idx], dtype=np.int32),
            local_frame_index=np.array(local_idx, dtype=np.int32),
            window_id=np.array(window_id, dtype=np.int32),
        )

    window_meta = {
        "window_id": window_id,
        "num_frames": len(source_frame_indices),
        "source_frame_indices": source_frame_indices,
        "elapsed_seconds": elapsed,
        "peak_memory_gb": peak_gb,
    }
    with open(os.path.join(window_dir, "metadata.json"), "w") as f:
        json.dump(window_meta, f, indent=2)
    return window_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos",
    )
    parser.add_argument("--episode_id", type=int, default=130010)
    parser.add_argument("--camera", type=str, default="head", choices=list(CAMERA_KEY_MAP.keys()))
    parser.add_argument("--sampling_stride", type=int, default=15)
    parser.add_argument("--max_sampled_frames", type=int, default=0)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--window_step", type=int, default=16)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--total_budget", type=int, default=1200000)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start_frame", type=int, default=-1)
    parser.add_argument("--end_frame", type=int, default=-1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for InfiniteVGGT inference")

    ensure_empty_dir(args.output_dir, args.overwrite)

    task_id, video_path = find_episode_video(args.data_root, args.episode_id, args.camera)
    if task_id is None or video_path is None:
        raise ValueError(f"Episode {args.episode_id} with camera={args.camera} not found under {args.data_root}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    clip_start = 0 if args.start_frame < 0 else max(0, int(args.start_frame))
    clip_end = total_frames if args.end_frame < 0 else min(total_frames, int(args.end_frame))
    if clip_end <= clip_start:
        raise ValueError(f"Invalid clip range: start={clip_start}, end={clip_end}, total_frames={total_frames}")

    sampled_indices = list(range(clip_start, clip_end, int(args.sampling_stride)))
    if args.max_sampled_frames > 0:
        sampled_indices = sampled_indices[: int(args.max_sampled_frames)]
    if len(sampled_indices) < 2:
        raise ValueError(f"Need at least 2 sampled frames, got {len(sampled_indices)} from clip {clip_start}:{clip_end}")

    windows = build_windows(sampled_indices, int(args.window_size), int(args.window_step))

    device = "cuda"
    model = load_model(args.checkpoint_path, device=device, total_budget=int(args.total_budget))

    os.makedirs(args.output_dir, exist_ok=True)
    top_meta = {
        "model": "InfiniteVGGT",
        "episode_id": args.episode_id,
        "task_id": task_id,
        "camera": args.camera,
        "video_path": video_path,
        "total_video_frames": total_frames,
        "clip_start_frame": clip_start,
        "clip_end_frame": clip_end,
        "fps": fps,
        "sampling_stride": args.sampling_stride,
        "max_sampled_frames": args.max_sampled_frames,
        "total_sampled_frames": len(sampled_indices),
        "window_size": args.window_size,
        "window_step": args.window_step,
        "num_windows": len(windows),
        "checkpoint_path": args.checkpoint_path,
        "total_budget": args.total_budget,
        "window_dirs": [],
        "window_metrics": [],
    }

    total_elapsed = 0.0
    peak_gb = 0.0
    for window_id, frame_indices in enumerate(windows):
        print(f"Window {window_id + 1}/{len(windows)}: {frame_indices[0]}-{frame_indices[-1]} ({len(frame_indices)} frames)")
        scratch_dir = os.path.join(args.output_dir, "_scratch", f"window_{window_id:04d}")
        raw_paths = extract_raw_frames_to_dir(video_path, frame_indices, scratch_dir)
        images, output, elapsed, window_peak_gb = run_stream_inference(model, raw_paths, device=device)
        window_dir = os.path.join(args.output_dir, f"window_{window_id:04d}")
        window_meta = save_window(
            window_dir=window_dir,
            window_id=window_id,
            source_frame_indices=frame_indices,
            images_chw=images,
            output=output,
            elapsed=elapsed,
            peak_gb=window_peak_gb,
        )
        top_meta["window_dirs"].append(os.path.basename(window_dir))
        top_meta["window_metrics"].append(window_meta)
        total_elapsed += elapsed
        peak_gb = max(peak_gb, window_peak_gb)

    top_meta["total_elapsed_seconds"] = total_elapsed
    top_meta["peak_memory_gb"] = peak_gb
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(top_meta, f, indent=2)

    print(f"Done! Results: {args.output_dir}")


if __name__ == "__main__":
    main()
