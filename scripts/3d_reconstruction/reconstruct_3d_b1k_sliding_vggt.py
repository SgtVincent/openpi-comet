#!/usr/bin/env python3
import os, sys, json, argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "openpi" / "third_party" / "vggt"))

CAMERA_KEY_MAP = {
    "head": "observation.images.rgb.head",
    "left_wrist": "observation.images.rgb.left_wrist",
    "right_wrist": "observation.images.rgb.right_wrist",
}


def find_episode_in_tasks(data_root: str, episode_id: int):
    episode_str = f"episode_{episode_id:08d}.mp4"
    videos_dir = Path(data_root) / "videos"
    for task_dir in sorted(videos_dir.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task-"):
            continue
        for cam_key in CAMERA_KEY_MAP.values():
            video_path = task_dir / cam_key / episode_str
            if video_path.exists():
                return task_dir.name.replace("task-", ""), video_path
    return None, None


def load_video_frames(video_path: str, frame_indices: list[int], target_size: int = 518):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    images = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Failed to read frame {idx} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame).resize((target_size, target_size), Image.Resampling.BICUBIC)
        images.append(img)
    cap.release()
    return images


def build_windows(sampled_indices: list[int], window_size: int, window_step: int):
    if len(sampled_indices) < window_size:
        raise ValueError(f"Need at least {window_size} sampled frames, got {len(sampled_indices)}")
    windows = []
    for start in range(0, len(sampled_indices) - window_size + 1, window_step):
        windows.append(sampled_indices[start:start + window_size])
    tail_window = sampled_indices[-window_size:]
    if not windows or windows[-1] != tail_window:
        windows.append(tail_window)
    return windows


def load_vggt_model(vggt_model: str):
    from vggt.models.vggt import VGGT

    local_ckpt = "checkpoints/openpi_comet/vggt/model.pt"
    if os.path.exists(local_ckpt):
        model = VGGT()
        model.load_state_dict(torch.load(local_ckpt, map_location="cpu"))
    else:
        model = VGGT.from_pretrained(vggt_model)
    return model


def run_window(images: list[Image.Image], model, device: str, dtype):
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    images_np = np.stack([np.array(img) for img in images])
    images_tensor = torch.from_numpy(images_np).float().permute(0, 3, 1, 2) / 255.0
    images_tensor = images_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            aggregated_tokens_list, ps_idx = model.aggregator(images_tensor)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_tensor.shape[-2:])
            point_map, point_conf = model.point_head(aggregated_tokens_list, images_tensor, ps_idx)
    return {
        "extrinsic": extrinsic.squeeze(0).cpu().numpy(),
        "intrinsic": intrinsic.squeeze(0).cpu().numpy(),
        "point_map": point_map.squeeze(0).cpu().numpy(),
        "point_conf": point_conf.squeeze(0).cpu().numpy(),
        "images": images_np.astype(np.uint8),
    }


def save_window(window_dir: str, window_id: int, frame_indices: list[int], results: dict):
    os.makedirs(window_dir, exist_ok=True)
    images_dir = os.path.join(window_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    for i, img in enumerate(results["images"]):
        Image.fromarray(img).save(os.path.join(images_dir, f"{i:04d}.png"))
        np.savez_compressed(
            os.path.join(window_dir, f"frame_{i:04d}.npz"),
            extrinsic=results["extrinsic"][i].astype(np.float32),
            intrinsic=results["intrinsic"][i].astype(np.float32),
            point_map=results["point_map"][i].astype(np.float32),
            point_conf=results["point_conf"][i].astype(np.float32),
            source_frame_index=np.array(frame_indices[i], dtype=np.int32),
            local_frame_index=np.array(i, dtype=np.int32),
            window_id=np.array(window_id, dtype=np.int32),
        )
    metadata = {
        "window_id": window_id,
        "num_frames": len(frame_indices),
        "source_frame_indices": frame_indices,
    }
    with open(os.path.join(window_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos")
    parser.add_argument("--episode_id", type=int, default=130010)
    parser.add_argument("--camera", type=str, default="head", choices=["head", "left_wrist", "right_wrist"])
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--sampling_stride", type=int, default=15)
    parser.add_argument("--window_step", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="./outputs/vggt_sliding_exp/task0013_stride15_w32")
    parser.add_argument("--vggt_model", type=str, default="facebook/VGGT-1B")
    parser.add_argument("--target_size", type=int, default=518)
    args = parser.parse_args()

    if args.window_step <= 0:
        raise ValueError("window_step must be positive")

    task_id, video_path = find_episode_in_tasks(args.data_root, args.episode_id)
    if task_id is None or video_path is None:
        raise ValueError(f"Episode {args.episode_id} not found")

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    sampled_indices = list(range(0, total_frames, args.sampling_stride))
    windows = build_windows(sampled_indices, args.num_frames, args.window_step)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = load_vggt_model(args.vggt_model).eval().to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    top_metadata = {
        "episode_id": args.episode_id,
        "task_id": task_id,
        "camera": args.camera,
        "num_frames": args.num_frames,
        "sampling_stride": args.sampling_stride,
        "window_step": args.window_step,
        "total_video_frames": total_frames,
        "total_sampled_frames": len(sampled_indices),
        "num_windows": len(windows),
        "fps": fps,
        "window_dirs": [],
    }

    print(f"Episode {args.episode_id} in task-{task_id}")
    print(f"Total video frames: {total_frames}, sampled every {args.sampling_stride} -> {len(sampled_indices)} frames")
    print(f"Sliding windows: {len(windows)}, num_frames={args.num_frames}, window_step={args.window_step}")

    for window_id, frame_indices in enumerate(windows):
        print(f"Window {window_id + 1}/{len(windows)}: {frame_indices[0]}-{frame_indices[-1]}")
        images = load_video_frames(str(video_path), frame_indices, args.target_size)
        results = run_window(images, model, device, dtype)
        window_dir = os.path.join(args.output_dir, f"window_{window_id:04d}")
        save_window(window_dir, window_id, frame_indices, results)
        top_metadata["window_dirs"].append(os.path.basename(window_dir))

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(top_metadata, f, indent=2)

    print(f"Done! Results: {args.output_dir}")


if __name__ == "__main__":
    main()
