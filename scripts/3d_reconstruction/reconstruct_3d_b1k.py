#!/usr/bin/env python3
"""3D Reconstruction Experiment for VLM2 on Behavior1K"""

import os, sys, json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import argparse
import numpy as np
import torch
from PIL import Image
import cv2

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "openpi" / "third_party" / "vggt"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "openpi" / "third_party" / "cut3r"))


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


CAMERA_KEY_MAP = {
    "head": "observation.images.rgb.head",
    "left_wrist": "observation.images.rgb.left_wrist",
    "right_wrist": "observation.images.rgb.right_wrist",
}

MAX_VGGT_FRAMES = 128


def find_episode_in_tasks(data_root: str, episode_id: int) -> Tuple[str, Path]:
    episode_str = f"episode_{episode_id:08d}.mp4"
    videos_dir = Path(data_root) / "videos"
    for task_dir in sorted(videos_dir.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task-"):
            continue
        for cam_key in CAMERA_KEY_MAP.values():
            video_path = task_dir / cam_key / episode_str
            if video_path.exists():
                return (task_dir.name.replace("task-", ""), video_path)
    return (None, None)


def run_experiment(data_root: str, episode_id: int, cameras: List[str], num_frames: int,
    sampling_stride: int, sampling_mode: str, output_dir: str,
    vggt_model: str = "facebook/VGGT-1B", target_size: int = 518, seed: int = 42) -> dict:
    set_seed(seed)
    print(f"\n{'='*60}\n3D Reconstruction Experiment\n{'='*60}")
    print(f"  Episode ID: {episode_id}")
    print(f"  Cameras: {cameras}")
    print(f"  Num frames: {num_frames}")
    print(f"  Sampling: {sampling_mode} (stride={sampling_stride})")
    print(f"  Output: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    task_id, _ = find_episode_in_tasks(data_root, episode_id)
    if task_id is None:
        return {"error": f"Episode {episode_id} not found"}
    
    episode_str = f"episode_{episode_id:08d}.mp4"
    print(f"  Found episode {episode_id} in task-{task_id}")
    
    video_paths = {}
    for cam in cameras:
        cam_key = CAMERA_KEY_MAP.get(cam, cam)
        video_path = os.path.join(data_root, "videos", f"task-{task_id}", cam_key, episode_str)
        if os.path.exists(video_path):
            video_paths[cam] = video_path
            print(f"  Found: {cam}")
        else:
            print(f"  Not found: {cam}")
    
    if not video_paths:
        return {"error": f"No videos found for episode {episode_id}"}
    
    print("\nLoading images...")
    all_images = {cam: [] for cam in cameras}
    
    for cam, video_path in video_paths.items():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  Warning: Cannot open: {cam}")
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if sampling_mode == "uniform" and num_frames is not None:
            indices = list(range(total_frames)) if total_frames <= num_frames else np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
        elif sampling_mode == "stride":
            indices = list(range(0, total_frames, sampling_stride))
            if num_frames is not None and num_frames > 0:
                indices = indices[:num_frames]
        else:
            indices = list(range(min(num_frames or total_frames, total_frames)))
        
        print(f"  Camera {cam}: total={total_frames}, sampled={len(indices)}, fps={fps:.2f}")
        if len(indices) > MAX_VGGT_FRAMES:
            raise ValueError(
                f"VGGT full-episode / long-sequence reconstruction is disabled: sampled {len(indices)} frames exceeds limit {MAX_VGGT_FRAMES}. "
                "Use bounded sampling for VGGT or switch to Point3R for full-episode reconstruction."
            )
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                all_images[cam].append(Image.fromarray(frame))
        cap.release()
    
    total_images = sum(len(imgs) for imgs in all_images.values())
    if total_images == 0:
        return {"error": "No images loaded"}
    print(f"Total images: {total_images}")
    
    all_images_list = []
    for cam in sorted(all_images.keys()):
        all_images_list.extend(all_images[cam])
    
    images_np = np.stack([np.array(img.resize((target_size, target_size), Image.BICUBIC)) for img in all_images_list])
    images_tensor = torch.from_numpy(images_np).float() / 255.0
    images_tensor = images_tensor.permute(0, 3, 1, 2)
    images_tensor = images_tensor.unsqueeze(0)
    print(f"  Input shape: {images_tensor.shape}")

    print(f"\nSaving images to {output_dir}/images...")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    for i, img in enumerate(all_images_list):
        img.save(os.path.join(images_dir, f"{i:04d}.png"))
    
    print("\nRunning VGGT reconstruction...")
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading VGGT model...")
    
    local_ckpt = "checkpoints/openpi_comet/vggt/model.pt"
    if os.path.exists(local_ckpt):
        print(f"  Loading from local checkpoint: {local_ckpt}")
        model = VGGT()
        model.load_state_dict(torch.load(local_ckpt, map_location="cpu"))
    else:
        print(f"  Loading from HuggingFace: {vggt_model}")
        model = VGGT.from_pretrained(vggt_model)
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            aggregated_tokens_list, ps_idx = model.aggregator(images_tensor.to(device))
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_tensor.shape[-2:])
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_tensor.to(device), ps_idx)
            point_map, point_conf = model.point_head(aggregated_tokens_list, images_tensor.to(device), ps_idx)
    
    results = {
        "extrinsic": extrinsic.squeeze(0).cpu().numpy(),
        "intrinsic": intrinsic.squeeze(0).cpu().numpy(),
        "depth_map": depth_map.squeeze(0).cpu().numpy(),
        "depth_conf": depth_conf.squeeze(0).cpu().numpy(),
        "point_map": point_map.squeeze(0).cpu().numpy(),
        "point_conf": point_conf.squeeze(0).cpu().numpy(),
    }
    
    for i in range(len(results["extrinsic"])):
        np.savez(os.path.join(output_dir, f"frame_{i:04d}.npz"),
            extrinsic=results["extrinsic"][i], intrinsic=results["intrinsic"][i],
            depth=results["depth_map"][i], conf=results["depth_conf"][i])
    print(f"  Saved {len(results['extrinsic'])} frames to {output_dir}")
    
    metadata = {"episode_id": episode_id, "cameras": cameras, "num_frames": num_frames,
        "sampling_stride": sampling_stride, "sampling_mode": sampling_mode,
        "total_images": total_images, "vggt_model": vggt_model, "target_size": target_size}
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
        default="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos")
    parser.add_argument("--episode_id", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./outputs/3d_recon_exp_v2")
    parser.add_argument("--cameras", type=str, default="head,left_wrist,right_wrist")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--sampling_stride", type=int, default=4)
    parser.add_argument("--sampling_mode", type=str, default="uniform",
        choices=["uniform", "stride", "all"])
    parser.add_argument("--vggt_model", type=str, default="facebook/VGGT-1B")
    parser.add_argument("--target_size", type=int, default=518)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_frames is not None and args.num_frames <= 0:
        raise SystemExit(
            "VGGT full-episode reconstruction is disabled. Please set a positive --num_frames value or use Point3R for full-episode runs."
        )
    cameras = [c.strip() for c in args.cameras.split(",")]
    exp_name = f"ep{args.episode_id}_vggt_{'-'.join(cameras)}_n{args.num_frames}_s{args.sampling_stride}_{args.sampling_mode}"
    output_dir = os.path.join(args.output_dir, exp_name)
    run_experiment(args.data_root, args.episode_id, cameras, args.num_frames,
        args.sampling_stride, args.sampling_mode, output_dir, args.vggt_model,
        args.target_size, args.seed)


if __name__ == "__main__":
    main()
