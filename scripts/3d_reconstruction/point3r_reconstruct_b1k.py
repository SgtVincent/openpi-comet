#!/usr/bin/env python3
"""Point3R inference on Behavior1K video for 3D reconstruction."""

import os
import sys
import glob
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POINT3R_ROOT = PROJECT_ROOT / "src" / "openpi" / "third_party" / "Point3R"
sys.path.insert(0, str(POINT3R_ROOT / "src"))

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

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
                return (task_dir.name.replace("task-", ""), video_path)
    return (None, None)


def load_video_frames(video_path: str, num_frames: int, stride: int = 1,
                     sampling_mode: str = "stride", target_size: int = 512,
                     start_frame: int | None = None, end_frame: int | None = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    clip_start = 0 if start_frame is None else max(0, int(start_frame))
    clip_end = total_frames if end_frame is None or int(end_frame) <= 0 else min(total_frames, int(end_frame))
    if clip_end <= clip_start:
        raise ValueError(f"Invalid clip range: start={clip_start}, end={clip_end}, total_frames={total_frames}")
    clip_indices = list(range(clip_start, clip_end))

    if sampling_mode == "uniform" and num_frames is not None:
        indices = clip_indices if len(clip_indices) <= num_frames else \
                  np.linspace(clip_start, clip_end - 1, num_frames, dtype=int).tolist()
    elif sampling_mode == "stride":
        indices = clip_indices[::stride][:num_frames]
    else:
        limit = min(num_frames or len(clip_indices), len(clip_indices))
        indices = clip_indices[:limit]

    cap.release()

    images_list = []
    for idx in indices:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame).resize((target_size, target_size), Image.BICUBIC)
            images_list.append(np.array(img_pil))

    return images_list, indices, fps, total_frames, clip_start, clip_end


def build_point3r_views(images_list, cam_name="head"):
    views = []
    H, W = images_list[0].shape[:2]

    for i, img_np in enumerate(images_list):
        img_tensor = torch.from_numpy(img_np).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)

        view = {
            "img": img_tensor.unsqueeze(0),
            "depth": torch.ones(1, H, W).float() * 5.0,
            "camera_pose": [torch.eye(4).float()],
            "instance": [f"{cam_name}_{i}"],
            "idx": [i],
            "valid_mask": torch.ones(1, H, W).bool(),
            "pseudo_focal": torch.tensor([W]).float(),
            "img_mask": torch.tensor(True),
            "ray_mask": torch.tensor(False),
            "true_shape": torch.tensor([[H, W]], dtype=torch.long),
            "world_to_camera": [torch.eye(4).float()],
        }
        views.append(view)

    return views


def build_point3r_model(checkpoint_path: str, device: str = "cuda"):
    import torch
    from math import inf
    from dust3r.point3r import Point3R, Point3RConfig
    from copy import deepcopy

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args_dict = ckpt["args"]
    args_str = args_dict["model"].replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if "landscape_only" not in args_str:
        args_str = args_str[:-2] + ", landscape_only=False))"
    else:
        args_str = args_str.replace(" ", "").replace("landscape_only=True", "landscape_only=False")
    assert "landscape_only=False" in args_str
    net = eval(args_str)
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    print(f"Loaded Point3R, missing={len(missing)}, unexpected={len(unexpected)}")
    return net.to(device)


def run_point3r_inference(images_list, checkpoint_path: str, device: str = "cuda",
                          size: int = 512, revisit: int = 1):
    from dust3r.utils.geometry import geotrf
    from copy import deepcopy

    print(f"Loading Point3R model from {checkpoint_path}...")
    model = build_point3r_model(checkpoint_path, device)
    model.eval()

    views = build_point3r_views(images_list)

    results = []
    with torch.no_grad():
        for ri in range(revisit):
            batch_r = deepcopy(views)
            for v in batch_r:
                for key in v:
                    if isinstance(v[key], torch.Tensor):
                        v[key] = v[key].to(device)

            with torch.cuda.amp.autocast(enabled=False):
                output = model(batch_r, point3r_tag=True)
            preds, views_out = output.ress, output.views

            for j, (pred, view) in enumerate(zip(preds, views_out)):
                cam_pose = pred.get("camera_pose")
                if cam_pose is None: cam_pose = np.eye(4)
                elif isinstance(cam_pose, torch.Tensor): cam_pose = cam_pose.cpu().numpy()
                
                results.append({
                    "frame_idx": j,
                    "img": view["img"][0].cpu().numpy().transpose(1, 2, 0),
                    "depth": pred.get("depth", None),
                    "conf": pred.get("conf", None),
                    "pts3d": pred.get("pts3d_in_other_view", pred.get("pts3d_in_self_view", None)),
                    "camera_pose": cam_pose,
                })

    return results


def save_pointcloud(results, output_dir: str, max_points: int = 500000):
    os.makedirs(output_dir, exist_ok=True)

    all_pts = []
    all_colors = []

    for r in results:
        img = r["img"]
        pts3d = r["pts3d"]
        if pts3d is None:
            continue

        if isinstance(pts3d, torch.Tensor):
            pts3d = pts3d.cpu().numpy()

        H, W = img.shape[:2]
        pts_flat = pts3d.reshape(-1, 3)
        color_flat = img.reshape(-1, 3)

        all_pts.append(pts_flat)
        all_colors.append(color_flat)

    all_pts = np.concatenate(all_pts, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    valid_mask = np.isfinite(all_pts).all(axis=1)
    all_pts = all_pts[valid_mask]
    all_colors = all_colors[valid_mask]

    center = np.mean(all_pts, axis=0)
    all_pts -= center

    if len(all_pts) > max_points:
        idx = np.linspace(0, len(all_pts) - 1, max_points, dtype=int)
        all_pts = all_pts[idx]
        all_colors = all_colors[idx]

    ply_path = os.path.join(output_dir, "point3r_reconstruction.ply")
    write_ply(ply_path, all_pts, all_colors)
    print(f"Saved point cloud to {ply_path} ({len(all_pts)} points)")

    for i, r in enumerate(results):
        npz_path = os.path.join(output_dir, f"frame_{i:04d}.npz")
        pts3d = r["pts3d"]
        if isinstance(pts3d, torch.Tensor):
            pts3d = pts3d.cpu().numpy()
        conf_val = r.get("conf")
        if isinstance(conf_val, torch.Tensor):
            conf_val = conf_val.cpu().numpy()
        cam_pose = r["camera_pose"]
        if isinstance(cam_pose, torch.Tensor):
            cam_pose = cam_pose.cpu().numpy()
        np.savez(npz_path,
                 img=r["img"],
                 pts3d=pts3d,
                 camera_pose=cam_pose,
                 conf=conf_val)
    print(f"Saved {len(results)} frame results")

    return ply_path


def write_ply(path, points, colors):
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        pts_i = np.ascontiguousarray(points.astype(np.float32))
        col_i = np.ascontiguousarray((colors * 255).astype(np.uint8))
        for pt, col in zip(pts_i, col_i):
            f.write(pt.tobytes())
            f.write(col.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos")
    parser.add_argument("--episode_id", type=int, default=10)
    parser.add_argument("--camera", type=str, default="head",
                        choices=["head", "left_wrist", "right_wrist"])
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--sampling_mode", type=str, default="stride",
                        choices=["stride", "uniform"])
    parser.add_argument("--checkpoint", type=str,
                        default="src/openpi/third_party/Point3R/src/checkpoints/point3r_512.pth")
    parser.add_argument("--output_dir", type=str, default="./outputs/point3r_exp")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max_points", type=int, default=500000)
    parser.add_argument("--start_frame", type=int, default=-1)
    parser.add_argument("--end_frame", type=int, default=-1)
    args = parser.parse_args()

    task_id, video_path = find_episode_in_tasks(args.data_root, args.episode_id)
    if task_id is None:
        print(f"Episode {args.episode_id} not found")
        return

    print(f"Episode {args.episode_id} found in task-{task_id}")
    print(f"Loading video frames from {video_path}...")

    clip_start = None if args.start_frame < 0 else args.start_frame
    clip_end = None if args.end_frame < 0 else args.end_frame

    images_list, indices, fps, total_frames, clip_start, clip_end = load_video_frames(
        str(video_path), args.num_frames, args.stride, args.sampling_mode, args.size, clip_start, clip_end
    )
    if not images_list:
        raise ValueError(f"No frames loaded from clip range {clip_start}:{clip_end}")
    print(f"Loaded {len(images_list)} frames, indices={indices}")

    os.makedirs(args.output_dir, exist_ok=True)
    for i, (img_np, idx) in enumerate(zip(images_list, indices)):
        img_path = os.path.join(args.output_dir, f"images/{i:04d}.png")
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        Image.fromarray(img_np).save(img_path)

    results = run_point3r_inference(
        images_list, args.checkpoint,
        device="cuda", size=args.size
    )

    ply_path = save_pointcloud(results, args.output_dir, args.max_points)

    metadata = {
        "episode_id": args.episode_id, "task_id": task_id,
        "camera": args.camera, "num_frames": args.num_frames,
        "stride": args.stride, "sampling_mode": args.sampling_mode,
        "indices": indices, "fps": fps,
        "clip_start_frame": clip_start,
        "clip_end_frame": clip_end,
        "total_video_frames": total_frames,
        "ply_path": ply_path
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
