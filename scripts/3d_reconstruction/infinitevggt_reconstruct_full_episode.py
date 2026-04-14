#!/usr/bin/env python3
"""InfiniteVGGT streaming reconstruction on full Behavior1K episode with sliding window.

This mirrors scripts/3d_reconstruction/point3r_reconstruct_full_episode.py:
- Sample frames with stride S across the whole video.
- Run sliding windows of N frames (advance by window_step, default N//2).
- Align windows to a global frame using overlap anchor camera poses.
- Save per-source-frame frame_{frame_index:04d}.npz in Point3R-compatible format:
    img, pts3d, conf, camera_pose
- Save a merged point cloud PLY.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INFINITEVGGT_ROOT = PROJECT_ROOT / "src" / "openpi" / "third_party" / "InfiniteVGGT"
sys.path.insert(0, str(INFINITEVGGT_ROOT))
sys.path.insert(0, str(INFINITEVGGT_ROOT / "src"))

from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.load_fn import load_and_preprocess_images
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
from streamvggt.utils.geometry import closed_form_inverse_se3


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


def extract_raw_frames(video_path: str, frame_indices: list[int], out_dir: str):
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
            break
        out_path = os.path.join(out_dir, f"{local_idx:04d}.png")
        cv2.imwrite(out_path, frame)
        paths.append(out_path)
    cap.release()
    return paths


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


def pose_enc_to_c2w(pose_enc: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    pose_enc = pose_enc.reshape(1, 1, 9)
    extrinsics, _ = pose_encoding_to_extri_intri(pose_enc, image_hw)
    extrinsic = extrinsics[0, 0]
    extrinsic_h = torch.eye(4, dtype=extrinsic.dtype, device=extrinsic.device)
    extrinsic_h[:3, :4] = extrinsic
    c2w = closed_form_inverse_se3(extrinsic_h[None])[0]
    return c2w


def transform_points(pts3d: np.ndarray, transform: np.ndarray):
    pts = np.asarray(pts3d, dtype=np.float32)
    original_shape = pts.shape
    if pts.ndim == 4:
        pts = pts.squeeze(0)
    pts_flat = pts.reshape(-1, 3)
    pts_global = (pts_flat @ transform[:3, :3].T) + transform[:3, 3]
    pts_global = pts_global.reshape(pts.shape)
    if len(original_shape) == 4:
        return pts_global[None]
    return pts_global


def write_ply(path: str, points: np.ndarray, colors: np.ndarray):
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


def run_window(model, image_paths: list[str], device: str):
    images = load_and_preprocess_images(image_paths).to(device)
    frames = [{"img": images[i].unsqueeze(0)} for i in range(images.shape[0])]
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            output = model.inference(frames, frame_writer=None, cache_results=True)
    return images.detach().cpu(), output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos")
    parser.add_argument("--episode_id", type=int, default=130010)
    parser.add_argument("--camera", type=str, default="head", choices=["head", "left_wrist", "right_wrist"])
    parser.add_argument("--num_frames_per_window", type=int, default=16, help="Number of frames per InfiniteVGGT inference window")
    parser.add_argument("--stride", type=int, default=15, help="Stride between consecutive sampled frames")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--total_budget", type=int, default=1200000)
    parser.add_argument("--output_dir", type=str, default="./outputs/infinitevggt_exp/task0000_stride15_full")
    parser.add_argument("--max_points", type=int, default=500000)
    parser.add_argument("--window_step", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for InfiniteVGGT inference")

    task_id, video_path = find_episode_video(args.data_root, args.episode_id, args.camera)
    if task_id is None or video_path is None:
        raise ValueError(f"Episode {args.episode_id} not found")

    if os.path.exists(args.output_dir) and (not args.overwrite) and os.listdir(args.output_dir):
        raise FileExistsError(f"Output dir not empty: {args.output_dir} (use --overwrite)")
    os.makedirs(args.output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Episode {args.episode_id} in task-{task_id}")
    print(f"Total video frames: {total_frames}, FPS: {fps}")

    all_indices = list(range(0, total_frames, int(args.stride)))
    print(f"Stride={args.stride} -> {len(all_indices)} sampled frames: {all_indices[:5]}...{all_indices[-3:]}")

    n_win = int(args.num_frames_per_window)
    window_step = int(args.window_step) if int(args.window_step) > 0 else max(1, n_win // 2)
    if window_step >= n_win:
        raise ValueError(f"window_step must be smaller than num_frames_per_window, got {window_step} >= {n_win}")

    windows = []
    for start in range(0, len(all_indices), window_step):
        win = all_indices[start:start + n_win]
        if len(win) < 2:
            break
        windows.append(win)
    if len(all_indices) >= 2:
        tail = all_indices[max(0, len(all_indices) - n_win):]
        if len(tail) >= 2 and (not windows or windows[-1] != tail):
            windows.append(tail)
    print(f"Formed {len(windows)} windows, last window has {len(windows[-1])} frames, window_step={window_step}")

    device = "cuda"
    print(f"Loading StreamVGGT on {device}...")
    model = load_model(args.checkpoint_path, device=device, total_budget=int(args.total_budget))

    all_pts_global = []
    all_colors_global = []
    global_camera_poses = {}
    saved_frame_indices = set()

    for w_idx, win_indices in enumerate(windows):
        print(f"\nWindow {w_idx+1}/{len(windows)}: frames {win_indices[0]}-{win_indices[-1]} ({len(win_indices)} frames)")
        scratch_dir = os.path.join(args.output_dir, "_scratch", f"window_{w_idx:04d}")
        image_paths = extract_raw_frames(video_path, win_indices, scratch_dir)
        if len(image_paths) < 2:
            print(f"  Skipping window {w_idx}: only {len(image_paths)} frames loaded")
            continue

        images_chw, output = run_window(model, image_paths, device=device)
        images_uint8 = (images_chw.permute(0, 2, 3, 1).numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)

        ress = output.ress or []
        if len(ress) != len(win_indices):
            raise ValueError(f"Unexpected output frames: win_indices={len(win_indices)} ress={len(ress)}")

        local_camera_poses = {}
        local_pts_by_idx = {}
        local_conf_by_idx = {}
        for res, idx_glob in zip(ress, win_indices):
            pose_enc = res["camera_pose"]
            if isinstance(pose_enc, np.ndarray):
                pose_enc_t = torch.from_numpy(pose_enc).float()
            else:
                pose_enc_t = pose_enc.float()
            c2w = pose_enc_to_c2w(pose_enc_t, image_hw=(images_chw.shape[-2], images_chw.shape[-1])).detach().cpu().numpy().astype(np.float32)
            local_camera_poses[idx_glob] = c2w

            pts = res["pts3d_in_other_view"]
            conf = res["conf"]
            if isinstance(pts, torch.Tensor):
                pts = pts.detach().cpu().numpy()
            if isinstance(conf, torch.Tensor):
                conf = conf.detach().cpu().numpy()
            local_pts_by_idx[idx_glob] = pts.astype(np.float32)
            local_conf_by_idx[idx_glob] = conf.astype(np.float32)

        overlap = [idx for idx in win_indices if idx in global_camera_poses]
        if overlap:
            anchor = overlap[-1]
            transform_local_to_global = global_camera_poses[anchor] @ np.linalg.inv(local_camera_poses[anchor])
            print(f"  Aligning window with anchor frame {anchor}")
        else:
            transform_local_to_global = np.eye(4, dtype=np.float32)
            print("  No overlap anchor found, using identity alignment")

        win_pts = []
        win_colors = []
        for local_pos, idx_glob in enumerate(win_indices):
            if idx_glob in saved_frame_indices:
                continue

            pts_local = local_pts_by_idx.get(idx_glob)
            if pts_local is None:
                continue
            pts_global = transform_points(pts_local, transform_local_to_global)
            global_pose = transform_local_to_global @ local_camera_poses[idx_glob]
            if idx_glob not in global_camera_poses:
                global_camera_poses[idx_glob] = global_pose

            img = images_uint8[local_pos].astype(np.float32) / 255.0
            conf = local_conf_by_idx.get(idx_glob)

            pts_flat = pts_global.reshape(-1, 3)
            col_flat = img.reshape(-1, 3)
            valid = np.isfinite(pts_flat).all(axis=1)
            win_pts.append(pts_flat[valid])
            win_colors.append(col_flat[valid])

            npz_path = os.path.join(args.output_dir, f"frame_{idx_glob:04d}.npz")
            np.savez_compressed(
                npz_path,
                img=img.astype(np.float32),
                pts3d=pts_global.astype(np.float32),
                conf=None if conf is None else conf.astype(np.float32),
                camera_pose=global_pose.astype(np.float32),
            )
            saved_frame_indices.add(idx_glob)

        if win_pts:
            win_pts_all = np.concatenate(win_pts, axis=0)
            win_col_all = np.concatenate(win_colors, axis=0)
            all_pts_global.append(win_pts_all)
            all_colors_global.append(win_col_all)
            print(f"  New window pts: {len(win_pts_all)}, Saved frames: {len(saved_frame_indices)}/{len(all_indices)}, Running total: {sum(len(p) for p in all_pts_global)}")

    if not all_pts_global:
        print("No valid point clouds produced!")
        return

    all_pts = np.concatenate(all_pts_global, axis=0)
    all_colors = np.concatenate(all_colors_global, axis=0)
    valid = np.isfinite(all_pts).all(axis=1)
    all_pts, all_colors = all_pts[valid], all_colors[valid]
    center = np.mean(all_pts, axis=0)
    all_pts -= center
    if len(all_pts) > int(args.max_points):
        idx = np.linspace(0, len(all_pts) - 1, int(args.max_points), dtype=int)
        all_pts, all_colors = all_pts[idx], all_colors[idx]

    ply_path = os.path.join(args.output_dir, "infinitevggt_reconstruction.ply")
    write_ply(ply_path, all_pts, all_colors)
    print(f"\nSaved merged point cloud: {ply_path} ({len(all_pts)} pts)")

    metadata = {
        "model": "InfiniteVGGT",
        "episode_id": args.episode_id,
        "task_id": task_id,
        "camera": args.camera,
        "num_windows": len(windows),
        "num_frames_per_window": int(args.num_frames_per_window),
        "window_step": window_step,
        "stride": int(args.stride),
        "total_sampled_frames": len(all_indices),
        "total_video_frames": total_frames,
        "fps": fps,
        "checkpoint_path": args.checkpoint_path,
        "total_budget": int(args.total_budget),
        "ply_path": ply_path,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Done! Results: {args.output_dir}")
    print(f"Metadata: {metadata}")


if __name__ == "__main__":
    main()
