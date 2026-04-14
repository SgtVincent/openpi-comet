#!/usr/bin/env python3
"""Point3R streaming reconstruction on full Behavior1K episode with sliding window.

Point3R processes N frames jointly. For a long episode with stride S,
we run a sliding window: each window advances by S frames.
All windows' pts3d predictions are merged into a single global point cloud.
"""
import os, sys, json, argparse
from pathlib import Path
import cv2, numpy as np, torch
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

def find_episode_video(data_root: str, episode_id: int):
    episode_str = f"episode_{episode_id:08d}.mp4"
    videos_dir = Path(data_root) / "videos"
    for task_dir in sorted(videos_dir.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task-"):
            continue
        for cam_key in CAMERA_KEY_MAP.values():
            video_path = task_dir / cam_key / episode_str
            if video_path.exists():
                return task_dir.name.replace("task-", ""), str(video_path)
    return (None, None)

def load_video_frames(video_path: str, frame_indices: list, target_size: int = 512):
    images_list = []
    for idx in frame_indices:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame).resize((target_size, target_size), Image.Resampling.BICUBIC)
        images_list.append(np.array(img_pil))
    return images_list

def build_point3r_views(images_list, cam_name="head"):
    from math import inf
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

def run_point3r_window(images_list, model, device: str = "cuda"):
    from copy import deepcopy
    views = build_point3r_views(images_list)
    with torch.no_grad():
        batch = deepcopy(views)
        for v in batch:
            for key in v:
                if isinstance(v[key], torch.Tensor):
                    v[key] = v[key].to(device)
        with torch.cuda.amp.autocast(enabled=False):
            output = model(batch, point3r_tag=True)
        preds, views_out = output.ress, output.views
    window_results = []
    for j, (pred, view) in enumerate(zip(preds, views_out)):
        cam_pose = pred.get("camera_pose")
        if cam_pose is None: cam_pose = np.eye(4)
        elif isinstance(cam_pose, torch.Tensor): cam_pose = cam_pose.cpu().numpy()
        
        window_results.append({
            "frame_idx": j,
            "img": view["img"][0].cpu().numpy().transpose(1, 2, 0),
            "pts3d": pred.get("pts3d_in_other_view", pred.get("pts3d_in_self_view", None)),
            "conf": pred.get("conf", None),
            "camera_pose": cam_pose,
        })
    return window_results

def to_pose_matrix(pose):
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape == (4, 4):
        return pose
    if pose.shape == (3, 4):
        pose4 = np.eye(4, dtype=np.float32)
        pose4[:3, :] = pose
        return pose4
    if pose.shape == (1, 7) or pose.shape == (7,):
        from dust3r.utils.camera import pose_encoding_to_camera
        pose_t = torch.from_numpy(pose.reshape(1, 7)).float()
        c2w = pose_encoding_to_camera(pose_t)[0].cpu().numpy().astype(np.float32)
        return c2w
    raise ValueError(f"Unsupported pose shape: {pose.shape}")

def transform_points(pts3d, transform):
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

def write_ply(path, points, colors):
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        pts_i = np.ascontiguousarray(points.astype(np.float32))
        col_i = np.ascontiguousarray((colors * 255).astype(np.uint8))
        for pt, col in zip(pts_i, col_i):
            f.write(pt.tobytes())
            f.write(col.tobytes())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos")
    parser.add_argument("--episode_id", type=int, default=10)
    parser.add_argument("--camera", type=str, default="head", choices=["head","left_wrist","right_wrist"])
    parser.add_argument("--num_frames_per_window", type=int, default=16,
                        help="Number of frames per Point3R inference window")
    parser.add_argument("--stride", type=int, default=15,
                        help="Stride between consecutive sampled frames")
    parser.add_argument("--checkpoint", type=str, default="src/openpi/third_party/Point3R/src/checkpoints/point3r_512.pth")
    parser.add_argument("--output_dir", type=str, default="./outputs/point3r_exp/task0000_stride15")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max_points", type=int, default=500000)
    parser.add_argument("--window_step", type=int, default=0)
    args = parser.parse_args()

    task_id, video_path = find_episode_video(args.data_root, args.episode_id)
    if task_id is None:
        print(f"Episode {args.episode_id} not found"); return
    print(f"Episode {args.episode_id} in task-{task_id}")
    if video_path is None:
        print(f"Episode {args.episode_id} video not found"); return

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Total video frames: {total_frames}, FPS: {fps}")

    all_indices = list(range(0, total_frames, args.stride))
    print(f"Stride={args.stride} -> {len(all_indices)} sampled frames: {all_indices[:5]}...{all_indices[-3:]}")

    n_win = args.num_frames_per_window
    window_step = args.window_step if args.window_step > 0 else max(1, n_win // 2)
    if window_step >= n_win:
        raise ValueError(f"window_step must be smaller than num_frames_per_window, got {window_step} >= {n_win}")
    windows = []
    for start in range(0, len(all_indices), window_step):
        window_indices = all_indices[start:start + n_win]
        if len(window_indices) < 2:
            break
        windows.append(window_indices)
    if len(all_indices) >= 2:
        tail_window = all_indices[max(0, len(all_indices) - n_win):]
        if len(tail_window) >= 2 and (not windows or windows[-1] != tail_window):
            windows.append(tail_window)

    print(f"Formed {len(windows)} windows, last window has {len(windows[-1])} frames, window_step={window_step}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Point3R model on {device}...")
    model = build_point3r_model(args.checkpoint, device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    all_pts_global = []
    all_colors_global = []
    global_camera_poses = {}
    saved_frame_indices = set()

    for w_idx, win_indices in enumerate(windows):
        print(f"\nWindow {w_idx+1}/{len(windows)}: frames {win_indices[0]}-{win_indices[-1]} ({len(win_indices)} frames)")
        images = load_video_frames(video_path, win_indices, args.size)
        if len(images) < 2:
            print(f"  Skipping window {w_idx}: only {len(images)} frames loaded")
            continue

        window_results = run_point3r_window(images, model, device)

        local_camera_poses = {}
        for r, idx_glob in zip(window_results, win_indices):
            local_camera_poses[idx_glob] = to_pose_matrix(r["camera_pose"])

        overlap_indices = [idx for idx in win_indices if idx in global_camera_poses]
        if overlap_indices:
            anchor_idx = overlap_indices[-1]
            transform_local_to_global = global_camera_poses[anchor_idx] @ np.linalg.inv(local_camera_poses[anchor_idx])
            print(f"  Aligning window with anchor frame {anchor_idx}")
        else:
            transform_local_to_global = np.eye(4, dtype=np.float32)
            print("  No overlap anchor found, using identity alignment")

        win_pts = []
        win_colors = []
        for r, idx_glob in zip(window_results, win_indices):
            pts3d = r["pts3d"]
            if pts3d is None:
                continue
            if isinstance(pts3d, torch.Tensor):
                pts3d = pts3d.cpu().numpy()
            pts3d_global = transform_points(pts3d, transform_local_to_global)
            global_pose = transform_local_to_global @ to_pose_matrix(r["camera_pose"])
            if idx_glob not in global_camera_poses:
                global_camera_poses[idx_glob] = global_pose
            if idx_glob in saved_frame_indices:
                continue
            pts_flat = pts3d_global.reshape(-1, 3)
            color_flat = r["img"].reshape(-1, 3)
            valid = np.isfinite(pts_flat).all(axis=1)
            win_pts.append(pts_flat[valid])
            win_colors.append(color_flat[valid])

            npz_path = os.path.join(args.output_dir, f"frame_{idx_glob:04d}.npz")
            conf_val = r.get("conf")
            if isinstance(conf_val, torch.Tensor):
                conf_val = conf_val.cpu().numpy()
            np.savez_compressed(npz_path,
                               img=r["img"].astype(np.float32),
                               pts3d=pts3d_global.astype(np.float32),
                               conf=conf_val,
                               camera_pose=global_pose.astype(np.float32))
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
    if len(all_pts) > args.max_points:
        idx = np.linspace(0, len(all_pts)-1, args.max_points, dtype=int)
        all_pts, all_colors = all_pts[idx], all_colors[idx]
    ply_path = os.path.join(args.output_dir, "point3r_reconstruction.ply")
    write_ply(ply_path, all_pts, all_colors)
    print(f"\nSaved merged point cloud: {ply_path} ({len(all_pts)} pts)")

    metadata = {
        "episode_id": args.episode_id, "task_id": task_id,
        "camera": args.camera, "num_windows": len(windows),
        "num_frames_per_window": args.num_frames_per_window,
        "window_step": window_step,
        "stride": args.stride,
        "total_sampled_frames": len(all_indices),
        "total_video_frames": total_frames, "fps": fps,
        "ply_path": ply_path
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Done! Results: {args.output_dir}")
    print(f"Metadata: {metadata}")

if __name__ == "__main__":
    main()
