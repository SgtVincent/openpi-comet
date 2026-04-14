#!/usr/bin/env python3
"""Visualize Easi3R outputs using viser."""

import argparse
import glob
import os
import time
from pathlib import Path

import cv2
import numpy as np
import viser


def _sorted_by_frame_idx(paths: list[str]):
    def _idx(p: str) -> int:
        stem = Path(p).stem
        parts = stem.split("_")
        return int(parts[-1]) if parts and parts[-1].isdigit() else 0

    return sorted(paths, key=_idx)


def _quat_wxyz_to_rotmat(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in wxyz]
    n = w * w + x * x + y * y + z * z
    if n <= 0:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def _load_intrinsics(intrinsics_path: str) -> np.ndarray:
    intrinsics = np.loadtxt(intrinsics_path).astype(np.float32)
    if intrinsics.ndim == 1:
        intrinsics = intrinsics[None, :]
    if intrinsics.shape[1] == 9:
        intrinsics = intrinsics.reshape(-1, 3, 3)
    elif intrinsics.shape[0] % 3 == 0 and intrinsics.shape[1] == 3:
        intrinsics = intrinsics.reshape(-1, 3, 3)
    else:
        raise ValueError(f"Unexpected intrinsics shape: {intrinsics.shape}")
    return intrinsics


def _load_poses_tum(pose_path: str, xyzw: bool) -> np.ndarray:
    poses = np.loadtxt(pose_path).astype(np.float32)
    if poses.ndim == 1:
        poses = poses[None, :]
    if poses.shape[1] < 8:
        raise ValueError(f"Unexpected pose shape: {poses.shape}")
    t = poses[:, 1:4]
    if xyzw:
        wxyz = np.concatenate([poses[:, 4:5], poses[:, 5:8]], axis=1)
    else:
        wxyz = np.concatenate([poses[:, 7:8], poses[:, 4:7]], axis=1)
    Twc = []
    for i in range(len(poses)):
        R = _quat_wxyz_to_rotmat(wxyz[i])
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = t[i]
        Twc.append(T)
    return np.stack(Twc, axis=0)


def _load_optional_npy(path: str):
    return np.load(path) if os.path.exists(path) else None


def _load_mask(mask_path: str, target_hw: tuple[int, int]) -> np.ndarray | None:
    if not mask_path or not os.path.exists(mask_path):
        return None
    img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = img[:, :, 0]
    if img.shape[:2] != target_hw:
        img = cv2.resize(img, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (img > 0).astype(np.bool_)


def _frame_paths(easi3r_seq_dir: str):
    rgb_paths = _sorted_by_frame_idx(glob.glob(os.path.join(easi3r_seq_dir, "frame_*.png")))
    depth_paths = _sorted_by_frame_idx(glob.glob(os.path.join(easi3r_seq_dir, "frame_*.npy")))
    conf_paths = _sorted_by_frame_idx(glob.glob(os.path.join(easi3r_seq_dir, "conf_*.npy")))
    init_conf_paths = _sorted_by_frame_idx(glob.glob(os.path.join(easi3r_seq_dir, "init_conf_*.npy")))
    mask_paths = _sorted_by_frame_idx(glob.glob(os.path.join(easi3r_seq_dir, "enlarged_dynamic_mask_*.png")))
    if not mask_paths:
        mask_paths = _sorted_by_frame_idx(glob.glob(os.path.join(easi3r_seq_dir, "dynamic_mask_*.png")))
    return rgb_paths, depth_paths, conf_paths, init_conf_paths, mask_paths


def _infer_seq_dir(easi3r_dir: str) -> str:
    p = Path(easi3r_dir)
    if (p / "pred_traj.txt").exists() and (p / "pred_intrinsics.txt").exists():
        return str(p)
    candidates = sorted([x for x in p.iterdir() if x.is_dir() and (x / "pred_traj.txt").exists()])
    if candidates:
        return str(candidates[0])
    raise ValueError(f"Cannot find Easi3R sequence directory under: {easi3r_dir}")


def _project_points(K: np.ndarray, T_world_camera: np.ndarray, rgb: np.ndarray, depth: np.ndarray, conf: np.ndarray, init_conf: np.ndarray | None, mask: np.ndarray | None, conf_thr: float, fg_conf_thr: float, downsample: int, bg_downsample: int, no_mask: bool):
    h, w = rgb.shape[:2]
    if downsample > 1:
        rgb_ds = rgb[::downsample, ::downsample]
        depth_ds = depth[::downsample, ::downsample]
        conf_ds = conf[::downsample, ::downsample]
        init_conf_ds = init_conf[::downsample, ::downsample] if init_conf is not None else None
        mask_ds = mask[::downsample, ::downsample] if mask is not None else None
    else:
        rgb_ds, depth_ds, conf_ds, init_conf_ds, mask_ds = rgb, depth, conf, init_conf, mask

    h2, w2 = rgb_ds.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w2, dtype=np.float32) + 0.5, np.arange(h2, dtype=np.float32) + 0.5)
    grid_x *= float(downsample)
    grid_y *= float(downsample)
    ones = np.ones_like(grid_x)
    pix = np.stack([grid_x, grid_y, ones], axis=-1).reshape(-1, 3)

    depth_flat = depth_ds.reshape(-1).astype(np.float32)
    conf_flat = conf_ds.reshape(-1).astype(np.float32)
    rgb_flat = rgb_ds.reshape(-1, 3).astype(np.uint8)
    if init_conf_ds is None:
        init_conf_flat = conf_flat
    else:
        init_conf_flat = init_conf_ds.reshape(-1).astype(np.float32)
    if mask_ds is None:
        mask_flat = np.zeros((len(depth_flat),), dtype=np.bool_)
    else:
        mask_flat = mask_ds.reshape(-1).astype(np.bool_)
    if no_mask:
        mask_flat = np.ones_like(mask_flat)

    valid_depth = np.isfinite(depth_flat) & (depth_flat > 1e-6)
    conf_mask = valid_depth & np.isfinite(conf_flat) & (conf_flat > conf_thr)
    fg_mask = valid_depth & np.isfinite(init_conf_flat) & (init_conf_flat > fg_conf_thr) & mask_flat
    bg_mask = conf_mask & (~mask_flat)

    Kinv = np.linalg.inv(K).astype(np.float32)
    dirs_local = (pix @ Kinv.T).astype(np.float32)
    R = T_world_camera[:3, :3].astype(np.float32)
    t = T_world_camera[:3, 3].astype(np.float32)
    dirs_world = (dirs_local @ R.T).astype(np.float32)
    pts_world = t[None, :] + dirs_world * depth_flat[:, None]

    pts_fg = pts_world[fg_mask]
    col_fg = rgb_flat[fg_mask]
    pts_bg = pts_world[bg_mask]
    col_bg = rgb_flat[bg_mask]
    if bg_downsample > 1 and len(pts_bg) > 0:
        pts_bg = pts_bg[::bg_downsample]
        col_bg = col_bg[::bg_downsample]
    pts = np.concatenate([pts_fg, pts_bg], axis=0) if len(pts_bg) else pts_fg
    col = np.concatenate([col_fg, col_bg], axis=0) if len(col_bg) else col_fg
    return pts.astype(np.float32), col.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--easi3r_dir", type=str, required=True)
    parser.add_argument("--port", type=int, default=9081)
    parser.add_argument("--downsample_factor", type=int, default=4)
    parser.add_argument("--bg_downsample_factor", type=int, default=4)
    parser.add_argument("--max_points", type=int, default=500000)
    parser.add_argument("--conf_threshold", type=float, default=1.0)
    parser.add_argument("--foreground_conf_threshold", type=float, default=0.1)
    parser.add_argument("--no_mask", action="store_true")
    parser.add_argument("--xyzw", action="store_true")
    parser.add_argument("--merge_all", action="store_true")
    parser.add_argument("--keep_global_coords", action="store_true")
    parser.add_argument("--point_size", type=float, default=0.015)
    args = parser.parse_args()

    seq_dir = _infer_seq_dir(args.easi3r_dir)
    intrinsics = _load_intrinsics(os.path.join(seq_dir, "pred_intrinsics.txt"))
    poses = _load_poses_tum(os.path.join(seq_dir, "pred_traj.txt"), xyzw=args.xyzw)

    rgb_paths, depth_paths, conf_paths, init_conf_paths, mask_paths = _frame_paths(seq_dir)
    n = min(len(rgb_paths), len(depth_paths), len(poses), len(intrinsics))
    if n <= 0:
        raise ValueError(f"No frames found under {seq_dir}")

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    status = server.gui.add_text("Status", f"Loading Easi3R from {seq_dir}")
    point_size = server.gui.add_slider("Point Size", min=0.001, max=0.1, step=0.001, initial_value=args.point_size)
    print(f"\nEasi3R viewer starting: http://localhost:{args.port}")

    def _load_frame(i: int):
        rgb = cv2.cvtColor(cv2.imread(rgb_paths[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        depth = np.load(depth_paths[i]).astype(np.float32).squeeze()
        conf = _load_optional_npy(conf_paths[i]) if i < len(conf_paths) else None
        conf = conf.astype(np.float32).squeeze() if conf is not None else np.ones_like(depth, dtype=np.float32)
        init_conf = _load_optional_npy(init_conf_paths[i]) if i < len(init_conf_paths) else None
        init_conf = init_conf.astype(np.float32).squeeze() if init_conf is not None else None
        mask_path = mask_paths[i] if i < len(mask_paths) else ""
        mask = _load_mask(mask_path, depth.shape[:2])
        return rgb, depth, conf, init_conf, mask

    if args.merge_all:
        pts_all = []
        col_all = []
        for i in range(n):
            rgb, depth, conf, init_conf, mask = _load_frame(i)
            pts, col = _project_points(
                intrinsics[i],
                poses[i],
                rgb,
                depth,
                conf,
                init_conf,
                mask,
                args.conf_threshold,
                args.foreground_conf_threshold,
                args.downsample_factor,
                args.bg_downsample_factor,
                args.no_mask,
            )
            if len(pts) == 0:
                continue
            pts_all.append(pts)
            col_all.append(col)
        if not pts_all:
            raise ValueError("No points to visualize")
        pts = np.concatenate(pts_all, axis=0)
        col = np.concatenate(col_all, axis=0)
        if not args.keep_global_coords and len(pts) > 0:
            pts = pts - np.mean(pts, axis=0, keepdims=True)
        if len(pts) > args.max_points:
            keep = np.linspace(0, len(pts) - 1, args.max_points, dtype=int)
            pts = pts[keep]
            col = col[keep]
        server.scene.add_point_cloud(
            name="easi3r",
            points=pts.astype(np.float32),
            colors=col.astype(np.uint8),
            point_size=float(point_size.value),
            point_shape="circle",
        )
        status.value = f"Ready (merged) frames={n} points={len(pts)}"
        print(f"Ready (merged): http://localhost:{args.port}")

        @point_size.on_update
        def _(_e):
            server.scene.add_point_cloud(
                name="easi3r",
                points=pts.astype(np.float32),
                colors=col.astype(np.uint8),
                point_size=float(point_size.value),
                point_shape="circle",
            )
            status.value = f"Ready (merged) frames={n} points={len(pts)}"

        while True:
            time.sleep(1.0)

    slider = server.gui.add_slider("Frame", min=0, max=n - 1, step=1, initial_value=0)

    def _render(i: int):
        rgb, depth, conf, init_conf, mask = _load_frame(i)
        pts, col = _project_points(
            intrinsics[i],
            poses[i],
            rgb,
            depth,
            conf,
            init_conf,
            mask,
            args.conf_threshold,
            args.foreground_conf_threshold,
            args.downsample_factor,
            args.bg_downsample_factor,
            args.no_mask,
        )
        if len(pts) > 0 and not args.keep_global_coords:
            pts = pts - np.mean(pts, axis=0, keepdims=True)
        if len(pts) > args.max_points:
            keep = np.linspace(0, len(pts) - 1, args.max_points, dtype=int)
            pts = pts[keep]
            col = col[keep]
        server.scene.add_point_cloud(
            name="easi3r",
            points=pts.astype(np.float32),
            colors=col.astype(np.uint8),
            point_size=float(point_size.value),
            point_shape="circle",
        )
        status.value = f"Ready frame={i}/{n-1} points={len(pts)}"

    @slider.on_update
    def _on_update(event):
        _render(int(event.value))

    @point_size.on_update
    def _(_e):
        _render(int(slider.value))

    _render(0)
    print(f"\nEasi3R visualization ready: http://localhost:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
