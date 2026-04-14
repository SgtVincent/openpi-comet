#!/usr/bin/env python3
"""Use viser to compare Point3R vs InfiniteVGGT reconstructions via sliders."""

import argparse
import glob
import os
import re
import time

import numpy as np
import viser
from PIL import Image


def _last_int(text: str) -> int:
    m = re.search(r"(\d+)(?!.*\d)", text)
    return int(m.group(1)) if m else 0


def _to_uint8(img):
    if img is None:
        return None
    if img.dtype == np.uint8:
        return img
    img = np.nan_to_num(img)
    if img.max() <= 1.0:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def _sorted_frame_npzs(root: str):
    return sorted(
        glob.glob(os.path.join(root, "frame_*.npz")),
        key=lambda p: _last_int(os.path.basename(p)),
    )


def _sorted_window_dirs(root: str):
    return sorted(
        glob.glob(os.path.join(root, "window_*")),
        key=lambda p: _last_int(os.path.basename(p)),
    )


def _sorted_pngs(root: str):
    return sorted(
        glob.glob(os.path.join(root, "*.png")),
        key=lambda p: _last_int(os.path.basename(p)),
    )


def _pc_from_point3r(npz_path: str, downsample: int, max_points: int):
    data = np.load(npz_path)
    pts = data["pts3d"]
    if pts.ndim == 4:
        pts = pts.squeeze(0)
    pts = pts.reshape(-1, 3)

    valid = np.isfinite(pts).all(axis=1)
    if "conf" in data.files and data["conf"] is not None:
        conf = np.asarray(data["conf"]).reshape(-1)
        valid = valid & np.isfinite(conf)

    img = _to_uint8(data["img"]) if "img" in data.files else None
    if img is None:
        col = np.full((pts.shape[0], 3), 180, dtype=np.uint8)
    else:
        col = img.reshape(-1, 3)

    pts = pts[valid]
    col = col[valid]

    if downsample > 1:
        pts, col = pts[::downsample], col[::downsample]

    if len(pts) > max_points:
        keep = np.linspace(0, len(pts) - 1, max_points, dtype=int)
        pts, col = pts[keep], col[keep]

    if len(pts) > 0:
        pts = pts - np.mean(pts, axis=0, keepdims=True)

    return pts.astype(np.float32), col.astype(np.uint8)


def _pc_from_infinitevggt(
    window_dir: str,
    local_idx: int,
    conf_percentile: float,
    downsample: int,
    max_points: int,
):
    frame_npzs = _sorted_frame_npzs(window_dir)
    if not frame_npzs:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), None, 0

    local_idx = max(0, min(local_idx, len(frame_npzs) - 1))
    data = np.load(frame_npzs[local_idx])

    pts = data["point_map"].reshape(-1, 3)
    conf = data["point_conf"].reshape(-1)

    valid = np.isfinite(pts).all(axis=1) & np.isfinite(conf)
    if np.any(valid):
        thr = np.percentile(conf[valid], conf_percentile)
        valid = valid & (conf >= thr)

    pts = pts[valid]

    img = None
    pngs = _sorted_pngs(os.path.join(window_dir, "images"))
    if 0 <= local_idx < len(pngs):
        img = np.array(Image.open(pngs[local_idx]).convert("RGB"))

    if img is None:
        col = np.full((pts.shape[0], 3), 180, dtype=np.uint8)
    else:
        col = img.reshape(-1, 3)[valid]

    if downsample > 1:
        pts, col = pts[::downsample], col[::downsample]

    if len(pts) > max_points:
        keep = np.linspace(0, len(pts) - 1, max_points, dtype=int)
        pts, col = pts[keep], col[keep]

    if len(pts) > 0:
        pts = pts - np.mean(pts, axis=0, keepdims=True)

    source_idx = int(data["source_frame_index"]) if "source_frame_index" in data.files else None
    return pts.astype(np.float32), col.astype(np.uint8), source_idx, len(frame_npzs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point3r_dir", type=str, required=True)
    parser.add_argument("--infinitevggt_dir", type=str, required=True)
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--point3r_downsample", type=int, default=4)
    parser.add_argument("--infinitevggt_conf_percentile", type=float, default=25.0)
    parser.add_argument("--infinitevggt_downsample", type=int, default=4)
    parser.add_argument("--max_points", type=int, default=500000)
    parser.add_argument("--point_size", type=float, default=0.02)
    parser.add_argument("--split_offset", type=float, default=1.5)
    args = parser.parse_args()

    point3r_files = _sorted_frame_npzs(args.point3r_dir)
    window_dirs = _sorted_window_dirs(args.infinitevggt_dir)

    if not point3r_files:
        raise ValueError(f"No Point3R frame_*.npz found in {args.point3r_dir}")
    if not window_dirs:
        raise ValueError(f"No InfiniteVGGT window_* found in {args.infinitevggt_dir}")

    max_local_frames = 0
    for w in window_dirs:
        max_local_frames = max(max_local_frames, len(_sorted_frame_npzs(w)))

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    s_p3r = server.gui.add_slider("Point3R Frame", min=0, max=len(point3r_files) - 1, step=1, initial_value=0)
    s_win = server.gui.add_slider("InfiniteVGGT Window", min=0, max=len(window_dirs) - 1, step=1, initial_value=0)
    s_loc = server.gui.add_slider(
        "InfiniteVGGT Local Frame",
        min=0,
        max=max(0, max_local_frames - 1),
        step=1,
        initial_value=0,
    )
    point_size = server.gui.add_slider("Point Size", min=0.001, max=0.1, step=0.001, initial_value=args.point_size)
    follow_match = server.gui.add_checkbox("Point3R 改变时尝试匹配 source_frame_index", initial_value=False)
    t_info = server.gui.add_text("Info", "")

    def _try_match_local():
        p3r_idx = _last_int(os.path.basename(point3r_files[int(s_p3r.value)]))
        win_dir = window_dirs[int(s_win.value)]
        frame_npzs = _sorted_frame_npzs(win_dir)
        for i, npz_path in enumerate(frame_npzs):
            try:
                src = np.load(npz_path)["source_frame_index"]
            except Exception:
                continue
            if int(src) == int(p3r_idx):
                s_loc.value = i
                return

    def _render():
        p3r_path = point3r_files[int(s_p3r.value)]
        win_dir = window_dirs[int(s_win.value)]

        p3r_pts, p3r_col = _pc_from_point3r(p3r_path, args.point3r_downsample, args.max_points)
        v_pts, v_col, src, n_local = _pc_from_infinitevggt(
            win_dir,
            int(s_loc.value),
            args.infinitevggt_conf_percentile,
            args.infinitevggt_downsample,
            args.max_points,
        )

        p3r_pts = p3r_pts + np.array([-args.split_offset * 0.5, 0.0, 0.0], dtype=np.float32)
        v_pts = v_pts + np.array([args.split_offset * 0.5, 0.0, 0.0], dtype=np.float32)

        server.scene.add_point_cloud(
            name="/point3r",
            points=p3r_pts,
            colors=p3r_col,
            point_size=float(point_size.value),
            point_shape="circle",
        )
        server.scene.add_point_cloud(
            name="/infinitevggt",
            points=v_pts,
            colors=v_col,
            point_size=float(point_size.value),
            point_shape="circle",
        )

        t_info.value = (
            f"Point3R={os.path.basename(p3r_path)} | "
            f"InfiniteVGGT={os.path.basename(win_dir)} local={int(s_loc.value)}/{max(0, n_local - 1)} source={src}"
        )

    @s_p3r.on_update
    def _(_e):
        if follow_match.value:
            _try_match_local()
        _render()

    @s_win.on_update
    def _(_e):
        _render()

    @s_loc.on_update
    def _(_e):
        _render()

    @follow_match.on_update
    def _(_e):
        if follow_match.value:
            _try_match_local()
        _render()

    @point_size.on_update
    def _(_e):
        _render()

    _render()
    print(f"Viewer ready: http://localhost:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
