#!/usr/bin/env python3
"""Use viser to compare two Point3R-style reconstruction dirs (frame_*.npz)."""

import argparse
import glob
import os
import re
import time

import numpy as np
import viser


def _last_int(text: str) -> int:
    m = re.search(r"(\d+)(?!.*\d)", text)
    return int(m.group(1)) if m else 0


def _sorted_frames(npz_dir: str):
    files = sorted(glob.glob(os.path.join(npz_dir, "frame_*.npz")), key=lambda p: _last_int(os.path.basename(p)))
    frames = []
    for p in files:
        d = np.load(p)
        frames.append(
            {
                "frame_index": _last_int(os.path.basename(p)),
                "pts3d": d["pts3d"],
                "img": d["img"] if "img" in d.files else None,
                "conf": d["conf"] if "conf" in d.files else None,
                "path": p,
            }
        )
    return frames


def _to_uint8(img):
    if img is None:
        return None
    if img.dtype == np.uint8:
        return img
    img = np.nan_to_num(img)
    if img.max() <= 1.0:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def _prepare_cloud(frame: dict, downsample: int, max_points: int):
    pts = frame["pts3d"]
    if pts.ndim == 4:
        pts = pts.squeeze(0)
    pts = pts.reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)

    conf = frame.get("conf", None)
    if conf is not None:
        conf = np.asarray(conf).reshape(-1)
        valid = valid & np.isfinite(conf)

    img = _to_uint8(frame.get("img", None))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left_dir", type=str, required=True)
    parser.add_argument("--right_dir", type=str, required=True)
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--left_downsample", type=int, default=6)
    parser.add_argument("--right_downsample", type=int, default=6)
    parser.add_argument("--max_points", type=int, default=500000)
    parser.add_argument("--point_size", type=float, default=0.02)
    parser.add_argument("--split_offset", type=float, default=1.5)
    args = parser.parse_args()

    left_frames = _sorted_frames(args.left_dir)
    right_frames = _sorted_frames(args.right_dir)
    if not left_frames:
        raise ValueError(f"No frame_*.npz found in {args.left_dir}")
    if not right_frames:
        raise ValueError(f"No frame_*.npz found in {args.right_dir}")

    right_by_idx = {f["frame_index"]: f for f in right_frames}
    common = [f for f in left_frames if f["frame_index"] in right_by_idx]
    if not common:
        raise ValueError("No common frame indices between left_dir and right_dir")

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    slider = server.gui.add_slider("Frame", min=0, max=len(common) - 1, step=1, initial_value=0)
    point_size = server.gui.add_slider("Point Size", min=0.001, max=0.1, step=0.001, initial_value=args.point_size)
    info = server.gui.add_text("Info", "")

    def render():
        f_left = common[int(slider.value)]
        f_right = right_by_idx[f_left["frame_index"]]
        pts_l, col_l = _prepare_cloud(f_left, args.left_downsample, args.max_points)
        pts_r, col_r = _prepare_cloud(f_right, args.right_downsample, args.max_points)

        pts_l = pts_l + np.array([-args.split_offset * 0.5, 0.0, 0.0], dtype=np.float32)
        pts_r = pts_r + np.array([args.split_offset * 0.5, 0.0, 0.0], dtype=np.float32)

        server.scene.add_point_cloud(
            name="/left",
            points=pts_l,
            colors=col_l,
            point_size=float(point_size.value),
            point_shape="circle",
        )
        server.scene.add_point_cloud(
            name="/right",
            points=pts_r,
            colors=col_r,
            point_size=float(point_size.value),
            point_shape="circle",
        )
        info.value = f"frame_index={f_left['frame_index']} left={os.path.basename(args.left_dir)} right={os.path.basename(args.right_dir)}"

    @slider.on_update
    def _(_e):
        render()

    @point_size.on_update
    def _(_e):
        render()

    render()
    print(f"Viewer ready: http://localhost:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
