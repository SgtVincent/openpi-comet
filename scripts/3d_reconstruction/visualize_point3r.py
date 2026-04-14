#!/usr/bin/env python3
"""Visualize Point3R reconstruction results using viser."""
import os, sys, glob, argparse, threading, time, re
import numpy as np
import viser

def load_point3r_results(npz_dir: str):
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "frame_*.npz")),
                       key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))
    results = []
    for f in npz_files:
        data = np.load(f)
        results.append({
            "pts3d": data["pts3d"],
            "img": data["img"],
            "conf": data.get("conf", None),
            "camera_pose": data["camera_pose"] if "camera_pose" in data else np.eye(4),
        })
    return results

def visualize_point3r(npz_dir: str, port: int = 8081,
                       downsample_factor: int = 1, max_points: int = 500000,
                       animate: bool = False, merge_all: bool = False,
                       keep_global_coords: bool = False,
                       point_size: float = 0.02):
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    status = server.gui.add_text("Status", f"Loading Point3R results from {npz_dir} ...")
    point_size_slider = server.gui.add_slider("Point Size", min=0.001, max=0.1, step=0.001, initial_value=point_size)
    print(f"\nPoint3R viewer starting: http://localhost:{port}")

    results = load_point3r_results(npz_dir)
    status.value = f"Loaded {len(results)} frames"
    print(f"Loaded {len(results)} frames, port={port}")

    if len(results) == 0:
        status.value = "No results found"
        print("No results found!")
        while True:
            time.sleep(1.0)

    if merge_all:
        all_pts = []
        all_colors = []
        for r in results:
            pts3d = r["pts3d"]
            img = r["img"]
            if pts3d is None:
                continue
            if pts3d.ndim == 4:
                pts3d = pts3d.squeeze(0)
            pts_flat = pts3d.reshape(-1, 3)
            color_flat = img.reshape(-1, 3)
            valid = np.isfinite(pts_flat).all(axis=1)
            pts_flat = pts_flat[valid]
            color_flat = color_flat[valid]
            if downsample_factor > 1:
                pts_flat = pts_flat[::downsample_factor]
                color_flat = color_flat[::downsample_factor]
            all_pts.append(pts_flat)
            all_colors.append(color_flat)
        points = np.concatenate(all_pts, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        if not keep_global_coords:
            points = points - np.mean(points, axis=0)
        if len(points) > max_points:
            idx = np.linspace(0, len(points) - 1, max_points, dtype=int)
            points = points[idx]
            colors = colors[idx]
        if colors.dtype != np.uint8:
            colors = (colors * 255).astype(np.uint8)
        server.scene.add_point_cloud(
            name="pcd",
            points=points.astype(np.float32),
            colors=colors,
            point_size=float(point_size_slider.value),
            point_shape="circle",
        )
        status.value = "Ready (merged)"
        print(f"Ready (merged): http://localhost:{port}")

        @point_size_slider.on_update
        def _(_e):
            server.scene.add_point_cloud(
                name="pcd",
                points=points.astype(np.float32),
                colors=colors,
                point_size=float(point_size_slider.value),
                point_shape="circle",
            )
            status.value = "Ready (merged)"

        while True:
            time.sleep(1.0)

    slider = server.gui.add_slider("Frame", min=0, max=len(results)-1, step=1, initial_value=0)

    def render_frame(frame_idx: int):
        r = results[frame_idx]
        img = r["img"]
        pts3d = r["pts3d"]
        if pts3d is None:
            return
        if pts3d.ndim == 4:
            pts3d = pts3d.squeeze(0)
        H, W = img.shape[:2]
        pts_flat = pts3d.reshape(-1, 3)
        color_flat = img.reshape(-1, 3)
        if downsample_factor > 1:
            pts_flat = pts_flat[::downsample_factor]
            color_flat = color_flat[::downsample_factor]
        valid = np.isfinite(pts_flat).all(axis=1)
        pts_flat = pts_flat[valid]
        color_flat = color_flat[valid]
        if not keep_global_coords:
            center = np.mean(pts_flat, axis=0)
            pts_flat = pts_flat - center
        if len(pts_flat) > max_points:
            idx = np.linspace(0, len(pts_flat)-1, max_points, dtype=int)
            pts_flat, color_flat = pts_flat[idx], color_flat[idx]
        if color_flat.dtype != np.uint8:
            color_flat = (color_flat * 255).astype(np.uint8)
        server.scene.add_point_cloud(
            name="pcd",
            points=pts_flat.astype(np.float32),
            colors=color_flat,
            point_size=float(point_size_slider.value),
            point_shape="circle",
        )

    @slider.on_update
    def on_slider_update(event):
        render_frame(int(event.value))

    @point_size_slider.on_update
    def _(_e):
        render_frame(int(slider.value))

    render_frame(0)
    status.value = "Ready"

    if animate:
        print(f"Animation mode: auto-playing {len(results)} frames...")
        idx = [0]
        def advance():
            while True:
                time.sleep(0.25)
                idx[0] = (idx[0] + 1) % len(results)
                render_frame(idx[0])
        t = threading.Thread(target=advance, daemon=True)
        t.start()

    print(f"\nPoint3R visualization ready: http://localhost:{port}")
    print(f"Use the 'Frame' slider to browse manually" + (" or watch auto-play" if animate else ""))
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", type=str, required=True)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--downsample_factor", type=int, default=1)
    parser.add_argument("--max_points", type=int, default=500000)
    parser.add_argument("--animate", action="store_true", help="Auto-play animation")
    parser.add_argument("--merge_all", action="store_true")
    parser.add_argument("--keep_global_coords", action="store_true")
    parser.add_argument("--point_size", type=float, default=0.02)
    args = parser.parse_args()
    visualize_point3r(args.npz_dir, args.port, args.downsample_factor,
                     args.max_points, args.animate, args.merge_all,
                     args.keep_global_coords, args.point_size)
