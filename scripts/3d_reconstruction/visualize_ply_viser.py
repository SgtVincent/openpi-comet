#!/usr/bin/env python3
"""Visualize a colored point cloud (.ply) in viser.

This is intentionally minimal and robust for common PLY produced by Pi3X.
It uses `plyfile` to load vertices and optional RGB colors.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import viser
from plyfile import PlyData


def load_ply_vertices(ply_path: str) -> tuple[np.ndarray, np.ndarray | None]:
    ply = PlyData.read(ply_path)
    if "vertex" not in ply:
        raise ValueError(f"No vertex element in PLY: {ply_path}")

    v = ply["vertex"].data
    points = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    colors = None
    if ("red" in v.dtype.names) and ("green" in v.dtype.names) and ("blue" in v.dtype.names):
        colors = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
    return points, colors


def downsample(points: np.ndarray, colors: np.ndarray | None, max_points: int) -> tuple[np.ndarray, np.ndarray | None]:
    if max_points <= 0 or len(points) <= max_points:
        return points, colors
    idx = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    if colors is None:
        return points[idx], None
    return points[idx], colors[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--max_points", type=int, default=500000)
    parser.add_argument("--point_size", type=float, default=0.01)
    parser.add_argument("--no_center", action="store_true", help="Do not subtract centroid (keep raw coordinates).")
    args = parser.parse_args()

    ply_path = Path(args.ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(str(ply_path))

    points, colors = load_ply_vertices(str(ply_path))
    points, colors = downsample(points, colors, int(args.max_points))
    if colors is None:
        colors = np.full((len(points), 3), 180, dtype=np.uint8)
    if not args.no_center:
        points = points - np.mean(points, axis=0, keepdims=True)

    server = viser.ViserServer(host="0.0.0.0", port=int(args.port))
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    status = server.gui.add_text("Status", f"Loaded {len(points)} points from {ply_path.name}")
    point_size = server.gui.add_slider("Point Size", min=0.001, max=0.05, step=0.001, initial_value=float(args.point_size))

    def render() -> None:
        server.scene.add_point_cloud(
            name="ply_pcd",
            points=points.astype(np.float32),
            colors=colors,
            point_size=float(point_size.value),
            point_shape="circle",
        )

    render()

    @point_size.on_update
    def _(_e) -> None:
        render()
        status.value = f"Loaded {len(points)} points from {ply_path.name}"

    print(f"PLY viewer: http://localhost:{int(args.port)}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()

