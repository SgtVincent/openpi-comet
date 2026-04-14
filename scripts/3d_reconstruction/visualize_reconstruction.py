#!/usr/bin/env python3
"""Visualize VGGT reconstruction results using viser with RGB coloring and downsampling support."""
# pyright: reportMissingImports=false

import os
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import viser
import viser.transforms as viser_tf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "openpi" / "third_party" / "vggt"))

from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3


def load_reconstruction_results(npz_dir: str):
    """Load all npz files from the reconstruction output directory."""
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "frame_*.npz")))
    results = []
    for f in npz_files:
        data = np.load(f)
        results.append({
            "extrinsic": data["extrinsic"],
            "intrinsic": data["intrinsic"],
            "depth": data["depth"],
            "conf": data["conf"],
        })
    return results


def load_images_for_frames(npz_dir: str, num_frames: int):
    """Load saved PNG images for RGB coloring if available."""
    images_dir = os.path.join(npz_dir, "images")
    if not os.path.isdir(images_dir):
        return None

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    if len(image_files) == 0:
        image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    if len(image_files) == 0:
        return None

    images = []
    for i in range(num_frames):
        img_path = image_files[i] if i < len(image_files) else image_files[i % len(image_files)]
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img.resize((518, 518), Image.Resampling.BICUBIC))
        images.append(img_np)
    return images


def downsample_points(points, colors, max_points: int):
    """Downsample points and colors uniformly."""
    if len(points) <= max_points:
        return points, colors

    indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
    return points[indices], colors[indices]


def visualize_with_viser(npz_dir: str, port: int = 8080, conf_threshold: float = 25.0,
                         downsample_factor: int = 1, use_rgb: bool = True,
                         max_points: int = 500000, point_size: float = 0.005):
    """Visualize reconstruction results with viser.

    Args:
        npz_dir: Directory containing frame_*.npz files
        port: Viser server port
        conf_threshold: Confidence percentile threshold for filtering points
        downsample_factor: Downsample every Nth point (1 = no downsampling)
        use_rgb: Whether to colorize point cloud with RGB from saved images
        max_points: Maximum number of points to render (GPU pressure control)
    """
    print(f"Loading results from {npz_dir}...")
    results = load_reconstruction_results(npz_dir)
    print(f"Loaded {len(results)} frames")

    if len(results) == 0:
        print("No results found!")
        return

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    point_size_slider = server.gui.add_slider("Point Size", min=0.001, max=0.1, step=0.001, initial_value=point_size)

    S = len(results)

    depth_list = []
    conf_list = []
    extrinsic_list = []
    intrinsic_list = []

    for r in results:
        depth = r["depth"]
        if len(depth.shape) == 2:
            depth = depth[..., np.newaxis]
        depth_list.append(depth)

        conf = r["conf"]
        if len(conf.shape) == 3:
            conf = conf.squeeze(-1)
        conf_list.append(conf)

        extrinsic = r["extrinsic"]
        if extrinsic.shape[0] == 4:
            extrinsic = extrinsic[:3, :]
        extrinsic_list.append(extrinsic)

        intrinsic_list.append(r["intrinsic"])

    depth_batch = np.stack(depth_list)
    conf_batch = np.stack(conf_list)
    extrinsic_batch = np.stack(extrinsic_list)
    intrinsic_batch = np.stack(intrinsic_list)

    rgb_images = None
    if use_rgb:
        print("Loading RGB images for coloring...")
        rgb_images = load_images_for_frames(npz_dir, S)
        if rgb_images is None:
            print("  No images found, falling back to gray coloring")
        else:
            print(f"  Loaded {len(rgb_images)} RGB images")

    print("Unprojecting depth maps to 3D points...")
    world_points = unproject_depth_map_to_point_map(
        depth_batch, extrinsic_batch, intrinsic_batch
    )

    S, H, W, _ = world_points.shape
    rgb_images_resized = None
    if rgb_images is not None:
        rgb_batch = np.stack([img.transpose(2, 0, 1) for img in rgb_images])
        rgb_images_resized = rgb_batch

    all_points = []
    all_colors = []

    for i in range(S):
        pts = world_points[i].reshape(-1, 3)
        conf = conf_batch[i].reshape(-1)

        threshold_val = np.percentile(conf, conf_threshold)
        valid_mask = (conf >= threshold_val) & (conf > 0.1)

        pts_valid = pts[valid_mask]

        if downsample_factor > 1:
            pts_valid = pts_valid[::downsample_factor]

        if rgb_images_resized is not None:
            rgb_frame = rgb_images_resized[i]
            rgb_flat = rgb_frame.reshape(3, -1).transpose(1, 0)
            colors_valid = rgb_flat[valid_mask]
            if downsample_factor > 1:
                colors_valid = colors_valid[::downsample_factor]
            if colors_valid.dtype != np.uint8:
                colors_valid = (colors_valid * 255).astype(np.uint8)
        else:
            gray = 180
            colors_valid = np.full((len(pts_valid), 3), gray, dtype=np.uint8)

        all_points.append(pts_valid)
        all_colors.append(colors_valid)
        print(f"Frame {i}: {len(pts_valid)} valid points")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    if len(points) > max_points:
        print(f"Downsampling from {len(points)} to {max_points} points (max_points limit)...")
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
        colors = colors[indices]

    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center

    print(f"Total points after filtering & downsampling: {len(points_centered)}")

    def render_point_cloud():
        server.scene.add_point_cloud(
            name="reconstruction_pcd",
            points=points_centered.astype(np.float32),
            colors=colors,
            point_size=float(point_size_slider.value),
            point_shape="circle",
        )

    render_point_cloud()

    @point_size_slider.on_update
    def _(_e):
        render_point_cloud()

    cam_to_world_batch = closed_form_inverse_se3(extrinsic_batch)

    for i in range(S):
        cam2world_3x4 = cam_to_world_batch[i][:3, :]
        cam2world_3x4[:, -1] -= scene_center
        T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

        server.scene.add_frame(
            f"frame_{i}",
            wxyz=T_world_camera.rotation().wxyz,
            position=T_world_camera.translation(),
            axes_length=0.05,
            axes_radius=0.002,
            origin_radius=0.002,
        )

        print(f"Added frame {i}")

    print(f"\nVisualization started at http://localhost:{port}")
    print("Press Ctrl+C to exit")

    while True:
        import time
        time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser(description="Visualize VGGT reconstruction results")
    parser.add_argument("--npz_dir", type=str, required=True,
                        help="Directory containing frame_*.npz files")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port for viser server")
    parser.add_argument("--conf_threshold", type=float, default=25.0,
                        help="Confidence threshold percentile for filtering low-quality depth points")
    parser.add_argument("--downsample_factor", type=int, default=4,
                        help="Downsample every Nth valid point to reduce GPU load (1=no downsampling)")
    parser.add_argument("--max_points", type=int, default=500000,
                        help="Maximum number of points to render (controls GPU memory pressure)")
    parser.add_argument("--no_rgb", action="store_true",
                        help="Disable RGB coloring from saved images (fall back to gray)")
    parser.add_argument("--point_size", type=float, default=0.005,
                        help="Initial point size in the GUI")
    args = parser.parse_args()

    use_rgb = not args.no_rgb
    visualize_with_viser(
        args.npz_dir, args.port, args.conf_threshold,
        downsample_factor=args.downsample_factor,
        use_rgb=use_rgb,
        max_points=args.max_points,
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()
