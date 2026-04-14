#!/usr/bin/env python3
"""4D-style viser viewer for Point3R-style reconstruction dirs (frame_*.npz).

Supports:
- Single dir visualization (frame slider + autoplay + camera frustum).
- Two dirs side-by-side visualization with synchronized frame index.
- Optional trajectory spline built from per-frame camera_pose.
"""

import argparse
import glob
from typing import TypeAlias
import os
import re
import threading
import time
from dataclasses import dataclass
import inspect

import numpy as np
import viser

Vec3Tuple: TypeAlias = tuple[float, float, float]


def _last_int(text: str) -> int:
    m = re.search(r"(\d+)(?!.*\d)", text)
    return int(m.group(1)) if m else 0


def _sorted_frame_paths(npz_dir: str) -> list[str]:
    return sorted(
        glob.glob(os.path.join(npz_dir, "frame_*.npz")),
        key=lambda p: _last_int(os.path.basename(p)),
    )


def _to_uint8(img: np.ndarray | None) -> np.ndarray | None:
    if img is None:
        return None
    if img.dtype == np.uint8:
        return img
    img = np.nan_to_num(img)
    if img.max() <= 1.0:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def _rotmat_to_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    q = q / np.linalg.norm(q)
    return q


def _load_npz(path: str):
    d = np.load(path)
    pts3d = d["pts3d"]
    img = d["img"] if "img" in d.files else None
    conf = d["conf"] if "conf" in d.files else None
    camera_pose = d["camera_pose"] if "camera_pose" in d.files else np.eye(4, dtype=np.float32)
    return pts3d, img, conf, camera_pose


def _prepare_cloud(
    pts3d: np.ndarray,
    img: np.ndarray | None,
    conf: np.ndarray | None,
    downsample: int,
    max_points: int,
    conf_threshold: float | None,
    keep_global_coords: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if pts3d.ndim == 4:
        pts3d = pts3d.squeeze(0)
    pts = pts3d.reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)

    if conf is not None:
        conf_f = np.asarray(conf).reshape(-1)
        valid = valid & np.isfinite(conf_f)
        if conf_threshold is not None:
            valid = valid & (conf_f >= conf_threshold)

    img_u8 = _to_uint8(img)
    if img_u8 is None:
        col = np.full((pts.shape[0], 3), 180, dtype=np.uint8)
    else:
        col = img_u8.reshape(-1, 3)

    pts = pts[valid]
    col = col[valid]

    if downsample > 1:
        pts, col = pts[::downsample], col[::downsample]

    if len(pts) > max_points:
        keep = np.linspace(0, len(pts) - 1, max_points, dtype=int)
        pts, col = pts[keep], col[keep]

    if (not keep_global_coords) and len(pts) > 0:
        pts = pts - np.mean(pts, axis=0, keepdims=True)

    return pts.astype(np.float32), col.astype(np.uint8)


@dataclass
class DirIndex:
    name: str
    npz_dir: str
    paths: list[str]
    idx_by_frame: dict[int, str]


def _build_index(npz_dir: str, label: str) -> DirIndex:
    paths = _sorted_frame_paths(npz_dir)
    idx_by_frame = {}
    for p in paths:
        idx_by_frame[_last_int(os.path.basename(p))] = p
    return DirIndex(name=label, npz_dir=npz_dir, paths=paths, idx_by_frame=idx_by_frame)


def _camera_pose_position_wxyz(camera_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(camera_pose, dtype=np.float32).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    return t, _rotmat_to_wxyz(R)


def _as_vec3_tuple_seq(points: np.ndarray) -> tuple[Vec3Tuple, ...]:
    return tuple((float(p[0]), float(p[1]), float(p[2])) for p in points)


def _add_camera_frustum(scene, *, name: str, fov: float, aspect: float, scale: float, wxyz, position, color, thickness: float):
    kwargs = {
        "name": name,
        "fov": fov,
        "aspect": aspect,
        "scale": scale,
        "wxyz": wxyz,
        "position": position,
        "color": color,
        "thickness": thickness,
    }
    scene.add_camera_frustum(**kwargs)


def main():
    try:
        from viser import _scene_api

        params = inspect.signature(_scene_api.SceneApi.add_camera_frustum).parameters
        if "thickness" not in params:
            _orig = _scene_api.SceneApi.add_camera_frustum

            def _patched(self, *args, thickness=None, **kwargs):
                return _orig(self, *args, **kwargs)

            _scene_api.SceneApi.add_camera_frustum = _patched
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--left_dir", type=str, required=True)
    parser.add_argument("--right_dir", type=str, default="")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--left_label", type=str, default="left")
    parser.add_argument("--right_label", type=str, default="right")
    parser.add_argument("--left_downsample", type=int, default=8)
    parser.add_argument("--right_downsample", type=int, default=8)
    parser.add_argument("--max_points", type=int, default=500000)
    parser.add_argument("--point_size", type=float, default=0.02)
    parser.add_argument("--conf_threshold", type=float, default=-1.0)
    parser.add_argument("--keep_global_coords", action="store_true")
    parser.add_argument("--split_offset", type=float, default=1.8)
    parser.add_argument("--camera_frustum_scale", type=float, default=0.08)
    parser.add_argument("--camera_fov_deg", type=float, default=70.0)
    parser.add_argument("--camera_thickness", type=float, default=2.0)
    parser.add_argument("--trajectory_stride", type=int, default=5)
    args = parser.parse_args()

    left = _build_index(args.left_dir, args.left_label)
    right = _build_index(args.right_dir, args.right_label) if args.right_dir else None
    if not left.paths:
        raise ValueError(f"No frame_*.npz found in {args.left_dir}")
    if right is not None and not right.paths:
        raise ValueError(f"No frame_*.npz found in {args.right_dir}")

    if right is None:
        timeline = sorted(left.idx_by_frame.keys())
    else:
        common = sorted(set(left.idx_by_frame.keys()) & set(right.idx_by_frame.keys()))
        if not common:
            raise ValueError("No common frame indices between left_dir and right_dir")
        timeline = common

    conf_threshold = None if args.conf_threshold < 0 else float(args.conf_threshold)
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    info = server.gui.add_text("Info", "")
    slider = server.gui.add_slider("Frame", min=0, max=len(timeline) - 1, step=1, initial_value=0)
    point_size = server.gui.add_slider("Point Size", min=0.001, max=0.1, step=0.001, initial_value=args.point_size)
    show_cameras = server.gui.add_checkbox("Show Cameras", initial_value=True)
    show_trajectory = server.gui.add_checkbox("Show Trajectory", initial_value=False)
    autoplay = server.gui.add_checkbox("Auto Play", initial_value=False)
    pause_resume = server.gui.add_button("Pause/Resume")
    playback = server.gui.add_text("Playback", "autoplay=off paused=false")
    playback_state = {"paused": False}

    traj_built = {"left": False, "right": False}

    @pause_resume.on_click
    def _(_e):
        if not autoplay.value:
            autoplay.value = True
            playback_state["paused"] = False
        else:
            playback_state["paused"] = not playback_state["paused"]
        playback.value = (
            f"autoplay={'on' if autoplay.value else 'off'} "
            f"paused={str(bool(playback_state['paused'])).lower()}"
        )

    def _ensure_trajectory():
        if not show_trajectory.value:
            return
        fov = np.deg2rad(float(args.camera_fov_deg))

        if not traj_built["left"]:
            pts = []
            for i, frame_idx in enumerate(timeline[:: max(1, int(args.trajectory_stride))]):
                p = left.idx_by_frame.get(frame_idx)
                if p is None:
                    continue
                _, _, _, pose = _load_npz(p)
                pos, _wxyz = _camera_pose_position_wxyz(pose)
                pts.append(pos.astype(np.float32))
            if len(pts) >= 2:
                pts_np = np.stack(pts, axis=0)
                if right is not None:
                    pts_np = pts_np + np.array([-args.split_offset * 0.5, 0.0, 0.0], dtype=np.float32)
                server.scene.add_spline_catmull_rom(
                    name="/trajectory/left",
                    positions=_as_vec3_tuple_seq(pts_np),
                    line_width=2,
                    color=(255, 60, 60),
                )
            traj_built["left"] = True

        if right is not None and (not traj_built["right"]):
            pts = []
            for i, frame_idx in enumerate(timeline[:: max(1, int(args.trajectory_stride))]):
                p = right.idx_by_frame.get(frame_idx)
                if p is None:
                    continue
                _, _, _, pose = _load_npz(p)
                pos, _wxyz = _camera_pose_position_wxyz(pose)
                pts.append(pos.astype(np.float32))
            if len(pts) >= 2:
                pts_np = np.stack(pts, axis=0)
                pts_np = pts_np + np.array([args.split_offset * 0.5, 0.0, 0.0], dtype=np.float32)
                server.scene.add_spline_catmull_rom(
                    name="/trajectory/right",
                    positions=_as_vec3_tuple_seq(pts_np),
                    line_width=2,
                    color=(60, 120, 255),
                )
            traj_built["right"] = True

    def _render():
        frame_idx = int(timeline[int(slider.value)])
        fov = np.deg2rad(float(args.camera_fov_deg))
        aspect = 1.0

        pts_l, img_l, conf_l, pose_l = _load_npz(left.idx_by_frame[frame_idx])
        cloud_l, col_l = _prepare_cloud(
            pts_l,
            img_l,
            conf_l,
            downsample=int(args.left_downsample),
            max_points=int(args.max_points),
            conf_threshold=conf_threshold,
            keep_global_coords=bool(args.keep_global_coords),
        )

        offset_l = np.zeros(3, dtype=np.float32)
        if right is not None:
            offset_l = np.array([-args.split_offset * 0.5, 0.0, 0.0], dtype=np.float32)
        cloud_l = cloud_l + offset_l
        pos_l, wxyz_l = _camera_pose_position_wxyz(pose_l)
        pos_l = pos_l + offset_l

        server.scene.add_point_cloud(
            name="/left/point_cloud",
            points=cloud_l,
            colors=col_l,
            point_size=float(point_size.value),
            point_shape="circle",
        )

        if show_cameras.value:
            _add_camera_frustum(
                server.scene,
                name="/left/camera",
                fov=fov,
                aspect=aspect,
                scale=float(args.camera_frustum_scale),
                wxyz=wxyz_l,
                position=pos_l,
                color=(255, 60, 60),
                thickness=float(args.camera_thickness),
            )
        else:
            server.scene.add_frame("/left/camera", visible=False)

        right_text = ""
        if right is not None:
            pts_r, img_r, conf_r, pose_r = _load_npz(right.idx_by_frame[frame_idx])
            cloud_r, col_r = _prepare_cloud(
                pts_r,
                img_r,
                conf_r,
                downsample=int(args.right_downsample),
                max_points=int(args.max_points),
                conf_threshold=conf_threshold,
                keep_global_coords=bool(args.keep_global_coords),
            )
            offset_r = np.array([args.split_offset * 0.5, 0.0, 0.0], dtype=np.float32)
            cloud_r = cloud_r + offset_r
            pos_r, wxyz_r = _camera_pose_position_wxyz(pose_r)
            pos_r = pos_r + offset_r

            server.scene.add_point_cloud(
                name="/right/point_cloud",
                points=cloud_r,
                colors=col_r,
                point_size=float(point_size.value),
                point_shape="circle",
            )
            if show_cameras.value:
                _add_camera_frustum(
                    server.scene,
                    name="/right/camera",
                    fov=fov,
                    aspect=aspect,
                    scale=float(args.camera_frustum_scale),
                    wxyz=wxyz_r,
                    position=pos_r,
                    color=(60, 120, 255),
                    thickness=float(args.camera_thickness),
                )
            else:
                server.scene.add_frame("/right/camera", visible=False)
            right_text = f" right={os.path.basename(right.npz_dir)}"

        _ensure_trajectory()
        info.value = (
            f"frame_index={frame_idx} "
            f"left={os.path.basename(left.npz_dir)}"
            f"{right_text} "
            f"keep_global_coords={str(bool(args.keep_global_coords)).lower()} "
            f"conf_threshold={'none' if conf_threshold is None else conf_threshold}"
        )

    @slider.on_update
    def _(_e):
        _render()

    @show_cameras.on_update
    def _(_e):
        _render()

    @point_size.on_update
    def _(_e):
        _render()

    @show_trajectory.on_update
    def _(_e):
        _ensure_trajectory()

    def _autoplay_loop():
        while True:
            time.sleep(0.15)
            if not autoplay.value or playback_state["paused"]:
                continue
            v = int(slider.value) + 1
            if v > int(slider.max):
                v = int(slider.min)
            slider.value = v

    threading.Thread(target=_autoplay_loop, daemon=True).start()

    _render()
    print(f"Viewer ready: http://localhost:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
