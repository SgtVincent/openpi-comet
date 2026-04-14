#!/usr/bin/env python3
"""Use viser to visualize InfiniteVGGT outputs in sliding-window or aggregated mode."""

import argparse
import glob
import os
import re
import threading
import time

import numpy as np
import viser
from PIL import Image


def _last_int(text: str) -> int:
    m = re.search(r"(\d+)(?!.*\d)", text)
    return int(m.group(1)) if m else 0


def _sorted_window_dirs(root: str):
    return sorted(
        glob.glob(os.path.join(root, "window_*")),
        key=lambda p: _last_int(os.path.basename(p)),
    )


def _sorted_frame_npzs(root: str):
    return sorted(
        glob.glob(os.path.join(root, "frame_*.npz")),
        key=lambda p: _last_int(os.path.basename(p)),
    )


def _sorted_pngs(root: str):
    return sorted(
        glob.glob(os.path.join(root, "*.png")),
        key=lambda p: _last_int(os.path.basename(p)),
    )


def _to_uint8(img):
    if img is None:
        return None
    if img.dtype == np.uint8:
        return img
    img = np.nan_to_num(img)
    if img.max() <= 1.0:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def _load_frame_cloud(npz_path: str, image_path: str | None, conf_percentile: float):
    data = np.load(npz_path)
    if "point_map" in data.files:
        pts = data["point_map"]
        conf = data["point_conf"] if "point_conf" in data.files else None
        embedded_img = None
    elif "pts3d" in data.files:
        pts = data["pts3d"]
        conf = data["conf"] if "conf" in data.files else None
        embedded_img = data["img"] if "img" in data.files else None
    else:
        raise KeyError(f"Neither point_map nor pts3d found in {npz_path}")

    if pts.ndim == 4:
        pts = pts.squeeze(0)
    pts = pts.reshape(-1, 3)
    if conf is None:
        conf = np.ones((pts.shape[0],), dtype=np.float32)
    else:
        conf = np.asarray(conf)
        if conf.ndim == 3:
            conf = conf.squeeze(0)
        conf = conf.reshape(-1)
    valid = np.isfinite(pts).all(axis=1) & np.isfinite(conf)
    if np.any(valid):
        thr = np.percentile(conf[valid], conf_percentile)
        valid = valid & (conf >= thr)

    pts = pts[valid]
    if image_path and os.path.exists(image_path):
        img = np.array(Image.open(image_path).convert("RGB"))
    else:
        img = embedded_img
    if img is None:
        col = np.full((len(pts), 3), 180, dtype=np.uint8)
    else:
        img_u8 = _to_uint8(img)
        if img_u8 is None:
            col = np.full((len(pts), 3), 180, dtype=np.uint8)
        else:
            col = img_u8.reshape(-1, 3)[valid]

    meta = {
        "source_frame_index": int(data["source_frame_index"]) if "source_frame_index" in data.files else _last_int(os.path.basename(npz_path)),
        "local_frame_index": int(data["local_frame_index"]) if "local_frame_index" in data.files else None,
        "window_id": int(data["window_id"]) if "window_id" in data.files else None,
        "path": npz_path,
    }
    return pts.astype(np.float32), col.astype(np.uint8), meta


def _downsample_limit(pts: np.ndarray, col: np.ndarray, downsample: int, max_points: int):
    if downsample > 1:
        pts = pts[::downsample]
        col = col[::downsample]
    if len(pts) > max_points:
        keep = np.linspace(0, len(pts) - 1, max_points, dtype=int)
        pts = pts[keep]
        col = col[keep]
    return pts, col


def _collect_window_records(root: str):
    records = []
    for window_dir in _sorted_window_dirs(root):
        frame_npzs = _sorted_frame_npzs(window_dir)
        image_pngs = _sorted_pngs(os.path.join(window_dir, "images"))
        for local_idx, npz_path in enumerate(frame_npzs):
            image_path = image_pngs[local_idx] if local_idx < len(image_pngs) else None
            records.append(
                {
                    "window_dir": window_dir,
                    "window_name": os.path.basename(window_dir),
                    "window_id": _last_int(os.path.basename(window_dir)),
                    "local_idx": local_idx,
                    "npz_path": npz_path,
                    "image_path": image_path,
                }
            )
    return records


def _collect_root_records(root: str):
    frame_npzs = _sorted_frame_npzs(root)
    image_pngs = _sorted_pngs(os.path.join(root, "images"))
    return [
        {
            "window_dir": root,
            "window_name": os.path.basename(root.rstrip("/")) or os.path.basename(root),
            "window_id": 0,
            "local_idx": idx,
            "npz_path": npz_path,
            "image_path": image_pngs[idx] if idx < len(image_pngs) else None,
        }
        for idx, npz_path in enumerate(frame_npzs)
    ]


def _collect_records(root: str):
    window_records = _collect_window_records(root)
    if window_records:
        return window_records, True
    root_records = _collect_root_records(root)
    return root_records, False


def _aggregate_cloud(records, conf_percentile: float, downsample: int, max_points: int, dedupe_by_source: bool):
    selected = []
    if dedupe_by_source:
        seen = set()
        for rec in records:
            data = np.load(rec["npz_path"])
            source_idx = int(data["source_frame_index"]) if "source_frame_index" in data.files else None
            key = source_idx if source_idx is not None else rec["npz_path"]
            if key in seen:
                continue
            seen.add(key)
            selected.append(rec)
    else:
        selected = list(records)

    pts_all = []
    col_all = []
    stats = []
    for rec in selected:
        pts, col, meta = _load_frame_cloud(rec["npz_path"], rec["image_path"], conf_percentile)
        if len(pts) == 0:
            continue
        pts_all.append(pts)
        col_all.append(col)
        stats.append(meta)

    if not pts_all:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), stats

    pts = np.concatenate(pts_all, axis=0)
    col = np.concatenate(col_all, axis=0)
    pts, col = _downsample_limit(pts, col, downsample, max_points)
    if len(pts) > 0:
        pts = pts - np.mean(pts, axis=0, keepdims=True)
    return pts.astype(np.float32), col.astype(np.uint8), stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infinitevggt_dir", type=str, required=True)
    parser.add_argument("--view_mode", type=str, default="sliding", choices=["sliding", "all_points"])
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--conf_percentile", type=float, default=25.0)
    parser.add_argument("--downsample", type=int, default=6)
    parser.add_argument("--max_points", type=int, default=500000)
    parser.add_argument("--point_size", type=float, default=0.02)
    parser.add_argument("--dedupe_by_source_frame", action="store_true")
    args = parser.parse_args()

    records, has_windows = _collect_records(args.infinitevggt_dir)
    if not records:
        raise ValueError(f"No InfiniteVGGT frame_*.npz found in {args.infinitevggt_dir}")

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    info = server.gui.add_text("Info", "")
    window_names = sorted({rec["window_name"] for rec in records}, key=_last_int)
    window_to_records = {name: [rec for rec in records if rec["window_name"] == name] for name in window_names}
    max_local_frames = max(len(v) for v in window_to_records.values())
    meta_cache = {}
    source_mapping = {}
    source_indices = []
    source_built = [False]

    mode_all_points = server.gui.add_checkbox("All Points Mode", initial_value=args.view_mode == "all_points")
    point_size = server.gui.add_slider("Point Size", min=0.001, max=0.1, step=0.001, initial_value=args.point_size)
    dedupe_source = server.gui.add_checkbox("All Points 去重 source_frame", initial_value=args.dedupe_by_source_frame)
    use_source_frame = server.gui.add_checkbox("Sliding 使用 source_frame 对齐", initial_value=False)
    s_win = server.gui.add_slider("InfiniteVGGT Window", min=0, max=len(window_names) - 1, step=1, initial_value=0)
    s_loc = server.gui.add_slider(
        "InfiniteVGGT Local Frame",
        min=0,
        max=max(0, max_local_frames - 1),
        step=1,
        initial_value=0,
    )
    s_src = server.gui.add_slider(
        "Source Frame Index",
        min=0,
        max=0,
        step=1,
        initial_value=0,
    )

    autoplay = server.gui.add_checkbox("Auto Play", initial_value=False)
    pause_resume = server.gui.add_button("Pause/Resume")
    playback = server.gui.add_text("Playback", "autoplay=off paused=false")
    playback_state = {"paused": False}

    aggregate_cache = {}

    def _record_meta(rec):
        cache_key = rec["npz_path"]
        if cache_key not in meta_cache:
            data = np.load(rec["npz_path"])
            meta_cache[cache_key] = {
                "source_frame_index": int(data["source_frame_index"]) if "source_frame_index" in data.files else None,
                "local_frame_index": int(data["local_frame_index"]) if "local_frame_index" in data.files else None,
                "window_id": int(data["window_id"]) if "window_id" in data.files else None,
            }
        return meta_cache[cache_key]

    def _ensure_source_mapping():
        if source_built[0]:
            return source_mapping, source_indices
        mapping = {}
        for rec in records:
            source_idx = _record_meta(rec)["source_frame_index"]
            if source_idx is None:
                continue
            mapping.setdefault(source_idx, []).append(rec)
        source_mapping.clear()
        source_mapping.update(mapping)
        source_indices.clear()
        source_indices.extend(sorted(mapping.keys()))
        source_built[0] = True
        s_src.max = max(0, len(source_indices) - 1)
        return source_mapping, source_indices

    def _render_all_points():
        cache_key = bool(dedupe_source.value)
        if cache_key not in aggregate_cache:
            aggregate_cache[cache_key] = _aggregate_cloud(
                records,
                args.conf_percentile,
                args.downsample,
                args.max_points,
                cache_key,
            )
        pts, col, stats = aggregate_cache[cache_key]
        server.scene.add_point_cloud(
            name="/infinitevggt",
            points=pts,
            colors=col,
            point_size=float(point_size.value),
            point_shape="circle",
        )
        unique_sources = sorted({s["source_frame_index"] for s in stats if s["source_frame_index"] is not None})
        info.value = (
            f"mode=all_points records={len(stats)} unique_source_frames={len(unique_sources)} "
            f"has_windows={has_windows} dedupe_by_source_frame={bool(dedupe_source.value)}"
        )

    def _render():
        if mode_all_points.value:
            _render_all_points()
            return

        if bool(use_source_frame.value):
            source_to_records, source_indices = _ensure_source_mapping()
            if source_indices:
                source_idx = source_indices[int(min(s_src.value, len(source_indices) - 1))]
                rec = source_to_records[source_idx][0]
                window_name = rec["window_name"]
                local_idx = rec["local_idx"]
            else:
                window_name = window_names[int(s_win.value)]
                candidates = window_to_records[window_name]
                local_idx = min(int(s_loc.value), len(candidates) - 1)
                rec = candidates[local_idx]
        else:
            window_name = window_names[int(s_win.value)]
            candidates = window_to_records[window_name]
            local_idx = min(int(s_loc.value), len(candidates) - 1)
            rec = candidates[local_idx]

        pts, col, meta = _load_frame_cloud(rec["npz_path"], rec["image_path"], args.conf_percentile)
        pts, col = _downsample_limit(pts, col, args.downsample, args.max_points)
        if len(pts) > 0:
            pts = pts - np.mean(pts, axis=0, keepdims=True)
        server.scene.add_point_cloud(
            name="/infinitevggt",
            points=pts.astype(np.float32),
            colors=col.astype(np.uint8),
            point_size=float(point_size.value),
            point_shape="circle",
        )
        info.value = (
            f"mode=sliding window={window_name} local={local_idx}/{len(window_to_records[window_name])-1} "
            f"source={meta['source_frame_index']} npz={os.path.basename(rec['npz_path'])}"
        )

    def _update_playback_text():
        playback.value = f"autoplay={'on' if autoplay.value else 'off'} paused={str(bool(playback_state['paused'])).lower()}"

    @autoplay.on_update
    def _(_e):
        if autoplay.value:
            playback_state["paused"] = False
        _update_playback_text()

    @pause_resume.on_click
    def _(_e):
        if not autoplay.value:
            autoplay.value = True
            playback_state["paused"] = False
        else:
            playback_state["paused"] = not playback_state["paused"]
        _update_playback_text()

    def _autoplay_loop():
        while True:
            time.sleep(0.5)
            if not autoplay.value or playback_state["paused"] or mode_all_points.value:
                continue
            if bool(use_source_frame.value):
                source_to_records, source_indices = _ensure_source_mapping()
                if not source_indices:
                    continue
                if int(s_src.value) >= len(source_indices) - 1:
                    s_src.value = 0
                else:
                    s_src.value = int(s_src.value) + 1
            else:
                win_name = window_names[int(s_win.value)]
                n_local = len(window_to_records[win_name])
                if n_local <= 0:
                    continue
                if int(s_loc.value) >= n_local - 1:
                    s_loc.value = 0
                    if int(s_win.value) >= len(window_names) - 1:
                        s_win.value = 0
                    else:
                        s_win.value = int(s_win.value) + 1
                else:
                    s_loc.value = int(s_loc.value) + 1
            _render()

    threading.Thread(target=_autoplay_loop, daemon=True).start()

    @mode_all_points.on_update
    def _(_e):
        _render()

    @dedupe_source.on_update
    def _(_e):
        _render()

    @point_size.on_update
    def _(_e):
        _render()

    @use_source_frame.on_update
    def _(_e):
        _render()

    @s_win.on_update
    def _(_e):
        _render()

    @s_loc.on_update
    def _(_e):
        _render()

    if s_src is not None:
        @s_src.on_update
        def _(_e):
            _render()

    _update_playback_text()
    _render()
    print(f"Viewer ready: http://localhost:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
