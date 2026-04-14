#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


CAMERA_SEG_KEY_MAP = {
    "head": "observation.images.seg_instance_id.head",
    "left_wrist": "observation.images.seg_instance_id.left_wrist",
    "right_wrist": "observation.images.seg_instance_id.right_wrist",
}

CAMERA_UNIQUE_IDS_KEY_MAP = {
    "head": "robot_r1::robot_r1:zed_link:Camera:0::unique_ins_ids",
    "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0::unique_ins_ids",
    "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0::unique_ins_ids",
}

ROBOT_ARM_KEYWORDS = (
    "robot_r1",
    "arm",
    "wrist",
    "gripper",
    "finger",
    "hand",
    "forearm",
    "upperarm",
    "link",
    "left",
    "right",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos",
    )
    parser.add_argument("--task_id", type=int, default=13)
    parser.add_argument("--episode_id", type=int, default=130010)
    parser.add_argument("--camera", type=str, default="head", choices=list(CAMERA_SEG_KEY_MAP.keys()))
    parser.add_argument("--clip_length", type=int, default=240)
    parser.add_argument("--clip_step", type=int, default=120)
    parser.add_argument("--sample_stride", type=int, default=15)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output_path", type=str, default="")
    return parser.parse_args()


def generate_yuv_palette(num_ids: int):
    side = int(np.ceil(num_ids ** (1 / 3)))
    y_vals = np.linspace(16, 235, side)
    u_vals = np.linspace(16, 240, side)
    v_vals = np.linspace(16, 240, side)
    palette = []
    for y in y_vals:
        for u in u_vals:
            for v in v_vals:
                palette.append([y, u, v])
                if len(palette) >= num_ids:
                    return np.asarray(palette, dtype=np.float32)
    return np.asarray(palette[:num_ids], dtype=np.float32)


def load_episode_json(data_root: str, task_id: int, episode_id: int):
    ep_name = f"episode_{episode_id:08d}.json"
    ann_path = Path(data_root) / "annotations" / f"task-{task_id:04d}" / ep_name
    meta_path = Path(data_root) / "meta" / "episodes" / f"task-{task_id:04d}" / ep_name
    with open(ann_path, "r") as f:
        annotations = json.load(f)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    return annotations, metadata, str(ann_path), str(meta_path)


def build_candidates(annotations: dict):
    candidates = []
    seen = set()
    for ann in annotations.get("skill_annotation", []):
        desc = ",".join(ann.get("skill_description", []))
        skill_type = ",".join(ann.get("skill_type", []))
        start, end = [int(x) for x in ann["frame_duration"]]
        if desc.strip().lower() == "move to" or skill_type.strip().lower() == "navigation":
            continue
        key = ("skill", start, end, desc, skill_type)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "source": "skill",
                "source_index": int(ann["skill_idx"]),
                "label": desc,
                "type": skill_type,
                "start_frame": start,
                "end_frame": end,
            }
        )
    for ann in annotations.get("primitive_annotation", []):
        desc = ",".join(ann.get("primitive_description", []))
        start, end = [int(x) for x in ann["frame_duration"]]
        key = ("primitive", start, end, desc)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "source": "primitive",
                "source_index": int(ann["primitive_idx"]),
                "label": desc,
                "type": "",
                "start_frame": start,
                "end_frame": end,
            }
        )
    return candidates


def parse_ins_id_mapping(metadata: dict):
    raw = metadata["ins_id_mapping"]
    mapping = json.loads(raw) if isinstance(raw, str) else raw
    return {int(k): v for k, v in mapping.items()}


def select_robot_ids(metadata: dict, camera: str):
    mapping = parse_ins_id_mapping(metadata)
    unique_ids = [int(x) for x in metadata[CAMERA_UNIQUE_IDS_KEY_MAP[camera]]]
    robot_specific = []
    robot_fallback = []
    for ins_id in unique_ids:
        name = mapping.get(ins_id, "")
        lowered = name.lower()
        if "robot_r1" not in lowered:
            continue
        robot_fallback.append(ins_id)
        if any(token in lowered for token in ROBOT_ARM_KEYWORDS[1:]):
            robot_specific.append(ins_id)
    selected = robot_specific or robot_fallback
    return mapping, unique_ids, selected


def seg_video_path(data_root: str, task_id: int, episode_id: int, camera: str):
    return str(
        Path(data_root)
        / "videos"
        / f"task-{task_id:04d}"
        / CAMERA_SEG_KEY_MAP[camera]
        / f"episode_{episode_id:08d}.mp4"
    )


def build_clip_windows(start_frame: int, end_frame: int, clip_length: int, clip_step: int):
    if end_frame <= start_frame:
        return []
    span = end_frame - start_frame
    if span <= clip_length:
        return [(start_frame, end_frame)]
    windows = []
    cursor = start_frame
    while cursor + clip_length < end_frame:
        windows.append((cursor, cursor + clip_length))
        cursor += clip_step
    tail = (max(start_frame, end_frame - clip_length), end_frame)
    if not windows or windows[-1] != tail:
        windows.append(tail)
    return windows


def compute_centroids(mask_stack: np.ndarray):
    h, w = mask_stack.shape[1:]
    yy, xx = np.mgrid[0:h, 0:w]
    centroids = []
    for mask in mask_stack:
        area = mask.sum()
        if area <= 0:
            centroids.append(np.array([np.nan, np.nan], dtype=np.float32))
            continue
        cy = float((yy * mask).sum() / area)
        cx = float((xx * mask).sum() / area)
        centroids.append(np.array([cy, cx], dtype=np.float32))
    centroids = np.stack(centroids, axis=0)
    valid = np.isfinite(centroids).all(axis=1)
    if not valid.any():
        centroids[:] = 0.0
        return centroids
    first_valid = np.argmax(valid)
    centroids[:first_valid] = centroids[first_valid]
    for i in range(first_valid + 1, len(centroids)):
        if not valid[i]:
            centroids[i] = centroids[i - 1]
    return centroids


def score_mask_sequence(mask_stack: np.ndarray):
    if mask_stack.shape[0] < 2:
        return None
    areas = mask_stack.sum(axis=(1, 2)).astype(np.float32)
    total_pixels = float(mask_stack.shape[1] * mask_stack.shape[2])
    areas_norm = areas / max(total_pixels, 1.0)
    centroids = compute_centroids(mask_stack.astype(np.float32))
    diag = math.sqrt(mask_stack.shape[1] ** 2 + mask_stack.shape[2] ** 2)
    centroid_disp = np.linalg.norm(np.diff(centroids, axis=0), axis=1) / max(diag, 1e-6)
    inter = (mask_stack[:-1] & mask_stack[1:]).sum(axis=(1, 2)).astype(np.float32)
    union = (mask_stack[:-1] | mask_stack[1:]).sum(axis=(1, 2)).astype(np.float32)
    iou = np.divide(inter, np.maximum(union, 1.0))
    area_change = np.abs(np.diff(areas_norm))
    visibility = float((areas > 0).mean())
    score = float(3.0 * np.mean(1.0 - iou) + 2.0 * np.mean(area_change) + 1.5 * np.mean(centroid_disp) + 0.5 * visibility)
    return {
        "score": score,
        "visibility": visibility,
        "mean_iou_loss": float(np.mean(1.0 - iou)),
        "mean_area_change": float(np.mean(area_change)),
        "mean_centroid_disp": float(np.mean(centroid_disp)),
        "num_sampled_frames": int(mask_stack.shape[0]),
        "mean_area_ratio": float(np.mean(areas_norm)),
        "max_area_ratio": float(np.max(areas_norm)),
    }


def decode_segmentation_frame(rgb_frame: np.ndarray, unique_ids: list[int], palette: np.ndarray):
    flat = rgb_frame.reshape(-1, 3).astype(np.float32)
    colors, inverse = np.unique(flat, axis=0, return_inverse=True)
    distances = ((colors[:, None, :] - palette[None, :, :]) ** 2).sum(axis=-1)
    palette_indices = np.argmin(distances, axis=1)
    ids = np.asarray(unique_ids, dtype=np.int32)[palette_indices]
    return ids[inverse].reshape(rgb_frame.shape[0], rgb_frame.shape[1])


def load_mask_stack(data_root: str, task_id: int, episode_id: int, camera: str, unique_ids: list[int], robot_ids: list[int], start_frame: int, end_frame: int, sample_stride: int):
    if not robot_ids:
        return None
    cap = cv2.VideoCapture(seg_video_path(data_root, task_id, episode_id, camera))
    if not cap.isOpened():
        raise ValueError("Cannot open segmentation video")
    palette = generate_yuv_palette(len(unique_ids))
    masks = []
    try:
        for frame_idx in range(start_frame, end_frame, sample_stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            seg = decode_segmentation_frame(rgb, unique_ids, palette)
            masks.append(np.isin(seg, np.asarray(robot_ids, dtype=seg.dtype)))
    finally:
        cap.release()
    if len(masks) < 2:
        return None
    return np.stack(masks, axis=0)


def evaluate_candidates(data_root: str, task_id: int, episode_id: int, camera: str, candidates: list[dict], unique_ids: list[int], robot_ids: list[int], clip_length: int, clip_step: int, sample_stride: int):
    scored = []
    for candidate in candidates:
        windows = build_clip_windows(candidate["start_frame"], candidate["end_frame"], clip_length, clip_step)
        for clip_start, clip_end in windows:
            mask_stack = load_mask_stack(
                data_root=data_root,
                task_id=task_id,
                episode_id=episode_id,
                camera=camera,
                unique_ids=unique_ids,
                robot_ids=robot_ids,
                start_frame=clip_start,
                end_frame=clip_end,
                sample_stride=sample_stride,
            )
            if mask_stack is None:
                continue
            metrics = score_mask_sequence(mask_stack)
            if metrics is None:
                continue
            scored.append(
                {
                    "source": candidate["source"],
                    "source_index": candidate["source_index"],
                    "label": candidate["label"],
                    "type": candidate["type"],
                    "candidate_start_frame": candidate["start_frame"],
                    "candidate_end_frame": candidate["end_frame"],
                    "clip_start_frame": clip_start,
                    "clip_end_frame": clip_end,
                    "clip_length": clip_end - clip_start,
                    **metrics,
                }
            )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def main():
    args = parse_args()
    annotations, metadata, ann_path, meta_path = load_episode_json(args.data_root, args.task_id, args.episode_id)
    candidates = build_candidates(annotations)
    mapping, unique_ids, robot_ids = select_robot_ids(metadata, args.camera)
    scored = evaluate_candidates(
        data_root=args.data_root,
        task_id=args.task_id,
        episode_id=args.episode_id,
        camera=args.camera,
        candidates=candidates,
        unique_ids=unique_ids,
        robot_ids=robot_ids,
        clip_length=args.clip_length,
        clip_step=args.clip_step,
        sample_stride=args.sample_stride,
    )
    summary = {
        "task_id": args.task_id,
        "episode_id": args.episode_id,
        "camera": args.camera,
        "annotation_path": ann_path,
        "metadata_path": meta_path,
        "num_candidates": len(candidates),
        "num_scored_clips": len(scored),
        "clip_length": args.clip_length,
        "clip_step": args.clip_step,
        "sample_stride": args.sample_stride,
        "robot_ids": robot_ids,
        "robot_paths": {str(k): mapping.get(k, "") for k in robot_ids},
        "top_candidates": scored[: args.top_k],
        "recommended_clip": scored[0] if scored else None,
    }
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = PROJECT_ROOT / "outputs" / "dynamic_segments" / f"task{args.task_id:04d}_ep{args.episode_id:08d}_{args.camera}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"recommended_clip": summary["recommended_clip"], "output_path": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
