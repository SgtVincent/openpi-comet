#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EASI3R_ROOT = PROJECT_ROOT / "src" / "openpi" / "third_party" / "Easi3R"
sys.path.insert(0, str(EASI3R_ROOT))
sys.path.insert(0, str(EASI3R_ROOT / "third_party" / "sam2"))

from demo import get_reconstructed_scene
from dust3r.model import AsymmetricCroCo3DStereo


CAMERA_KEY_MAP = {
    "head": "observation.images.rgb.head",
    "left_wrist": "observation.images.rgb.left_wrist",
    "right_wrist": "observation.images.rgb.right_wrist",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos",
    )
    parser.add_argument("--episode_id", type=int, default=130010)
    parser.add_argument("--camera", type=str, default="head", choices=list(CAMERA_KEY_MAP.keys()))
    parser.add_argument("--start_frame", type=int, required=True)
    parser.add_argument("--end_frame", type=int, required=True)
    parser.add_argument("--use_source_video", action="store_true")
    parser.add_argument("--sampling_mode", type=str, default="native_fps", choices=["native_fps", "uniform", "stride"])
    parser.add_argument("--sample_stride", type=int, default=15)
    parser.add_argument("--weights", type=str, default=str(EASI3R_ROOT / "checkpoints" / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"))
    parser.add_argument("--model_name", type=str, default="nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seq_name", type=str, default="")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--num_frames", type=int, default=65)
    parser.add_argument("--niter", type=int, default=300)
    parser.add_argument("--scenegraph_type", type=str, default="swinstride")
    parser.add_argument("--winsize", type=int, default=5)
    parser.add_argument("--refid", type=int, default=0)
    parser.add_argument("--flow_loss_weight", type=float, default=0.01)
    parser.add_argument("--flow_loss_start_iter", type=float, default=0.1)
    parser.add_argument("--flow_loss_threshold", type=float, default=25.0)
    parser.add_argument("--temporal_smoothing_weight", type=float, default=0.0)
    parser.add_argument("--translation_weight", type=str, default="1.0")
    parser.add_argument("--min_conf_thr", type=float, default=1.1)
    parser.add_argument("--cam_size", type=float, default=0.05)
    parser.add_argument("--sam2_mask_refine", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--not_batchify", action="store_true")
    return parser.parse_args()


def ensure_output_dir(path: str, overwrite: bool):
    output_path = Path(path)
    if not output_path.exists():
        return
    if overwrite:
        return
    entries = [p for p in output_path.iterdir() if p.name != ".gitkeep"]
    if entries:
        raise FileExistsError(f"Output dir not empty: {path} (use --overwrite)")


def find_episode_video(data_root: str, episode_id: int, camera: str):
    episode_str = f"episode_{episode_id:08d}.mp4"
    videos_dir = Path(data_root) / "videos"
    camera_key = CAMERA_KEY_MAP[camera]
    for task_dir in sorted(videos_dir.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task-"):
            continue
        video_path = task_dir / camera_key / episode_str
        if video_path.exists():
            return task_dir.name.replace("task-", ""), str(video_path)
    return None, None


def extract_clip(video_path: str, start_frame: int, end_frame: int, clip_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    clip_start = max(0, int(start_frame))
    clip_end = min(total_frames, int(end_frame))
    if clip_end <= clip_start:
        cap.release()
        raise ValueError(f"Invalid clip range: start={clip_start}, end={clip_end}, total_frames={total_frames}")
    os.makedirs(os.path.dirname(clip_path), exist_ok=True)
    writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
    written = 0
    for _ in range(clip_start, clip_end):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    writer.release()
    cap.release()
    if written <= 1:
        raise ValueError(f"Clip extraction failed or too short: written={written}")
    return {
        "clip_start_frame": clip_start,
        "clip_end_frame": clip_end,
        "written_frames": written,
        "fps": fps,
        "width": width,
        "height": height,
        "total_video_frames": total_frames,
    }


def extract_uniform_frames(video_path: str, start_frame: int, end_frame: int, num_frames: int, output_dir: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    clip_start = max(0, int(start_frame))
    clip_end = min(total_frames, int(end_frame))
    if clip_end <= clip_start:
        cap.release()
        raise ValueError(f"Invalid clip range: start={clip_start}, end={clip_end}, total_frames={total_frames}")
    if num_frames <= 0:
        cap.release()
        raise ValueError(f"num_frames must be positive for uniform sampling, got {num_frames}")
    sample_count = min(int(num_frames), clip_end - clip_start)
    frame_indices = np.linspace(clip_start, clip_end - 1, sample_count, dtype=int).tolist()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for sample_id, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            break
        image_path = output_path / f"{sample_id:04d}_f{frame_idx:05d}.png"
        if not cv2.imwrite(str(image_path), frame):
            cap.release()
            raise ValueError(f"Failed to write sampled frame to {image_path}")
        image_paths.append(str(image_path))
    cap.release()
    if len(image_paths) <= 1:
        raise ValueError(f"Uniform frame extraction failed or too short: written={len(image_paths)}")
    return image_paths, {
        "clip_start_frame": clip_start,
        "clip_end_frame": clip_end,
        "written_frames": clip_end - clip_start,
        "fps": fps,
        "width": width,
        "height": height,
        "total_video_frames": total_frames,
        "sample_count": len(image_paths),
        "sampled_frame_indices": frame_indices[: len(image_paths)],
    }


def extract_stride_frames(video_path: str, start_frame: int, end_frame: int, sample_stride: int, output_dir: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    clip_start = max(0, int(start_frame))
    clip_end = min(total_frames, int(end_frame))
    if clip_end <= clip_start:
        cap.release()
        raise ValueError(f"Invalid clip range: start={clip_start}, end={clip_end}, total_frames={total_frames}")
    if sample_stride <= 0:
        cap.release()
        raise ValueError(f"sample_stride must be positive, got {sample_stride}")
    frame_indices = list(range(clip_start, clip_end, int(sample_stride)))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for sample_id, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            break
        image_path = output_path / f"{sample_id:04d}_f{frame_idx:05d}.png"
        if not cv2.imwrite(str(image_path), frame):
            cap.release()
            raise ValueError(f"Failed to write sampled frame to {image_path}")
        image_paths.append(str(image_path))
    cap.release()
    if len(image_paths) <= 1:
        raise ValueError(f"Stride frame extraction failed or too short: written={len(image_paths)}")
    return image_paths, {
        "clip_start_frame": clip_start,
        "clip_end_frame": clip_end,
        "written_frames": clip_end - clip_start,
        "fps": fps,
        "width": width,
        "height": height,
        "total_video_frames": total_frames,
        "sample_stride": int(sample_stride),
        "sample_count": len(image_paths),
        "sampled_frame_indices": frame_indices[: len(image_paths)],
    }


def resolve_weights(weights: str, model_name: str):
    if weights:
        weight_path = Path(weights)
        if weight_path.exists():
            return str(weight_path.resolve())
        if not weight_path.is_absolute():
            project_relative = PROJECT_ROOT / weight_path
            if project_relative.exists():
                return str(project_relative.resolve())
    return model_name


def existing_path_or_none(path_str: str):
    if not path_str:
        return None
    path = Path(path_str)
    return str(path) if path.exists() else None


def main():
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for Easi3R inference")

    ensure_output_dir(args.output_dir, args.overwrite)
    task_id, video_path = find_episode_video(args.data_root, args.episode_id, args.camera)
    if task_id is None or video_path is None:
        raise ValueError(f"Episode {args.episode_id} with camera={args.camera} not found under {args.data_root}")

    seq_name = args.seq_name or f"task{int(task_id):04d}_ep{args.episode_id:08d}_{args.camera}_f{args.start_frame:05d}_{args.end_frame:05d}"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.use_source_video and args.sampling_mode == "uniform":
        clip_path = output_dir / "_scratch" / "uniform_frames"
        filelist, clip_meta = extract_uniform_frames(
            video_path,
            args.start_frame,
            args.end_frame,
            args.num_frames,
            str(clip_path),
        )
    elif args.use_source_video and args.sampling_mode == "stride":
        clip_path = output_dir / "_scratch" / "stride_frames"
        filelist, clip_meta = extract_stride_frames(
            video_path,
            args.start_frame,
            args.end_frame,
            args.sample_stride,
            str(clip_path),
        )
    elif args.use_source_video:
        clip_path = Path(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        clip_meta = {
            "clip_start_frame": 0,
            "clip_end_frame": total_frames,
            "written_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "total_video_frames": total_frames,
        }
        filelist = [str(clip_path)]
    else:
        clip_path = output_dir / "_scratch" / f"{seq_name}.mp4"
        clip_meta = extract_clip(video_path, args.start_frame, args.end_frame, str(clip_path))
        filelist = [str(clip_path)]

    easi3r_args = SimpleNamespace(
        weights=args.weights,
        output_dir=str(output_dir),
        not_batchify=args.not_batchify,
    )

    weights_path = resolve_weights(args.weights, args.model_name)
    previous_cwd = Path.cwd()
    os.chdir(EASI3R_ROOT)
    try:
        model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)
        model.eval()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        scene, glb_path, vis_video_path, attn_video_path, cluster_video_path = get_reconstructed_scene(
            easi3r_args,
            str(output_dir),
            model,
            args.device,
            args.silent,
            args.image_size,
            filelist,
            "linear",
            args.niter,
            args.min_conf_thr,
            True,
            False,
            True,
            False,
            args.cam_size,
            True,
            args.scenegraph_type,
            args.winsize,
            args.refid,
            seq_name,
            weights_path,
            args.temporal_smoothing_weight,
            args.translation_weight,
            True,
            args.flow_loss_weight,
            args.flow_loss_start_iter,
            args.flow_loss_threshold,
            False,
            args.fps,
            args.num_frames,
            args.sam2_mask_refine,
        )
        torch.cuda.synchronize()
    finally:
        os.chdir(previous_cwd)
    elapsed = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    result_dir = output_dir / seq_name
    artifact_paths = {
        "result_dir": str(result_dir),
        "glb_path": existing_path_or_none(glb_path),
        "visualization_video": existing_path_or_none(vis_video_path),
        "attention_video": existing_path_or_none(attn_video_path),
        "cluster_video": existing_path_or_none(cluster_video_path),
        "pred_traj": existing_path_or_none(str(result_dir / "pred_traj.txt")),
        "pred_intrinsics": existing_path_or_none(str(result_dir / "pred_intrinsics.txt")),
        "dynamic_masks_dir": existing_path_or_none(str(result_dir / "dynamic_masks")),
        "init_fused_dynamic_masks_dir": existing_path_or_none(str(result_dir / "init_fused_dynamic_masks")),
    }

    run_metadata = {
        "model": "Easi3R",
        "task_id": task_id,
        "episode_id": args.episode_id,
        "camera": args.camera,
        "source_video_path": video_path,
        "clip_video_path": str(clip_path),
        "clip": clip_meta,
        "weights_argument": args.weights,
        "weights_resolved": weights_path,
        "model_name": args.model_name,
        "device": args.device,
        "sampling_mode": args.sampling_mode,
        "sample_stride": args.sample_stride,
        "image_size": args.image_size,
        "fps": args.fps,
        "num_frames": args.num_frames,
        "niter": args.niter,
        "scenegraph_type": args.scenegraph_type,
        "winsize": args.winsize,
        "refid": args.refid,
        "sam2_mask_refine": args.sam2_mask_refine,
        "ffmpeg_available": shutil.which("ffmpeg") is not None or Path("/usr/bin/ffmpeg").exists(),
        "elapsed_seconds": elapsed,
        "peak_memory_gb": peak_gb,
        "artifacts": artifact_paths,
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2)

    print(json.dumps(run_metadata, indent=2))


if __name__ == "__main__":
    main()
