from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any


RIGHT_ARROW_KEY = 2555904
LEFT_ARROW_KEY = 2424832
UP_ARROW_KEY = 2490368
DOWN_ARROW_KEY = 2621440


def _parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _ensure_repo_import_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    extra_paths = [
        repo_root / "src",
        repo_root.parent / "BEHAVIOR-1K" / "OmniGibson",
        repo_root.parent / "BEHAVIOR-1K" / "bddl3",
        repo_root.parent / "BEHAVIOR-1K",
    ]
    for path in extra_paths:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
    return repo_root


def _load_runtime() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import cv2
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "缺少运行依赖，请在 openpi-comet 的训练环境中运行该脚本。"
        ) from exc

    try:
        from behavior.learning.datas.dataset import BehaviorLeRobotDataset
        from behavior.learning.datas.dataset import BehaviorLerobotDatasetMetadata
    except ImportError as exc:
        raise RuntimeError(
            "无法导入 BehaviorLeRobotDataset。请确认当前 Python 环境能访问 openpi-comet/src 和 "
            "BEHAVIOR-1K/OmniGibson。"
        ) from exc

    return cv2, np, torch, BehaviorLeRobotDataset, BehaviorLerobotDatasetMetadata


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _as_numpy(value: Any, np: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    return np.asarray(value)


def _format_vector(value: Any, np: Any, max_items: int = 8) -> str:
    array = _as_numpy(value, np)
    if array is None:
        return "None"
    flat = array.reshape(-1)
    preview = ", ".join(f"{float(x):.3f}" for x in flat[:max_items])
    if flat.size > max_items:
        preview += ", ..."
    return f"shape={list(array.shape)} [{preview}]"


def _quality_flags(sample: dict[str, Any], np: Any) -> list[str]:
    flags: list[str] = []
    for key in ("action", "observation.state", "observation.cam_rel_poses"):
        if key in sample:
            array = _as_numpy(sample[key], np)
            if array is not None and np.isnan(array).any():
                flags.append(f"{key}: NaN")
    for key, value in sample.items():
        if not key.startswith("observation.images.rgb."):
            continue
        array = _as_numpy(value, np)
        if array is None or array.size == 0:
            flags.append(f"{key}: empty")
            continue
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.moveaxis(array, 0, -1)
        if float(array.std()) < 1.0:
            flags.append(f"{key}: low-variance")
    for key, value in sample.items():
        if not key.startswith("observation.images.depth."):
            continue
        array = _as_numpy(value, np)
        if array is None or array.size == 0:
            flags.append(f"{key}: empty")
            continue
        if np.allclose(array, 0.0):
            flags.append(f"{key}: all-zero")
    return flags


def _seg_palette(max_items: int, np: Any) -> Any:
    palette = np.zeros((max_items, 3), dtype=np.uint8)
    for idx in range(1, max_items):
        palette[idx] = (
            (37 * idx) % 255,
            (97 * idx) % 255,
            (157 * idx) % 255,
        )
    return palette


def _prepare_rgb_image(array: Any, np: Any) -> Any:
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _prepare_depth_image(array: Any, np: Any, cv2: Any, depth_min: float, depth_max: float) -> tuple[Any, str]:
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    array = array.astype(np.float32)
    normalized = np.clip((array - depth_min) / max(depth_max - depth_min, 1e-6), 0.0, 1.0)
    image = (normalized * 255.0).astype(np.uint8)
    image = cv2.applyColorMap(image, cv2.COLORMAP_VIRIDIS)
    stats = f"min={float(array.min()):.3f}m max={float(array.max()):.3f}m"
    return image, stats


def _prepare_seg_image(array: Any, np: Any) -> tuple[Any, str]:
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    array = array.astype(np.int64)
    unique_ids = np.unique(array)
    remapped = np.zeros_like(array, dtype=np.int32)
    id_to_slot = {int(seg_id): slot for slot, seg_id in enumerate(unique_ids[:4096])}
    for seg_id, slot in id_to_slot.items():
        remapped[array == seg_id] = slot
    palette = _seg_palette(len(unique_ids[:4096]) + 1, np)
    image = palette[remapped % len(palette)]
    stats = f"unique_ids={len(unique_ids)}"
    return image, stats


def _resize_keep_aspect(image: Any, cv2: Any, target_height: int) -> Any:
    height, width = image.shape[:2]
    if height == target_height:
        return image
    scale = target_height / max(height, 1)
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _draw_text_block(
    canvas: Any,
    lines: list[str],
    origin: tuple[int, int],
    cv2: Any,
    *,
    color: tuple[int, int, int] = (235, 235, 235),
    font_scale: float = 0.52,
    line_gap: int = 22,
) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )
        y += line_gap


@dataclass
class ViewerArgs:
    dataset_root: str
    repo_id: str
    task: str | None
    episode: int
    frame: int
    modalities: list[str]
    cameras: list[str]
    fine_grained_level: int
    train_rgb_type: str
    window_name: str
    tile_height: int
    tile_cols: int
    info_panel_width: int
    cache_size: int
    depth_min: float
    depth_max: float
    screenshot_dir: str
    subtask_source: str
    subtask_template_path: str | None
    subtask_object_name_mapping_path: str | None
    subtask_joiner: str
    check_files: bool
    check_timestamp_sync: bool
    list_tasks: bool


class EpisodeViewerState:
    def __init__(
        self,
        args: ViewerArgs,
        np: Any,
        torch: Any,
        dataset_cls: Any,
        metadata_cls: Any,
    ) -> None:
        self.args = args
        self.np = np
        self.torch = torch
        self.dataset_cls = dataset_cls
        self.metadata_cls = metadata_cls
        self.status_message = ""
        self.dataset = None
        self.current_sample: dict[str, Any] | None = None
        self.current_frame_pos = 0
        self.next_frame_pos = 0
        self.cache: dict[int, dict[str, Any]] = {}
        self.task_names: list[str] = []
        self.task_to_episode_count: dict[str, int] = {}
        self.task_idx = 0
        self.episode_idx = 0
        self.global_episode_idx = -1
        self.episode_length = 0

        os.environ.setdefault("OPENPI_LOAD_DATASET_NUM_PROC_CAP", "8")

        self.meta = self.metadata_cls(
            repo_id=self.args.repo_id,
            root=self.args.dataset_root,
            tasks=None,
            modalities={"rgb", "depth", "seg_instance_id"},
            cameras={"head", "left_wrist", "right_wrist"},
        )

        task_order = sorted(self.meta.tasks.keys())
        self.task_names = [self.meta.task_names[task_index] for task_index in task_order]
        self.task_to_episode_count = {
            self.meta.task_names[task_index]: sum(
                1 for ep_index in self.meta.episodes if int(ep_index // 10000) == task_index
            )
            for task_index in task_order
        }
        if not self.task_names:
            raise RuntimeError("没有找到任何 task，无法启动 viewer。")

        if self.args.task is not None:
            if self.args.task not in self.task_names:
                valid = ", ".join(self.task_names[:10])
                raise ValueError(f"未知 task: {self.args.task}。可用 task 示例: {valid}")
            self.task_idx = self.task_names.index(self.args.task)

        self.episode_idx = self._clamp_episode_index(self.task_names[self.task_idx], self.args.episode)

        if "seg_instance_id" in self.args.modalities and not self.torch.cuda.is_available():
            raise RuntimeError("当前环境没有可用 CUDA，无法加载 seg_instance_id 视频。请去掉该模态或切到带 GPU 的环境。")

    def _clamp_episode_index(self, task_name: str, episode_idx: int) -> int:
        episode_count = self.task_to_episode_count[task_name]
        if episode_count <= 0:
            raise RuntimeError(f"task {task_name} 没有可用 episode。")
        return max(0, min(episode_idx, episode_count - 1))

    @property
    def current_task_name(self) -> str:
        return self.task_names[self.task_idx]

    def print_task_summary(self) -> None:
        print("Available tasks:")
        for task_name in self.task_names:
            episode_count = self.task_to_episode_count[task_name]
            print(f"  - {task_name}: {episode_count} episodes")

    def _build_dataset(self) -> None:
        self.dataset = self.dataset_cls(
            repo_id=self.args.repo_id,
            root=self.args.dataset_root,
            tolerance_s=1e-4,
            tasks=[self.current_task_name],
            modalities=self.args.modalities,
            cameras=self.args.cameras,
            local_only=True,
            check_files=self.args.check_files,
            check_timestamp_sync=self.args.check_timestamp_sync,
            delta_timestamps=None,
            episodes=[self.episode_idx],
            chunk_streaming_using_keyframe=True,
            shuffle=False,
            fine_grained_level=self.args.fine_grained_level,
            train_rgb_type=self.args.train_rgb_type,
            return_seg_instance=False,
            subtask_source=self.args.subtask_source,
            subtask_template_path=self.args.subtask_template_path,
            subtask_object_name_mapping_path=self.args.subtask_object_name_mapping_path,
            subtask_joiner=self.args.subtask_joiner,
        )
        self.global_episode_idx = int(self.dataset.episodes[0])
        self.episode_length = int(self.dataset.meta.episodes[self.global_episode_idx]["length"])
        self.cache = {}
        self.current_sample = None
        self.current_frame_pos = 0
        self.next_frame_pos = 0

    def reset_episode(self, *, frame_pos: int = 0, announce: bool = True) -> None:
        self._build_dataset()
        self.seek(frame_pos)
        if announce:
            self.status_message = (
                f"Task={self.current_task_name} episode={self.episode_idx} "
                f"global_episode={self.global_episode_idx}"
            )
            print(self.status_message)

    def _cache_sample(self, frame_pos: int, sample: dict[str, Any]) -> None:
        self.cache[frame_pos] = sample
        while len(self.cache) > self.args.cache_size:
            oldest_key = min(self.cache)
            if oldest_key == self.current_frame_pos and len(self.cache) > 1:
                sorted_keys = sorted(self.cache)
                oldest_key = sorted_keys[1]
            del self.cache[oldest_key]

    def _consume_until(self, target_frame_pos: int) -> None:
        while self.next_frame_pos <= target_frame_pos:
            if self.next_frame_pos in self.cache:
                self.next_frame_pos += 1
                continue
            sample = self.dataset[self.next_frame_pos]
            self._cache_sample(self.next_frame_pos, sample)
            self.next_frame_pos += 1

    def seek(self, target_frame_pos: int) -> dict[str, Any]:
        if self.episode_length <= 0:
            raise RuntimeError("当前 episode 长度为 0。")
        target_frame_pos = max(0, min(target_frame_pos, self.episode_length - 1))
        if target_frame_pos not in self.cache and target_frame_pos < self.next_frame_pos:
            self.status_message = f"回退到 frame {target_frame_pos}，重新从头顺序定位。"
            self._build_dataset()
        self._consume_until(target_frame_pos)
        self.current_frame_pos = target_frame_pos
        self.current_sample = self.cache[target_frame_pos]
        return self.current_sample

    def step_frame(self, delta: int) -> dict[str, Any]:
        return self.seek(self.current_frame_pos + delta)

    def step_episode(self, delta: int) -> dict[str, Any]:
        self.episode_idx = self._clamp_episode_index(self.current_task_name, self.episode_idx + delta)
        self.reset_episode(frame_pos=0)
        return self.current_sample

    def step_task(self, delta: int) -> dict[str, Any]:
        self.task_idx = (self.task_idx + delta) % len(self.task_names)
        self.episode_idx = self._clamp_episode_index(self.current_task_name, self.episode_idx)
        self.reset_episode(frame_pos=0)
        return self.current_sample


def _collect_visual_tiles(
    sample: dict[str, Any],
    np: Any,
    cv2: Any,
    *,
    modalities: list[str],
    cameras: list[str],
    tile_height: int,
    depth_min: float,
    depth_max: float,
) -> list[tuple[str, Any, str]]:
    tiles: list[tuple[str, Any, str]] = []
    for camera in cameras:
        for modality in modalities:
            key = f"observation.images.{modality}.{camera}"
            if key not in sample:
                continue
            array = _as_numpy(sample[key], np)
            if array is None:
                continue
            stats = ""
            if modality == "rgb":
                image = _prepare_rgb_image(array, np)
            elif modality == "depth":
                image, stats = _prepare_depth_image(array, np, cv2, depth_min, depth_max)
            elif modality == "seg_instance_id":
                image, stats = _prepare_seg_image(array, np)
            else:
                continue
            image = _resize_keep_aspect(image, cv2, tile_height)
            tiles.append((key, image, stats))
    return tiles


def _compose_grid(
    tiles: list[tuple[str, Any, str]],
    np: Any,
    cv2: Any,
    *,
    tile_cols: int,
    tile_height: int,
) -> Any:
    if not tiles:
        return np.zeros((tile_height, tile_height, 3), dtype=np.uint8)

    max_width = max(image.shape[1] for _, image, _ in tiles)
    rows: list[Any] = []
    for row_start in range(0, len(tiles), tile_cols):
        padded_tiles: list[Any] = []
        for key, image, stats in tiles[row_start : row_start + tile_cols]:
            canvas = np.zeros((tile_height + 38, max_width, 3), dtype=np.uint8)
            x_offset = (max_width - image.shape[1]) // 2
            canvas[38 : 38 + image.shape[0], x_offset : x_offset + image.shape[1]] = image
            title = key.replace("observation.images.", "")
            cv2.putText(
                canvas,
                title,
                (10, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if stats:
                cv2.putText(
                    canvas,
                    stats,
                    (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (180, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
            padded_tiles.append(canvas)
        while len(padded_tiles) < tile_cols:
            padded_tiles.append(np.zeros_like(padded_tiles[0]))
        rows.append(np.hstack(padded_tiles))
    return np.vstack(rows)


def _build_info_panel(state: EpisodeViewerState, sample: dict[str, Any], np: Any, cv2: Any) -> Any:
    panel = np.zeros((720, state.args.info_panel_width, 3), dtype=np.uint8)
    task_text = str(sample.get("task", state.current_task_name))
    subtask_text = str(sample.get("subtask_text", ""))
    timestamp = float(_scalar(sample.get("timestamp", 0.0)))
    frame_from_timestamp = int(round(timestamp * 30.0))
    lines = [
        "Behavior-1K Viewer",
        "",
        f"task_name: {state.current_task_name}",
        f"task_idx: {state.task_idx + 1}/{len(state.task_names)}",
        f"episode(local): {state.episode_idx}/{state.task_to_episode_count[state.current_task_name] - 1}",
        f"episode(global): {state.global_episode_idx}",
        f"frame(view): {state.current_frame_pos}/{state.episode_length - 1}",
        f"frame(ts*fps): {frame_from_timestamp}",
        f"timestamp: {timestamp:.3f}s",
        "",
        f"task: {task_text}",
    ]
    if subtask_text and subtask_text != "None":
        lines.append(f"subtask: {subtask_text}")
    lines.extend(
        [
            "",
            f"action: {_format_vector(sample.get('action'), np, max_items=6)}",
            f"state: {_format_vector(sample.get('observation.state'), np, max_items=6)}",
            f"cam_rel_poses: {_format_vector(sample.get('observation.cam_rel_poses'), np, max_items=6)}",
            "",
        ]
    )
    flags = _quality_flags(sample, np)
    if flags:
        lines.append("quality flags:")
        lines.extend(f"  - {flag}" for flag in flags[:12])
    else:
        lines.append("quality flags: none on current frame")
    lines.extend(
        [
            "",
            "keys:",
            "  space / d / -> : next frame",
            "  a / <-        : prev frame",
            "  w / s         : +/-10 frames",
            "  . / ,         : next/prev episode",
            "  ] / [         : next/prev task",
            "  r             : reload current episode",
            "  p             : save screenshot",
            "  q / esc       : quit",
            "",
            "note: parquet 里的 joint efforts 已知错误，",
            "      不要把它当作训练信号质量判断。",
        ]
    )

    _draw_text_block(panel, lines, (16, 26), cv2, font_scale=0.48, line_gap=21)
    if state.status_message:
        cv2.putText(
            panel,
            state.status_message[: max(10, state.args.info_panel_width // 8)],
            (16, panel.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (120, 220, 255),
            1,
            cv2.LINE_AA,
        )
    return panel


def _compose_screen(state: EpisodeViewerState, sample: dict[str, Any], np: Any, cv2: Any) -> Any:
    tiles = _collect_visual_tiles(
        sample,
        np,
        cv2,
        modalities=state.args.modalities,
        cameras=state.args.cameras,
        tile_height=state.args.tile_height,
        depth_min=state.args.depth_min,
        depth_max=state.args.depth_max,
    )
    grid = _compose_grid(
        tiles,
        np,
        cv2,
        tile_cols=state.args.tile_cols,
        tile_height=state.args.tile_height,
    )
    info_panel = _build_info_panel(state, sample, np, cv2)
    if info_panel.shape[0] < grid.shape[0]:
        pad = grid.shape[0] - info_panel.shape[0]
        info_panel = cv2.copyMakeBorder(info_panel, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    elif info_panel.shape[0] > grid.shape[0]:
        pad = info_panel.shape[0] - grid.shape[0]
        grid = cv2.copyMakeBorder(grid, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return np.hstack([grid, info_panel])


def _save_screenshot(state: EpisodeViewerState, screen: Any, cv2: Any) -> str:
    output_dir = Path(state.args.screenshot_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"{state.current_task_name}_ep{state.episode_idx:04d}_frame{state.current_frame_pos:06d}.png"
    )
    cv2.imwrite(str(output_path), screen)
    state.status_message = f"saved screenshot: {output_path}"
    print(state.status_message)
    return str(output_path)


def _handle_key(state: EpisodeViewerState, key: int, screen: Any, cv2: Any) -> bool:
    if key in (27, ord("q")):
        return False
    if key in (ord(" "), ord("d"), RIGHT_ARROW_KEY):
        state.step_frame(1)
    elif key in (ord("a"), LEFT_ARROW_KEY):
        state.step_frame(-1)
    elif key in (ord("w"), UP_ARROW_KEY):
        state.step_frame(10)
    elif key in (ord("s"), DOWN_ARROW_KEY):
        state.step_frame(-10)
    elif key == ord("."):
        state.step_episode(1)
    elif key == ord(","):
        state.step_episode(-1)
    elif key == ord("]"):
        state.step_task(1)
    elif key == ord("["):
        state.step_task(-1)
    elif key == ord("r"):
        state.reset_episode(frame_pos=state.current_frame_pos)
    elif key == ord("p"):
        _save_screenshot(state, screen, cv2)
    return True


def run_viewer(args: ViewerArgs) -> None:
    _ensure_repo_import_paths()
    cv2, np, torch, dataset_cls, metadata_cls = _load_runtime()

    state = EpisodeViewerState(args, np, torch, dataset_cls, metadata_cls)
    if args.list_tasks:
        state.print_task_summary()
        return

    state.reset_episode(frame_pos=args.frame, announce=True)
    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    while True:
        assert state.current_sample is not None
        screen = _compose_screen(state, state.current_sample, np, cv2)
        height, width = screen.shape[:2]
        max_height = 1400
        display = screen
        if height > max_height:
            scale = max_height / height
            display = cv2.resize(screen, (int(width * scale), max_height), interpolation=cv2.INTER_AREA)
        cv2.imshow(args.window_name, display)
        key = cv2.waitKeyEx(0)
        if not _handle_key(state, key, screen, cv2):
            break

    cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "基于 BehaviorLeRobotDataset 的本地 GUI viewer。"
            "默认按 task 内 episode 顺序加载，适合肉眼检查图像、depth、seg 和动作/状态摘要。"
        )
    )
    parser.add_argument("--dataset-root", required=True, help="本地 Behavior-1K LeRobot 数据根目录。")
    parser.add_argument("--repo-id", default="behavior-1k/2025-challenge-demos")
    parser.add_argument("--task", default=None, help="初始 task 名称，不传则默认第一个 task。")
    parser.add_argument("--episode", type=int, default=0, help="task 内的 episode 局部序号。")
    parser.add_argument("--frame", type=int, default=0, help="打开后定位到的帧序号。")
    parser.add_argument(
        "--modalities",
        default="rgb,depth",
        help="要展示的模态，逗号分隔。可选: rgb,depth,seg_instance_id",
    )
    parser.add_argument(
        "--cameras",
        default="head,left_wrist,right_wrist",
        help="要展示的相机，逗号分隔。",
    )
    parser.add_argument("--fine-grained-level", type=int, default=0)
    parser.add_argument("--train-rgb-type", default="regular")
    parser.add_argument("--window-name", default="Behavior-1K Dataset Viewer")
    parser.add_argument("--tile-height", type=int, default=320)
    parser.add_argument("--tile-cols", type=int, default=3)
    parser.add_argument("--info-panel-width", type=int, default=560)
    parser.add_argument("--cache-size", type=int, default=180)
    parser.add_argument("--depth-min", type=float, default=0.0)
    parser.add_argument("--depth-max", type=float, default=10.0)
    parser.add_argument("--screenshot-dir", default="data_process/viewer_captures")
    parser.add_argument(
        "--subtask-source",
        default="orchestrator",
        choices=["orchestrator", "annotations_primitive", "annotations_skill"],
    )
    parser.add_argument("--subtask-template-path", default=None)
    parser.add_argument("--subtask-object-name-mapping-path", default=None)
    parser.add_argument("--subtask-joiner", default=" then ")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--check-timestamp-sync", action="store_true")
    parser.add_argument("--list-tasks", action="store_true", help="打印 task 列表后退出。")
    return parser


def parse_args() -> ViewerArgs:
    parser = build_arg_parser()
    namespace = parser.parse_args()
    modalities = _parse_csv_list(namespace.modalities) or ["rgb", "depth"]
    cameras = _parse_csv_list(namespace.cameras) or ["head", "left_wrist", "right_wrist"]
    valid_modalities = {"rgb", "depth", "seg_instance_id"}
    invalid_modalities = [item for item in modalities if item not in valid_modalities]
    if invalid_modalities:
        parser.error(f"不支持的 modalities: {invalid_modalities}")

    valid_cameras = {"head", "left_wrist", "right_wrist"}
    invalid_cameras = [item for item in cameras if item not in valid_cameras]
    if invalid_cameras:
        parser.error(f"不支持的 cameras: {invalid_cameras}")

    if namespace.tile_cols <= 0:
        parser.error("--tile-cols 必须大于 0")
    if namespace.cache_size <= 0:
        parser.error("--cache-size 必须大于 0")
    if namespace.depth_max <= namespace.depth_min:
        parser.error("--depth-max 必须大于 --depth-min")

    return ViewerArgs(
        dataset_root=namespace.dataset_root,
        repo_id=namespace.repo_id,
        task=namespace.task,
        episode=namespace.episode,
        frame=namespace.frame,
        modalities=modalities,
        cameras=cameras,
        fine_grained_level=namespace.fine_grained_level,
        train_rgb_type=namespace.train_rgb_type,
        window_name=namespace.window_name,
        tile_height=namespace.tile_height,
        tile_cols=namespace.tile_cols,
        info_panel_width=namespace.info_panel_width,
        cache_size=namespace.cache_size,
        depth_min=namespace.depth_min,
        depth_max=namespace.depth_max,
        screenshot_dir=namespace.screenshot_dir,
        subtask_source=namespace.subtask_source,
        subtask_template_path=namespace.subtask_template_path,
        subtask_object_name_mapping_path=namespace.subtask_object_name_mapping_path,
        subtask_joiner=namespace.subtask_joiner,
        check_files=namespace.check_files,
        check_timestamp_sync=namespace.check_timestamp_sync,
        list_tasks=namespace.list_tasks,
    )


def main() -> None:
    args = parse_args()
    run_viewer(args)


if __name__ == "__main__":
    main()
