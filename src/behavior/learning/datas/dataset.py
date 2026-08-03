import bisect
from collections import defaultdict
from collections.abc import Callable, Iterable
import json
import os
from pathlib import Path
import random
import re
import time

import datasets
from datasets import load_dataset
from huggingface_hub import snapshot_download
from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import EPISODES_PATH
from lerobot.datasets.utils import EPISODES_STATS_PATH
from lerobot.datasets.utils import STATS_PATH
from lerobot.datasets.utils import TASKS_PATH
from lerobot.datasets.utils import backward_compatible_episodes_stats
from lerobot.datasets.utils import cast_stats_to_numpy
from lerobot.datasets.utils import check_delta_timestamps
from lerobot.datasets.utils import check_timestamps_sync
from lerobot.datasets.utils import check_version_compatibility
from lerobot.datasets.utils import get_delta_indices
from lerobot.datasets.utils import get_episode_data_index
from lerobot.datasets.utils import get_safe_version
from lerobot.datasets.utils import is_valid_version
from lerobot.datasets.utils import load_info
from lerobot.datasets.utils import load_json
from lerobot.datasets.utils import load_jsonlines
from lerobot.datasets.video_utils import get_safe_default_codec
import numpy as np
from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES
from omnigibson.learning.utils.lerobot_utils import aggregate_stats
from omnigibson.learning.utils.lerobot_utils import decode_video_frames
from omnigibson.learning.utils.lerobot_utils import hf_transform_to_torch
from omnigibson.learning.utils.obs_utils import OBS_LOADER_MAP
from omnigibson.learning.utils.obs_utils import instance_id_to_instance
from omnigibson.utils.ui_utils import create_module_logger
import packaging.version
import torch as th
from torch.utils.data import Dataset
from torch.utils.data import get_worker_info

from behavior.learning.datas.dataset_utils import SubtaskPhraseConverter, _duration_to_segments
from behavior.learning.datas.hf_cache_sync import DistributedCacheError
from behavior.learning.datas.hf_cache_sync import HfCacheSyncSettings
from behavior.learning.datas.hf_cache_sync import build_prepared_cache_manifest
from behavior.learning.datas.hf_cache_sync import coordinate_cache_attempt
from behavior.learning.datas.hf_cache_sync import coordinate_global_cache_setup
from behavior.learning.datas.hf_cache_sync import load_with_local_cache_sync
from behavior.learning.datas.hf_cache_sync import make_cache_request_id
from behavior.learning.datas.hf_cache_sync import make_cache_selection_id
from behavior.learning.datas.hf_cache_sync import next_cache_invocation_index
from behavior.learning.datas.hf_cache_sync import observe_global_cache_failure
from behavior.learning.datas.hf_cache_sync import prepared_arrow_paths
from behavior.learning.datas.hf_cache_sync import publish_global_cache_failure
from behavior.learning.datas.hf_cache_sync import resolve_cache_run_id
from behavior.learning.datas.hf_cache_sync import setup_node_cache_paths
from behavior.learning.datas.hf_cache_sync import snapshot_cache_tree
from behavior.learning.datas.hf_cache_sync import wait_for_global_cache_readiness

ANNOTATIONS_PATH = "annotations"
ORCHESTRATORS_PATH = "orchestrators"
logger = create_module_logger("BehaviorLeRobotDataset")

_B1K_ANCHOR_STRIDE_ENV = "OPENPI_B1K_ANCHOR_STRIDE"
_B1K_ANCHOR_OFFSET_ENV = "OPENPI_B1K_ANCHOR_OFFSET"
_B1K_DROP_INCOMPLETE_HORIZON_ENV = "OPENPI_B1K_DROP_INCOMPLETE_HORIZON"


def _read_streaming_anchor_env() -> tuple[int, int, bool]:
    """Read one immutable chunk-streaming anchor contract for this dataset instance."""

    raw_stride = os.environ.get(_B1K_ANCHOR_STRIDE_ENV, "1")
    raw_offset = os.environ.get(_B1K_ANCHOR_OFFSET_ENV, "0")
    raw_drop = os.environ.get(_B1K_DROP_INCOMPLETE_HORIZON_ENV, "0")
    try:
        stride = int(raw_stride)
    except ValueError as exc:
        raise ValueError(f"{_B1K_ANCHOR_STRIDE_ENV} must be an integer, got {raw_stride!r}") from exc
    try:
        offset = int(raw_offset)
    except ValueError as exc:
        raise ValueError(f"{_B1K_ANCHOR_OFFSET_ENV} must be an integer, got {raw_offset!r}") from exc
    if stride < 1:
        raise ValueError(f"{_B1K_ANCHOR_STRIDE_ENV} must be >= 1, got {stride}")
    if not 0 <= offset < stride:
        raise ValueError(
            f"{_B1K_ANCHOR_OFFSET_ENV} must satisfy 0 <= offset < stride; got offset={offset}, stride={stride}"
        )
    if raw_drop not in {"0", "1"}:
        raise ValueError(f"{_B1K_DROP_INCOMPLETE_HORIZON_ENV} must be 0 or 1, got {raw_drop!r}")
    return stride, offset, raw_drop == "1"


def _aligned_streaming_chunk_start(chunk: tuple[int, int, int], *, stride: int, offset: int) -> int | None:
    """Return the first episode-local aligned global cursor in ``chunk``."""

    global_start, global_end, episode_local_start = chunk
    delta = (offset - episode_local_start) % stride
    cursor = global_start + delta
    return cursor if cursor < global_end else None


class BehaviorLeRobotDataset(LeRobotDataset):
    """
    BehaviorLeRobotDataset is a customized dataset class for loading and managing LeRobot datasets,
    with additional filtering and loading options tailored for the BEHAVIOR-1K benchmark.
    This class extends LeRobotDataset and introduces the following customizations:
        - Task-based filtering: Load only episodes corresponding to specific tasks.
        - Modality and camera selection: Load only specified modalities (e.g., "rgb", "depth", "seg_instance_id")
          and cameras (e.g., "left_wrist", "right_wrist", "head").
        - Ability to download and use additional annotation and metainfo files.
        - Local-only mode: Optionally restrict dataset usage to local files, disabling downloads.
        - Optional batch streaming using keyframe for faster access.
    These customizations allow for more efficient and targeted dataset usage in the context of B1K tasks
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = "pyav",
        batch_encoding_size: int = 1,
        # === Customized arguments for BehaviorLeRobotDataset ===
        tasks: Iterable[str] = None,
        modalities: Iterable[str] = None,
        cameras: Iterable[str] = None,
        local_only: bool = False,
        check_files: bool = True,
        check_timestamp_sync: bool = True,
        chunk_streaming_using_keyframe: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        fine_grained_level: int = 0,  # 0, 1, 2, 3
        train_rgb_type: str = "regular",  # regular | bbox | point
        return_seg_instance: bool = False,
        skill_list: list[str] = ["all"],
        resample_group_by: str | None = None,  # None | task_skill | skill_type | skill_description
        resample_weights: dict[str, float] | None = None,
        resample_default_weight: float = 1.0,
        subtask_source: str = "orchestrator",  # orchestrator | annotations_primitive | annotations_skill
        subtask_template_path: str | Path | None = None,
        subtask_object_name_mapping_path: str | Path | None = None,
        subtask_joiner: str = " then ",
    ):
        """
        Custom args:
            episodes (List[int]): list of episodes to use PER TASK.
                NOTE: This is different from the actual episode indices in the dataset.
                Rather, this is meant to be used for train/val split, or loading a specific amount of partial data.
                If set to None, all episodes will be loaded for a given task.
            tasks (List[str]): list of task names to load. If None, all tasks will be loaded.
            modalities (List[str]): list of modality names to load. If None, all modalities will be loaded.
                must be a subset of ["rgb", "depth", "seg_instance_id"]
            cameras (List[str]): list of camera names to load. If None, all cameras will be loaded.
                must be a subset of ["left_wrist", "right_wrist", "head"]
            local_only (bool): whether to only use local data (not download from HuggingFace).
                NOTE: set this to False and force_cache_sync to True if you want to force re-syncing the local cache with the remote dataset.
                For more details, please refer to the `force_cache_sync` argument in the base class.
            check_timestamp_sync (bool): whether to check timestamp synchronization between different modalities and the state/action data.
                While it is set to True in the original LeRobotDataset and is set to True here by default, it can be set to False to skip the check for faster loading.
                This will especially save time if you are loading the complete challenge demo dataset.
            chunk_streaming_using_keyframe (bool): whether to use chunk streaming mode for loading the dataset using keyframes.
                When this is enabled, the dataset will pseudo-randomly load data in chunks based on keyframes, allowing for faster access to the data.
                NOTE: As B1K challenge demos has GOP size of 250 frames for efficient storage, it is STRONGLY recommended to set this to True if you don't need true frame-level random access.
                When this is enabled, it is recommended to set shuffle to True for better randomness in chunk selection.
                We also enforce that segmentation instance ID videos can only be loaded in chunk_streaming_using_keyframe mode for faster access.
            shuffle (bool): whether to shuffle the chunks after loading. This ONLY applies in chunk streaming mode. Recommended to be set to True for better randomness in chunk selection.
            seed (int): random seed for shuffling chunks.
            fine_grained_level (int): fine-grained level of orchestrators to use for training.
            train_rgb_type (str): type of rgb to use for training.
            return_seg_instance (bool): whether to return seg instance.
            skill_list (list[str]): ["all", "move_to:0.5"] etc.
        """
        Dataset.__init__(self)
        self.repo_id = repo_id
        self.root = Path(os.path.expanduser(str(root))) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.revision = revision or CODEBASE_VERSION
        self.video_backend = video_backend or get_safe_default_codec()
        self.delta_indices = None
        self.batch_encoding_size = batch_encoding_size
        self.episodes_since_last_encoding = 0
        self.return_seg_instance = return_seg_instance
        self.train_rgb_type = train_rgb_type
        self.skill_list = skill_list
        self.resample_group_by = None if resample_group_by is None else str(resample_group_by)
        self.resample_weights = (
            None if resample_weights is None else {str(k): float(v) for k, v in dict(resample_weights).items()}
        )
        self.resample_default_weight = float(resample_default_weight)
        if self.resample_weights is None:
            self._resample_weight_norm = 1.0
        else:
            self._resample_weight_norm = max(
                1.0,
                float(max([self.resample_default_weight, *list(self.resample_weights.values())])),
            )
        self.subtask_source = subtask_source
        self.subtask_template_path = Path(subtask_template_path) if subtask_template_path is not None else None
        self.subtask_object_name_mapping_path = (
            Path(subtask_object_name_mapping_path) if subtask_object_name_mapping_path is not None else None
        )
        self.subtask_joiner = subtask_joiner
        self._subtask_templates = None
        self._subtask_object_name_mapping = None
        self._subtask_segments = {}
        self._subtask_segment_ends = {}
        self._resample_skill_segments = {}
        self._resample_skill_segment_ends = {}
        self._accept_rng = None

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        # ========== Customizations ==========
        self.seed = seed
        (
            self._streaming_anchor_stride,
            self._streaming_anchor_offset,
            self._streaming_drop_incomplete_horizon,
        ) = _read_streaming_anchor_env()
        if modalities is None:
            modalities = ["rgb", "depth", "seg_instance_id"]
        if "seg_instance_id" in modalities:
            assert chunk_streaming_using_keyframe, "For the sake of data loading speed, please use chunk_streaming_using_keyframe=True when loading segmentation instance ID videos."
        if "depth" in modalities:
            assert self.video_backend == "pyav", (
                "Depth videos can only be decoded with the 'pyav' backend. "
                "Please set video_backend='pyav' when initializing the dataset."
            )
        if cameras is None:
            cameras = ["head", "left_wrist", "right_wrist"]
        self.task_names = set(tasks) if tasks is not None else set(TASK_NAMES_TO_INDICES.keys())
        self.task_indices = [TASK_NAMES_TO_INDICES[task] for task in self.task_names]
        # Load metadata
        self.meta = BehaviorLerobotDatasetMetadata(
            repo_id=self.repo_id,
            root=self.root,
            revision=self.revision,
            force_cache_sync=force_cache_sync,
            tasks=self.task_names,
            modalities=modalities,
            cameras=cameras,
        )
        # overwrite episode based on task
        all_episodes = load_jsonlines(self.root / EPISODES_PATH)
        # get the episodes grouped by task
        epi_by_task = defaultdict(list)
        for item in all_episodes:
            if item["episode_index"] // 1e4 in self.meta.tasks:
                epi_by_task[item["episode_index"] // 1e4].append(item["episode_index"])
        # sort and cherrypick episodes within each task
        for task_id, ep_indices in epi_by_task.items():
            epi_by_task[task_id] = sorted(ep_indices)
            if episodes is not None:
                epi_by_task[task_id] = [epi_by_task[task_id][i] for i in episodes if i < len(epi_by_task[task_id])]
        # now put episodes back together
        self.episodes = sorted([ep for eps in epi_by_task.values() for ep in eps])
        # handle streaming mode and shuffling of episodes
        self._chunk_streaming_using_keyframe = chunk_streaming_using_keyframe
        if self._chunk_streaming_using_keyframe:
            if not shuffle:
                logger.warning(
                    "chunk_streaming_using_keyframe mode is enabled but shuffle is set to False. This may lead to less randomness in chunk selection."
                )
            self.chunks = self._get_keyframe_chunk_indices()
            # Now, we randomly permute the episodes if shuffle is True
            if shuffle:
                self.current_streaming_chunk_idx = None
                self.current_streaming_frame_idx = None
                self._active_chunks = None
            else:
                self._active_chunks = self.chunks
                self.current_streaming_chunk_idx = 0
                self.current_streaming_frame_idx = None
                self._select_aligned_streaming_chunk(start_at_current=True)
            self.obs_loaders = dict()
            self._should_obs_loaders_reload = True
        # record the positional index of each episode index within self.episodes
        self.episode_data_index_pos = {ep_idx: i for i, ep_idx in enumerate(self.episodes)}
        logger.info(f"Total episodes: {len(self.episodes)}")
        # ====================================

        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)

        # Load actual data
        try:
            if force_cache_sync:
                raise FileNotFoundError
            if check_files:
                for fpath in self.get_episodes_file_paths():
                    assert (self.root / fpath).is_file(), f"Missing file: {self.root / fpath}"
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError) as e:
            if local_only:
                raise e
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        # Check timestamps
        if check_timestamp_sync:
            timestamps = th.stack(self.hf_dataset["timestamp"]).numpy()
            episode_indices = th.stack(self.hf_dataset["episode_index"]).numpy()
            ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
            check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

        # Setup delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        self.prepare_task(fine_grained_level)

        self.omnigibson_mapping = {ep_idx: defaultdict(dict) for ep_idx in self.episodes}
        self._init_subtask_assets()

    def _init_subtask_assets(self):
        self._subtask_phrase_converter = None
        if self.subtask_source == "orchestrator":
            return
        if self.subtask_template_path is None:
            raise ValueError("subtask_template_path is required when subtask_source is not orchestrator")
        if self.subtask_object_name_mapping_path is None:
            raise ValueError("subtask_object_name_mapping_path is required when subtask_source is not orchestrator")
        with open(self.subtask_template_path, "r", encoding="utf-8") as f:
            self._subtask_templates = json.load(f)
        with open(self.subtask_object_name_mapping_path, "r", encoding="utf-8") as f:
            self._subtask_object_name_mapping = json.load(f)
        self._subtask_phrase_converter = SubtaskPhraseConverter(
            subtask_source=self.subtask_source,
            subtask_template_path=None,
            subtask_object_name_mapping_path=None,
            subtask_templates=self._subtask_templates,
            object_name_mapping=self._subtask_object_name_mapping,
            subtask_joiner=self.subtask_joiner,
        )

    def _subtask_obj_name(self, raw_id: str | None) -> str | None:
        if raw_id is None:
            return None
        s = str(raw_id).strip()
        if not s:
            return None
        if self._subtask_object_name_mapping is not None and s in self._subtask_object_name_mapping:
            v = self._subtask_object_name_mapping.get(s)
            if v is None:
                return None
            if isinstance(v, str) and v.strip():
                return v.strip()
        return self._canonicalize_object_id_fallback(s)

    def _canonicalize_object_id_fallback(self, obj_id: str) -> str | None:
        s = str(obj_id).strip()
        if not s:
            return None
        if s.lower() in {"left", "right"}:
            return None
        if s.startswith("[") and s.endswith("]"):
            return None
        s = s.replace("-", "_")
        parts = [p for p in s.split("_") if p]
        if not parts:
            return None
        while parts and re.fullmatch(r"\d+", parts[-1]):
            parts.pop()
        phrase = " ".join(parts).strip().lower()
        phrase = re.sub(r"\s+", " ", phrase)
        return phrase if phrase else None

    def _subtask_flatten(self, x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            out = []
            for y in x:
                out.extend(self._subtask_flatten(y))
            return out
        return [x]

    def _subtask_first_text(self, x) -> str | None:
        for y in self._subtask_flatten(x):
            if y is None:
                continue
            s = str(y).strip()
            if s:
                return s
        return None

    def _duration_to_segments(self, dur):
        if isinstance(dur, list) and len(dur) == 2 and all(isinstance(z, (int, np.integer)) for z in dur):
            return [(int(dur[0]), int(dur[1]))]
        if isinstance(dur, list) and dur and all(isinstance(z, list) and len(z) == 2 for z in dur):
            out = []
            for z in dur:
                if all(isinstance(t, (int, np.integer)) for t in z):
                    out.append((int(z[0]), int(z[1])))
            return out
        ints = [int(z) for z in self._subtask_flatten(dur) if isinstance(z, (int, np.integer))]
        if len(ints) >= 2:
            return [(min(ints), max(ints))]
        return []

    def _extract_main_target(self, object_id_val):
        if isinstance(object_id_val, list) and object_id_val and isinstance(object_id_val[0], list):
            g = object_id_val[0]
            if isinstance(g, list) and len(g) >= 2:
                if len(g) >= 3:
                    return self._subtask_first_text(g[0]), self._subtask_first_text(g[-1])
                return self._subtask_first_text(g[0]), self._subtask_first_text(g[1])
            if isinstance(g, list) and len(g) == 1:
                return self._subtask_first_text(g[0]), None
        flat = [self._subtask_first_text(object_id_val)] if object_id_val is not None else []
        flat = [x for x in flat if x]
        if not flat:
            return None, None
        if len(flat) == 1:
            return flat[0], None
        return flat[0], flat[-1]

    def _apply_template(self, template: dict, **kwargs) -> str:
        s = template.get("template", "")
        if not s:
            return ""
        for k, v in kwargs.items():
            s = s.replace("{" + k + "}", v or "")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _phrase_for_action(self, action: str, obj: str | None, src: str | None, dst: str | None, target: str | None):
        if self._subtask_templates is None:
            return ""
        tpl = self._subtask_templates.get("skill", {}).get(action) or self._subtask_templates.get("primitive", {}).get(
            action
        )
        if tpl is None:
            return ""
        return self._apply_template(tpl, obj=obj, src=src, dst=dst, target=target, verb=action.lower())

    def _phrase_from_skill_ann(self, ann: dict) -> str | None:
        if self._subtask_phrase_converter is None:
            return None
        return self._subtask_phrase_converter.phrase_from_skill_ann(ann)

    def _phrase_from_primitive_ann(self, ann: dict) -> str | None:
        if self._subtask_phrase_converter is None:
            return None
        return self._subtask_phrase_converter.phrase_from_primitive_ann(ann)

    def _build_subtask_segments_for_episode(self, ep_idx: int):
        ann = self.meta.annotations.get(ep_idx)
        if not isinstance(ann, dict) or self._subtask_phrase_converter is None:
            return [], []
        return self._subtask_phrase_converter.build_subtask_segments_for_episode(ann)

    def _get_subtask_text(self, item: dict) -> str | None:
        if self.subtask_source == "orchestrator":
            return self._get_task_at_level(item, 1)
        ep_idx = item["episode_index"].item()
        frame_index = round(item["timestamp"].item() * self.fps)
        if ep_idx not in self._subtask_segments:
            segs, ends = self._build_subtask_segments_for_episode(ep_idx)
            self._subtask_segments[ep_idx] = segs
            self._subtask_segment_ends[ep_idx] = ends
        segs = self._subtask_segments[ep_idx]
        ends = self._subtask_segment_ends[ep_idx]
        if not segs:
            return self._get_task_at_level(item, 1)
        i = bisect.bisect_left(ends, frame_index)
        if 0 <= i < len(segs):
            s, e, t = segs[i]
            if s <= frame_index <= e:
                return t
        return self._get_task_at_level(item, 1)

    def prepare_task(self, fine_grained_level: int):
        """set train subtask mode for lerobot dataset"""
        self.fine_grained_level = fine_grained_level

        # calculate the start and end indices of each episode
        self.task_sizes = {}
        try:
            for ep_id, ep_orch in self.meta.orchestrators.items():
                self.task_sizes[ep_id] = [task_info["end_frame"] for task_info in ep_orch[fine_grained_level]]
        except Exception as e:
            print(f"[warn] {self.repo_id} failed to calculate episode subtask cumulate: {e}")

        print(f"prepare task with fine_grained_level {self.fine_grained_level} for {self.root}")

    def get_episodes_file_paths(self) -> list[str]:
        """
        Overwrite the original method to use the episodes indices instead of range(self.meta.total_episodes)
        """
        episodes = self.episodes if self.episodes is not None else list(self.meta.episodes.keys())
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        # append metainfo and language annotations
        fpaths += [str(self.meta.get_metainfo_path(ep_idx)) for ep_idx in episodes]
        # TODO: add this back once we have all the language annotations
        # fpaths += [str(self.meta.get_annotation_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files

        return fpaths

    def download_episodes(self, download_videos: bool = True) -> None:
        """
        Overwrite base method to allow more flexible pattern matching.
        Here, we do coarse filtering based on tasks, cameras, and modalities.
        We do this instead of filename patterns to speed up pattern checking and download speed.
        """
        allow_patterns = []
        if set(self.task_indices) != set(TASK_NAMES_TO_INDICES.values()):
            for task in self.task_indices:
                allow_patterns.append(f"**/task-{task:04d}/**")
        if len(self.meta.modalities) != 3:
            for modality in self.meta.modalities:
                if len(self.meta.camera_names) != 3:
                    for camera in self.meta.camera_names:
                        allow_patterns.append(f"**/observation.images.{modality}.{camera}/**")
                else:
                    allow_patterns.append(f"**/observation.images.{modality}.*/**")
        elif len(self.meta.camera_names) != 3:
            for camera in self.meta.camera_names:
                allow_patterns.append(f"**/observation.images.*.{camera}/**")
        ignore_patterns = []
        if not download_videos:
            ignore_patterns.append("videos/")
        if set(self.task_indices) != set(TASK_NAMES_TO_INDICES.values()):
            for task in set(TASK_NAMES_TO_INDICES.values()).difference(self.task_indices):
                ignore_patterns.append(f"**/task-{task:04d}/**")

        allow_patterns = None if allow_patterns == [] else allow_patterns
        ignore_patterns = None if ignore_patterns == [] else ignore_patterns
        self.pull_from_repo(allow_patterns=allow_patterns, ignore_patterns=ignore_patterns)

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        """
        Overwrite base class to increase max workers to num of CPUs - 2
        """
        logger.info(f"Pulling dataset {self.repo_id} from HuggingFace hub...")
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            max_workers=os.cpu_count() - 2,
            local_files_only=os.environ.get("HF_HUB_OFFLINE", "0") == "1",
        )

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        # Calculate num_proc carefully to avoid oversubscribing the system when running with DDP.
        total_cpus = os.cpu_count() or 1
        nproc_per_node = max(1, int(os.environ.get("NPROC_PER_NODE", "1")))
        num_proc = max(1, total_cpus // nproc_per_node)

        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        rank = max(0, int(os.environ.get("RANK", "0")))
        local_rank = max(0, int(os.environ.get("LOCAL_RANK", "0")))
        default_num_proc_cap = "32" if world_size > 1 else "0"
        num_proc_cap = int(os.environ.get("OPENPI_LOAD_DATASET_NUM_PROC_CAP", default_num_proc_cap))
        if num_proc_cap > 0:
            num_proc = min(num_proc, num_proc_cap)

        # Node-local Arrow cache construction must never hold an NCCL collective
        # open. The c10d store coordinates one fresh generation per invocation;
        # generation-scoped local markers and store keys cannot be satisfied by a
        # previous load of the same request. Persistent prepared identity remains
        # request-scoped so rank 0 can deliberately validate and reuse it.
        is_distributed = th.distributed.is_available() and th.distributed.is_initialized()
        force_load_cache = os.environ.get("OPENPI_FORCE_LOAD_CACHE", "0") == "1"
        if world_size > 1 and not is_distributed:
            raise RuntimeError(
                f"WORLD_SIZE={world_size} requests distributed HF cache loading, but the torch "
                "process group is not initialized; refusing concurrent unsynchronized load_dataset calls."
            )
        if is_distributed:
            actual_world_size = th.distributed.get_world_size()
            actual_rank = th.distributed.get_rank()
            if (world_size, rank) != (actual_world_size, actual_rank):
                raise RuntimeError(
                    "Distributed environment/process-group mismatch during HF cache setup: "
                    f"env world/rank={world_size}/{rank}, process-group={actual_world_size}/{actual_rank}"
                )

        local_world_size = max(
            1,
            int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("NPROC_PER_NODE", "1"))),
        )
        per_node_cache = os.environ.get("OPENPI_HF_DATASETS_CACHE_PER_RANK", "1") == "1"
        run_id = resolve_cache_run_id(distributed=is_distributed)

        data_dir: str | None = None
        data_files: list[str] | None = None
        if self.episodes is None:
            data_dir = str(self.root / "data")
            request_sources = [data_dir]
            source_mode = "data_dir"
        else:
            data_files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            request_sources = data_files
            source_mode = "data_files"
        load_identity_options = {
            "builder": "parquet",
            "split": "train",
            "datasets_version": datasets.__version__,
            "source_mode": source_mode,
            "episodes_count": None if self.episodes is None else len(self.episodes),
            "cache_layout": "node_local" if world_size > 1 else "single_process",
        }
        selection_id = make_cache_selection_id(
            dataset_root=self.root,
            source_mode=source_mode,
            source_paths=request_sources,
            load_options=load_identity_options,
        )
        invocation_index = next_cache_invocation_index(selection_id)
        sync_settings = HfCacheSyncSettings.from_env()

        store = None
        if is_distributed:
            try:
                store = th.distributed.distributed_c10d._get_default_store()
            except Exception as exc:
                raise RuntimeError(
                    "Unable to access the c10d control-plane store required for HF cache readiness; "
                    "refusing to fall back to an NCCL barrier."
                ) from exc

        attempt = coordinate_cache_attempt(
            store,
            rank=rank,
            selection_id=selection_id,
            invocation_index=invocation_index,
            run_id=run_id,
            request_id_factory=lambda: make_cache_request_id(
                dataset_root=self.root,
                source_mode=source_mode,
                source_paths=request_sources,
                load_options=load_identity_options,
            ),
            timeout_s=sync_settings.timeout_s,
            poll_s=sync_settings.poll_s,
        )

        def _setup_cache_paths():
            return setup_node_cache_paths(
                os.environ.get("HF_DATASETS_CACHE"),
                world_size=world_size,
                rank=rank,
                local_world_size=local_world_size,
                per_node_cache=per_node_cache,
                run_id=run_id,
                request_id=attempt.request_id,
                generation_id=attempt.generation_id,
                force_load_cache=force_load_cache,
            )

        cache_dir, sync_paths = coordinate_global_cache_setup(
            _setup_cache_paths,
            store=store,
            request_id=attempt.request_id,
            generation_id=attempt.generation_id,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            settings=sync_settings,
        )

        logger.info(
            "Loading dataset with %s processes (world_size=%s, rank=%s, local_rank=%s, cache_dir=%s)",
            num_proc,
            world_size,
            rank,
            local_rank,
            cache_dir,
        )

        load_kwargs: dict[str, object] = {"split": "train", "num_proc": num_proc}
        if cache_dir is not None:
            load_kwargs["cache_dir"] = cache_dir
        logger.info(
            "HF cache sync selection_id=%s request_id=%s generation_id=%s invocation=%s "
            "timeout_s=%s poll_s=%s force_load=%s distributed=%s run_id=%s",
            selection_id,
            attempt.request_id,
            attempt.generation_id,
            attempt.invocation_index,
            sync_settings.timeout_s,
            sync_settings.poll_s,
            force_load_cache,
            is_distributed,
            run_id,
        )

        def _do_load() -> datasets.Dataset:
            max_retries = max(1, int(os.environ.get("OPENPI_HF_LOAD_DATASET_RETRIES", "5")))
            retry_sleep_s = float(os.environ.get("OPENPI_HF_LOAD_DATASET_RETRY_SLEEP_S", "2"))
            for load_attempt in range(1, max_retries + 1):
                try:
                    if data_dir is not None:
                        return load_dataset("parquet", data_dir=data_dir, **load_kwargs)
                    assert data_files is not None
                    return load_dataset("parquet", data_files=data_files, **load_kwargs)
                except FileNotFoundError as exc:
                    # filelock can sporadically raise ENOENT on shared filesystems under contention.
                    lock_race = exc.filename is None and world_size > 1
                    if (not lock_race) or load_attempt >= max_retries:
                        raise
                    delay = retry_sleep_s * load_attempt
                    logger.warning(
                        "Transient filelock ENOENT while loading dataset (attempt %s/%s). Retrying in %.1fs.",
                        load_attempt,
                        max_retries,
                        delay,
                    )
                    time.sleep(delay)

        def _prepared_manifest(hf_dataset: datasets.Dataset) -> dict[str, object]:
            if cache_dir is None:
                raise RuntimeError("Cannot publish prepared-cache manifest without cache_dir")
            arrow_files = [cache_file["filename"] for cache_file in hf_dataset.cache_files]
            return build_prepared_cache_manifest(
                cache_dir,
                arrow_files,
                dataset_fingerprint=getattr(hf_dataset, "_fingerprint", None),
            )

        def _strict_force_load(manifest: dict[str, object]) -> datasets.Dataset:
            assert sync_paths is not None
            before = snapshot_cache_tree(sync_paths.cache_dir)
            arrow_paths = prepared_arrow_paths(sync_paths, manifest)
            parts = [datasets.Dataset.from_file(str(arrow_path)) for arrow_path in arrow_paths]
            hf_dataset = parts[0] if len(parts) == 1 else datasets.concatenate_datasets(parts)
            after = snapshot_cache_tree(sync_paths.cache_dir)
            if after != before:
                raise RuntimeError(
                    "Strict force-load detected cache-tree writes while opening prepared Arrow artifacts; "
                    "refusing to continue."
                )
            return hf_dataset

        def _publish_store_failure(error: BaseException) -> None:
            assert store is not None
            publish_global_cache_failure(
                store,
                request_id=attempt.request_id,
                generation_id=attempt.generation_id,
                rank=rank,
                local_rank=local_rank,
                error=error,
            )

        def _check_store_failure() -> None:
            assert store is not None
            observe_global_cache_failure(
                store,
                request_id=attempt.request_id,
                generation_id=attempt.generation_id,
                rank=rank,
                world_size=world_size,
                timeout_s=sync_settings.timeout_s,
                poll_s=sync_settings.poll_s,
            )

        try:
            if sync_paths is None:
                if force_load_cache:
                    raise RuntimeError(
                        "--force_load_cache is enabled in a single-process run, but HF_DATASETS_CACHE "
                        f"is unset, so request_id={attempt.request_id} cannot be verified. "
                        "Strict force-load will not create or rebuild a cache."
                    )
                hf_dataset = _do_load()
            else:
                hf_dataset = load_with_local_cache_sync(
                    _do_load,
                    paths=sync_paths,
                    is_builder=local_rank == 0 or not is_distributed,
                    force_load_cache=force_load_cache,
                    rank=rank,
                    local_rank=local_rank,
                    settings=sync_settings,
                    prepared_manifest_factory=_prepared_manifest,
                    force_load_fn=_strict_force_load,
                    external_failure_publisher=_publish_store_failure if store is not None else None,
                    external_failure_check=_check_store_failure if store is not None else None,
                )
        except DistributedCacheError:
            raise
        except Exception as exc:
            if store is not None:
                _publish_store_failure(exc)
                _check_store_failure()
            raise DistributedCacheError.from_exception(
                exc,
                request_id=attempt.request_id,
                generation_id=attempt.generation_id,
                rank=rank,
                local_rank=local_rank,
            ) from exc

        if store is not None:
            try:
                wait_for_global_cache_readiness(
                    store,
                    request_id=attempt.request_id,
                    generation_id=attempt.generation_id,
                    rank=rank,
                    world_size=world_size,
                    timeout_s=sync_settings.timeout_s,
                    poll_s=sync_settings.poll_s,
                )
            except DistributedCacheError:
                raise
            except Exception as exc:
                _publish_store_failure(exc)
                _check_store_failure()
                raise AssertionError("unreachable after cache failure consensus")

        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    @property
    def streaming_anchor_stride(self) -> int:
        return self._streaming_anchor_stride

    @property
    def streaming_anchor_offset(self) -> int:
        return self._streaming_anchor_offset

    @property
    def streaming_drop_incomplete_horizon(self) -> bool:
        return self._streaming_drop_incomplete_horizon

    def _select_aligned_streaming_chunk(self, *, start_at_current: bool) -> None:
        if not self._active_chunks:
            raise RuntimeError("Chunk streaming has no active chunks for this worker")
        start_idx = int(self.current_streaming_chunk_idx or 0)
        if not start_at_current:
            start_idx = (start_idx + 1) % len(self._active_chunks)
        for shift in range(len(self._active_chunks)):
            chunk_idx = (start_idx + shift) % len(self._active_chunks)
            cursor = _aligned_streaming_chunk_start(
                self._active_chunks[chunk_idx],
                stride=self._streaming_anchor_stride,
                offset=self._streaming_anchor_offset,
            )
            if cursor is not None:
                self.current_streaming_chunk_idx = chunk_idx
                self.current_streaming_frame_idx = cursor
                return
        raise RuntimeError(
            "No active chunk contains an anchor aligned to "
            f"stride={self._streaming_anchor_stride}, offset={self._streaming_anchor_offset}"
        )

    def _ensure_streaming_cursor_initialized(self) -> None:
        if self.current_streaming_chunk_idx is not None:
            return
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        ddp_rank = int(os.environ.get("RANK", "0"))
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        global_num_workers = max(1, num_workers * world_size)
        global_worker_id = ddp_rank * num_workers + worker_id
        if self._accept_rng is None:
            self._accept_rng = random.Random(self.seed + 1000003 * global_worker_id + 17)
        if self._active_chunks is None:
            # Existing approximate worker partitioning is intentionally retained.
            indices = list(range(global_worker_id, len(self.chunks), global_num_workers))
            if len(indices) == 0:
                indices = list(range(worker_id, len(self.chunks), num_workers))
            worker_chunks = [self.chunks[i] for i in indices]
            rng = np.random.default_rng(self.seed + global_worker_id)
            rng.shuffle(worker_chunks)
            self._active_chunks = worker_chunks
        rng = np.random.default_rng(self.seed + global_worker_id)
        self.current_streaming_chunk_idx = rng.integers(0, len(self._active_chunks)).item()
        self._select_aligned_streaming_chunk(start_at_current=True)

    def _move_to_next_streaming_chunk(self) -> None:
        self._select_aligned_streaming_chunk(start_at_current=False)
        self._should_obs_loaders_reload = True

    def _next_streaming_observation(self, key: str, *, context: str):
        try:
            return next(self.obs_loaders[key])[0]
        except StopIteration as exc:
            chunk = self._active_chunks[self.current_streaming_chunk_idx]
            raise RuntimeError(
                "Observation loader ended unexpectedly while "
                f"{context}: modality={key}, cursor={self.current_streaming_frame_idx}, chunk={chunk}, "
                f"stride={self._streaming_anchor_stride}, offset={self._streaming_anchor_offset}"
            ) from exc

    def _advance_streaming_anchor(self, *, observation_consumed: bool, context: str) -> None:
        """Advance all modality readers and the HF cursor to the next aligned anchor."""

        _, chunk_end, _ = self._active_chunks[self.current_streaming_chunk_idx]
        next_cursor = self.current_streaming_frame_idx + self._streaming_anchor_stride
        if next_cursor < chunk_end:
            frames_to_consume = self._streaming_anchor_stride - int(observation_consumed)
            for _ in range(frames_to_consume):
                for key in self.meta.video_keys:
                    self._next_streaming_observation(key, context=context)
            self.current_streaming_frame_idx = next_cursor
            return

        # Consume the current rejected/dropped observation, but never decode past
        # the chunk boundary. The next call closes and reopens at its aligned start.
        if not observation_consumed:
            for key in self.meta.video_keys:
                self._next_streaming_observation(key, context=context)
        self.current_streaming_frame_idx = chunk_end
        self._should_obs_loaders_reload = True

    @staticmethod
    def _action_horizon_is_padded(padding: dict[str, th.Tensor]) -> bool:
        action_padding = padding.get("action_is_pad")
        return action_padding is not None and bool(th.as_tensor(action_padding).any().item())

    def _reload_streaming_observation_loaders(self, item: dict, ep_idx: int) -> None:
        for loader in self.obs_loaders.values():
            loader.close()
        self.obs_loaders = {}
        self.current_streaming_episode_idx = ep_idx
        chunk_global_start, _, chunk_episode_local_start = self._active_chunks[self.current_streaming_chunk_idx]
        episode_local_cursor = chunk_episode_local_start + self.current_streaming_frame_idx - chunk_global_start
        for vid_key in self.meta.video_keys:
            kwargs = {}
            task_id = item["task_index"].item()
            if "seg_instance_id" in vid_key:
                with open(
                    self.root / "meta/episodes" / f"task-{task_id:04d}" / f"episode_{ep_idx:08d}.json",
                ) as f:
                    meta = json.load(f)
                    instance_id_mapping = json.loads(meta["ins_id_mapping"])
                    instance_id_mapping = {int(k): v for k, v in instance_id_mapping.items()}
                    self.omnigibson_mapping[ep_idx]["instance_id_mapping"] = instance_id_mapping
                    self.omnigibson_mapping[ep_idx]["unique_ins_ids"][vid_key.split(".")[-1]] = meta[
                        f"{ROBOT_CAMERA_NAMES['R1Pro'][vid_key.split('.')[-1]]}::unique_ins_ids"
                    ]
                    kwargs["id_list"] = th.tensor(
                        self.omnigibson_mapping[ep_idx]["unique_ins_ids"][vid_key.split(".")[-1]]
                    )
            if "rgb" in vid_key:
                kwargs["train_rgb_type"] = self.train_rgb_type
            self.obs_loaders[vid_key] = iter(
                OBS_LOADER_MAP[vid_key.split(".")[2]](
                    data_path=self.root,
                    task_id=task_id,
                    camera_id=vid_key.split(".")[-1],
                    demo_id=f"{ep_idx:08d}",
                    start_idx=episode_local_cursor,
                    start_idx_is_keyframe=False,
                    batch_size=1,
                    stride=1,
                    **kwargs,
                )
            )
        self._should_obs_loaders_reload = False

    def __getitem__(self, idx) -> dict:
        if not self._chunk_streaming_using_keyframe:
            item = super().__getitem__(idx)
            item["task"] = self._get_fine_grained_task(item)
            subtask_text = self._get_subtask_text(item)
            if subtask_text is not None:
                item["subtask_text"] = subtask_text
            return item

        # Rejections and incomplete horizons advance iteratively; recursion here
        # could overflow when a long tail or low resampling weight is encountered.
        while True:
            self._ensure_streaming_cursor_initialized()
            _, chunk_end, _ = self._active_chunks[self.current_streaming_chunk_idx]
            if self.current_streaming_frame_idx >= chunk_end:
                self._move_to_next_streaming_chunk()

            item = self.hf_dataset[self.current_streaming_frame_idx]
            item.pop("observation.task_info")
            ep_idx = item["episode_index"].item()

            if self._should_obs_loaders_reload:
                self._reload_streaming_observation_loaders(item, ep_idx)

            if self.delta_indices is not None:
                query_indices, padding = self._get_query_indices(self.current_streaming_frame_idx, ep_idx)
                if self._streaming_drop_incomplete_horizon and self._action_horizon_is_padded(padding):
                    self._advance_streaming_anchor(
                        observation_consumed=False,
                        context="dropping an incomplete action horizon",
                    )
                    continue
                query_result = self._query_hf_dataset(query_indices)
                item = {**item, **padding, **query_result}

            weight = self._get_resample_weight(item)
            if self._accept_rng is not None and self._accept_rng.random() >= weight:
                self._advance_streaming_anchor(observation_consumed=False, context="rejecting a resampled anchor")
                continue

            # The current observation consumes one decoded frame per modality.
            for key in self.meta.video_keys:
                item[key] = self._next_streaming_observation(key, context="returning an aligned anchor")

                if self.return_seg_instance and "seg_instance_id" in key:
                    seg_instance, instance_mapping = instance_id_to_instance(
                        obs=item[key],
                        instance_id_mapping=self.omnigibson_mapping[ep_idx]["instance_id_mapping"],
                        unique_ins_ids=np.array(
                            self.omnigibson_mapping[ep_idx]["unique_ins_ids"][key.split(".")[-1]]
                        ),
                    )
                    instance_mapping = {instance_name: id for id, instance_name in instance_mapping.items()}

                    frame_index = round(item["timestamp"].item() * self.fps)
                    sub_idx = bisect.bisect_right(
                        self.task_sizes[ep_idx], frame_index, hi=len(self.task_sizes[ep_idx]) - 1
                    )
                    skill_annotation = self.meta.annotations[ep_idx]["skill_annotation"]
                    relative_obj_names = skill_annotation[sub_idx]["object_id"][0]
                    for i, relative_obj_name in enumerate(relative_obj_names):
                        instance_id = instance_mapping[relative_obj_name]
                        seg_instance[seg_instance == instance_id] = -(i + 1)
                    seg_instance[seg_instance > 0] = 0
                    seg_instance *= -1
                    item[key.replace("seg_instance_id", "seg_instance")] = seg_instance

            if self.image_transforms is not None:
                for cam in self.meta.camera_keys:
                    item[cam] = self.image_transforms(item[cam])

            item["task"] = self._get_fine_grained_task(item)
            subtask_text = self._get_subtask_text(item)
            if subtask_text is not None:
                item["subtask_text"] = subtask_text
            self._advance_streaming_anchor(observation_consumed=True, context="advancing after an aligned anchor")
            return item

    def _get_resample_key_from_skill_ann(self, item: dict) -> tuple[str | None, str | None]:
        if self.resample_group_by not in {"skill_type", "skill_description"}:
            return None, None
        ep_idx = item["episode_index"].item()
        frame_index = round(item["timestamp"].item() * self.fps)
        if ep_idx not in self._resample_skill_segments:
            ann = self.meta.annotations.get(ep_idx)
            segs = []
            if isinstance(ann, dict):
                for a in ann.get("skill_annotation", []) or []:
                    if not isinstance(a, dict):
                        continue
                    skill_type = a.get("skill_type", "")
                    if skill_type is None:
                        skill_type = ""
                    skill_desc = ""
                    desc_list = a.get("skill_description", []) or []
                    if isinstance(desc_list, list):
                        for d in desc_list:
                            if isinstance(d, str) and d.strip():
                                skill_desc = d.strip()
                                break
                    for s, e in _duration_to_segments(a.get("frame_duration")):
                        segs.append((int(s), int(e), str(skill_type), str(skill_desc)))
            segs.sort(key=lambda x: (x[0], x[1]))
            ends = [e for _, e, _, _ in segs]
            self._resample_skill_segments[ep_idx] = segs
            self._resample_skill_segment_ends[ep_idx] = ends
        segs = self._resample_skill_segments[ep_idx]
        ends = self._resample_skill_segment_ends[ep_idx]
        if not segs:
            return "", ""
        i = bisect.bisect_left(ends, frame_index)
        if 0 <= i < len(segs):
            s, e, stype, sdesc = segs[i]
            if s <= frame_index <= e:
                return stype, sdesc
        return "", ""

    def _get_resample_weight(self, item: dict) -> float:
        group_by = self.resample_group_by
        weights = self.resample_weights
        default_weight = float(self.resample_default_weight)
        weight = 1.0
        if group_by is None:
            task_skill = self._get_current_task_skill(item)
            weight = skill_weight(task_skill, self.skill_list)
        elif group_by == "task_skill":
            task_skill = self._get_current_task_skill(item)
            if weights is None:
                weight = skill_weight(task_skill, self.skill_list)
            else:
                weight = float(weights.get(str(task_skill), default_weight)) / float(self._resample_weight_norm)
        elif group_by in {"skill_type", "skill_description"}:
            skill_type, skill_desc = self._get_resample_key_from_skill_ann(item)
            key = skill_type if group_by == "skill_type" else skill_desc
            if weights is None:
                weight = default_weight
            else:
                weight = float(weights.get(str(key), default_weight)) / float(self._resample_weight_norm)
        else:
            raise ValueError(f"Unsupported resample_group_by={group_by}")
        if weight <= 0.0:
            return 0.0
        if weight >= 1.0:
            return 1.0
        return float(weight)

    def _get_current_task_skill(self, item: dict) -> str:
        ep_idx = item["episode_index"].item()
        frame_index = round(item["timestamp"].item() * self.fps)
        sub_idx = bisect.bisect_right(self.task_sizes[ep_idx], frame_index, hi=len(self.task_sizes[ep_idx]) - 1)
        task_skill = self.meta.orchestrators[ep_idx][1][sub_idx]["task"]
        return task_skill

    def _get_fine_grained_task(self, item: dict) -> str:
        ep_idx = item["episode_index"].item()
        task_idx = item["task_index"].item()
        frame_index = round(item["timestamp"].item() * self.fps)
        try:
            sub_idx = bisect.bisect_right(self.task_sizes[ep_idx], frame_index, hi=len(self.task_sizes[ep_idx]) - 1)
            task_text = self.meta.orchestrators[ep_idx][self.fine_grained_level][sub_idx]["task"]

        except Exception as e:
            print(f"[warn] {self.repo_id} failed to get subtask {item}: {e}")
            task_text = self.meta.tasks[task_idx]
        return task_text

    def _get_task_at_level(self, item: dict, level: int) -> str | None:
        """Get the task description at a specific orchestrator level.

        Returns None if the level is not available.
        """
        ep_idx = item["episode_index"].item()
        task_idx = item["task_index"].item()
        frame_index = round(item["timestamp"].item() * self.fps)
        try:
            sub_idx = bisect.bisect_right(self.task_sizes[ep_idx], frame_index, hi=len(self.task_sizes[ep_idx]) - 1)
            return self.meta.orchestrators[ep_idx][level][sub_idx]["task"]
        except Exception:
            return None

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        ep_idx = self.episode_data_index_pos[ep_idx]
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": th.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, th.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)

        return item

    def _get_keyframe_chunk_indices(self, chunk_size=250) -> list[tuple[int, int, int]]:
        """
        Divide each episode into chunks of data based on GOP of the data (here for B1K, GOP size is 250 frames).
        Args:
            chunk_size (int): size of each chunk in number of frames. Default is 250 for B1K. Should be the GOP size of the video data.
        Returns:
            List of tuples, where each tuple contains (start_index, end_index, local_start_index) for each chunk.
        """
        episode_lengths = {ep_idx: ep_dict["length"] for ep_idx, ep_dict in self.meta.episodes.items()}
        episode_lengths = [episode_lengths[ep_idx] for ep_idx in self.episodes]
        chunks = []
        offset = 0
        for L in episode_lengths:
            local_starts = list(range(0, L, chunk_size))
            local_ends = local_starts[1:] + [L]
            for ls, le in zip(local_starts, local_ends):
                chunks.append((offset + ls, offset + le, ls))
            offset += L
        return chunks


class BehaviorLerobotDatasetMetadata(LeRobotDatasetMetadata):
    """
    BehaviorLerobotDatasetMetadata extends LeRobotDatasetMetadata with the following customizations:
        1. Restricts the set of allowed modalities to {"rgb", "depth", "seg_instance_id"}.
        2. Restricts the set of allowed camera names to those defined in ROBOT_CAMERA_NAMES["R1Pro"].
        3. Provides a filtered view of dataset features, including only those corresponding to the selected modalities and camera names.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
        # === Customized arguments for BehaviorLeRobotDataset ===
        tasks: Iterable[str] = None,
        modalities: Iterable[str] = None,
        cameras: Iterable[str] = None,
    ):
        # ========== Customizations ==========
        self.task_name_candidates = set(tasks) if tasks is not None else set(TASK_NAMES_TO_INDICES.keys())
        self.modalities = set(modalities)
        self.camera_names = set(cameras)
        assert self.modalities.issubset(
            {"rgb", "depth", "seg_instance_id"}
        ), f"Modalities must be a subset of ['rgb', 'depth', 'seg_instance_id'], but got {self.modalities}"
        assert self.camera_names.issubset(
            ROBOT_CAMERA_NAMES["R1Pro"]
        ), f"Camera names must be a subset of {ROBOT_CAMERA_NAMES['R1Pro']}, but got {self.camera_names}"
        # ===================================

        self.repo_id = repo_id
        self.revision = revision or CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)

            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/**", ignore_patterns="meta/episodes/**")
            self.load_metadata()

    def load_metadata(self):
        self.info = load_info(self.root)
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks, self.task_to_task_index, self.task_names = self.load_tasks(self.root)
        # filter based on self.task_name_candidates
        valid_task_indices = [idx for idx, name in self.task_names.items() if name in self.task_name_candidates]
        self.task_names = set([self.task_names[idx] for idx in valid_task_indices])
        self.tasks = {idx: self.tasks[idx] for idx in valid_task_indices}
        self.task_to_task_index = {v: k for k, v in self.tasks.items()}

        self.episodes = self.load_episodes(self.root)
        self.annotations = self.load_annotations(self.root)
        self.orchestrators = self.load_orchestrators(self.root)
        if self._version < packaging.version.parse("v2.1"):
            self.stats = self.load_stats(self.root)
            self.episodes_stats = backward_compatible_episodes_stats(self.stats, self.episodes)
        else:
            self.episodes_stats = self.load_episodes_stats(self.root)
            self.stats = aggregate_stats(list(self.episodes_stats.values()))
        logger.info(f"Loaded metadata for {len(self.episodes)} episodes.")

    def load_tasks(self, local_dir: Path) -> tuple[dict, dict]:
        tasks = load_jsonlines(local_dir / TASKS_PATH)
        task_names = {item["task_index"]: item["task_name"] for item in sorted(tasks, key=lambda x: x["task_index"])}
        tasks = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
        task_to_task_index = {task: task_index for task_index, task in tasks.items()}
        return tasks, task_to_task_index, task_names

    def load_episodes(self, local_dir: Path) -> dict:
        episodes = load_jsonlines(local_dir / EPISODES_PATH)
        return {
            item["episode_index"]: item
            for item in sorted(episodes, key=lambda x: x["episode_index"])
            if item["episode_index"] // 1e4 in self.tasks
        }

    def load_stats(self, local_dir: Path) -> dict[str, dict[str, np.ndarray]]:
        if not (local_dir / STATS_PATH).exists():
            return None
        stats = load_json(local_dir / STATS_PATH)
        return cast_stats_to_numpy(stats)

    def load_episodes_stats(self, local_dir: Path) -> dict:
        episodes_stats = load_jsonlines(local_dir / EPISODES_STATS_PATH)
        return {
            item["episode_index"]: cast_stats_to_numpy(item["stats"])
            for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
            if item["episode_index"] in self.episodes
        }

    def load_annotations(self, local_dir: Path) -> dict:
        annotations = local_dir / ANNOTATIONS_PATH
        task_list = [task_id for task_id in annotations.iterdir() if task_id.is_dir()]
        return {
            int(episode.stem[8:]): load_json(episode)
            for task_id in task_list
            if int(task_id.name[5:]) in self.tasks
            for episode in sorted(task_id.iterdir())
        }

    def load_orchestrators(self, local_dir: Path) -> dict:
        orchestrators_path = local_dir / ORCHESTRATORS_PATH
        orchestrators = {
            episode_key: load_orchestrators_data(episode_data["tasks"][0], episode_data["length"])
            for episode_key, episode_data in sorted(self.episodes.items())
        }
        if orchestrators_path.exists():
            for task in self.tasks:
                if (orchestrators_path / f"task-{task:04d}").exists():
                    orchestrators.update(
                        {
                            int(episode.stem[8:]): load_orchestrators_data(
                                episode, self.episodes[int(episode.stem[8:])]["length"]
                            )
                            for episode in sorted((orchestrators_path / f"task-{task:04d}").iterdir())
                        }
                    )
        return orchestrators

    def get_annotation_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.annotation_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_metainfo_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.metainfo_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    @property
    def annotation_path(self) -> str | None:
        """Formattable string for the annotation files."""
        return self.info["annotation_path"]

    @property
    def metainfo_path(self) -> str | None:
        """Formattable string for the metainfo files."""
        return self.info["metainfo_path"]

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        features = dict()
        # pop not required features
        for name in self.info["features"].keys():
            if (
                name.startswith("observation.images.")
                and name.split(".")[-1] in self.camera_names
                and name.split(".")[-2] in self.modalities
            ):
                features[name] = self.info["features"][name]
        return features


def load_orchestrators_data(episode_path_or_level_0_task, episode_len):
    output_data = defaultdict(list)
    if type(episode_path_or_level_0_task) == str:
        for i in range(4):
            output_data[i] = [
                {
                    "task": episode_path_or_level_0_task,
                    "start_frame": 0,
                    "end_frame": episode_len - 1,
                }
            ]
        return output_data
    episode_path = episode_path_or_level_0_task
    task_annotated_data = load_json(episode_path / "task_annotated.json")
    level_0_task = task_annotated_data["cot_task_description"]
    output_data[0].append(
        {
            "task": level_0_task,
            "start_frame": 0,
            "end_frame": episode_len - 1,
        }
    )
    try:
        num_level1_tasks = len(task_annotated_data["cot_subtask_description_list"])
        for i in range(num_level1_tasks):
            subtask_data = load_json(episode_path / f"subtask_{i}_annotated.json")
            subtask = subtask_data["cot_subtask_description"]
            start_frame, end_frame = subtask_data["start_frame"], subtask_data["end_frame"] - 1
            skill = subtask_data["skill_description"]
            output_data[1].append(
                {
                    "task": skill,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                }
            )
            output_data[2].append(
                {
                    "task": subtask,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                }
            )
            for event_data_path in sorted(episode_path.glob(f"event_{i}_*_annotated.json")):
                event_data = load_json(event_data_path)
                event_task = event_data["subtask_answer_detailed"]
                start_frame, end_frame = event_data["start_frame"], event_data["end_frame"] - 1
                output_data[3].append(
                    {
                        "task": event_task,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                    }
                )
    except Exception as e:
        print(f"[warn] {episode_path} failed to load orchestrators data: {e}, falling back to default task.")
        for i in range(len(output_data)):
            output_data[i] = output_data[0]
    return output_data


def skill_weight(cur_skill, skill_list: list[str]) -> float:
    if "all" in skill_list:
        skill_list = [skill for skill in skill_list if skill != "all"]
        for skill_item in skill_list:
            skill, weight = skill_item.split(":")
            if skill == cur_skill:
                return float(weight)
        return 1.0
    for skill_item in skill_list:
        skill, weight = skill_item.split(":")
        if skill == cur_skill:
            return float(weight)
    return 0.0


class MultiBehaviorLeRobotDataset:
    def __init__(self, datasets: list[BehaviorLeRobotDataset], sample_weights: list[float] | None = None):
        if sample_weights is None:
            sample_weights = [1.0 / len(datasets)] * len(datasets)
        assert len(datasets) == len(sample_weights), "Length of datasets and sample weights must be the same"
        if sum(sample_weights) != 1.0:
            sample_weights = [weight / sum(sample_weights) for weight in sample_weights]

        self.datasets = datasets
        self.sample_weights = sample_weights

    def __len__(self):
        return max(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        index = np.random.choice(range(len(self.datasets)), p=self.sample_weights)
        return self.datasets[index][idx]
