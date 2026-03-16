from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _configure_runtime_env(config_name: str) -> None:
    os.environ.setdefault("OPENPI_DATA_HOME", str(REPO_ROOT / ".cache" / "openpi"))
    os.environ.setdefault("OPENPI_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("OPENPI_LOAD_DATASET_NUM_PROC_CAP", "8")
    os.environ.setdefault("HF_DATASETS_CACHE", f"/opt/tiger/hf_datasets_cache/{config_name}/")


def _parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_episodes(value: str | None) -> list[int] | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if ":" in value:
        start_str, end_str = value.split(":", 1)
        start = int(start_str)
        end = int(end_str)
        if end < start:
            raise ValueError(f"Invalid episode range: {value}")
        return list(range(start, end))
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="pi05_subtask_b1k-make_pizza_ann-skill_lr2.5e-6_5ep_sft",
    )
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--episodes", type=str, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument(
        "--subtask-source",
        type=str,
        default=None,
        choices=("orchestrator", "annotations_skill", "annotations_primitive"),
    )
    parser.add_argument("--subtask-template-path", type=str, default=None)
    parser.add_argument("--subtask-object-name-mapping-path", type=str, default=None)
    parser.add_argument("--subtask-joiner", type=str, default=None)
    return parser.parse_args()


def _fallback_data_config(config_name: str):
    if config_name != "pi05_subtask_b1k-make_pizza_ann-skill_lr2.5e-6_5ep_sft":
        raise ValueError(f"Unknown config for fallback path: {config_name}")
    from openpi.training.data_config import DataConfig
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    subtask_template_path = repo_root / "src/behavior/learning/datas/b1k_subtask_phrase_templates.json"
    subtask_object_name_mapping_path = repo_root / "src/behavior/learning/datas/b1k_object_id_name_mapping.json"

    return DataConfig(
        repo_id="behavior-1k/2025-challenge-demos",
        behavior_dataset_root="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos/",
        prompt_from_task=True,
        tasks=["make_pizza"],
        modalities=["rgb"],
        fine_grained_level=0,
        action_sequence_keys=("action",),
        tolerance_s=1e-4,
        subtask_source="annotations_skill",
        subtask_template_path=str(subtask_template_path),
        subtask_object_name_mapping_path=str(subtask_object_name_mapping_path),
        subtask_joiner=" then ",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    args = _parse_args()
    _configure_runtime_env(args.config)

    train_config_name = args.config
    action_horizon = 32
    try:
        from openpi.training.train_config import get_config

        train_config = get_config(args.config)
        if getattr(train_config, "name", None) != args.config:
            raise ValueError(f"Config {args.config} not registered in openpi; using fallback DataConfig for this script.")
        train_config_name = train_config.name
        action_horizon = int(getattr(train_config.model, "action_horizon", 32))
        data_factory = train_config.data[0]
        data_config = data_factory.create(train_config.assets_dirs, train_config.model)
    except Exception:
        data_config = _fallback_data_config(args.config)

    overrides = {}
    if args.dataset_root is not None:
        overrides["behavior_dataset_root"] = args.dataset_root
    tasks = _parse_csv_list(args.tasks)
    if tasks is not None:
        overrides["tasks"] = tasks
    episodes = _parse_episodes(args.episodes)
    if episodes is not None:
        overrides["episodes_index"] = episodes
    if overrides:
        data_config = dataclasses.replace(data_config, **overrides)

    logger = logging.getLogger("test_pi05_loader")
    logger.info("train_config=%s", train_config_name)
    logger.info(
        "data_config repo_id=%s root=%s tasks=%s episodes_index=%s modalities=%s fine_grained_level=%s subtask_source=%s",
        data_config.repo_id,
        data_config.behavior_dataset_root,
        data_config.tasks,
        data_config.episodes_index,
        data_config.modalities,
        data_config.fine_grained_level,
        args.subtask_source or getattr(data_config, "subtask_source", None),
    )
    logger.info(
        "subtask_template_path=%s subtask_object_name_mapping_path=%s subtask_joiner=%s",
        getattr(data_config, "subtask_template_path", None),
        getattr(data_config, "subtask_object_name_mapping_path", None),
        getattr(data_config, "subtask_joiner", None),
    )

    from behavior.learning.datas.dataset import BehaviorLeRobotDataset

    args_kwargs = {}
    if data_config.skill_list != ["all"]:
        args_kwargs["skill_list"] = data_config.skill_list
    subtask_source = args.subtask_source or getattr(data_config, "subtask_source", "orchestrator")
    if subtask_source != "orchestrator":
        args_kwargs["subtask_source"] = subtask_source
        args_kwargs["subtask_template_path"] = args.subtask_template_path or getattr(data_config, "subtask_template_path", None)
        args_kwargs["subtask_object_name_mapping_path"] = (
            args.subtask_object_name_mapping_path or getattr(data_config, "subtask_object_name_mapping_path", None)
        )
        args_kwargs["subtask_joiner"] = args.subtask_joiner or getattr(data_config, "subtask_joiner", " then ")

    shuffle = False if args.no_shuffle else True
    if args.shuffle:
        shuffle = True

    dataset = BehaviorLeRobotDataset(
        repo_id=data_config.repo_id,
        root=data_config.behavior_dataset_root,
        tolerance_s=data_config.tolerance_s,
        tasks=data_config.tasks,
        modalities=data_config.modalities,
        local_only=True,
        check_files=False,
        check_timestamp_sync=False,
        delta_timestamps={
            key: [t / 30.0 for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        episodes=data_config.episodes_index,
        chunk_streaming_using_keyframe=True,
        shuffle=shuffle,
        fine_grained_level=data_config.fine_grained_level,
        return_seg_instance=data_config.return_seg_instance,
        train_rgb_type=data_config.train_rgb_type,
        **args_kwargs,
    )

    for i in range(max(1, int(args.num_samples))):
        item = dataset[args.sample_index]
        logger.info("sample=%d keys=%s", i, sorted(item.keys()))
        if "task" in item:
            logger.info("task=%r", item["task"])
        if "subtask_text" in item:
            logger.info("subtask_text=%r", item["subtask_text"])


if __name__ == "__main__":
    main()
