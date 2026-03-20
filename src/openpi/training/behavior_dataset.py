from __future__ import annotations

import openpi.training.config as _config


def is_behavior_dataset(data_config: _config.DataConfig) -> bool:
    if data_config.behavior_dataset_root is not None:
        return True
    if data_config.repo_id is None:
        return False
    return data_config.repo_id.startswith("behavior-1k/") or data_config.repo_id.startswith("delinqu/")


def create_behavior_dataset(data_config: _config.DataConfig, action_horizon: int):
    from behavior.learning.datas.dataset import BehaviorLeRobotDataset

    args = {}
    if data_config.skill_list != ["all"]:
        args["skill_list"] = data_config.skill_list
    if getattr(data_config, "resample_group_by", None) is not None:
        args["resample_group_by"] = data_config.resample_group_by
        args["resample_weights"] = getattr(data_config, "resample_weights", None)
        args["resample_default_weight"] = float(getattr(data_config, "resample_default_weight", 1.0))
    if getattr(data_config, "subtask_source", "orchestrator") != "orchestrator":
        args["subtask_source"] = data_config.subtask_source
        args["subtask_template_path"] = data_config.subtask_template_path
        args["subtask_object_name_mapping_path"] = data_config.subtask_object_name_mapping_path
        args["subtask_joiner"] = data_config.subtask_joiner

    dataset = BehaviorLeRobotDataset(
        repo_id=data_config.repo_id,
        root=data_config.behavior_dataset_root,
        tolerance_s=data_config.tolerance_s,
        tasks=data_config.tasks,
        modalities=data_config.modalities,
        local_only=True,
        check_files=False,
        check_timestamp_sync=False,
        delta_timestamps={key: [t / 30.0 for t in range(action_horizon)] for key in data_config.action_sequence_keys},
        episodes=data_config.episodes_index,
        chunk_streaming_using_keyframe=True,
        shuffle=True,
        fine_grained_level=data_config.fine_grained_level,
        return_seg_instance=data_config.return_seg_instance,
        train_rgb_type=data_config.train_rgb_type,
        **args,
    )

    return dataset


def create_multi_behavior_dataset(
    data_configs: list[_config.DataConfig], sample_weights: list[float] | None, action_horizon: int
):
    from behavior.learning.datas.dataset import MultiBehaviorLeRobotDataset

    datasets = [create_behavior_dataset(data_config, action_horizon) for data_config in data_configs]
    return MultiBehaviorLeRobotDataset(datasets, sample_weights=sample_weights)
