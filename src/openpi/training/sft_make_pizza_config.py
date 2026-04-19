import openpi.models.pi0_config as pi0_config
import openpi.models.pi05_subtask_config as pi05_subtask_config
from openpi.models.vlm2_vla_config import VLM2VLAConfig
import openpi.training.optimizer as _optimizer
from openpi.training.data_config import AssetsConfig, DataConfig, LeRobotB1KDataConfig
from openpi.training.train_config import TrainConfig
import openpi.training.weight_loaders as weight_loaders
from pathlib import Path


_SFT_MAKE_PIZZA_CONFIGS = [
        TrainConfig(
            name="vlm2_b1k-make_pizza_lr1e-4_5ep_sft",
            exp_name="openpi",
            project_name="B1K",
            model=VLM2VLAConfig(action_horizon=32, max_token_len=512),
            pytorch_model_name="vlm2",
            vlm2_num_frames=3,
            vlm2_geometry_dim=512,
            vlm2_view_dim=512,
            vlm2_working_memory_size=8,
            vlm2_episodic_memory_capacity=32,
            vlm2_episodic_similarity_threshold=0.7,
            vlm2_episodic_fusion_alpha=0.5,
            data=LeRobotB1KDataConfig(
                repo_id="behavior-1k/2025-challenge-demos",
                assets=AssetsConfig(
                    assets_dir="checkpoints/openpi_comet/pi05-b1kpt50-cs32/assets",
                    asset_id="behavior-1k/2025-challenge-demos",
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                    behavior_dataset_root="/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos/",
                    tasks=["make_pizza"],
                    fine_grained_level=0,
                ),
            ),
            pytorch_weight_path="checkpoints/openpi_comet/pi05-b1kpt50-cs32",
            weight_loader=weight_loaders.CheckpointWeightLoader("sunshk/openpi_comet/pi05-b1kpt50-cs32"),
            num_train_steps=0,
            num_train_epochs=5,
            save_interval=10000,
            keep_period=100000,
            # save_at_epoch_end_only=True,
            lr_schedule=_optimizer.CosineDecaySchedule(
                peak_lr=1e-4,
                # decay_steps must span the full training duration.
                # 5 epochs × ~40k steps/epoch ≈ 200k total steps.
                # Setting 0 = auto-match to num_train_steps (handled by train_pytorch.py).
                decay_steps=0,
                # Non-zero end LR prevents complete learning stall at end of training.
                # pi-0 uses end_lr = 75% of peak; we use ~10% as a conservative floor.
                decay_lr=1e-5,
            ),
            freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32, max_token_len=512).get_freeze_filter(),
            ema_decay=None,
            checkpoint_base_dir="./outputs/checkpoints",
            num_workers=10,
            # batch_size=4 * 12,
            batch_size_per_gpu=12,
        ),
        TrainConfig(
            name="pi05_b1k-make_pizza_lr1e-4_5ep_sft",
            exp_name="openpi",
            project_name="B1K",
            model=pi0_config.Pi0Config(pi05=True, action_horizon=32, max_token_len=512),
            pytorch_model_name="pi0",
            data=LeRobotB1KDataConfig(
                repo_id="behavior-1k/2025-challenge-demos",
                assets=AssetsConfig(
                    assets_dir="checkpoints/openpi_comet/pi05-b1kpt50-cs32/assets",
                    asset_id="behavior-1k/2025-challenge-demos",
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                    behavior_dataset_root="/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos/",
                    tasks=["make_pizza"],
                    fine_grained_level=0,
                ),
            ),
            pytorch_weight_path="checkpoints/openpi_comet/pi05-b1kpt50-cs32",
            weight_loader=weight_loaders.CheckpointWeightLoader("sunshk/openpi_comet/pi05-b1kpt50-cs32"),
            num_train_steps=0,
            num_train_epochs=5,
            save_interval=10000,
            keep_period=100000,
            # save_at_epoch_end_only=True,
            lr_schedule=_optimizer.CosineDecaySchedule(
                peak_lr=1e-4,
                # decay_steps must span the full training duration.
                # Setting 0 = auto-match to num_train_steps (handled by train_pytorch.py).
                decay_steps=0,
                decay_lr=1e-5,
            ),
            freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32, max_token_len=512).get_freeze_filter(),
            ema_decay=None,
            checkpoint_base_dir="./outputs/checkpoints",
            num_workers=10,
            # batch_size=4 * 12,
            batch_size_per_gpu=12,
        ),
        TrainConfig(
            name="pi05_subtask_b1k-make_pizza_lr1e-4_5ep_sft",
            exp_name="openpi",
            project_name="B1K",
            model=pi05_subtask_config.Pi05SubtaskConfig(
                alpha=10.0,
                subtask_max_len=128,
                action_horizon=32,
                max_token_len=512,
            ),
            pytorch_model_name="subtask",
            data=LeRobotB1KDataConfig(
                repo_id="behavior-1k/2025-challenge-demos",
                assets=AssetsConfig(
                    assets_dir="checkpoints/openpi_comet/pi05-b1kpt50-cs32/assets",
                    asset_id="behavior-1k/2025-challenge-demos",
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                    behavior_dataset_root="/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos/",
                    tasks=["make_pizza"],
                    fine_grained_level=0,
                ),
            ),
            pytorch_weight_path="checkpoints/openpi_comet/pi05-b1kpt50-cs32",
            weight_loader=weight_loaders.CheckpointWeightLoader("sunshk/openpi_comet/pi05-b1kpt50-cs32"),
            num_train_steps=0,
            num_train_epochs=5,
            save_interval=10000,
            keep_period=100000,
            lr_schedule=_optimizer.CosineDecaySchedule(
                peak_lr=1e-4,
                decay_steps=0,
                decay_lr=1e-5,
            ),
            freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32, max_token_len=512).get_freeze_filter(),
            ema_decay=None,
            checkpoint_base_dir="./outputs/checkpoints",
            num_workers=10,
            batch_size_per_gpu=12,
        ),
        TrainConfig(
            name="pi05_subtask_b1k-make_pizza_ann-skill_lr1e-4_5ep_sft",
            exp_name="openpi",
            project_name="B1K",
            model=pi05_subtask_config.Pi05SubtaskConfig(
                alpha=10.0,
                subtask_max_len=128,
                action_horizon=32,
                max_token_len=512,
            ),
            pytorch_model_name="subtask",
            data=LeRobotB1KDataConfig(
                repo_id="behavior-1k/2025-challenge-demos",
                assets=AssetsConfig(
                    assets_dir="checkpoints/openpi_comet/pi05-b1kpt50-cs32/assets",
                    asset_id="behavior-1k/2025-challenge-demos",
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                    behavior_dataset_root="/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos/",
                    tasks=["make_pizza"],
                    fine_grained_level=0,
                    subtask_source="annotations_skill",
                    subtask_template_path=str(
                        Path(__file__).resolve().parents[3] / "src/behavior/learning/datas/b1k_subtask_phrase_templates.json"
                    ),
                    subtask_object_name_mapping_path=str(
                        Path(__file__).resolve().parents[3] / "src/behavior/learning/datas/b1k_object_id_name_mapping.json"
                    ),
                    subtask_joiner=" then ",
                ),
            ),
            pytorch_weight_path="checkpoints/openpi_comet/pi05-b1kpt50-cs32",
            weight_loader=weight_loaders.CheckpointWeightLoader("sunshk/openpi_comet/pi05-b1kpt50-cs32"),
            num_train_steps=0,
            num_train_epochs=5,
            save_interval=10000,
            keep_period=100000,
            lr_schedule=_optimizer.CosineDecaySchedule(
                peak_lr=1e-4,
                decay_steps=0,
                decay_lr=1e-5,
            ),
            freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32, max_token_len=512).get_freeze_filter(),
            ema_decay=None,
            checkpoint_base_dir="./outputs/checkpoints",
            num_workers=10,
            batch_size_per_gpu=12,
        ),
        TrainConfig(
            name="vlm2_subtask_b1k-make_pizza_ann-skill_lr1e-4_5ep_sft",
            exp_name="openpi",
            project_name="B1K",
            model=pi05_subtask_config.Pi05SubtaskConfig(
                alpha=10.0,
                subtask_max_len=128,
                action_horizon=32,
                max_token_len=512,
            ),
            pytorch_model_name="vlm2_subtask",
            vlm2_num_frames=3,
            vlm2_geometry_dim=512,
            vlm2_view_dim=512,
            vlm2_working_memory_size=8,
            vlm2_episodic_memory_capacity=32,
            vlm2_episodic_similarity_threshold=0.7,
            vlm2_episodic_fusion_alpha=0.5,
            data=LeRobotB1KDataConfig(
                repo_id="behavior-1k/2025-challenge-demos",
                assets=AssetsConfig(
                    assets_dir="checkpoints/openpi_comet/pi05-b1kpt50-cs32/assets",
                    asset_id="behavior-1k/2025-challenge-demos",
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                    behavior_dataset_root="/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos/",
                    tasks=["make_pizza"],
                    fine_grained_level=0,
                    subtask_source="annotations_skill",
                    subtask_template_path=str(
                        Path(__file__).resolve().parents[3] / "src/behavior/learning/datas/b1k_subtask_phrase_templates.json"
                    ),
                    subtask_object_name_mapping_path=str(
                        Path(__file__).resolve().parents[3] / "src/behavior/learning/datas/b1k_object_id_name_mapping.json"
                    ),
                    subtask_joiner=" then ",
                ),
            ),
            pytorch_weight_path="checkpoints/openpi_comet/pi05-b1kpt50-cs32",
            weight_loader=weight_loaders.CheckpointWeightLoader("sunshk/openpi_comet/pi05-b1kpt50-cs32"),
            num_train_steps=0,
            num_train_epochs=5,
            save_interval=10000,
            keep_period=100000,
            lr_schedule=_optimizer.CosineDecaySchedule(
                peak_lr=1e-4,
                decay_steps=0,
                decay_lr=1e-5,
            ),
            freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32, max_token_len=512).get_freeze_filter(),
            ema_decay=None,
            checkpoint_base_dir="./outputs/checkpoints",
            num_workers=10,
            batch_size_per_gpu=12,
        ),
        TrainConfig(
            name="pi05_subtask_b1k-make_pizza_ann-skill_lr1e-4_5ep_sft_resample",
            exp_name="openpi",
            project_name="B1K",
            model=pi05_subtask_config.Pi05SubtaskConfig(
                alpha=10.0,
                subtask_max_len=128,
                action_horizon=32,
                max_token_len=512,
            ),
            pytorch_model_name="subtask",
            data=LeRobotB1KDataConfig(
                repo_id="behavior-1k/2025-challenge-demos",
                assets=AssetsConfig(
                    assets_dir="checkpoints/openpi_comet/pi05-b1kpt50-cs32/assets",
                    asset_id="behavior-1k/2025-challenge-demos",
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                    behavior_dataset_root="/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos/",
                    tasks=["make_pizza"],
                    fine_grained_level=0,
                    subtask_source="annotations_skill",
                    subtask_template_path=str(
                        Path(__file__).resolve().parents[3]
                        / "src/behavior/learning/datas/b1k_subtask_phrase_templates.json"
                    ),
                    subtask_object_name_mapping_path=str(
                        Path(__file__).resolve().parents[3] / "src/behavior/learning/datas/b1k_object_id_name_mapping.json"
                    ),
                    subtask_joiner=" then ",
                    resample_group_by="skill_type",
                    resample_weights={
                        "coordinated": 5.64,
                        "navigation": 1.68,
                        "uncoordinated": 1.0,
                    },
                    resample_default_weight=1.0,
                ),
            ),
            pytorch_weight_path="checkpoints/openpi_comet/pi05-b1kpt50-cs32",
            weight_loader=weight_loaders.CheckpointWeightLoader("sunshk/openpi_comet/pi05-b1kpt50-cs32"),
            num_train_steps=0,
            num_train_epochs=5,
            save_interval=10000,
            keep_period=100000,
            lr_schedule=_optimizer.CosineDecaySchedule(
                peak_lr=1e-4,
                decay_steps=0,
                decay_lr=1e-5,
            ),
            freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32, max_token_len=512).get_freeze_filter(),
            ema_decay=None,
            checkpoint_base_dir="./outputs/checkpoints",
            num_workers=10,
            batch_size_per_gpu=12,
        ),
]
