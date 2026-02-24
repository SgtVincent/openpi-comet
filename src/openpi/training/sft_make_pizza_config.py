import openpi.models.pi0_config as pi0_config
from openpi.models.vlm2_vla_config import VLM2VLAConfig
import openpi.training.optimizer as _optimizer
from openpi.training.data_config import AssetsConfig, DataConfig, LeRobotB1KDataConfig
from openpi.training.train_config import TrainConfig
import openpi.training.weight_loaders as weight_loaders


_SFT_MAKE_PIZZA_CONFIGS = [
        TrainConfig(
            name="vlm2_b1k-make_pizza_lr2.5e-6_5ep_sft",
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
                    behavior_dataset_root="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos/",
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
                peak_lr=2.5e-6,
                decay_steps=20_000,
            ),
            freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32, max_token_len=512).get_freeze_filter(),
            ema_decay=None,
            checkpoint_base_dir="checkpoints",
            num_workers=10,
            # batch_size=4 * 12,
            batch_size_per_gpu=12,
        ),
        TrainConfig(
            name="pi05_b1k-make_pizza_lr2.5e-6_5ep_sft",
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
                    behavior_dataset_root="/mnt/bn/robot-mllm-data-lf-3/mlx/users/chenjunting/data/2025-challenge-demos/",
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
                peak_lr=2.5e-6,
                decay_steps=20_000,
            ),
            freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32, max_token_len=512).get_freeze_filter(),
            ema_decay=None,
            checkpoint_base_dir="checkpoints",
            num_workers=10,
            # batch_size=4 * 12,
            batch_size_per_gpu=12,
        ),
]
