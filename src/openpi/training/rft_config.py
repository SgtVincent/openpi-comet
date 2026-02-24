import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
from openpi.training.data_config import DataConfig, LeRobotB1KDataConfig
from openpi.training.train_config import TrainConfig
import openpi.training.weight_loaders as weight_loaders


_RFT_CONFIGS = [
    TrainConfig(
        name="pi05_b1k-turning_on_radio_lr2.5e-6_step20k_rft",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=True,
                behavior_dataset_root="../DATASETS/behavior/2025-challenge-demos-rft",
                tasks=["turning_on_radio"],
                fine_grained_level=0,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("path_to_your_pretrained_checkpoint"),
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=20_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        checkpoint_base_dir="checkpoints",
        num_workers=8,
        batch_size=8 * 32,
    ),
]
