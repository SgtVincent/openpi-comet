import openpi.models.pi0_config as pi0_config
from openpi.models.vlm2_vla_config import VLM2VLAConfig
import openpi.training.optimizer as _optimizer
from openpi.training.data_config import FakeDataConfig
from openpi.training.train_config import TrainConfig
import openpi.training.weight_loaders as weight_loaders


_TEST_CONFIGS = [
    TrainConfig(
        name="vlm2_8gpu_test",
        exp_name="vlm2_test_run",
        project_name="B1K",
        model=VLM2VLAConfig(action_horizon=32),
        pytorch_model_name="vlm2",
        vlm2_num_frames=3,
        vlm2_geometry_dim=512,
        vlm2_view_dim=512,
        vlm2_working_memory_size=8,
        vlm2_episodic_memory_capacity=32,
        vlm2_episodic_similarity_threshold=0.7,
        vlm2_episodic_fusion_alpha=0.5,
        data=FakeDataConfig(),
        weight_loader=weight_loaders.CheckpointWeightLoader("sunshk/openpi_comet/pi05-b1kpt50-cs32"),
        num_train_steps=10,
        log_interval=1,
        save_interval=5,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=10,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        checkpoint_base_dir="checkpoints",
        num_workers=8,
        batch_size=8 * 4,
    ),
]
