import argparse
import json
import logging
import platform

import numpy as np

import openpi.models.pi0_config as pi0_config
import openpi.training.data_loader as _data_loader
from openpi.training.data_config import AssetsConfig
from openpi.training.data_config import DataConfig
from openpi.training.data_config import LeRobotB1KDataConfig
from openpi.training.train_config import TrainConfig


def _init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.handlers[0].setFormatter(formatter)


def _maybe_string(x):
    if isinstance(x, str):
        return x
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/mnt/bn/mllm-data-yg/chenjunting/data/2025-challenge-demos/")
    parser.add_argument("--repo-id", default="behavior-1k/2025-challenge-demos")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--precision", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--debug-raw-text", action="store_true")
    args = parser.parse_args()

    _init_logging()
    logging.info("Running on: %s", platform.node())

    model = pi0_config.Pi0Config(pi05=True, action_horizon=args.action_horizon)
    config = TrainConfig(
        name="__debug_b1k_skill_phrase_batch__",
        exp_name="__debug__",
        project_name="B1K",
        model=model,
        pytorch_model_name="pi0",
        pytorch_training_precision=args.precision,
        data=LeRobotB1KDataConfig(
            repo_id=args.repo_id,
            assets=AssetsConfig(
                assets_dir="checkpoints/pi05_base_pytorch/assets",
                asset_id=args.repo_id,
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                prompt_from_skill_description=True,
                episodes_index=list(range(args.episodes)),
                behavior_dataset_root=args.dataset_root,
                fine_grained_level=0,
            ),
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    data_loader = _data_loader.create_data_loader(config, framework="pytorch", shuffle=False, skip_norm_stats=True)
    observation, actions = next(iter(data_loader))

    logging.info("batch observation keys: %s", list(observation.to_dict().keys()))
    logging.info("batch actions shape: %s dtype=%s", tuple(actions.shape), actions.dtype)

    obs_dict = observation.to_dict()
    if "tokenized_prompt" in obs_dict and obs_dict["tokenized_prompt"] is not None:
        toks = obs_dict["tokenized_prompt"]
        mask = obs_dict.get("tokenized_prompt_mask")
        logging.info("tokenized_prompt: shape=%s dtype=%s", tuple(toks.shape), toks.dtype)
        if mask is not None:
            logging.info("tokenized_prompt_mask: shape=%s dtype=%s", tuple(mask.shape), mask.dtype)

    if args.debug_raw_text:
        from openpi.training.behavior_skill_dataset import BehaviorLeRobotSkillDataset

        raw_dataset = BehaviorLeRobotSkillDataset(
            repo_id=args.repo_id,
            root=args.dataset_root,
            tolerance_s=1e-4,
            tasks=None,
            modalities=["rgb"],
            local_only=True,
            check_files=False,
            check_timestamp_sync=False,
            delta_timestamps={"action": [t / 30.0 for t in range(args.action_horizon)]},
            episodes=list(range(args.episodes)),
            chunk_streaming_using_keyframe=True,
            shuffle=False,
            fine_grained_level=0,
            return_seg_instance=False,
            train_rgb_type="regular",
        )

        for i in range(min(3, len(raw_dataset))):
            item = raw_dataset[i]
            task_text = item.get("task")
            ep_idx = item.get("episode_index")
            ts = item.get("timestamp")
            task_text = _maybe_string(task_text) if task_text is not None else None
            ep_idx = int(ep_idx.item()) if hasattr(ep_idx, "item") else ep_idx
            ts = float(ts.item()) if hasattr(ts, "item") else ts
            logging.info("raw[%d]: episode_index=%s timestamp=%.3f task(phrase)=%s", i, ep_idx, ts, task_text)

        ex = (
            raw_dataset.root
            / "subtasks"
            / "task-0000"
            / "episode_00000010.json"
        )
        try:
            with open(ex, "r", encoding="utf-8") as f:
                payload = json.load(f)
            logging.info("example subtasks file: %s", str(ex))
            logging.info("example skill.merged_segments[1].phrase: %s", payload["skill"]["merged_segments"][1]["phrase"])
        except Exception:
            pass

    logging.info("OK")


if __name__ == "__main__":
    main()
