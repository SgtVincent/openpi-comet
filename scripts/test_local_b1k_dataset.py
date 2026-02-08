import argparse
import logging
import time
from collections.abc import Mapping, Sequence

import numpy as np
import torch

import openpi.training.data_loader as _openpi_data_loader


def _parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


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
            raise ValueError(f"Invalid episodes range: {value}")
        return list(range(start, end))
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _summarize(obj, *, max_depth: int = 4, _depth: int = 0):
    if _depth >= max_depth:
        return type(obj).__name__
    if isinstance(obj, torch.Tensor):
        return {"type": "torch.Tensor", "shape": list(obj.shape), "dtype": str(obj.dtype), "device": str(obj.device)}
    if isinstance(obj, np.ndarray):
        return {"type": "np.ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Mapping):
        out = {}
        for k, v in list(obj.items())[:50]:
            out[str(k)] = _summarize(v, max_depth=max_depth, _depth=_depth + 1)
        if len(obj) > 50:
            out["..."] = f"+{len(obj) - 50} keys"
        return out
    if isinstance(obj, Sequence) and not isinstance(obj, (bytes, bytearray)):
        seq = list(obj)
        return [_summarize(v, max_depth=max_depth, _depth=_depth + 1) for v in seq[:20]] + (
            [f"+{len(seq) - 20} items"] if len(seq) > 20 else []
        )
    return type(obj).__name__


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo_id", default="behavior-1k/2025-challenge-demos")
    parser.add_argument("--action_horizon", type=int, default=32)
    parser.add_argument("--action_keys", default="action")
    parser.add_argument("--episodes", default="0:200")
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--modalities", default="rgb")
    parser.add_argument("--tolerance_s", type=float, default=1e-4)
    parser.add_argument("--fine_grained_level", type=int, default=0)
    parser.add_argument("--return_seg_instance", action="store_true")
    parser.add_argument("--train_rgb_type", default="regular")
    parser.add_argument("--skill_list", default="all")
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--test_dataloader", action="store_true")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_batches", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    episodes = _parse_episodes(args.episodes)
    tasks = _parse_csv_list(args.tasks)
    modalities = _parse_csv_list(args.modalities) or ["rgb"]
    action_sequence_keys = tuple(_parse_csv_list(args.action_keys) or ["action"])
    delta_timestamps = {key: [t / 30.0 for t in range(args.action_horizon)] for key in action_sequence_keys}
    shuffle = not args.no_shuffle

    from behavior.learning.datas.dataset import BehaviorLeRobotDataset

    dataset_kwargs = {}
    if args.skill_list and args.skill_list != "all":
        dataset_kwargs["skill_list"] = _parse_csv_list(args.skill_list) or ["all"]

    dataset = BehaviorLeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        tolerance_s=args.tolerance_s,
        tasks=tasks,
        modalities=modalities,
        local_only=True,
        check_files=False,
        check_timestamp_sync=False,
        delta_timestamps=delta_timestamps,
        episodes=episodes,
        chunk_streaming_using_keyframe=True,
        shuffle=shuffle,
        fine_grained_level=args.fine_grained_level,
        return_seg_instance=args.return_seg_instance,
        train_rgb_type=args.train_rgb_type,
        **dataset_kwargs,
    )

    logging.info("Dataset created: %s", dataset)
    try:
        length = len(dataset)
        logging.info("len(dataset)=%d", length)
    except Exception as e:
        logging.exception("len(dataset) failed: %s", e)

    for i in range(args.num_samples):
        t0 = time.time()
        sample = dataset[i]
        dt = time.time() - t0
        logging.info("dataset[%d] loaded in %.3fs", i, dt)
        logging.info("sample[%d] summary: %s", i, _summarize(sample))

    if args.test_dataloader:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            collate_fn=_openpi_data_loader._collate_fn,
        )
        it = iter(loader)
        for b in range(args.num_batches):
            t0 = time.time()
            batch = next(it)
            dt = time.time() - t0
            logging.info("batch %d loaded in %.3fs", b, dt)
            logging.info("batch %d summary: %s", b, _summarize(batch))


if __name__ == "__main__":
    main()
