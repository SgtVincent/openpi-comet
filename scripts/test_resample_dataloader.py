from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        type=str,
        default="pi05_subtask_b1k-make_pizza_ann-skill_lr1e-4_5ep_sft,pi05_subtask_b1k-make_pizza_ann-skill_lr1e-4_5ep_sft_resample",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _configure_env() -> None:
    os.environ.setdefault("OPENPI_DATA_HOME", str(REPO_ROOT / ".cache" / "openpi"))
    os.environ.setdefault("OPENPI_DOWNLOAD_TIMEOUT_S", "60")


def _iter_one_batch(cfg_name: str, *, batch_size: int, seed: int) -> None:
    from openpi.training.data_loader import create_data_loader
    from openpi.training.train_config import get_config

    cfg = get_config(cfg_name)
    cfg = dataclasses.replace(
        cfg,
        num_workers=0,
        batch_size=int(batch_size),
        batch_size_per_gpu=int(batch_size),
        seed=int(seed),
    )
    dl = create_data_loader(cfg, shuffle=True, num_batches=1, skip_norm_stats=True, framework="pytorch")
    obs, actions = next(iter(dl))
    obs_dict = obs.to_dict()

    print("config", cfg_name)
    print("obs_keys_head", sorted(obs_dict.keys())[:20])
    print("has_subtask_tokens", obs_dict.get("subtask_tokens") is not None)
    if hasattr(actions, "keys"):
        print("actions_keys", sorted(actions.keys()))
        print("action_shapes", {k: getattr(v, "shape", None) for k, v in actions.items()})
    else:
        print("actions_type", type(actions).__name__)
        print("actions_shape", getattr(actions, "shape", None))


def main() -> None:
    _configure_env()
    args = _parse_args()
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    if not configs:
        raise ValueError("No configs provided")
    for c in configs:
        _iter_one_batch(c, batch_size=int(args.batch_size), seed=int(args.seed))


if __name__ == "__main__":
    main()
