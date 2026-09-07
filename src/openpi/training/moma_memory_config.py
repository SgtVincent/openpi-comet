"""MoMA-VLA HeldMemory-K training config (fixed compact Memory dataset).

Why this file exists
--------------------
Everything needed to train on the fixed-compact-Memory dataset was implemented
-- the dataset lookup, the repack/forwarding path, the `require_memory`
fail-closed switch, the anchor schedule -- but **no TrainConfig selected it**.
`subtask_source` is what drives every one of those switches, and no registered
config set it to the memory source, so the entire memory path was unreachable
from any launchable entry point.  That is the same "implemented but no consumer"
shape as the breaks this work started from, one level up.

Budgets are hardcoded here on purpose
-------------------------------------
`subtask_max_len` and `max_token_len` were deliberately left with no defaults
while the numbers were undecided, so that a run could not start on a guess.
They are now decided (192 / 320) and set explicitly:

* Memory segment: measured production max is 105 tokens, so 192 leaves 87
  tokens of headroom.  128 would also have fit today, but the overflow path
  truncates the LAST fields (`Next skill` / `Next primitive`), which looks like
  "the model is bad at predicting next-skill" rather than like truncation.
* Prompt segment: the `State:` block is 23 discretised joint values, not 32.
  32 is the model's padded action width, and `PadStatesAndActions` runs AFTER
  tokenization, so those 9 padding slots never reach the prompt text.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import openpi.models.pi05_subtask_config as pi05_subtask_config
import openpi.training.optimizer as _optimizer
from openpi.training.data_config import AssetsConfig, DataConfig, LeRobotB1KDataConfig
from openpi.training.memory_annotation import MEMORY_SUBTASK_SOURCE

if TYPE_CHECKING:
    from openpi.training.train_config import TrainConfig

#: Root of the dataset whose `derived/fixed_compact_memory_annotations` subtree
#: holds the Memory intervals.  The dataset resolves that subpath itself.
MEMORY_DATA_ROOT = "/mnt/bn/behavior-data-hl/chenjunting/data/2025-challenge-demos"

#: Absolute, deliberately.  The sibling configs use the relative
#: "checkpoints/pi05_base_pytorch", which only resolves when the process happens
#: to run from the main tree.  From anywhere else `assets.assets_dir` misses, and
#: `_load_norm_stats` logs at INFO and returns None (data_config.py:394-396).
#: The consumer does raise (data_loader.py:201), so this fails loudly rather than
#: training unnormalised -- but it fails at a point that says "run
#: compute_norm_stats" rather than "your assets path depended on the cwd".
_PI05_BASE_CKPT = "/mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet/checkpoints/pi05_base_pytorch"

#: Decided budgets.  See module docstring for the measurements behind them.
MEMORY_SUBTASK_MAX_LEN = 192
MEMORY_PROMPT_MAX_LEN = 320


def make_memory_data_config(
    *,
    episodes_index: list[int] | None = None,
    data_root: str = MEMORY_DATA_ROOT,
) -> LeRobotB1KDataConfig:
    """Data config that routes the dataset through the Memory source.

    `subtask_source` is the single switch the whole memory path keys off: the
    dataset attaches `memory_text`/`previous_memory_text`, the repack allowlist
    carries them, `B1kInputs` forwards them, and `require_memory` fails closed if
    they are missing.  Setting it anywhere other than here would leave those four
    consumers disagreeing about whether this is a memory run.
    """
    return LeRobotB1KDataConfig(
        repo_id="behavior-1k/2025-challenge-demos",
        assets=AssetsConfig(
            assets_dir=f"{_PI05_BASE_CKPT}/assets",
            asset_id="behavior-1k/2025-challenge-demos",
        ),
        base_config=DataConfig(
            prompt_from_task=True,
            episodes_index=episodes_index,
            behavior_dataset_root=data_root,
            fine_grained_level=0,
            subtask_source=MEMORY_SUBTASK_SOURCE,
            # Deliberately no subtask_template_path / object_name_mapping: the
            # Memory text is generated offline, so those assets are never read.
            # Passing them would make them look configured while being ignored.
        ),
    )


def make_memory_train_config(
    *,
    name: str,
    planner_stride: int = 5,
    planner_stride_weights: tuple[tuple[int, float], ...] | None = None,
    num_train_steps: int = 30_000,
    peak_lr: float = 1e-4,
    episodes_index: list[int] | None = None,
    data_root: str = MEMORY_DATA_ROOT,
    ce_weight: float = 1.0,
    alpha: float = 10.0,
) -> "TrainConfig":
    """Build a MoMA-VLA memory TrainConfig.

    `planner_stride_weights` exists so that the mixed-K sampling arm (K drawn
    per sample from a weighted set) does not require reshaping this signature
    later.  It is validated but NOT yet implemented: passing it raises rather
    than quietly training single-K, because a silently ignored sweep parameter
    produces a run that looks like the mixed arm and is not.
    """
    model = pi05_subtask_config.Pi05SubtaskConfig(
        subtask_max_len=MEMORY_SUBTASK_MAX_LEN,
        max_token_len=MEMORY_PROMPT_MAX_LEN,
        action_horizon=32,
        planner_stride=planner_stride,
        ce_weight=ce_weight,
        alpha=alpha,
    )
    # Validate the stride through the same guards the schedule uses, at config
    # build time -- an invalid K must not survive until the first anchor
    # decision, where `-1 % 5 == 4` would yield a plausible-looking schedule.
    from openpi.training.memory_anchor import validate_planner_stride_spec

    validate_planner_stride_spec(planner_stride, planner_stride_weights)

    # Imported here, not at module scope: train_config imports this module at its
    # own bottom to register these configs, so a module-level import would be a
    # cycle whose symptom (ImportError on a "partially initialized module")
    # depends on which module the process happens to import first.
    from openpi.training.train_config import TrainConfig

    return TrainConfig(
        name=name,
        exp_name=name,
        project_name="moma_vla",
        # Model factory dispatch uses the established name "subtask"
        # (model.py:359 and the other Pi05SubtaskConfig TrainConfigs). The
        # ModelType value is "pi05_subtask", but pytorch_model_name is a separate
        # registry key; using that value here would silently fall through to the
        # base PI0Pytorch constructor.
        pytorch_model_name="subtask",
        model=model,
        data=make_memory_data_config(episodes_index=episodes_index, data_root=data_root),
        pytorch_weight_path=_PI05_BASE_CKPT,
        num_train_steps=num_train_steps,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=peak_lr,
            decay_steps=num_train_steps,
        ),
        pytorch_training_precision="bfloat16",
        accelerate_mixed_precision="bf16",
        ema_decay=None,
        wandb_enabled=False,
    )


def memory_configs() -> "tuple[TrainConfig, ...]":
    """Build the registered MoMA memory configs.

    A function rather than a module-level tuple so that importing this module
    does not require ``train_config`` to be fully initialised. The smoke variant
    exists so the wiring can be exercised on a couple of episodes without
    committing a full run.
    """
    return (
        make_memory_train_config(
            name="pi05_moma_memory_b1k-k5_smoke",
            planner_stride=5,
            num_train_steps=200,
            episodes_index=list(range(2)),
        ),
        make_memory_train_config(
            name="pi05_moma_memory_b1k-k5",
            planner_stride=5,
        ),
    )
