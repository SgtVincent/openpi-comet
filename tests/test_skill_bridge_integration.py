"""Integration tests exercising real BehaviorLeRobotDataset._get_bridge_subtask_text.

These tests import the actual BehaviorLeRobotDataset class and call its
_get_bridge_subtask_text method on a minimally-configured mock instance.

All heavy imports are deferred to test time with pytest.importorskip so the
file can be collected in any environment; individual tests are skipped when
dependencies aren't available.

Covers:
  - Real unbound _get_bridge_subtask_text method (global→local conversion,
    padding detection, source gating, config gating, episode-end rejection)
  - Bridge config resolution via get_config() and path verification
  - Data-only smoke test (no model loading, just dataset instantiation)
"""
from __future__ import annotations

import os
import types

import pytest

from openpi.training.skill_bridge_config import SkillBridgeConfig


# Local path constants (must match feat-skill-bridge worktree + canonical paths)
_CANONICAL_BASE_CKPT = (
    "/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/openpi-comet/checkpoints/pi05_base_pytorch"
)
_B1K_DATA_ROOT = "/mnt/bn/saiwenresearch/mlx/users/chenjunting/data/2025-challenge-demos/"
_WORKTREE_ROOT = "/mnt/bn/saiwenresearch/mlx/users/chenjunting/repo/feat-skill-bridge"


def _import_dataset_cls():
    """Import BehaviorLeRobotDataset; raise ImportError if not available."""
    from behavior.learning.datas.dataset import BehaviorLeRobotDataset
    return BehaviorLeRobotDataset


def _import_get_config():
    """Import get_config; raise ImportError if not available."""
    from openpi.training.train_config import get_config
    return get_config


# ===========================================================================
# Tests against actual unbound BehaviorLeRobotDataset._get_bridge_subtask_text
# ===========================================================================

class TestRealDatasetBridgeMethod:
    """Tests that call the real unbound _get_bridge_subtask_text method.

    Constructs a mock instance with the same attribute structure as the real
    BehaviorLeRobotDataset, then calls the REAL method via the unbound
    function:
        BehaviorLeRobotDataset._get_bridge_subtask_text(mock_self, item, ...)

    Validates that the actual method code in dataset.py handles:
      - Global → local frame conversion (CRITICAL bug fix from round 2)
      - Non-streaming padding detection from item dict
      - Episode end computation (padded-tail rejection when no pad mask)
      - Source gating (Phase 1: annotations_skill only)
      - Config gating (enabled / None)
    """

    @pytest.fixture
    def mock_ds(self):
        """Mock with same attribute structure as real BehaviorLeRobotDataset.

        Simulates episode 7 starting at global frame 5000 (not the first
        episode in the dataset) to exercise the global→local conversion.
        """
        import torch
        import bisect
        _ = torch, bisect  # ensure available
        pytest.importorskip("behavior.learning.datas.dataset")

        two_skill_segs = [
            (0, 99, "move to the radio"),
            (100, 199, "pick up the radio"),
        ]
        EP_START = 5000   # global frame where episode 7 starts
        EP_LEN = 200      # number of frames in episode 7

        ds = types.SimpleNamespace()
        ds.subtask_source = "annotations_skill"
        ds._skill_bridge_config = SkillBridgeConfig(enabled=True)
        ds._subtask_segments = {7: two_skill_segs}
        ds._subtask_segment_ends = {7: [99, 199]}
        ds.delta_indices = {"action": list(range(32))}
        ds.fps = 30

        # episode_data_index_pos: maps episode_index → positional index
        # (the real one is {ep_idx: i for i, ep_idx in enumerate(self.episodes)})
        ds.episode_data_index_pos = {7: 0}

        # episode_data_index: dict where ["from"][pos] and ["to"][pos] are
        # tensor-like with .item() — matches get_episode_data_index output.
        # Since we only have episode 7 at position 0, we make 1-element tensors.
        class _PosTensor:
            """1-element tensor-like subscriptable by integer position."""
            def __init__(self, values):
                self._values = values
            def __getitem__(self, pos):
                class _Scalar:
                    def __init__(self, v):
                        self._v = v
                    def item(self):
                        return self._v
                return _Scalar(self._values[pos])

        ds.episode_data_index = {
            "from": _PosTensor([EP_START]),
            "to": _PosTensor([EP_START + EP_LEN]),
        }

        # Base _get_subtask_text — same logic as the real one for annotations_skill
        def _mock_get_subtask_text(item):
            ep_idx = (
                int(item["episode_index"].item())
                if hasattr(item["episode_index"], "item")
                else item["episode_index"]
            )
            ts = item["timestamp"]
            ts_val = ts.item() if hasattr(ts, "item") else ts
            frame = round(ts_val * ds.fps)
            segs = ds._subtask_segments.get(ep_idx, [])
            ends = ds._subtask_segment_ends.get(ep_idx, [])
            if not segs:
                return "fallback_task"
            i = bisect.bisect_left(ends, frame)
            if 0 <= i < len(segs):
                s, e, t = segs[i]
                if s <= frame <= e:
                    return t
            return "fallback_task"

        ds._get_subtask_text = _mock_get_subtask_text
        ds._build_subtask_segments_for_episode = lambda ep_idx: (
            ds._subtask_segments.get(ep_idx, []),
            ds._subtask_segment_ends.get(ep_idx, []),
        )

        # _action_horizon — same as real method
        def _action_horizon() -> int:
            if ds.delta_indices is not None:
                for key, deltas in ds.delta_indices.items():
                    if key.endswith("action") or key.startswith("action"):
                        return len(deltas)
            return 0

        ds._action_horizon = _action_horizon
        return ds

    def test_global_to_local_conversion_real_method(self, mock_ds):
        """CRITICAL: Real method converts global HF query indices to episode-local.

        This is the most important test — it proves the actual code in
        dataset.py correctly subtracts the episode start offset.
        """
        import torch
        BehaviorLeRobotDataset = _import_dataset_cls()

        item = {
            "episode_index": torch.tensor(7),
            "timestamp": torch.tensor(85 / 30.0),  # local frame 85
        }
        # Global query indices: episode starts at 5000, local 85 = global 5085
        query_indices = {"action": list(range(5085, 5085 + 32))}
        padding = {"action_is_pad": torch.BoolTensor([False] * 32)}

        result = BehaviorLeRobotDataset._get_bridge_subtask_text(
            mock_ds, item, query_indices=query_indices, padding=padding
        )
        # Local frame 85 in skill_a, boundary at 100 → valid bridge
        assert result == "move to the radio then pick up the radio"

    def test_non_streaming_padding_real_method(self, mock_ds):
        """Real method detects padding from item dict (non-streaming path)."""
        import torch
        BehaviorLeRobotDataset = _import_dataset_cls()

        item = {
            "episode_index": torch.tensor(7),
            "timestamp": torch.tensor(85 / 30.0),
            "action_is_pad": torch.BoolTensor([False] * 30 + [True, True]),
        }
        # No explicit query_indices or padding passed → method reads from item
        result = BehaviorLeRobotDataset._get_bridge_subtask_text(mock_ds, item)
        # Any pad → reject bridge → fallback
        assert result == "move to the radio"

    def test_non_streaming_no_padding_valid_bridge(self, mock_ds):
        """Real method: non-streaming with all-valid steps → bridge works."""
        import torch
        BehaviorLeRobotDataset = _import_dataset_cls()

        item = {
            "episode_index": torch.tensor(7),
            "timestamp": torch.tensor(85 / 30.0),
            "action_is_pad": torch.BoolTensor([False] * 32),
        }
        result = BehaviorLeRobotDataset._get_bridge_subtask_text(mock_ds, item)
        assert result == "move to the radio then pick up the radio"

    def test_disabled_config_real_method(self, mock_ds):
        """Real method: disabled config → same as _get_subtask_text."""
        import torch
        BehaviorLeRobotDataset = _import_dataset_cls()

        mock_ds._skill_bridge_config = SkillBridgeConfig(enabled=False)
        item = {
            "episode_index": torch.tensor(7),
            "timestamp": torch.tensor(85 / 30.0),
        }
        result = BehaviorLeRobotDataset._get_bridge_subtask_text(mock_ds, item)
        assert result == "move to the radio"

    def test_orchestrator_source_real_method(self, mock_ds):
        """Real method: orchestrator source → no bridge even when enabled."""
        import torch
        BehaviorLeRobotDataset = _import_dataset_cls()

        mock_ds.subtask_source = "orchestrator"
        item = {
            "episode_index": torch.tensor(7),
            "timestamp": torch.tensor(85 / 30.0),
        }
        result = BehaviorLeRobotDataset._get_bridge_subtask_text(mock_ds, item)
        assert result == "move to the radio"

    def test_padding_rejects_streaming_real_method(self, mock_ds):
        """Real method: streaming path with any padding → no bridge."""
        import torch
        BehaviorLeRobotDataset = _import_dataset_cls()

        item = {
            "episode_index": torch.tensor(7),
            "timestamp": torch.tensor(85 / 30.0),
        }
        query_indices = {"action": list(range(5085, 5085 + 32))}
        padding = {"action_is_pad": torch.BoolTensor([False] * 31 + [True])}
        result = BehaviorLeRobotDataset._get_bridge_subtask_text(
            mock_ds, item, query_indices=query_indices, padding=padding
        )
        assert result == "move to the radio"

    def test_episode_end_no_pad_real_method(self, mock_ds):
        """Real method: near episode end with no pad mask → padded-tail rejection."""
        import torch
        BehaviorLeRobotDataset = _import_dataset_cls()

        item = {
            "episode_index": torch.tensor(7),
            "timestamp": torch.tensor(190 / 30.0),  # local frame 190
        }
        # No pad mask, no query indices → uses chunk_size + episode_end
        result = BehaviorLeRobotDataset._get_bridge_subtask_text(mock_ds, item)
        # 32 steps from 190 → extends to 221, past episode end (199) → reject
        assert result == "pick up the radio"

    def test_none_config_real_method(self, mock_ds):
        """Real method: config is None → no bridge."""
        import torch
        BehaviorLeRobotDataset = _import_dataset_cls()

        mock_ds._skill_bridge_config = None
        item = {
            "episode_index": torch.tensor(7),
            "timestamp": torch.tensor(85 / 30.0),
        }
        result = BehaviorLeRobotDataset._get_bridge_subtask_text(mock_ds, item)
        assert result == "move to the radio"

    def test_annotations_primitive_real_method(self, mock_ds):
        """Real method: annotations_primitive source → no bridge (Phase 1)."""
        import torch
        BehaviorLeRobotDataset = _import_dataset_cls()

        mock_ds.subtask_source = "annotations_primitive"
        item = {
            "episode_index": torch.tensor(7),
            "timestamp": torch.tensor(85 / 30.0),
        }
        result = BehaviorLeRobotDataset._get_bridge_subtask_text(mock_ds, item)
        assert result == "move to the radio"


# ===========================================================================
# Config path verification tests
# ===========================================================================

_BRIDGE_CONFIGS = [
    "pi05_ki_joint_query_b1k-single_task-radio-ki_on_skillbridge_fp32",
    "pi05_ki_joint_query_b1k-single_task-radio-ki_on_skillbridge_bf16",
]


class TestBridgeConfigPaths:
    """Resolve bridge training config via get_config and verify all paths.

    Note: cfg.data is a LeRobotB1KDataConfig *factory*, not a flat DataConfig.
    The skill_bridge field lives at cfg.data.base_config.skill_bridge.
    """

    @pytest.mark.parametrize("config_name", _BRIDGE_CONFIGS)
    def test_config_resolves(self, config_name):
        """Config name resolves without error."""
        get_config = _import_get_config()
        cfg = get_config(config_name)
        assert cfg is not None
        assert cfg.name == config_name

    @pytest.mark.parametrize("config_name", _BRIDGE_CONFIGS)
    def test_skill_bridge_enabled_in_config(self, config_name):
        """skill_bridge.enabled is True in bridge configs."""
        get_config = _import_get_config()
        cfg = get_config(config_name)
        # cfg.data is a list of DataConfigFactory objects
        # LeRobotB1KDataConfig wraps base_config which has skill_bridge
        assert len(cfg.data) > 0, "cfg.data is empty"
        data_cfg = cfg.data[0]
        bridge_cfg = getattr(data_cfg.base_config, "skill_bridge", None)
        assert bridge_cfg is not None, "skill_bridge not on base_config"
        assert getattr(bridge_cfg, "enabled", False) is True, (
            "skill_bridge.enabled is not True"
        )

    def test_baseline_config_still_disabled(self):
        """Non-bridge config still has skill_bridge disabled (no global change)."""
        get_config = _import_get_config()
        cfg = get_config("pi05_ki_joint_query_b1k-single_task-radio-ki_on_fp32")
        data_cfg = cfg.data[0]
        bridge_cfg = getattr(data_cfg.base_config, "skill_bridge", None)
        assert bridge_cfg is not None, "skill_bridge not on base_config"
        assert getattr(bridge_cfg, "enabled", True) is False, (
            "Baseline config should have skill_bridge disabled"
        )

    @pytest.mark.parametrize("config_name", _BRIDGE_CONFIGS)
    def test_data_paths_exist(self, config_name):
        """Dataset root exists on disk."""
        get_config = _import_get_config()
        cfg = get_config(config_name)
        data_cfg = cfg.data[0]
        data_root = getattr(data_cfg.base_config, "behavior_dataset_root", None)
        assert data_root is not None, "behavior_dataset_root not set"
        assert os.path.exists(data_root), f"Dataset root not found: {data_root}"

    @pytest.mark.parametrize("config_name", _BRIDGE_CONFIGS)
    def test_assets_dir_exists(self, config_name):
        """Assets dir exists on disk."""
        get_config = _import_get_config()
        cfg = get_config(config_name)
        data_cfg = cfg.data[0]
        assets = getattr(data_cfg, "assets", None)
        assert assets is not None, "assets not on data config"
        assets_dir = getattr(assets, "assets_dir", None)
        assert assets_dir is not None, "assets_dir not set"
        assert os.path.exists(assets_dir), f"Assets dir not found: {assets_dir}"

    @pytest.mark.parametrize("config_name", _BRIDGE_CONFIGS)
    def test_base_checkpoint_exists(self, config_name):
        """Base checkpoint (pytorch_weight_path) exists on disk."""
        get_config = _import_get_config()
        cfg = get_config(config_name)
        ckpt_path = getattr(cfg, "pytorch_weight_path", None)
        assert ckpt_path is not None, "pytorch_weight_path not set"
        assert os.path.exists(ckpt_path), f"Base checkpoint not found: {ckpt_path}"
        # Check that model weights exist inside
        safetensors_path = os.path.join(ckpt_path, "model.safetensors")
        assert os.path.exists(safetensors_path), (
            f"model.safetensors not found in {ckpt_path}"
        )

    @pytest.mark.parametrize("config_name", _BRIDGE_CONFIGS)
    def test_output_dirs_exist(self, config_name):
        """Output dirs (checkpoint/log/assets base dirs) exist on disk."""
        get_config = _import_get_config()
        cfg = get_config(config_name)
        for attr in ["checkpoint_base_dir", "log_base_dir", "assets_base_dir"]:
            path = getattr(cfg, attr, None)
            if path is not None:
                # The actual run dir may not exist yet, but its parent should
                parent = os.path.dirname(path) if os.path.dirname(path) else path
                assert os.path.exists(parent), (
                    f"{attr} parent not found: {parent}"
                )


# ===========================================================================
# Data-only smoke test (no model loading)
# ===========================================================================

class TestDatasetDataOnlySmoke:
    """Data-only smoke test: create dataset with skill bridge config.

    No model loading, no GPU.  Just verifies that:
      1. Dataset can be instantiated with skill bridge config
      2. __getitem__ returns items with subtask_text
    """

    def test_dataset_instantiation_with_bridge(self):
        """Dataset can be created with skill bridge via create_behavior_dataset.

        create_behavior_dataset takes a flat DataConfig (not a factory).
        We construct one with all fields the dataset constructor expects.
        """
        pytest.importorskip("behavior.learning.datas.dataset")
        from openpi.training.behavior_dataset import create_behavior_dataset
        from openpi.training.data_config import DataConfig
        from openpi.training.skill_bridge_config import SkillBridgeConfig

        subtask_templates = os.path.join(
            _WORKTREE_ROOT,
            "src/behavior/learning/datas/b1k_subtask_phrase_templates.json",
        )
        object_mapping = os.path.join(
            _WORKTREE_ROOT,
            "src/behavior/learning/datas/b1k_object_id_name_mapping.json",
        )
        # Build a flat DataConfig with all fields create_behavior_dataset reads
        tiny_config = DataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            prompt_from_task=True,
            tasks=["turning_on_radio"],
            episodes_index=[0],
            behavior_dataset_root=_B1K_DATA_ROOT,
            fine_grained_level=0,
            subtask_source="annotations_skill",
            subtask_template_path=subtask_templates,
            subtask_object_name_mapping_path=object_mapping,
            subtask_joiner=" then ",
            skill_bridge=SkillBridgeConfig(enabled=True),
            action_sequence_keys=("action",),
        )
        dataset = create_behavior_dataset(tiny_config, action_horizon=32)
        assert dataset is not None
        assert hasattr(dataset, "_skill_bridge_config")
        assert dataset._skill_bridge_config.enabled is True

    def test_dataset_sample_has_subtask_text(self):
        """First sample has subtask_text field with bridge-aware content."""
        pytest.importorskip("behavior.learning.datas.dataset")
        from openpi.training.behavior_dataset import create_behavior_dataset
        from openpi.training.data_config import DataConfig
        from openpi.training.skill_bridge_config import SkillBridgeConfig

        subtask_templates = os.path.join(
            _WORKTREE_ROOT,
            "src/behavior/learning/datas/b1k_subtask_phrase_templates.json",
        )
        object_mapping = os.path.join(
            _WORKTREE_ROOT,
            "src/behavior/learning/datas/b1k_object_id_name_mapping.json",
        )
        tiny_config = DataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            prompt_from_task=True,
            tasks=["turning_on_radio"],
            episodes_index=[0],
            behavior_dataset_root=_B1K_DATA_ROOT,
            fine_grained_level=0,
            subtask_source="annotations_skill",
            subtask_template_path=subtask_templates,
            subtask_object_name_mapping_path=object_mapping,
            subtask_joiner=" then ",
            skill_bridge=SkillBridgeConfig(enabled=True),
            action_sequence_keys=("action",),
        )
        dataset = create_behavior_dataset(tiny_config, action_horizon=32)
        item = dataset[0]
        assert "subtask_text" in item
        assert item["subtask_text"] is not None
        assert len(item["subtask_text"]) > 0
