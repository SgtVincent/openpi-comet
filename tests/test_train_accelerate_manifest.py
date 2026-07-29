"""Tests for _compute_data_manifest probe_batches=0 metadata-only mode.

Verifies that:
1. probe_batches=0 skips loader iteration entirely (no data loading overhead)
2. Metadata fields are still populated from config/data_config
3. probe_skipped flag is set correctly
4. probe_batches>0 still works and actually iterates the loader
5. val_loader handling works in both modes
"""

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Module import: scripts/train_accelerate.py is a script, not a package module.
# We load it via importlib so we can test its pure functions.
# ---------------------------------------------------------------------------
def _load_train_accelerate():
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    script_path = repo_root / "scripts" / "train_accelerate.py"
    spec = importlib.util.spec_from_file_location("train_accelerate_script", str(script_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_accelerate_script"] = mod
    spec.loader.exec_module(mod)
    return mod


train_accel = None


def _get_module():
    global train_accel
    if train_accel is None:
        train_accel = _load_train_accelerate()
    return train_accel


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------
class MockDataConfig:
    """Minimal mock for DataConfig with attributes used by _compute_data_manifest."""

    def __init__(self, repo_id="test/repo", tasks=None, modalities=None,
                 episodes_index=None, fine_grained_level=0, subtask_source=None,
                 prompt_from_task=False, fps=30, norm_stats=None):
        self.repo_id = repo_id
        self.tasks = tasks or ["turning_on_radio"]
        self.modalities = modalities or ["image"]
        self.episodes_index = episodes_index or list(range(10))
        self.fine_grained_level = fine_grained_level
        self.subtask_source = subtask_source
        self.prompt_from_task = prompt_from_task
        self.fps = fps
        self.norm_stats = norm_stats


class MockConfig:
    """Minimal mock for TrainConfig with attributes used by _compute_data_manifest."""

    def __init__(self, seed=42, batch_size=32, num_train_steps=1000,
                 action_horizon=10):
        self.seed = seed
        self.batch_size = batch_size
        self.num_train_steps = num_train_steps

        class _ModelCfg:
            pass

        _ModelCfg.action_horizon = action_horizon
        self.model = _ModelCfg()


class CountingIterableLoader:
    """A loader-like object that counts how many times it's iterated.

    Records the number of times __iter__ is called and how many items
    were consumed. This lets us verify that probe_batches=0 never
    touches the loader.
    """

    def __init__(self, length=100, batch_size=4):
        self._length = length
        self._batch_size = batch_size
        self.iter_count = 0
        self.items_consumed = 0

    def __len__(self):
        return self._length

    def __iter__(self):
        self.iter_count += 1
        bs = self._batch_size

        class _Counter:
            def __init__(self2, total, on_consume):
                self2._total = total
                self2._consumed = 0
                self2._on_consume = on_consume

            def __iter__(self2):
                return self2

            def __next__(self2):
                if self2._consumed >= self2._total:
                    raise StopIteration
                self2._consumed += 1
                self2._on_consume()
                # Return minimal (observation, actions) tuple that won't crash probing
                import torch
                obs = {
                    "state": torch.zeros(bs, 7),
                    "images": {"head": torch.zeros(bs, 3, 224, 224)},
                }
                actions = torch.zeros(bs, 10, 7)  # [B, horizon, action_dim]
                return obs, actions

        def _tick():
            self.items_consumed += 1

        return _Counter(self._length, _tick)


# ---------------------------------------------------------------------------
# Tests: probe_batches=0 (metadata-only mode)
# ---------------------------------------------------------------------------
class TestProbeBatchesZero:
    """Verify that probe_batches=0 skips all loader iteration."""

    def test_probe_skipped_flag_true(self):
        mod = _get_module()
        cfg = MockConfig()
        dcfg = MockDataConfig()
        loader = CountingIterableLoader(length=100, batch_size=4)

        manifest = mod._compute_data_manifest(
            config=cfg,
            data_config=dcfg,
            train_loader=loader,
            val_loader=None,
            val_data_config=None,
            steps_per_epoch=25,
            world_size=1,
            grad_accum_steps=1,
            seed=42,
            num_probe_batches=0,
        )

        assert manifest["probe_skipped"] is True

    def test_no_loader_iteration(self):
        mod = _get_module()
        cfg = MockConfig()
        dcfg = MockDataConfig()
        loader = CountingIterableLoader(length=100, batch_size=4)

        mod._compute_data_manifest(
            config=cfg,
            data_config=dcfg,
            train_loader=loader,
            val_loader=None,
            val_data_config=None,
            steps_per_epoch=25,
            world_size=1,
            grad_accum_steps=1,
            seed=42,
            num_probe_batches=0,
        )

        # The loader should never have been iterated (len() only, not __iter__)
        assert loader.iter_count == 0, (
            f"probe_batches=0 should never iterate the loader, "
            f"but __iter__ was called {loader.iter_count} time(s)"
        )
        assert loader.items_consumed == 0

    def test_metadata_fields_present(self):
        """All metadata fields should be populated even without probing."""
        mod = _get_module()
        cfg = MockConfig()
        dcfg = MockDataConfig(
            repo_id="test/b1k",
            tasks=["task_a", "task_b"],
            modalities=["image", "state"],
            episodes_index=list(range(50)),
        )
        loader = CountingIterableLoader(length=200, batch_size=4)

        manifest = mod._compute_data_manifest(
            config=cfg,
            data_config=dcfg,
            train_loader=loader,
            val_loader=None,
            val_data_config=None,
            steps_per_epoch=50,
            world_size=2,
            grad_accum_steps=2,
            seed=42,
            num_probe_batches=0,
        )

        # Top-level metadata
        assert "generated_at" in manifest
        assert "generated_at_iso" in manifest
        assert manifest["action_horizon"] == 10

        # Episode / task metadata
        assert manifest["n_train_episodes"] == 50
        assert manifest["train_tasks"] == ["task_a", "task_b"]
        assert manifest["train_repo_id"] == "test/b1k"
        assert manifest["train_modalities"] == ["image", "state"]

        # Loader length (len() doesn't iterate)
        assert manifest["n_train_microbatches_per_rank"] == 200
        assert manifest["n_train_microbatches_total_estimate"] == 400  # 200 * 2
        assert manifest["steps_per_epoch"] == 50
        assert manifest["world_size"] == 2
        assert manifest["grad_accum_steps"] == 2

        # FPS
        assert manifest["fps"] == 30

        # Data fingerprint
        assert "data_sha" in manifest
        assert "sha256" in manifest["data_sha"]

        # Streaming semantics
        assert "streaming_dataset" in manifest
        assert manifest["streaming_dataset"]["is_streaming"] is True

        # Val metadata
        assert manifest["has_val_data"] is False
        assert manifest["val_probe"] is None

    def test_probe_fields_are_none_or_empty(self):
        """Probe-derived fields should be None/empty when probe is skipped."""
        mod = _get_module()
        cfg = MockConfig()
        dcfg = MockDataConfig()
        loader = CountingIterableLoader(length=100, batch_size=4)

        manifest = mod._compute_data_manifest(
            config=cfg,
            data_config=dcfg,
            train_loader=loader,
            val_loader=None,
            val_data_config=None,
            steps_per_epoch=25,
            world_size=1,
            grad_accum_steps=1,
            seed=42,
            num_probe_batches=0,
        )

        # Action/state dims come from probing only
        assert manifest["action_dim"] is None
        assert manifest["state_dim"] is None
        assert manifest["image_keys"] == []
        assert manifest["has_subtask_tokens"] is False
        assert manifest["subtask_max_len"] is None

        # Sample count estimates require batch_size from probe
        assert manifest["n_train_samples_per_rank_estimate"] is None
        assert manifest["n_train_samples_total_estimate"] is None
        assert manifest["n_train_frames"] is None

        # Train probe dict structure
        train_probe = manifest["train_probe"]
        assert train_probe["num_batches_requested"] == 0
        assert train_probe["num_batches_sampled"] == 0
        assert train_probe["probe_skipped"] is True
        assert train_probe["action_dim"] is None
        assert train_probe["state_dim"] is None
        assert train_probe["image_keys"] == []
        assert train_probe["batch_size"] is None

    def test_negative_probe_batches_also_skips(self):
        """Negative probe_batches should also be treated as skip."""
        mod = _get_module()
        cfg = MockConfig()
        dcfg = MockDataConfig()
        loader = CountingIterableLoader(length=100, batch_size=4)

        manifest = mod._compute_data_manifest(
            config=cfg,
            data_config=dcfg,
            train_loader=loader,
            val_loader=None,
            val_data_config=None,
            steps_per_epoch=25,
            world_size=1,
            grad_accum_steps=1,
            seed=42,
            num_probe_batches=-1,
        )

        assert manifest["probe_skipped"] is True
        assert loader.iter_count == 0

    def test_val_loader_also_skipped(self):
        """With probe_batches=0, val loader should also not be iterated."""
        mod = _get_module()
        cfg = MockConfig()
        dcfg = MockDataConfig()
        val_dcfg = MockDataConfig(repo_id="test/val", tasks=["task_c"])
        train_loader = CountingIterableLoader(length=100, batch_size=4)
        val_loader = CountingIterableLoader(length=20, batch_size=2)

        manifest = mod._compute_data_manifest(
            config=cfg,
            data_config=dcfg,
            train_loader=train_loader,
            val_loader=val_loader,
            val_data_config=val_dcfg,
            steps_per_epoch=25,
            world_size=1,
            grad_accum_steps=1,
            seed=42,
            num_probe_batches=0,
        )

        assert train_loader.iter_count == 0
        assert val_loader.iter_count == 0

        # Val metadata should still be populated
        assert manifest["has_val_data"] is True
        assert manifest["n_val_episodes"] == 10
        assert manifest["val_tasks"] == ["task_c"]
        assert manifest["val_repo_id"] == "test/val"
        assert manifest["n_val_microbatches_per_rank"] == 20

        # Val probe should be skipped
        val_probe = manifest["val_probe"]
        assert val_probe is not None
        assert val_probe["probe_skipped"] is True
        assert val_probe["num_batches_sampled"] == 0


# ---------------------------------------------------------------------------
# Tests: probe_batches > 0 (normal mode still works)
# ---------------------------------------------------------------------------
class TestProbeBatchesPositive:
    """Verify that probe_batches>0 still iterates and populates probe fields."""

    def test_probe_skipped_flag_false(self):
        mod = _get_module()
        cfg = MockConfig()
        dcfg = MockDataConfig()
        loader = CountingIterableLoader(length=100, batch_size=4)

        manifest = mod._compute_data_manifest(
            config=cfg,
            data_config=dcfg,
            train_loader=loader,
            val_loader=None,
            val_data_config=None,
            steps_per_epoch=25,
            world_size=1,
            grad_accum_steps=1,
            seed=42,
            num_probe_batches=3,
        )

        assert manifest["probe_skipped"] is False

    def test_loader_is_iterated(self):
        mod = _get_module()
        cfg = MockConfig()
        dcfg = MockDataConfig()
        loader = CountingIterableLoader(length=100, batch_size=4)

        mod._compute_data_manifest(
            config=cfg,
            data_config=dcfg,
            train_loader=loader,
            val_loader=None,
            val_data_config=None,
            steps_per_epoch=25,
            world_size=1,
            grad_accum_steps=1,
            seed=42,
            num_probe_batches=3,
        )

        assert loader.iter_count == 1
        assert loader.items_consumed == 3

    def test_probe_fields_populated(self):
        mod = _get_module()
        cfg = MockConfig()
        dcfg = MockDataConfig()
        loader = CountingIterableLoader(length=100, batch_size=4)

        manifest = mod._compute_data_manifest(
            config=cfg,
            data_config=dcfg,
            train_loader=loader,
            val_loader=None,
            val_data_config=None,
            steps_per_epoch=25,
            world_size=1,
            grad_accum_steps=1,
            seed=42,
            num_probe_batches=2,
        )

        train_probe = manifest["train_probe"]
        assert train_probe["num_batches_requested"] == 2
        assert train_probe["num_batches_sampled"] == 2
        assert "probe_skipped" not in train_probe or train_probe.get("probe_skipped") is not True

    def test_val_loader_probe(self):
        mod = _get_module()
        cfg = MockConfig()
        dcfg = MockDataConfig()
        val_dcfg = MockDataConfig()
        train_loader = CountingIterableLoader(length=100, batch_size=4)
        val_loader = CountingIterableLoader(length=20, batch_size=2)

        manifest = mod._compute_data_manifest(
            config=cfg,
            data_config=dcfg,
            train_loader=train_loader,
            val_loader=val_loader,
            val_data_config=val_dcfg,
            steps_per_epoch=25,
            world_size=1,
            grad_accum_steps=1,
            seed=42,
            num_probe_batches=2,
        )

        assert val_loader.iter_count == 1
        val_probe = manifest["val_probe"]
        assert val_probe["num_batches_sampled"] == 2


# ---------------------------------------------------------------------------
# Tests: Runtime config validation
# ---------------------------------------------------------------------------
def _runtime_config(**overrides):
    values = {
        "prepare_hf_cache_only": False,
        "force_load_cache": False,
        "log_interval": 10,
        "val_log_interval": 100,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestRuntimeConfigValidation:
    """Verify the actual fail-fast checks used by ``train_loop``."""

    def test_mutual_exclusion_raises_value_error(self):
        mod = _get_module()
        config = _runtime_config(prepare_hf_cache_only=True, force_load_cache=True)

        with pytest.raises(ValueError, match="mutually exclusive"):
            mod._validate_runtime_config(config)

    @pytest.mark.parametrize(
        ("prepare_only", "force_load"),
        [(True, False), (False, True), (False, False)],
    )
    def test_either_cache_mode_alone_is_ok(self, prepare_only, force_load):
        mod = _get_module()
        config = _runtime_config(
            prepare_hf_cache_only=prepare_only,
            force_load_cache=force_load,
        )

        mod._validate_runtime_config(config)

    def test_error_message_mentions_both_modes(self):
        mod = _get_module()
        config = _runtime_config(prepare_hf_cache_only=True, force_load_cache=True)

        with pytest.raises(ValueError) as exc_info:
            mod._validate_runtime_config(config)

        msg = str(exc_info.value)
        assert "prepare_hf_cache_only" in msg
        assert "force_load_cache" in msg
        assert "builds" in msg.lower()
        assert "missing" in msg.lower()

    @pytest.mark.parametrize("log_interval", [0, -1])
    def test_nonpositive_log_interval_is_rejected(self, log_interval):
        mod = _get_module()

        with pytest.raises(ValueError, match="log-interval"):
            mod._validate_runtime_config(_runtime_config(log_interval=log_interval))

    @pytest.mark.parametrize("val_log_interval", [0, -1])
    def test_nonpositive_val_log_interval_is_rejected(self, val_log_interval):
        mod = _get_module()

        with pytest.raises(ValueError, match="val-log-interval"):
            mod._validate_runtime_config(_runtime_config(val_log_interval=val_log_interval))

    def test_log_interval_one_is_valid(self):
        mod = _get_module()

        mod._validate_runtime_config(_runtime_config(log_interval=1, val_log_interval=1))


class _CacheAccelerator:
    def __init__(self, *, num_processes=1, is_main_process=True, is_local_main_process=True):
        self.num_processes = num_processes
        self.is_main_process = is_main_process
        self.is_local_main_process = is_local_main_process
        self.wait_calls = 0

    def wait_for_everyone(self):
        self.wait_calls += 1


def _cache_config(tmp_path, *, force_load_cache):
    checkpoint_root = tmp_path / "checkpoints"
    return SimpleNamespace(
        checkpoint_base_dir=checkpoint_root,
        checkpoint_dir=checkpoint_root / "experiment",
        force_load_cache=force_load_cache,
    )


@pytest.mark.parametrize("force_load_cache", [False, True])
def test_configure_hf_cache_only_exports_dataset_path_without_filesystem_touch(
    tmp_path,
    monkeypatch,
    force_load_cache,
):
    mod = _get_module()
    config = _cache_config(tmp_path, force_load_cache=force_load_cache)
    missing_cache = tmp_path / "missing-hf-datasets-cache"
    config.checkpoint_base_dir = missing_cache
    config.checkpoint_dir = missing_cache / "experiment"
    missing_cache_absolute = Path(os.path.abspath(missing_cache))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(missing_cache))
    # Deliberately overlap HF_HOME: configure must not indirectly create the
    # dataset root or descendants through model-cache parent creation either.
    monkeypatch.setenv("HF_HOME", str(missing_cache))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    accelerator = _CacheAccelerator()

    def guard_dataset_operation(name, original):
        def guarded(path, *args, **kwargs):
            absolute = Path(os.path.abspath(path))
            if absolute == missing_cache_absolute or missing_cache_absolute in absolute.parents:
                raise AssertionError(f"configure_hf_cache called Path.{name} on dataset tree: {path}")
            return original(path, *args, **kwargs)

        return guarded

    for method_name in ("exists", "is_dir", "mkdir", "stat"):
        original = getattr(Path, method_name)
        monkeypatch.setattr(Path, method_name, guard_dataset_operation(method_name, original))

    mod.configure_hf_cache(config, accelerator=accelerator)

    assert os.environ["HF_DATASETS_CACHE"] == str(missing_cache)
    assert not os.path.lexists(missing_cache)
    assert not os.path.lexists(config.checkpoint_base_dir)
    assert accelerator.wait_calls == 0


def test_configure_hf_cache_source_defers_dataset_filesystem_setup():
    source = Path(_get_module().__file__).read_text()
    configure_source = source[
        source.index("def configure_hf_cache("):source.index("def init_wandb(")
    ]

    for forbidden in (
        "datasets_cache.exists(",
        "datasets_cache.is_dir(",
        "datasets_cache.mkdir(",
        "datasets_cache.stat(",
        "datasets_cache.resolve(",
        "accelerator.wait_for_everyone(",
    ):
        assert forbidden not in configure_source


def test_default_hf_datasets_cache_is_tmp_rank_consistent_and_not_checkpoint_or_log(tmp_path, monkeypatch):
    mod = _get_module()
    checkpoint_root = tmp_path / "nas-checkpoints"
    log_root = tmp_path / "nas-logs"
    config = SimpleNamespace(
        checkpoint_base_dir=checkpoint_root,
        checkpoint_dir=checkpoint_root / "experiment",
        log_dir=log_root / "experiment",
    )
    monkeypatch.setenv("OPENPI_HF_CACHE_RUN_ID", "shared-distributed-run")

    default_hf_datasets_cache = vars(mod)["_default_hf_datasets_cache"]
    paths = [
        default_hf_datasets_cache(
            config,
            accelerator=_CacheAccelerator(num_processes=32, is_main_process=rank == 0),
        )
        for rank in (0, 7, 8, 31)
    ]

    assert len(set(paths)) == 1
    cache_path = paths[0]
    assert cache_path.is_relative_to(Path("/tmp"))
    assert not cache_path.is_relative_to(checkpoint_root)
    assert not cache_path.is_relative_to(log_root)
    assert str(checkpoint_root) not in str(cache_path)
    assert str(log_root) not in str(cache_path)
