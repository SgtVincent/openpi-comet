"""Focused CPU tests for node-local Hugging Face cache synchronization."""

import ast
from concurrent.futures import ThreadPoolExecutor
import contextlib
from datetime import timedelta
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import socket
import sys
import threading
import time
import traceback
import uuid

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
# Ensure imports resolve to this worktree instead of an editable installation
# from another checkout.
sys.path.insert(0, str(_REPO_ROOT / "src"))

from behavior.learning.datas import hf_cache_sync as sync

_DATASET_SOURCE = _REPO_ROOT / "src/behavior/learning/datas/dataset.py"
_TRAIN_ACCELERATE_SOURCE = _REPO_ROOT / "scripts/train_accelerate.py"


class _FakeStore:
    def __init__(self):
        self._values: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: str | bytes) -> None:
        encoded = value.encode("utf-8") if isinstance(value, str) else value
        with self._lock:
            self._values[key] = encoded

    def get(self, key: str) -> bytes:
        with self._lock:
            return self._values[key]

    def check(self, keys: list[str]) -> bool:
        with self._lock:
            return all(key in self._values for key in keys)

    def compare_set(self, key: str, expected: str | bytes, desired: str | bytes) -> bytes:
        expected_bytes = expected.encode("utf-8") if isinstance(expected, str) else expected
        desired_bytes = desired.encode("utf-8") if isinstance(desired, str) else desired
        with self._lock:
            current = self._values.get(key, b"")
            if current == expected_bytes:
                self._values[key] = desired_bytes
            return self._values.get(key, current)


def _dummy_manifest() -> dict[str, object]:
    return {"manifest_version": 1, "dataset_fingerprint": "test", "arrow_files": [], "artifacts": []}


def _make_sources(tmp_path: Path, count: int = 3) -> tuple[Path, list[str]]:
    root = tmp_path / "dataset"
    root.mkdir(parents=True, exist_ok=True)
    sources = []
    for index in range(count):
        source = root / f"part-{index:03d}.parquet"
        source.write_bytes(f"parquet-{index}".encode())
        sources.append(str(source))
    return root, sources


def _selection_and_request(tmp_path: Path) -> tuple[str, str]:
    root, sources = _make_sources(tmp_path)
    options = {"builder": "parquet", "split": "train", "datasets_version": "test"}
    selection_id = sync.make_cache_selection_id(
        dataset_root=root,
        source_mode="data_files",
        source_paths=sources,
        load_options=options,
    )
    request_id = sync.make_cache_request_id(
        dataset_root=root,
        source_mode="data_files",
        source_paths=sources,
        load_options=options,
    )
    return selection_id, request_id


def _paths(
    tmp_path: Path,
    request_id: str,
    generation_id: str = "a" * 24,
) -> sync.HfCacheSyncPaths:
    return sync.make_cache_sync_paths(tmp_path / "cache", request_id, generation_id)


def _tree_state(root: Path) -> dict[str, tuple[str, int, int, str | None]]:
    state = {}
    for path in [root, *sorted(root.rglob("*"))]:
        stat_result = path.lstat()
        relative_path = "." if path == root else path.relative_to(root).as_posix()
        if path.is_symlink():
            kind = "symlink"
            digest = hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest()
        elif path.is_file():
            kind = "file"
            file_digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    file_digest.update(chunk)
            digest = file_digest.hexdigest()
        else:
            kind = "directory"
            digest = None
        state[relative_path] = (kind, stat_result.st_size, stat_result.st_mtime_ns, digest)
    return state


def _prepare_tiny_arrow_cache(tmp_path: Path):
    import datasets
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "tiny.parquet"
    cache_dir = tmp_path / "cache"
    pq.write_table(pa.table({"value": [1, 2, 3]}), source)
    hf_dataset = datasets.load_dataset(
        "parquet",
        data_files=[str(source)],
        split="train",
        cache_dir=str(cache_dir),
        num_proc=1,
    )
    request_id = sync.make_cache_request_id(
        dataset_root=tmp_path,
        source_mode="data_files",
        source_paths=[str(source)],
        load_options={"builder": "parquet", "split": "train", "datasets_version": datasets.__version__},
    )
    paths = sync.make_cache_sync_paths(cache_dir, request_id, "a" * 24)
    manifest = sync.build_prepared_cache_manifest(
        cache_dir,
        [cache_file["filename"] for cache_file in hf_dataset.cache_files],
        dataset_fingerprint=getattr(hf_dataset, "_fingerprint", None),
    )
    sync.publish_local_cache_ready(
        paths,
        rank=0,
        local_rank=0,
        prepared_manifest=manifest,
    )
    return datasets, paths, manifest


def _multiprocess_retry_worker(rank: int, world_size: int, port: int, root: str, result_queue) -> None:
    try:
        import torch
        import torch.distributed.distributed_c10d as c10d

        torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=15),
        )
        store = c10d._get_default_store()  # noqa: SLF001
        selection_id = "d" * 24
        request_id = "e" * 24
        settings = sync.HfCacheSyncSettings(10, 0.005)
        cache_dir = Path(root) / "multiprocess-cache"

        def coordinate(invocation_index: int):
            return sync.coordinate_cache_attempt(
                store,
                rank=rank,
                selection_id=selection_id,
                invocation_index=invocation_index,
                run_id="multiprocess-default-store",
                request_id_factory=lambda: request_id,
                timeout_s=settings.timeout_s,
                poll_s=settings.poll_s,
            )

        def callbacks(attempt):
            def publish(error: BaseException):
                sync.publish_global_cache_failure(
                    store,
                    request_id=attempt.request_id,
                    generation_id=attempt.generation_id,
                    rank=rank,
                    local_rank=rank,
                    error=error,
                )

            def observe():
                sync.observe_global_cache_failure(
                    store,
                    request_id=attempt.request_id,
                    generation_id=attempt.generation_id,
                    rank=rank,
                    world_size=world_size,
                    timeout_s=settings.timeout_s,
                    poll_s=settings.poll_s,
                )

            return publish, observe

        first = coordinate(0)
        first_paths = sync.make_cache_sync_paths(cache_dir, request_id, first.generation_id)
        first_publish, first_observe = callbacks(first)
        try:
            sync.load_with_local_cache_sync(
                (
                    (lambda: (_ for _ in ()).throw(FileNotFoundError("transient filelock ENOENT")))
                    if rank == 0
                    else (lambda: "peer-should-not-load")
                ),
                paths=first_paths,
                is_builder=rank == 0,
                force_load_cache=False,
                rank=rank,
                local_rank=rank,
                settings=settings,
                prepared_manifest_factory=lambda _: _dummy_manifest(),
                external_failure_publisher=first_publish,
                external_failure_check=first_observe,
            )
            raise AssertionError("generation 1 should fail retryably")
        except sync.DistributedCacheError as first_error:
            first_result = (first_error.retryable, str(first_error), first_error.generation_id)

        second = coordinate(1)
        second_paths = sync.make_cache_sync_paths(cache_dir, request_id, second.generation_id)
        second_publish, second_observe = callbacks(second)
        second_value = sync.load_with_local_cache_sync(
            (lambda: "builder-success") if rank == 0 else (lambda: "peer-success"),
            paths=second_paths,
            is_builder=rank == 0,
            force_load_cache=False,
            rank=rank,
            local_rank=rank,
            settings=settings,
            prepared_manifest_factory=lambda _: _dummy_manifest(),
            external_failure_publisher=second_publish,
            external_failure_check=second_observe,
        )
        sync.wait_for_global_cache_readiness(
            store,
            request_id=request_id,
            generation_id=second.generation_id,
            rank=rank,
            world_size=world_size,
            timeout_s=settings.timeout_s,
            poll_s=settings.poll_s,
        )

        third = coordinate(2)
        third_paths = sync.make_cache_sync_paths(cache_dir, request_id, third.generation_id)
        third_publish, third_observe = callbacks(third)
        try:
            sync.load_with_local_cache_sync(
                (
                    (lambda: (_ for _ in ()).throw(ValueError("nonretryable parquet schema error")))
                    if rank == 0
                    else (lambda: "peer-should-not-load")
                ),
                paths=third_paths,
                is_builder=rank == 0,
                force_load_cache=False,
                rank=rank,
                local_rank=rank,
                settings=settings,
                prepared_manifest_factory=lambda _: _dummy_manifest(),
                external_failure_publisher=third_publish,
                external_failure_check=third_observe,
            )
            raise AssertionError("generation 3 should fail nonretryably")
        except sync.DistributedCacheError as third_error:
            third_result = (third_error.retryable, str(third_error), third_error.generation_id)

        result_queue.put(
            {
                "rank": rank,
                "store_type": type(store).__name__,
                "first": first_result,
                "second_generation": second.generation_id,
                "second_value": second_value,
                "third": third_result,
            }
        )
    except Exception:
        result_queue.put({"rank": rank, "worker_error": traceback.format_exc()})
    finally:
        with contextlib.suppress(Exception):
            import torch

            torch.distributed.destroy_process_group()


def _multiprocess_setup_failure_worker(
    rank: int,
    world_size: int,
    port: int,
    root: str,
    failure_mode: str,
    result_queue,
) -> None:
    try:
        import torch
        import torch.distributed.distributed_c10d as c10d

        torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=15),
        )
        store = c10d._get_default_store()  # noqa: SLF001
        settings = sync.HfCacheSyncSettings(10, 0.005)
        selection_id = "6" * 24
        request_id = "7" * 24

        def coordinate(invocation_index: int):
            return sync.coordinate_cache_attempt(
                store,
                rank=rank,
                selection_id=selection_id,
                invocation_index=invocation_index,
                run_id=f"setup-failure-{failure_mode}",
                request_id_factory=lambda: request_id,
                timeout_s=settings.timeout_s,
                poll_s=settings.poll_s,
            )

        first = coordinate(0)
        root_path = Path(root)
        first_cache_root = root_path / f"first-{failure_mode}"

        def first_setup():
            return sync.setup_node_cache_paths(
                str(first_cache_root),
                world_size=world_size,
                rank=rank,
                local_world_size=1,
                per_node_cache=True,
                run_id=f"setup-failure-{failure_mode}",
                request_id=first.request_id,
                generation_id=first.generation_id,
                force_load_cache=failure_mode == "strict-missing",
            )

        start = time.monotonic()
        try:
            sync.coordinate_global_cache_setup(
                first_setup,
                store=store,
                request_id=first.request_id,
                generation_id=first.generation_id,
                rank=rank,
                local_rank=rank,
                world_size=world_size,
                settings=settings,
            )
            raise AssertionError("first setup generation should fail on every rank")
        except sync.DistributedCacheError as first_error:
            first_result = (
                first_error.retryable,
                str(first_error),
                first_error.generation_id,
                time.monotonic() - start,
            )

        second = coordinate(1)
        second_cache_root = root_path / f"second-{failure_mode}"
        second_value, _ = sync.coordinate_global_cache_setup(
            lambda: sync.setup_node_cache_paths(
                str(second_cache_root),
                world_size=world_size,
                rank=rank,
                local_world_size=1,
                per_node_cache=True,
                run_id=f"setup-recovery-{failure_mode}",
                request_id=second.request_id,
                generation_id=second.generation_id,
                force_load_cache=False,
            ),
            store=store,
            request_id=second.request_id,
            generation_id=second.generation_id,
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            settings=settings,
        )
        result_queue.put(
            {
                "rank": rank,
                "store_type": type(store).__name__,
                "first": first_result,
                "second_generation": second.generation_id,
                "second_value": second_value,
            }
        )
    except Exception:
        result_queue.put({"rank": rank, "worker_error": traceback.format_exc()})
    finally:
        with contextlib.suppress(Exception):
            import torch

            torch.distributed.destroy_process_group()


def test_sync_settings_defaults_and_validation():
    assert sync.HfCacheSyncSettings.from_env({}) == sync.HfCacheSyncSettings(7200, 2)
    assert sync.HfCacheSyncSettings.from_env(
        {
            "OPENPI_HF_LOCAL_SYNC_TIMEOUT_S": "12.5",
            "OPENPI_HF_LOCAL_SYNC_POLL_S": "0.25",
        }
    ) == sync.HfCacheSyncSettings(12.5, 0.25)
    for env in (
        {"OPENPI_HF_LOCAL_SYNC_TIMEOUT_S": "0"},
        {"OPENPI_HF_LOCAL_SYNC_TIMEOUT_S": "nan"},
        {"OPENPI_HF_LOCAL_SYNC_POLL_S": "-1"},
        {
            "OPENPI_HF_LOCAL_SYNC_TIMEOUT_S": "1",
            "OPENPI_HF_LOCAL_SYNC_POLL_S": "2",
        },
    ):
        with pytest.raises(ValueError, match="OPENPI_HF_LOCAL_SYNC"):
            sync.HfCacheSyncSettings.from_env(env)


def test_run_identity_is_rank_consistent_and_distributed_fail_closed():
    env = {"OPENPI_HF_CACHE_RUN_ID": "arnold-123"}
    assert sync.resolve_cache_run_id(env, distributed=True) == "arnold-123"
    fallback_env = {"MASTER_ADDR": "host", "MASTER_PORT": "29500", "WORLD_SIZE": "32"}
    assert sync.resolve_cache_run_id(fallback_env, distributed=True) == "torchrun:host:29500:32"
    with pytest.raises(RuntimeError, match="rank-consistent run identity"):
        sync.resolve_cache_run_id({}, distributed=True)


def test_request_fingerprint_covers_order_options_and_changed_file_metadata(tmp_path):
    root, sources = _make_sources(tmp_path)
    options = {"builder": "parquet", "split": "train", "datasets_version": "1"}

    def fingerprint(paths=sources, current_options=options):
        return sync.make_cache_request_id(
            dataset_root=root,
            source_mode="data_files",
            source_paths=paths,
            load_options=current_options,
        )

    baseline = fingerprint()
    assert baseline != fingerprint(list(reversed(sources)))
    assert baseline != fingerprint(current_options={**options, "split": "validation"})

    source = Path(sources[0])
    source.write_bytes(b"changed-in-place-and-size")
    stat_result = source.stat()
    os.utime(source, ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns + 1_000_000))
    assert baseline != fingerprint()


def test_request_fingerprint_stats_unique_paths_once_semantically(tmp_path):
    root, sources = _make_sources(tmp_path, count=2)
    repeated = [sources[0]] * 4500 + [sources[1]] * 4500
    request_id = sync.make_cache_request_id(
        dataset_root=root,
        source_mode="data_files",
        source_paths=repeated,
        load_options={"builder": "parquet", "split": "train"},
    )
    assert len(request_id) == 24


def test_distributed_unset_cache_derives_one_node_local_path_per_node(tmp_path):
    kwargs = {
        "configured_cache": None,
        "world_size": 32,
        "local_world_size": 8,
        "per_node_cache": True,
        "run_id": "job-123",
        "temp_dir": tmp_path,
    }
    rank0 = sync.resolve_node_cache_dir(rank=0, **kwargs)
    rank7 = sync.resolve_node_cache_dir(rank=7, **kwargs)
    rank8 = sync.resolve_node_cache_dir(rank=8, **kwargs)
    assert rank0 == rank7
    assert rank0 is not None
    assert rank0.name == "node0"
    assert rank8 is not None
    assert rank8.name == "node1"
    assert rank0.parent == rank8.parent


def test_distributed_unset_cache_default_ignores_tmpdir_and_uses_literal_tmp(monkeypatch):
    monkeypatch.setenv("TMPDIR", "/shared/checkpoint-or-log-tmp")

    cache_path = sync.resolve_node_cache_dir(
        None,
        world_size=32,
        rank=7,
        local_world_size=8,
        per_node_cache=True,
        run_id="job-uses-node-local-tmp",
    )

    assert cache_path is not None
    assert cache_path.is_relative_to(Path("/tmp/openpi-hf-datasets-cache"))
    assert "/shared/checkpoint-or-log-tmp" not in str(cache_path)


def test_distributed_shared_cache_mode_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match="OPENPI_HF_DATASETS_CACHE_PER_RANK=0 is unsafe"):
        sync.resolve_node_cache_dir(
            str(tmp_path / "shared"),
            world_size=32,
            rank=0,
            local_world_size=8,
            per_node_cache=False,
            run_id="job-123",
        )
    assert (
        sync.resolve_node_cache_dir(
            None,
            world_size=1,
            rank=0,
            local_world_size=1,
            per_node_cache=False,
            run_id="standalone",
        )
        is None
    )


def test_attempt_generation_is_all_rank_agreed_and_changes_per_invocation(tmp_path):
    selection_id, request_id = _selection_and_request(tmp_path)
    store = _FakeStore()
    settings = sync.HfCacheSyncSettings(1, 0.001)

    first_rank0 = sync.coordinate_cache_attempt(
        store,
        rank=0,
        selection_id=selection_id,
        invocation_index=0,
        run_id="run-1",
        request_id_factory=lambda: request_id,
        timeout_s=settings.timeout_s,
        poll_s=settings.poll_s,
    )
    first_rank1 = sync.coordinate_cache_attempt(
        store,
        rank=1,
        selection_id=selection_id,
        invocation_index=0,
        run_id="run-1",
        request_id_factory=lambda: pytest.fail("only rank 0 may fingerprint"),
        timeout_s=settings.timeout_s,
        poll_s=settings.poll_s,
    )
    second = sync.coordinate_cache_attempt(
        store,
        rank=0,
        selection_id=selection_id,
        invocation_index=1,
        run_id="run-1",
        request_id_factory=lambda: request_id,
        timeout_s=settings.timeout_s,
        poll_s=settings.poll_s,
    )
    assert first_rank0 == first_rank1
    assert first_rank0.request_id == second.request_id
    assert first_rank0.generation_id != second.generation_id


def test_local_rank0_atomically_publishes_prepared_and_attempt_ready(tmp_path):
    _, request_id = _selection_and_request(tmp_path)
    paths = _paths(tmp_path, request_id)
    sync.publish_local_cache_ready(
        paths,
        rank=0,
        local_rank=0,
        prepared_manifest=_dummy_manifest(),
    )

    prepared = json.loads(paths.prepared_ready.read_text())
    attempt = json.loads(paths.ready.read_text())
    assert prepared["status"] == "prepared"
    assert prepared["generation_id"] is None
    assert attempt["status"] == "ready"
    assert attempt["generation_id"] == paths.generation_id
    assert not paths.failure.exists()
    assert not list(paths.ready.parent.glob("*.tmp.*"))


def test_second_generation_does_not_consume_first_generation_ready(tmp_path):
    _, request_id = _selection_and_request(tmp_path)
    first = _paths(tmp_path, request_id, "1" * 24)
    second = _paths(tmp_path, request_id, "2" * 24)
    settings = sync.HfCacheSyncSettings(2, 0.005)
    sync.publish_local_cache_ready(
        first,
        rank=0,
        local_rank=0,
        prepared_manifest=_dummy_manifest(),
    )
    peer_loaded = threading.Event()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            sync.load_with_local_cache_sync,
            lambda: peer_loaded.set() or "peer",
            paths=second,
            is_builder=False,
            force_load_cache=False,
            rank=1,
            local_rank=1,
            settings=settings,
        )
        time.sleep(0.03)
        assert not future.done()
        assert not peer_loaded.is_set()
        assert sync.load_with_local_cache_sync(
            lambda: "builder",
            paths=second,
            is_builder=True,
            force_load_cache=False,
            rank=0,
            local_rank=0,
            settings=settings,
            prepared_manifest_factory=lambda _: _dummy_manifest(),
        ) == "builder"
        assert future.result(timeout=1) == "peer"


def test_builder_failure_marker_is_actionable_and_propagates_to_peer(tmp_path):
    _, request_id = _selection_and_request(tmp_path)
    paths = _paths(tmp_path, request_id)
    settings = sync.HfCacheSyncSettings(1, 0.005)

    with pytest.raises(OSError, match="local disk full"):
        sync.load_with_local_cache_sync(
            lambda: (_ for _ in ()).throw(OSError("local disk full")),
            paths=paths,
            is_builder=True,
            force_load_cache=False,
            rank=8,
            local_rank=0,
            settings=settings,
        )
    with pytest.raises(sync.DistributedCacheError, match=r"generation_id=.*origin_rank=8.*local disk full"):
        sync.load_with_local_cache_sync(
            lambda: pytest.fail("peer must not load"),
            paths=paths,
            is_builder=False,
            force_load_cache=False,
            rank=9,
            local_rank=1,
            settings=settings,
        )


def test_marker_publication_failures_reach_local_peers_through_store(tmp_path, monkeypatch):
    _, request_id = _selection_and_request(tmp_path)
    generation_id = "f" * 24
    paths = _paths(tmp_path, request_id, generation_id)
    settings = sync.HfCacheSyncSettings(2, 0.005)
    store = _FakeStore()
    peer_loaded = threading.Event()

    def publish_external(error: BaseException):
        sync.publish_global_cache_failure(
            store,
            request_id=request_id,
            generation_id=generation_id,
            rank=0,
            local_rank=0,
            error=error,
        )

    def check_external(rank: int):
        return lambda: sync.observe_global_cache_failure(
            store,
            request_id=request_id,
            generation_id=generation_id,
            rank=rank,
            world_size=2,
            timeout_s=settings.timeout_s,
            poll_s=settings.poll_s,
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        peer = pool.submit(
            sync.load_with_local_cache_sync,
            lambda: peer_loaded.set(),
            paths=paths,
            is_builder=False,
            force_load_cache=False,
            rank=1,
            local_rank=1,
            settings=settings,
            external_failure_check=check_external(1),
        )
        time.sleep(0.02)
        monkeypatch.setattr(
            sync,
            "_atomic_write_json",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("marker filesystem unavailable")),
        )
        with pytest.raises(sync.DistributedCacheError, match="marker filesystem unavailable") as builder_error:
            sync.load_with_local_cache_sync(
                lambda: "built",
                paths=paths,
                is_builder=True,
                force_load_cache=False,
                rank=0,
                local_rank=0,
                settings=settings,
                prepared_manifest_factory=lambda _: _dummy_manifest(),
                external_failure_publisher=publish_external,
                external_failure_check=check_external(0),
            )
        with pytest.raises(sync.DistributedCacheError, match="marker filesystem unavailable") as peer_error:
            peer.result(timeout=1)
        assert str(builder_error.value) == str(peer_error.value)
    assert not peer_loaded.is_set()


def test_local_wait_has_bounded_actionable_timeout(tmp_path):
    _, request_id = _selection_and_request(tmp_path)
    paths = _paths(tmp_path, request_id)
    with pytest.raises(TimeoutError, match=r"generation_id=.*ready_marker=.*timeout_s=0.03"):
        sync.wait_for_local_cache_ready(paths, timeout_s=0.03, poll_s=0.005)


def test_strict_force_load_complete_tiny_parquet_cache_is_read_only(tmp_path):
    datasets, prepared_paths, _ = _prepare_tiny_arrow_cache(tmp_path)
    current_paths = sync.make_cache_sync_paths(
        prepared_paths.cache_dir,
        prepared_paths.request_id,
        "b" * 24,
    )
    before = _tree_state(prepared_paths.cache_dir)

    def force_load(manifest):
        arrow_paths = sync.prepared_arrow_paths(current_paths, manifest)
        parts = [datasets.Dataset.from_file(str(path)) for path in arrow_paths]
        return parts[0] if len(parts) == 1 else datasets.concatenate_datasets(parts)

    loaded = sync.load_with_local_cache_sync(
        lambda: pytest.fail("strict force-load must not call load_dataset/build path"),
        paths=current_paths,
        is_builder=False,
        force_load_cache=True,
        rank=0,
        local_rank=0,
        settings=sync.HfCacheSyncSettings(1, 0.01),
        force_load_fn=force_load,
    )
    assert len(loaded) == 3
    assert _tree_state(prepared_paths.cache_dir) == before


def test_strict_force_load_missing_cache_root_fails_without_creating_it(tmp_path):
    cache_root = tmp_path / "missing-cache"
    paths = sync.make_cache_sync_paths(cache_root, "a" * 24, "b" * 24)

    with pytest.raises(sync.DistributedCacheError, match="no manifest-backed prepared cache exists"):
        sync.load_with_local_cache_sync(
            lambda: pytest.fail("strict force-load must not build a missing cache"),
            paths=paths,
            is_builder=True,
            force_load_cache=True,
            rank=0,
            local_rank=0,
            settings=sync.HfCacheSyncSettings(1, 0.01),
            force_load_fn=lambda _: pytest.fail("missing manifest must fail before artifact loading"),
        )

    assert not cache_root.exists()


def test_full_file_digest_rejects_same_size_middle_corruption_with_restored_mtime(tmp_path):
    cache_root = tmp_path / "cache"
    arrow_path = cache_root / "builder" / "large.arrow"
    arrow_path.parent.mkdir(parents=True)
    arrow_path.write_bytes(b"A" * (512 * 1024))
    prepared_paths = sync.make_cache_sync_paths(cache_root, "c" * 24, "d" * 24)
    manifest = sync.build_prepared_cache_manifest(
        cache_root,
        [arrow_path],
        dataset_fingerprint="large-artifact",
    )
    assert manifest["artifacts"][0]["sha256"] == hashlib.sha256(arrow_path.read_bytes()).hexdigest()
    sync.publish_local_cache_ready(
        prepared_paths,
        rank=0,
        local_rank=0,
        prepared_manifest=manifest,
    )

    original_stat = arrow_path.stat()
    with arrow_path.open("r+b") as artifact_file:
        artifact_file.seek(original_stat.st_size // 2)
        artifact_file.write(b"B")
    os.utime(arrow_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    corrupted_stat = arrow_path.stat()
    assert corrupted_stat.st_size == original_stat.st_size
    assert corrupted_stat.st_mtime_ns == original_stat.st_mtime_ns

    current_paths = sync.make_cache_sync_paths(cache_root, prepared_paths.request_id, "e" * 24)
    with pytest.raises(sync.DistributedCacheError, match="artifact identity mismatch"):
        sync.load_with_local_cache_sync(
            lambda: pytest.fail("strict force-load must not rebuild a corrupt cache"),
            paths=current_paths,
            is_builder=False,
            force_load_cache=True,
            rank=0,
            local_rank=0,
            settings=sync.HfCacheSyncSettings(1, 0.01),
            force_load_fn=lambda _: pytest.fail("digest validation must fail before artifact loading"),
        )


def test_strict_force_load_missing_arrow_fails_without_rebuild_or_writes(tmp_path):
    datasets, prepared_paths, manifest = _prepare_tiny_arrow_cache(tmp_path)
    del datasets
    arrow_path = sync.prepared_arrow_paths(prepared_paths, manifest)[0]
    arrow_path.unlink()
    before = _tree_state(prepared_paths.cache_dir)
    current_paths = sync.make_cache_sync_paths(
        prepared_paths.cache_dir,
        prepared_paths.request_id,
        "c" * 24,
    )

    with pytest.raises(sync.DistributedCacheError, match="artifact is missing.*will not rebuild") as error:
        sync.load_with_local_cache_sync(
            lambda: pytest.fail("strict force-load must not rebuild missing Arrow artifacts"),
            paths=current_paths,
            is_builder=False,
            force_load_cache=True,
            rank=0,
            local_rank=0,
            settings=sync.HfCacheSyncSettings(1, 0.01),
            force_load_fn=lambda _: pytest.fail("manifest validation must fail before opening artifacts"),
        )
    assert error.value.retryable is False
    assert _tree_state(prepared_paths.cache_dir) == before
    assert not list(prepared_paths.cache_dir.rglob("*.arrow"))


def test_strict_force_load_rejects_unverifiable_legacy_marker(tmp_path):
    _, request_id = _selection_and_request(tmp_path)
    paths = _paths(tmp_path, request_id)
    paths.legacy_ready.parent.mkdir(parents=True, exist_ok=True)
    paths.legacy_ready.write_text("ready\n")
    with pytest.raises(sync.DistributedCacheError, match="Legacy HF cache markers cannot prove"):
        sync.require_prepared_cache(paths)


def test_global_store_failure_and_timeout_are_generation_scoped(tmp_path):
    _, request_id = _selection_and_request(tmp_path)
    store = _FakeStore()
    old_generation = "1" * 24
    current_generation = "2" * 24
    sync.publish_global_cache_failure(
        store,
        request_id=request_id,
        generation_id=old_generation,
        rank=16,
        local_rank=0,
        error=OSError("old failure"),
    )
    with pytest.raises(TimeoutError, match=r"missing_ranks=\[1, 2, 3\].*No NCCL collective"):
        sync.wait_for_global_cache_readiness(
            store,
            request_id=request_id,
            generation_id=current_generation,
            rank=0,
            world_size=4,
            timeout_s=0.03,
            poll_s=0.005,
        )


def test_four_node_eight_local_rank_protocol_simulation(tmp_path):
    world_size = 32
    local_world_size = 8
    _, request_id = _selection_and_request(tmp_path)
    generation_id = "a" * 24
    paths_by_node = [
        _paths(tmp_path / f"node{node_rank}", request_id, generation_id) for node_rank in range(4)
    ]
    builder_finished = [threading.Event() for _ in range(4)]
    peer_loads = [0, 0, 0, 0]
    peer_lock = threading.Lock()
    store = _FakeStore()
    settings = sync.HfCacheSyncSettings(3, 0.001)

    def run_rank(rank: int) -> str:
        node_rank = rank // local_world_size
        local_rank = rank % local_world_size
        if local_rank == 0:
            def load():
                time.sleep(0.005 * (node_rank + 1))
                builder_finished[node_rank].set()
                return f"builder-{node_rank}"
        else:
            def load():
                assert builder_finished[node_rank].is_set()
                with peer_lock:
                    peer_loads[node_rank] += 1
                return f"peer-{rank}"

        result = sync.load_with_local_cache_sync(
            load,
            paths=paths_by_node[node_rank],
            is_builder=local_rank == 0,
            force_load_cache=False,
            rank=rank,
            local_rank=local_rank,
            settings=settings,
            prepared_manifest_factory=lambda _: _dummy_manifest(),
        )
        sync.wait_for_global_cache_readiness(
            store,
            request_id=request_id,
            generation_id=generation_id,
            rank=rank,
            world_size=world_size,
            timeout_s=settings.timeout_s,
            poll_s=settings.poll_s,
        )
        return result

    with ThreadPoolExecutor(max_workers=world_size) as pool:
        results = list(pool.map(run_rank, range(world_size)))
    assert len(results) == world_size
    assert peer_loads == [7, 7, 7, 7]


def test_real_prefix_store_repeated_generation_waits_for_current_rank(tmp_path):
    torch = pytest.importorskip("torch")
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    tcp_store = torch.distributed.TCPStore(
        host_name="127.0.0.1",
        port=port,
        world_size=1,
        is_master=True,
        timeout=timedelta(seconds=5),
        wait_for_workers=False,
    )
    store = torch.distributed.PrefixStore(f"openpi-test-{uuid.uuid4().hex}", tcp_store)
    selection_id, request_id = _selection_and_request(tmp_path)
    first = sync.coordinate_cache_attempt(
        store,
        rank=0,
        selection_id=selection_id,
        invocation_index=0,
        run_id="real-store-run",
        request_id_factory=lambda: request_id,
        timeout_s=2,
        poll_s=0.001,
    )
    second = sync.coordinate_cache_attempt(
        store,
        rank=0,
        selection_id=selection_id,
        invocation_index=1,
        run_id="real-store-run",
        request_id_factory=lambda: request_id,
        timeout_s=2,
        poll_s=0.001,
    )
    assert first.generation_id != second.generation_id

    def wait_rank(rank: int, generation_id: str):
        sync.wait_for_global_cache_readiness(
            store,
            request_id=request_id,
            generation_id=generation_id,
            rank=rank,
            world_size=4,
            timeout_s=2,
            poll_s=0.001,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda rank: wait_rank(rank, first.generation_id), range(4)))
        current = [pool.submit(wait_rank, rank, second.generation_id) for rank in range(3)]
        time.sleep(0.03)
        assert all(not future.done() for future in current)
        delayed = pool.submit(wait_rank, 3, second.generation_id)
        for future in [*current, delayed]:
            future.result(timeout=2)


def test_real_multiprocess_default_store_has_symmetric_retry_and_abort(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("MKL_THREADING_LAYER", "GNU")
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_multiprocess_retry_worker,
            args=(rank, 2, port, str(tmp_path), result_queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    results = [result_queue.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    assert all("worker_error" not in result for result in results), results
    assert all("PrefixStore" in result["store_type"] for result in results)
    first_results = [result["first"] for result in sorted(results, key=lambda item: item["rank"])]
    assert first_results[0] == first_results[1]
    assert first_results[0][0] is True
    assert "FileNotFoundError: transient filelock ENOENT" in first_results[0][1]
    assert len({result["second_generation"] for result in results}) == 1
    assert {result["second_value"] for result in results} == {"builder-success", "peer-success"}
    third_results = [result["third"] for result in sorted(results, key=lambda item: item["rank"])]
    assert third_results[0] == third_results[1]
    assert third_results[0][0] is False
    assert "ValueError: nonretryable parquet schema error" in third_results[0][1]


@pytest.mark.parametrize("failure_mode", ["strict-missing", "mkdir-failure"])
def test_real_multiprocess_cache_setup_failure_is_prompt_symmetric_and_generation_scoped(
    tmp_path,
    monkeypatch,
    failure_mode,
):
    pytest.importorskip("torch")
    monkeypatch.setenv("MKL_THREADING_LAYER", "GNU")
    first_cache_root = tmp_path / f"first-{failure_mode}"
    if failure_mode == "strict-missing":
        (first_cache_root / "node1").mkdir(parents=True)
    else:
        first_cache_root.mkdir()
        (first_cache_root / "node0").write_text("not a directory\n")

    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_multiprocess_setup_failure_worker,
            args=(rank, 2, port, str(tmp_path), failure_mode, result_queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    results = [result_queue.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    assert all("worker_error" not in result for result in results), results
    assert all("PrefixStore" in result["store_type"] for result in results)
    ordered = sorted(results, key=lambda item: item["rank"])
    first_results = [result["first"] for result in ordered]
    assert first_results[0][:3] == first_results[1][:3]
    assert first_results[0][0] is False
    assert all(result[3] < 5 for result in first_results)
    assert len({result["second_generation"] for result in results}) == 1
    assert first_results[0][2] != results[0]["second_generation"]
    second_cache_root = tmp_path / f"second-{failure_mode}"
    assert {result["second_value"] for result in results} == {
        str(second_cache_root / "node0"),
        str(second_cache_root / "node1"),
    }
    if failure_mode == "strict-missing":
        assert not (first_cache_root / "node0").exists()
        assert "Strict force-load requires the resolved HF datasets cache" in first_results[0][1]
    else:
        assert (first_cache_root / "node0").is_file()
        assert "FileExistsError" in first_results[0][1]


def test_dataset_cache_build_path_contains_no_distributed_barrier():
    source = _DATASET_SOURCE.read_text()
    tree = ast.parse(source)
    load_method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "load_hf_dataset"
    )
    method_source = ast.get_source_segment(source, load_method)
    assert method_source is not None
    assert "th.distributed.barrier" not in method_source
    assert "coordinate_cache_attempt" in method_source
    assert "coordinate_global_cache_setup" in method_source
    assert "generation_id=attempt.generation_id" in method_source
    assert "setup_node_cache_paths" in method_source
    assert method_source.index("coordinate_cache_attempt(") < method_source.index("setup_node_cache_paths(")


def test_prepare_only_and_force_load_trainer_order_remains_compatible():
    source = _TRAIN_ACCELERATE_SOURCE.read_text()
    configure_index = source.index("configure_hf_cache(config, accelerator=accelerator)")
    force_index = source.index('os.environ["OPENPI_FORCE_LOAD_CACHE"]', configure_index)
    build_index = source.index("loader, data_config = build_datasets(config)", force_index)
    val_build_index = source.index("val_loader, val_data_config = build_val_datasets(config)", build_index)
    prepare_exit_index = source.index("if config.prepare_hf_cache_only:", val_build_index)
    assert configure_index < force_index < build_index < val_build_index < prepare_exit_index
