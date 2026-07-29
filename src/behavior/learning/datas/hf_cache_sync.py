"""CPU/file synchronization for Hugging Face Arrow cache preparation.

Large parquet conversion must not hold an NCCL collective open. This module
coordinates each cache attempt through atomic node-local markers and the c10d
control-plane store. Persistent prepared-cache identity is deliberately separate
from per-invocation attempt readiness so an old success can be reused only after
the current node builder validates it.
"""

from collections.abc import Callable, Iterable, Mapping
import contextlib
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import threading
import time
from typing import Any, Protocol, TypeVar
import uuid

_LOCAL_SYNC_TIMEOUT_ENV = "OPENPI_HF_LOCAL_SYNC_TIMEOUT_S"
_LOCAL_SYNC_POLL_ENV = "OPENPI_HF_LOCAL_SYNC_POLL_S"
_DEFAULT_LOCAL_SYNC_TIMEOUT_S = 7200.0
_DEFAULT_LOCAL_SYNC_POLL_S = 2.0
_MARKER_PROTOCOL_VERSION = 3
_SYNC_DIR_NAME = ".openpi_hf_cache_sync"
_LEGACY_READY_NAME = ".hf_cache_ready"
_LEGACY_FAILURE_NAME = ".hf_cache_failed"
_MAX_ERROR_TEXT_CHARS = 16_384

_T = TypeVar("_T")
_INVOCATION_COUNTS: dict[str, int] = {}
_INVOCATION_COUNTS_LOCK = threading.Lock()


class StoreLike(Protocol):
    """Minimal c10d Store interface used by the readiness protocol."""

    def set(self, key: str, value: str | bytes) -> None: ...

    def get(self, key: str) -> bytes: ...

    def check(self, keys: list[str]) -> bool: ...

    def compare_set(self, key: str, expected: str | bytes, desired: str | bytes) -> bytes: ...


class DistributedCacheError(RuntimeError):
    """Canonical cache-attempt failure observed identically by every rank."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        request_id: str | None,
        generation_id: str | None,
        origin_rank: int,
        error_type: str,
        error_text: str,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.request_id = request_id
        self.generation_id = generation_id
        self.origin_rank = origin_rank
        self.error_type = error_type
        self.error_text = error_text

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DistributedCacheError":
        message = (
            "Distributed HF cache attempt failed: "
            f"request_id={payload.get('request_id')}, generation_id={payload.get('generation_id')}, "
            f"origin_rank={payload.get('rank')}, local_rank={payload.get('local_rank')}, "
            f"retryable={bool(payload.get('retryable'))}, host={payload.get('hostname')}, "
            f"error={payload.get('error_type')}: {payload.get('error')}"
        )
        return cls(
            message,
            retryable=bool(payload.get("retryable")),
            request_id=payload.get("request_id"),
            generation_id=payload.get("generation_id"),
            origin_rank=int(payload.get("rank", -1)),
            error_type=str(payload.get("error_type", "UnknownError")),
            error_text=str(payload.get("error", "")),
        )

    @classmethod
    def from_exception(
        cls,
        error: BaseException,
        *,
        request_id: str | None,
        generation_id: str | None,
        rank: int,
        local_rank: int,
    ) -> "DistributedCacheError":
        payload = _failure_payload(
            error,
            request_id=request_id,
            generation_id=generation_id,
            rank=rank,
            local_rank=local_rank,
        )
        return cls.from_payload(payload)


def is_retryable_cache_error(error: BaseException) -> bool:
    if isinstance(error, DistributedCacheError):
        return error.retryable
    return isinstance(error, FileNotFoundError) and error.filename is None


@dataclass(frozen=True)
class HfCacheSyncSettings:
    timeout_s: float
    poll_s: float

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "HfCacheSyncSettings":
        env = os.environ if environ is None else environ
        timeout_s = _positive_finite_float(
            env.get(_LOCAL_SYNC_TIMEOUT_ENV, str(_DEFAULT_LOCAL_SYNC_TIMEOUT_S)),
            name=_LOCAL_SYNC_TIMEOUT_ENV,
        )
        poll_s = _positive_finite_float(
            env.get(_LOCAL_SYNC_POLL_ENV, str(_DEFAULT_LOCAL_SYNC_POLL_S)),
            name=_LOCAL_SYNC_POLL_ENV,
        )
        if poll_s > timeout_s:
            raise ValueError(
                f"{_LOCAL_SYNC_POLL_ENV}={poll_s} must not exceed "
                f"{_LOCAL_SYNC_TIMEOUT_ENV}={timeout_s}"
            )
        return cls(timeout_s=timeout_s, poll_s=poll_s)


@dataclass(frozen=True)
class HfCacheAttempt:
    selection_id: str
    request_id: str
    generation_id: str
    invocation_index: int


@dataclass(frozen=True)
class HfCacheSyncPaths:
    cache_dir: Path
    request_id: str
    generation_id: str
    ready: Path
    failure: Path
    prepared_ready: Path
    legacy_ready: Path
    legacy_failure: Path


def _positive_finite_float(raw: str, *, name: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number, got {raw!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {raw!r}")
    return value


def _hash_text(value: str, *, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _failure_payload(
    error: BaseException,
    *,
    request_id: str | None,
    generation_id: str | None,
    rank: int,
    local_rank: int,
) -> dict[str, Any]:
    if isinstance(error, DistributedCacheError):
        error_type = error.error_type
        error_text = error.error_text
        retryable = error.retryable
        origin_rank = error.origin_rank
    else:
        error_type = type(error).__name__
        error_text = str(error)[:_MAX_ERROR_TEXT_CHARS]
        retryable = is_retryable_cache_error(error)
        origin_rank = rank
    return {
        "protocol_version": _MARKER_PROTOCOL_VERSION,
        "status": "failed",
        "request_id": request_id,
        "generation_id": generation_id,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "rank": origin_rank,
        "local_rank": local_rank,
        "retryable": retryable,
        "error_type": error_type,
        "error": error_text,
        "published_at_ns": time.time_ns(),
    }


def resolve_cache_run_id(
    environ: Mapping[str, str] | None = None,
    *,
    distributed: bool,
) -> str:
    """Resolve one rank-consistent run identity without using process-local PIDs."""
    env = os.environ if environ is None else environ
    for variable in ("OPENPI_HF_CACHE_RUN_ID", "ARNOLD_JOB_ID", "TORCHELASTIC_RUN_ID"):
        value = env.get(variable, "").strip()
        if value:
            return value
    master_addr = env.get("MASTER_ADDR", "").strip()
    master_port = env.get("MASTER_PORT", "").strip()
    world_size = env.get("WORLD_SIZE", "").strip()
    if master_addr and master_port and world_size:
        return f"torchrun:{master_addr}:{master_port}:{world_size}"
    if distributed:
        raise RuntimeError(
            "Distributed HF cache synchronization requires a rank-consistent run identity. "
            "Set OPENPI_HF_CACHE_RUN_ID (preferred) or provide ARNOLD_JOB_ID/TORCHELASTIC_RUN_ID."
        )
    return "standalone"


def resolve_node_cache_dir(
    configured_cache: str | None,
    *,
    world_size: int,
    rank: int,
    local_world_size: int,
    per_node_cache: bool,
    run_id: str,
    temp_dir: str | Path | None = None,
) -> Path | None:
    """Resolve a deterministic node-local cache directory.

    Distributed shared-cache mode is rejected because independent node builders
    would race on one path. If no cache is configured, a run-isolated cache is
    derived below the node-local temporary directory.
    """
    if world_size <= 1:
        return Path(configured_cache).expanduser() if configured_cache else None
    if not per_node_cache:
        raise RuntimeError(
            "OPENPI_HF_DATASETS_CACHE_PER_RANK=0 is unsafe in distributed mode: multiple node "
            "builders can race on one Arrow cache. Set OPENPI_HF_DATASETS_CACHE_PER_RANK=1."
        )
    if local_world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(
            f"Invalid distributed cache topology: world_size={world_size}, rank={rank}, "
            f"local_world_size={local_world_size}"
        )
    if configured_cache:
        cache_root = Path(configured_cache).expanduser()
    else:
        if not run_id:
            raise RuntimeError("Cannot derive distributed HF cache path without a non-empty run_id")
        tmp_root = Path("/tmp" if temp_dir is None else temp_dir)
        cache_root = tmp_root / "openpi-hf-datasets-cache" / _hash_text(run_id)
    node_rank = rank // local_world_size
    return cache_root / f"node{node_rank}"


def _update_length_prefixed(digest: Any, value: str | bytes) -> None:
    encoded = value if isinstance(value, bytes) else value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _canonical_options(load_options: Mapping[str, Any]) -> str:
    return json.dumps(dict(load_options), sort_keys=True, separators=(",", ":"), default=str)


def make_cache_selection_id(
    *,
    dataset_root: str | Path,
    source_mode: str,
    source_paths: Iterable[str | os.PathLike[str]],
    load_options: Mapping[str, Any],
) -> str:
    """Hash loading semantics and ordered source paths without filesystem stats."""
    digest = hashlib.sha256(f"openpi-hf-selection-v{_MARKER_PROTOCOL_VERSION}\0".encode())
    _update_length_prefixed(digest, os.path.abspath(os.fspath(dataset_root)))
    _update_length_prefixed(digest, source_mode)
    _update_length_prefixed(digest, _canonical_options(load_options))
    count = 0
    for source_path in source_paths:
        _update_length_prefixed(digest, os.path.abspath(os.fspath(source_path)))
        count += 1
    digest.update(count.to_bytes(8, "big"))
    return digest.hexdigest()[:24]


def _identity_paths(source_paths: Iterable[str | os.PathLike[str]]) -> list[Path]:
    unique_paths: dict[str, Path] = {}
    for raw_path in source_paths:
        path = Path(raw_path)
        absolute = Path(os.path.abspath(os.fspath(path)))
        if absolute.is_dir():
            children = sorted(candidate for candidate in absolute.rglob("*.parquet") if candidate.is_file())
            if children:
                for child in children:
                    unique_paths.setdefault(os.fspath(child), child)
                continue
        unique_paths.setdefault(os.fspath(absolute), absolute)
    return [unique_paths[key] for key in sorted(unique_paths)]


def make_cache_request_id(
    *,
    dataset_root: str | Path,
    source_mode: str,
    source_paths: Iterable[str | os.PathLike[str]],
    load_options: Mapping[str, Any],
) -> str:
    """Fingerprint loading semantics, ordered selection, and source metadata.

    Ordered paths are included through ``selection_id``. File metadata is then
    stat'ed once per unique parquet path, bounding the expensive work to at most
    the number of unique source files (9,000 in the full-B1K envelope), and this
    function is executed by global rank 0 only in distributed runs.
    """
    paths = list(source_paths)
    selection_id = make_cache_selection_id(
        dataset_root=dataset_root,
        source_mode=source_mode,
        source_paths=paths,
        load_options=load_options,
    )
    digest = hashlib.sha256(f"openpi-hf-request-v{_MARKER_PROTOCOL_VERSION}\0{selection_id}".encode())
    identity_paths = _identity_paths(paths)
    digest.update(len(identity_paths).to_bytes(8, "big"))
    for path in identity_paths:
        _update_length_prefixed(digest, os.fspath(path))
        try:
            stat_result = path.stat()
        except FileNotFoundError:
            _update_length_prefixed(digest, "missing")
            continue
        _update_length_prefixed(
            digest,
            f"size={stat_result.st_size};mtime_ns={stat_result.st_mtime_ns};"
            f"ctime_ns={stat_result.st_ctime_ns};mode={stat_result.st_mode}",
        )
    return digest.hexdigest()[:24]


def next_cache_invocation_index(selection_id: str) -> int:
    """Return the process-local phase ordinal shared by identical rank code paths."""
    with _INVOCATION_COUNTS_LOCK:
        invocation_index = _INVOCATION_COUNTS.get(selection_id, 0)
        _INVOCATION_COUNTS[selection_id] = invocation_index + 1
    return invocation_index


def _coordination_namespace(run_id: str, selection_id: str, invocation_index: int) -> str:
    return (
        f"openpi/hf-cache/coord/v{_MARKER_PROTOCOL_VERSION}/"
        f"{_hash_text(run_id)}/{selection_id}/{invocation_index}"
    )


def _decode_json_store_value(store: StoreLike, key: str, *, what: str) -> dict[str, Any]:
    raw_payload = store.get(key)
    try:
        payload = json.loads(raw_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{what} is malformed at c10d store key {key}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{what} must be a JSON object at c10d store key {key}")
    return payload


def coordinate_cache_attempt(
    store: StoreLike | None,
    *,
    rank: int,
    selection_id: str,
    invocation_index: int,
    run_id: str,
    request_id_factory: Callable[[], str],
    timeout_s: float,
    poll_s: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> HfCacheAttempt:
    """Allocate one generation agreed by every rank for this invocation phase."""
    if store is None:
        request_id = request_id_factory()
        generation_id = _hash_text(
            f"{run_id}\0{selection_id}\0{invocation_index}\0{uuid.uuid4().hex}"
        )
        return HfCacheAttempt(selection_id, request_id, generation_id, invocation_index)

    namespace = _coordination_namespace(run_id, selection_id, invocation_index)
    descriptor_key = f"{namespace}/descriptor"
    failure_key = f"{namespace}/failure"
    if rank == 0:
        try:
            request_id = request_id_factory()
            generation_id = _hash_text(
                f"{run_id}\0{selection_id}\0{invocation_index}\0{uuid.uuid4().hex}"
            )
            descriptor = {
                "protocol_version": _MARKER_PROTOCOL_VERSION,
                "selection_id": selection_id,
                "request_id": request_id,
                "generation_id": generation_id,
                "invocation_index": invocation_index,
            }
            store.set(descriptor_key, json.dumps(descriptor, sort_keys=True))
        except Exception as exc:
            failure = {
                "error_type": type(exc).__name__,
                "error": str(exc)[:_MAX_ERROR_TEXT_CHARS],
                "rank": rank,
            }
            store.set(failure_key, json.dumps(failure, sort_keys=True))
            raise

    deadline = monotonic() + timeout_s
    while True:
        if store.check([failure_key]):
            failure = _decode_json_store_value(store, failure_key, what="HF cache attempt coordination failure")
            raise RuntimeError(
                "Global rank 0 failed to fingerprint/coordinate the HF cache attempt: "
                f"rank={failure.get('rank')}, error={failure.get('error_type')}: {failure.get('error')}"
            )
        if store.check([descriptor_key]):
            descriptor = _decode_json_store_value(store, descriptor_key, what="HF cache attempt descriptor")
            expected = {
                "protocol_version": _MARKER_PROTOCOL_VERSION,
                "selection_id": selection_id,
                "invocation_index": invocation_index,
            }
            for field, expected_value in expected.items():
                if descriptor.get(field) != expected_value:
                    raise RuntimeError(
                        f"HF cache attempt descriptor mismatch for {field}: "
                        f"expected {expected_value!r}, got {descriptor.get(field)!r}"
                    )
            return HfCacheAttempt(
                selection_id=selection_id,
                request_id=str(descriptor["request_id"]),
                generation_id=str(descriptor["generation_id"]),
                invocation_index=invocation_index,
            )
        now = monotonic()
        if now >= deadline:
            raise TimeoutError(
                "Timed out waiting for global rank 0 to coordinate HF cache attempt: "
                f"selection_id={selection_id}, invocation_index={invocation_index}, "
                f"timeout_s={timeout_s}, poll_s={poll_s}"
            )
        sleep(min(poll_s, max(0.0, deadline - now)))


def make_cache_sync_paths(
    cache_dir: str | Path,
    request_id: str,
    generation_id: str,
) -> HfCacheSyncPaths:
    for name, value in (("request_id", request_id), ("generation_id", generation_id)):
        if not value or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"Invalid HF cache {name}: {value!r}")
    cache_path = Path(cache_dir)
    sync_dir = cache_path / _SYNC_DIR_NAME
    attempt_dir = sync_dir / "attempts" / generation_id
    prepared_dir = sync_dir / "prepared"
    return HfCacheSyncPaths(
        cache_dir=cache_path,
        request_id=request_id,
        generation_id=generation_id,
        ready=attempt_dir / f"{request_id}.ready.json",
        failure=attempt_dir / f"{request_id}.failed.json",
        prepared_ready=prepared_dir / f"{request_id}.ready.json",
        legacy_ready=cache_path / _LEGACY_READY_NAME,
        legacy_failure=cache_path / _LEGACY_FAILURE_NAME,
    )


def setup_node_cache_paths(
    configured_cache: str | None,
    *,
    world_size: int,
    rank: int,
    local_world_size: int,
    per_node_cache: bool,
    run_id: str,
    request_id: str,
    generation_id: str,
    force_load_cache: bool,
) -> tuple[str | None, HfCacheSyncPaths | None]:
    """Resolve and perform the generation-scoped node-local cache setup."""
    cache_dir_path = resolve_node_cache_dir(
        configured_cache,
        world_size=world_size,
        rank=rank,
        local_world_size=local_world_size,
        per_node_cache=per_node_cache,
        run_id=run_id,
    )
    if cache_dir_path is None:
        return None, None
    if force_load_cache:
        if not cache_dir_path.is_dir():
            raise RuntimeError(
                "Strict force-load requires the resolved HF datasets cache to be an existing "
                f"directory; refusing to create or repair it: {cache_dir_path}"
            )
    else:
        cache_dir_path.mkdir(parents=True, exist_ok=True)
    cache_dir = str(cache_dir_path)
    return cache_dir, make_cache_sync_paths(cache_dir, request_id, generation_id)


def _marker_payload(
    *,
    status: str,
    request_id: str,
    generation_id: str | None,
    rank: int,
    local_rank: int,
    error: BaseException | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": _MARKER_PROTOCOL_VERSION,
        "status": status,
        "request_id": request_id,
        "generation_id": generation_id,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "rank": rank,
        "local_rank": local_rank,
        "published_at_ns": time.time_ns(),
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)[:_MAX_ERROR_TEXT_CHARS]
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish one complete marker without exposing a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with tmp_path.open("x", encoding="utf-8") as marker_file:
            json.dump(payload, marker_file, sort_keys=True)
            marker_file.write("\n")
            marker_file.flush()
            os.fsync(marker_file.fileno())
        os.replace(tmp_path, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()


def _remove_if_exists(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


def _full_artifact_sha256(path: Path) -> str:
    """Hash every artifact byte with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        while chunk := artifact_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_relative_path(cache_root: Path, artifact_path: Path) -> str:
    root = cache_root.resolve(strict=True)
    artifact = artifact_path.resolve(strict=True)
    try:
        return artifact.relative_to(root).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"Prepared HF cache artifact is outside cache root {root}: {artifact}") from exc


def build_prepared_cache_manifest(
    cache_dir: str | Path,
    arrow_files: Iterable[str | Path],
    *,
    dataset_fingerprint: str | None,
) -> dict[str, Any]:
    """Capture the exact Arrow artifacts required by strict force-load."""
    cache_root = Path(cache_dir)
    ordered_arrow_paths = [Path(path) for path in arrow_files]
    if not ordered_arrow_paths:
        raise RuntimeError("HF dataset produced no cache_files; cannot publish strict prepared-cache manifest")

    artifact_paths: dict[str, tuple[Path, str]] = {}
    arrow_relative_paths = []
    for arrow_path in ordered_arrow_paths:
        if arrow_path.suffix != ".arrow":
            raise RuntimeError(f"Unexpected non-Arrow HF cache file: {arrow_path}")
        relative = _artifact_relative_path(cache_root, arrow_path)
        arrow_relative_paths.append(relative)
        artifact_paths.setdefault(relative, (arrow_path, "arrow"))
        dataset_info = arrow_path.parent / "dataset_info.json"
        if dataset_info.is_file():
            info_relative = _artifact_relative_path(cache_root, dataset_info)
            artifact_paths.setdefault(info_relative, (dataset_info, "dataset_info"))

    artifacts = []
    for relative_path in sorted(artifact_paths):
        path, kind = artifact_paths[relative_path]
        stat_result = path.stat()
        artifacts.append(
            {
                "relative_path": relative_path,
                "kind": kind,
                "size": stat_result.st_size,
                "mtime_ns": stat_result.st_mtime_ns,
                "sha256": _full_artifact_sha256(path),
            }
        )
    return {
        "manifest_version": 1,
        "dataset_fingerprint": dataset_fingerprint,
        "arrow_files": arrow_relative_paths,
        "artifacts": artifacts,
    }


def validate_prepared_cache_manifest(
    paths: HfCacheSyncPaths,
    marker_payload: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = marker_payload.get("prepared_manifest")
    if not isinstance(manifest, dict) or manifest.get("manifest_version") != 1:
        raise DistributedCacheError.from_exception(
            RuntimeError(
                "Prepared cache marker has no verifiable artifact manifest. Re-run cache preparation "
                "with the current code before using --force_load_cache."
            ),
            request_id=paths.request_id,
            generation_id=paths.generation_id,
            rank=-1,
            local_rank=-1,
        )
    artifacts = manifest.get("artifacts")
    arrow_files = manifest.get("arrow_files")
    if not isinstance(artifacts, list) or not artifacts or not isinstance(arrow_files, list) or not arrow_files:
        raise DistributedCacheError.from_exception(
            RuntimeError("Prepared cache artifact manifest is empty or malformed"),
            request_id=paths.request_id,
            generation_id=paths.generation_id,
            rank=-1,
            local_rank=-1,
        )

    root = paths.cache_dir.resolve(strict=True)
    artifact_by_relative: dict[str, Mapping[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("relative_path"), str):
            raise DistributedCacheError.from_exception(
                RuntimeError("Prepared cache artifact entry is malformed"),
                request_id=paths.request_id,
                generation_id=paths.generation_id,
                rank=-1,
                local_rank=-1,
            )
        relative_path = artifact["relative_path"]
        candidate = (root / relative_path).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise DistributedCacheError.from_exception(
                RuntimeError(f"Prepared cache artifact escapes cache root: {relative_path}"),
                request_id=paths.request_id,
                generation_id=paths.generation_id,
                rank=-1,
                local_rank=-1,
            ) from exc
        if not candidate.is_file():
            raise DistributedCacheError.from_exception(
                RuntimeError(
                    f"Prepared HF cache artifact is missing: {candidate}. "
                    "Strict force-load will not rebuild it; re-run cache preparation."
                ),
                request_id=paths.request_id,
                generation_id=paths.generation_id,
                rank=-1,
                local_rank=-1,
            )
        stat_result = candidate.stat()
        expected_size = int(artifact.get("size", -1))
        expected_mtime_ns = int(artifact.get("mtime_ns", -1))
        expected_digest = artifact.get("sha256")
        if (
            stat_result.st_size != expected_size
            or stat_result.st_mtime_ns != expected_mtime_ns
            or _full_artifact_sha256(candidate) != expected_digest
        ):
            raise DistributedCacheError.from_exception(
                RuntimeError(
                    "Prepared HF cache artifact identity mismatch: "
                    f"{candidate}; expected size/mtime/digest from {paths.prepared_ready}. "
                    "Strict force-load will not rebuild it."
                ),
                request_id=paths.request_id,
                generation_id=paths.generation_id,
                rank=-1,
                local_rank=-1,
            )
        artifact_by_relative[relative_path] = artifact

    for arrow_relative in arrow_files:
        artifact = artifact_by_relative.get(arrow_relative)
        if artifact is None or artifact.get("kind") != "arrow":
            raise DistributedCacheError.from_exception(
                RuntimeError(f"Prepared Arrow artifact is absent from manifest: {arrow_relative}"),
                request_id=paths.request_id,
                generation_id=paths.generation_id,
                rank=-1,
                local_rank=-1,
            )
    return manifest


def prepared_arrow_paths(paths: HfCacheSyncPaths, manifest: Mapping[str, Any]) -> list[Path]:
    root = paths.cache_dir.resolve(strict=True)
    return [(root / relative_path).resolve(strict=True) for relative_path in manifest["arrow_files"]]


def snapshot_cache_tree(cache_dir: str | Path) -> dict[str, tuple[int, int]]:
    root = Path(cache_dir)
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def begin_local_cache_build(paths: HfCacheSyncPaths) -> None:
    """Clear only this generation's state; prepared identity is retained."""
    _remove_if_exists(paths.ready)
    _remove_if_exists(paths.failure)
    _remove_if_exists(paths.legacy_failure)


def publish_local_cache_ready(
    paths: HfCacheSyncPaths,
    *,
    rank: int,
    local_rank: int,
    prepared_manifest: Mapping[str, Any],
) -> None:
    prepared_payload = _marker_payload(
        status="prepared",
        request_id=paths.request_id,
        generation_id=None,
        rank=rank,
        local_rank=local_rank,
    )
    prepared_payload["prepared_manifest"] = dict(prepared_manifest)
    attempt_payload = _marker_payload(
        status="ready",
        request_id=paths.request_id,
        generation_id=paths.generation_id,
        rank=rank,
        local_rank=local_rank,
    )
    _remove_if_exists(paths.failure)
    _remove_if_exists(paths.legacy_failure)
    # Attempt readiness is published last, after all persistent cache identity.
    _atomic_write_json(paths.prepared_ready, prepared_payload)
    _atomic_write_json(paths.legacy_ready, prepared_payload)
    _atomic_write_json(paths.ready, attempt_payload)


def publish_local_cache_failure(
    paths: HfCacheSyncPaths,
    *,
    rank: int,
    local_rank: int,
    error: BaseException,
) -> None:
    payload = _failure_payload(
        error,
        request_id=paths.request_id,
        generation_id=paths.generation_id,
        rank=rank,
        local_rank=local_rank,
    )
    _remove_if_exists(paths.ready)
    _remove_if_exists(paths.prepared_ready)
    _remove_if_exists(paths.legacy_ready)
    _atomic_write_json(paths.failure, payload)
    _atomic_write_json(paths.legacy_failure, payload)


def _read_json_marker(
    path: Path,
    *,
    expected_status: str,
    request_id: str,
    generation_id: str | None,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"HF cache marker is unreadable or malformed: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"HF cache marker must contain a JSON object: {path}")
    expected = {
        "protocol_version": _MARKER_PROTOCOL_VERSION,
        "status": expected_status,
        "request_id": request_id,
        "generation_id": generation_id,
    }
    for field, expected_value in expected.items():
        if payload.get(field) != expected_value:
            raise RuntimeError(
                f"HF cache marker mismatch at {path} for {field}: "
                f"expected {expected_value!r}, got {payload.get(field)!r}"
            )
    return payload


def wait_for_local_cache_ready(
    paths: HfCacheSyncPaths,
    *,
    timeout_s: float,
    poll_s: float,
    external_failure_check: Callable[[], None] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Wait for this generation's node builder without touching NCCL."""
    deadline = monotonic() + timeout_s
    while True:
        if external_failure_check is not None:
            external_failure_check()
        if paths.failure.exists():
            payload = _read_json_marker(
                paths.failure,
                expected_status="failed",
                request_id=paths.request_id,
                generation_id=paths.generation_id,
            )
            if external_failure_check is None:
                raise DistributedCacheError.from_payload(payload)
            # The builder publishes the local marker before the canonical store
            # failure. Do not let a peer escape with a different exception type.
            now = monotonic()
            if now >= deadline:
                raise DistributedCacheError.from_payload(payload)
            sleep(min(poll_s, max(0.0, deadline - now)))
            continue
        if paths.ready.exists():
            _read_json_marker(
                paths.ready,
                expected_status="ready",
                request_id=paths.request_id,
                generation_id=paths.generation_id,
            )
            return
        now = monotonic()
        if now >= deadline:
            raise TimeoutError(
                "Timed out waiting for node-local HF Arrow cache preparation: "
                f"request_id={paths.request_id}, generation_id={paths.generation_id}, "
                f"ready_marker={paths.ready}, failure_marker={paths.failure}, "
                f"timeout_s={timeout_s}, poll_s={poll_s}. Inspect local-rank-0 logs on this node."
            )
        sleep(min(poll_s, max(0.0, deadline - now)))


def require_prepared_cache(paths: HfCacheSyncPaths) -> dict[str, Any]:
    """Validate strict force-load identity and all required artifacts."""
    if paths.prepared_ready.exists():
        payload = _read_json_marker(
            paths.prepared_ready,
            expected_status="prepared",
            request_id=paths.request_id,
            generation_id=None,
        )
        return validate_prepared_cache_manifest(paths, payload)

    if paths.legacy_failure.exists() or paths.legacy_ready.exists():
        raise DistributedCacheError.from_exception(
            RuntimeError(
                "Legacy HF cache markers cannot prove Arrow artifact completeness. Strict force-load "
                "refuses to rebuild; remove/migrate the legacy marker by running cache preparation "
                f"with the current code. legacy_ready={paths.legacy_ready}"
            ),
            request_id=paths.request_id,
            generation_id=paths.generation_id,
            rank=-1,
            local_rank=-1,
        )

    raise DistributedCacheError.from_exception(
        RuntimeError(
            "--force_load_cache is enabled, but no manifest-backed prepared cache exists for "
            f"request_id={paths.request_id}. Expected {paths.prepared_ready}. "
            "Strict force-load will not build; run cache preparation first."
        ),
        request_id=paths.request_id,
        generation_id=paths.generation_id,
        rank=-1,
        local_rank=-1,
    )


def load_with_local_cache_sync(
    load_fn: Callable[[], _T],
    *,
    paths: HfCacheSyncPaths,
    is_builder: bool,
    force_load_cache: bool,
    rank: int,
    local_rank: int,
    settings: HfCacheSyncSettings,
    prepared_manifest_factory: Callable[[_T], Mapping[str, Any]] | None = None,
    force_load_fn: Callable[[Mapping[str, Any]], _T] | None = None,
    external_failure_publisher: Callable[[BaseException], None] | None = None,
    external_failure_check: Callable[[], None] | None = None,
) -> _T:
    """Run one cache attempt and canonicalize every rank's failure outcome."""

    def propagate_failure(error: BaseException, *, publish_local: bool) -> None:
        if external_failure_publisher is None:
            if publish_local:
                with contextlib.suppress(Exception):
                    publish_local_cache_failure(
                        paths,
                        rank=rank,
                        local_rank=local_rank,
                        error=error,
                    )
            raise error
        if publish_local:
            with contextlib.suppress(Exception):
                publish_local_cache_failure(
                    paths,
                    rank=rank,
                    local_rank=local_rank,
                    error=error,
                )
        if external_failure_publisher is not None:
            external_failure_publisher(error)
        if external_failure_check is not None:
            # This acknowledges the canonical store failure and waits for every
            # rank to make the same retry/abort decision before raising.
            external_failure_check()
        raise DistributedCacheError.from_exception(
            error,
            request_id=paths.request_id,
            generation_id=paths.generation_id,
            rank=rank,
            local_rank=local_rank,
        )

    if force_load_cache:
        try:
            manifest = require_prepared_cache(paths)
            if force_load_fn is None:
                raise RuntimeError("Strict force-load requires a reuse-only force_load_fn")
            return force_load_fn(manifest)
        except Exception as exc:
            propagate_failure(exc, publish_local=False)

    if is_builder:
        begin_local_cache_build(paths)
        try:
            result = load_fn()
            if prepared_manifest_factory is None:
                raise RuntimeError("Cache builder requires prepared_manifest_factory")
            prepared_manifest = prepared_manifest_factory(result)
            publish_local_cache_ready(
                paths,
                rank=rank,
                local_rank=local_rank,
                prepared_manifest=prepared_manifest,
            )
            return result
        except Exception as exc:
            propagate_failure(exc, publish_local=True)

    wait_for_local_cache_ready(
        paths,
        timeout_s=settings.timeout_s,
        poll_s=settings.poll_s,
        external_failure_check=external_failure_check,
    )
    try:
        return load_fn()
    except Exception as exc:
        propagate_failure(exc, publish_local=False)


def _store_namespace(request_id: str, generation_id: str) -> str:
    return f"openpi/hf-cache/v{_MARKER_PROTOCOL_VERSION}/{request_id}/{generation_id}"


def _global_failure_key(request_id: str, generation_id: str) -> str:
    return f"{_store_namespace(request_id, generation_id)}/failure"


def publish_global_cache_failure(
    store: StoreLike,
    *,
    request_id: str,
    generation_id: str,
    rank: int,
    local_rank: int,
    error: BaseException,
) -> dict[str, Any]:
    """Publish the first failure as the canonical generation outcome."""
    payload = _failure_payload(
        error,
        request_id=request_id,
        generation_id=generation_id,
        rank=rank,
        local_rank=local_rank,
    )
    failure_key = _global_failure_key(request_id, generation_id)
    canonical = store.compare_set(failure_key, "", json.dumps(payload, sort_keys=True))
    try:
        canonical_payload = json.loads(canonical.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Canonical HF cache failure is malformed at store key {failure_key}") from exc
    if not isinstance(canonical_payload, dict):
        raise RuntimeError(f"Canonical HF cache failure must be a JSON object at store key {failure_key}")
    return canonical_payload


def observe_global_cache_failure(
    store: StoreLike,
    *,
    request_id: str,
    generation_id: str,
    rank: int,
    world_size: int,
    timeout_s: float,
    poll_s: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Acknowledge one canonical failure before every rank raises it."""
    namespace = _store_namespace(request_id, generation_id)
    failure_key = _global_failure_key(request_id, generation_id)
    if not store.check([failure_key]):
        return
    payload = _decode_json_store_value(store, failure_key, what="Global HF cache failure status")
    ack_keys = [f"{namespace}/failure-observed/{peer_rank}" for peer_rank in range(world_size)]
    store.set(ack_keys[rank], "observed")
    deadline = monotonic() + timeout_s
    while not store.check(ack_keys):
        now = monotonic()
        if now >= deadline:
            missing_ranks = [peer_rank for peer_rank, key in enumerate(ack_keys) if not store.check([key])]
            raise DistributedCacheError(
                "Timed out establishing rank-symmetric HF cache failure decision: "
                f"request_id={request_id}, generation_id={generation_id}, missing_ranks={missing_ranks}",
                retryable=False,
                request_id=request_id,
                generation_id=generation_id,
                origin_rank=int(payload.get("rank", -1)),
                error_type="FailureConsensusTimeout",
                error_text=f"missing_ranks={missing_ranks}",
            )
        sleep(min(poll_s, max(0.0, deadline - now)))
    raise DistributedCacheError.from_payload(payload)


def _wait_for_global_cache_phase(
    store: StoreLike,
    *,
    request_id: str,
    generation_id: str,
    rank: int,
    world_size: int,
    timeout_s: float,
    poll_s: float,
    phase: str,
    description: str,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    namespace = _store_namespace(request_id, generation_id)
    ready_keys = [f"{namespace}/{phase}/{peer_rank}" for peer_rank in range(world_size)]
    store.set(ready_keys[rank], "ready")

    deadline = monotonic() + timeout_s
    while True:
        observe_global_cache_failure(
            store,
            request_id=request_id,
            generation_id=generation_id,
            rank=rank,
            world_size=world_size,
            timeout_s=timeout_s,
            poll_s=poll_s,
        )
        if store.check(ready_keys):
            return
        now = monotonic()
        if now >= deadline:
            missing_ranks = [peer_rank for peer_rank, key in enumerate(ready_keys) if not store.check([key])]
            raise TimeoutError(
                f"Timed out waiting for all ranks to finish {description} through the c10d store: "
                f"request_id={request_id}, generation_id={generation_id}, rank={rank}, "
                f"missing_ranks={missing_ranks}, timeout_s={timeout_s}, poll_s={poll_s}. "
                "No NCCL collective is pending."
            )
        sleep(min(poll_s, max(0.0, deadline - now)))


def wait_for_global_cache_setup(
    store: StoreLike,
    *,
    request_id: str,
    generation_id: str,
    rank: int,
    world_size: int,
    timeout_s: float,
    poll_s: float,
) -> None:
    """Coordinate generation-scoped local filesystem setup before cache loading."""
    _wait_for_global_cache_phase(
        store,
        request_id=request_id,
        generation_id=generation_id,
        rank=rank,
        world_size=world_size,
        timeout_s=timeout_s,
        poll_s=poll_s,
        phase="setup-ready",
        description="node-local HF cache setup",
    )


def coordinate_global_cache_setup(
    setup_fn: Callable[[], _T],
    *,
    store: StoreLike | None,
    request_id: str,
    generation_id: str,
    rank: int,
    local_rank: int,
    world_size: int,
    settings: HfCacheSyncSettings,
) -> _T:
    """Run rank-local cache setup under one canonical generation outcome."""

    def propagate_failure(error: BaseException) -> None:
        if store is None:
            raise DistributedCacheError.from_exception(
                error,
                request_id=request_id,
                generation_id=generation_id,
                rank=rank,
                local_rank=local_rank,
            ) from error
        publish_global_cache_failure(
            store,
            request_id=request_id,
            generation_id=generation_id,
            rank=rank,
            local_rank=local_rank,
            error=error,
        )
        observe_global_cache_failure(
            store,
            request_id=request_id,
            generation_id=generation_id,
            rank=rank,
            world_size=world_size,
            timeout_s=settings.timeout_s,
            poll_s=settings.poll_s,
        )
        raise AssertionError("unreachable after cache setup failure consensus")

    try:
        result = setup_fn()
    except Exception as exc:
        propagate_failure(exc)

    if store is not None:
        try:
            wait_for_global_cache_setup(
                store,
                request_id=request_id,
                generation_id=generation_id,
                rank=rank,
                world_size=world_size,
                timeout_s=settings.timeout_s,
                poll_s=settings.poll_s,
            )
        except DistributedCacheError:
            raise
        except Exception as exc:
            propagate_failure(exc)
    return result


def wait_for_global_cache_readiness(
    store: StoreLike,
    *,
    request_id: str,
    generation_id: str,
    rank: int,
    world_size: int,
    timeout_s: float,
    poll_s: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Coordinate current-generation post-load readiness through TCPStore."""
    _wait_for_global_cache_phase(
        store,
        request_id=request_id,
        generation_id=generation_id,
        rank=rank,
        world_size=world_size,
        timeout_s=timeout_s,
        poll_s=poll_s,
        phase="ready",
        description="HF Arrow cache loading",
        monotonic=monotonic,
        sleep=sleep,
    )
