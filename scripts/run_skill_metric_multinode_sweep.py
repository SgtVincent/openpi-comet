#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shlex
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR_DIR_DEFAULT = Path("/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K")
DEMO_DATA_PATH_DEFAULT = Path("/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos")
RAWDATA_PATH_DEFAULT = Path("/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-rawdata")
DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "openpi_comet" / "pi05-b1kpt50-cs32"
DEFAULT_CONFIG = "pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k"
PERF_OPT_DOC_HINT = "BEHAVIOR-1K/.trae/rules/performance_optimization.md"
REGISTRY_REL_PATH = Path("OmniGibson/omnigibson/learning/utils/segment_skill_metric_registry.py")
TASK_MAPPING_PATH = REPO_ROOT / "scripts" / "task_mapping.json"
CONDA_SH = "/mnt/bn/behavior-data-hl/chenjunting/miniconda3/etc/profile.d/conda.sh"
RESULT_ENV_TASK_SUCCESS_BEFORE_SEGMENT_SUCCESS = "env_task_success_before_segment_success"
ATTEMPTABLE_RESULT_TYPES = {
    "predicate_satisfied",
    "short_proxy_success",
    "likely_proxy_false_positive",
    "timeout",
    "truncated",
    "env_terminated",
    "short_video_problem",
    RESULT_ENV_TASK_SUCCESS_BEFORE_SEGMENT_SUCCESS,
}
METRIC_INVALID_PREFIX = "metric_invalid"
RESULT_METRIC_INVALID_MISSING_OBJECT = "metric_invalid_missing_object"
RESULT_PRE_SATISFIED_START = "pre_satisfied_start"
RESULT_SHORT_PROXY_SUCCESS = "short_proxy_success"
RESULT_LIKELY_PROXY_FALSE_POSITIVE = "likely_proxy_false_positive"
RESULT_SHORT_VIDEO_PROBLEM = "short_video_problem"
RESULT_STATE_INVALID_RESTORE_COMPONENTS = "state_invalid_restore_components"
RESULT_STATE_INVALID_ROLLOUT_COMPONENTS = "state_invalid_rollout_components"
RESULT_SPEC_INVALID_COMPONENT_DEPENDENCY = "spec_invalid_component_dependency"
COMPONENT_INVALID_RESULT_TYPES = {
    RESULT_STATE_INVALID_RESTORE_COMPONENTS,
    RESULT_STATE_INVALID_ROLLOUT_COMPONENTS,
    RESULT_SPEC_INVALID_COMPONENT_DEPENDENCY,
}
COMPONENT_EVALUABILITY_MAX_LIST_ITEMS = 64
COMPONENT_EVALUABILITY_MAX_DEPENDENCY_ROWS = 64
COMPONENT_EVALUABILITY_MAX_RESTORE_STAGES = 3
COMPONENT_EVALUABILITY_MAX_REASON_CHARS = 512
COMPONENT_EVALUABILITY_MAX_NAME_CHARS = 128
SHORT_VIDEO_PROBLEM_STEP_THRESHOLD = 150
TRANSFER_POSE_PROXY_FAMILY = "transfer_pose_proxy"
CONTACT_EFFECT_PROXY_FAMILY = "contact_effect_proxy"
ARTICULATION_CLOSE_FAMILY = "articulation_close"
ARTICULATION_CLOSE_PROXY_FAMILY = "articulation_close_proxy"
ARTICULATION_OPEN_FAMILY = "articulation_open"
ARTICULATION_OPEN_PROXY_FAMILY = "articulation_open_proxy"
GEOMETRY_BASE_FACING_FAMILY = "geometry_base_facing"
ORIENTATION_PROXY_FAMILY = "orientation_proxy"
RELATION_TRANSFER_PROXY_FAMILY = "relation_transfer_proxy"


def is_short_video_problem_row(row: Dict[str, Any]) -> bool:
    """Flag predicate successes that terminate before producing a meaningful review video.

    Rollout videos are written at roughly 30 FPS in the segment eval pipeline, so 150
    rollout steps corresponds to the manual-review threshold of about five seconds.
    User review found these very short successful videos are not representative of
    actual skill completion, even when they are not a metric-family-specific proxy.
    """
    if row.get("result_type") != "predicate_satisfied" or not bool(row.get("success")):
        return False
    final_step = row.get("final_step")
    try:
        return final_step is not None and int(final_step) < SHORT_VIDEO_PROBLEM_STEP_THRESHOLD
    except (TypeError, ValueError):
        return False


def is_transfer_pose_proxy_success_unconfirmed(row: Dict[str, Any]) -> bool:
    """Treat transfer-pose proxy successes as review-needed, not clean success.

    These metrics only check that the object reaches the demonstration end pose
    while still grasped. Manual review found this can be a proxy-only hit for
    hand-over skills: the object pose matches numerically, but the video does
    not show a stable semantic hand-over.
    """
    return (
        row.get("result_type") == "predicate_satisfied"
        and bool(row.get("success"))
        and row.get("metric_family") == TRANSFER_POSE_PROXY_FAMILY
    )


def is_contact_effect_proxy_success_unconfirmed(row: Dict[str, Any]) -> bool:
    """Treat contact/effect proxy successes as review-needed, not clean success.

    Manual video audit found `contact_effect_proxy` metrics can be satisfied by
    brief or static tool-object contact without completing the intended semantic
    action (for example, a scrub brush merely touching a trumpet instead of
    visibly wiping it).
    """
    return (
        row.get("result_type") == "predicate_satisfied"
        and bool(row.get("success"))
        and row.get("metric_family") == CONTACT_EFFECT_PROXY_FAMILY
    )


def is_articulation_close_success_unconfirmed(row: Dict[str, Any]) -> bool:
    """Treat articulation-close successes as review-needed, not clean success.

    Manual review of can_meat close-lid videos found `open == False` can be
    satisfied without a stable semantic lid/container close.
    """
    return (
        row.get("result_type") == "predicate_satisfied"
        and bool(row.get("success"))
        and row.get("metric_family") == ARTICULATION_CLOSE_FAMILY
    )


def is_articulation_close_proxy_success_unconfirmed(row: Dict[str, Any]) -> bool:
    """Treat articulation-close proxy successes as review-needed, not clean success.

    `push tray` currently proxies success via `open == False` on the carrier
    articulation. This is useful for raw analysis but still too weak for the
    conservative clean-success bucket.
    """
    return (
        row.get("result_type") == "predicate_satisfied"
        and bool(row.get("success"))
        and row.get("metric_family") == ARTICULATION_CLOSE_PROXY_FAMILY
    )


def is_geometry_base_facing_success_unconfirmed(row: Dict[str, Any]) -> bool:
    """Treat base-facing geometry successes as review-needed, not clean success.

    This family only checks a yaw-facing proxy against a demo-derived target,
    so it is useful as a raw metric but too weak for conservative clean success.
    """
    return (
        row.get("result_type") == "predicate_satisfied"
        and bool(row.get("success"))
        and row.get("metric_family") == GEOMETRY_BASE_FACING_FAMILY
    )


def is_relation_transfer_proxy_success_unconfirmed(row: Dict[str, Any]) -> bool:
    """Treat relation-transfer proxy successes as review-needed, not clean success.

    Pour/transfer relation proxies can collapse rich transfer semantics into a
    loose `ontop`/`inside` relation, sometimes with payload roles falling back to
    the manipulated object.
    """
    return (
        row.get("result_type") == "predicate_satisfied"
        and bool(row.get("success"))
        and row.get("metric_family") == RELATION_TRANSFER_PROXY_FAMILY
    )


def is_orientation_proxy_success_unconfirmed(row: Dict[str, Any]) -> bool:
    """Treat orientation-only proxy successes as review-needed, not clean success."""
    return (
        row.get("result_type") == "predicate_satisfied"
        and bool(row.get("success"))
        and row.get("metric_family") == ORIENTATION_PROXY_FAMILY
    )


def is_articulation_open_proxy_success_unconfirmed(row: Dict[str, Any]) -> bool:
    """Treat articulation-open proxy successes as review-needed, not clean success."""
    return (
        row.get("result_type") == "predicate_satisfied"
        and bool(row.get("success"))
        and row.get("metric_family") == ARTICULATION_OPEN_PROXY_FAMILY
    )


def is_articulation_open_success_unconfirmed(row: Dict[str, Any]) -> bool:
    """Treat ordinary open-door/drawer articulation successes as review-needed.

    Manual review found fridge-door and cabinet/drawer `open(...)` successes can
    enter conservative clean success even when final videos still look closed.
    Keep the raw articulation-open metric, but exclude these high-risk skills
    from the conservative clean-success count until the articulation predicate
    can be strengthened with final-state / magnitude checks.
    """
    return (
        row.get("result_type") == "predicate_satisfied"
        and bool(row.get("success"))
        and row.get("metric_family") == ARTICULATION_OPEN_FAMILY
        and row.get("skill") in {"open door", "open drawer"}
    )


def is_can_meat_lid_success_unconfirmed(row: Dict[str, Any]) -> bool:
    """Treat can_meat lid articulation successes as review-needed.

    Manual video audit found can_meat lid metrics can be satisfied even when
    the lid/can is dropped or does not remain in a semantically valid terminal
    state. Keep the raw metric but exclude it from conservative clean success.
    """
    return (
        row.get("result_type") == "predicate_satisfied"
        and bool(row.get("success"))
        and row.get("task_name") == "can_meat"
        and row.get("skill") in {"open lid", "close lid"}
        and row.get("metric_family") in {ARTICULATION_OPEN_FAMILY, ARTICULATION_CLOSE_FAMILY}
    )


def _safe_int(value: Any) -> Optional[int]:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _normalize_optional_bool(value: Any) -> Optional[bool]:
    """Normalize producer booleans without conflating a missing value with false."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    return None


def _compact_component_text(value: Any, *, max_chars: int) -> Optional[str]:
    if not isinstance(value, (str, int, float, bool)):
        return None
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _normalize_component_list(value: Any) -> Tuple[Optional[List[str]], bool]:
    if not isinstance(value, (list, tuple)):
        return None, False
    normalized = {
        text
        for item in value[:COMPONENT_EVALUABILITY_MAX_LIST_ITEMS]
        if (text := _compact_component_text(item, max_chars=COMPONENT_EVALUABILITY_MAX_NAME_CHARS)) is not None
    }
    return sorted(normalized), len(value) > COMPONENT_EVALUABILITY_MAX_LIST_ITEMS


def _normalize_dependency_ambiguity(value: Any, *, depth: int = 0) -> Any:
    if isinstance(value, bool):
        return value
    scalar = _compact_component_text(value, max_chars=COMPONENT_EVALUABILITY_MAX_REASON_CHARS)
    if scalar is not None:
        return scalar
    if depth >= 2:
        return None
    if isinstance(value, (list, tuple)):
        normalized_list = []
        for item in value[:COMPONENT_EVALUABILITY_MAX_LIST_ITEMS]:
            normalized_item = _normalize_dependency_ambiguity(item, depth=depth + 1)
            if normalized_item is not None:
                normalized_list.append(normalized_item)
        return normalized_list
    if isinstance(value, dict):
        normalized_dict: Dict[str, Any] = {}
        for index, (raw_key, raw_value) in enumerate(value.items()):
            if index >= COMPONENT_EVALUABILITY_MAX_LIST_ITEMS:
                break
            key = _compact_component_text(raw_key, max_chars=COMPONENT_EVALUABILITY_MAX_NAME_CHARS)
            if key is None:
                continue
            normalized_value = _normalize_dependency_ambiguity(raw_value, depth=depth + 1)
            if normalized_value is not None:
                normalized_dict[key] = normalized_value
        return dict(sorted(normalized_dict.items()))
    return None


def _normalize_dependency_rows(value: Any) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
    if not isinstance(value, (list, tuple)):
        return None, False
    rows: List[Dict[str, Any]] = []
    truncated = len(value) > COMPONENT_EVALUABILITY_MAX_DEPENDENCY_ROWS
    for raw_row in value[:COMPONENT_EVALUABILITY_MAX_DEPENDENCY_ROWS]:
        if not isinstance(raw_row, dict):
            continue
        row: Dict[str, Any] = {}
        for key in ("metric_type", "name", "metric_family"):
            text = _compact_component_text(raw_row.get(key), max_chars=COMPONENT_EVALUABILITY_MAX_NAME_CHARS)
            if text is not None:
                row[key] = text
        if "required_components" in raw_row:
            components, components_truncated = _normalize_component_list(raw_row.get("required_components"))
            if components is not None:
                row["required_components"] = components
            truncated = truncated or components_truncated
        if "dependency_ambiguity" in raw_row:
            raw_ambiguity = raw_row.get("dependency_ambiguity")
            ambiguity = _normalize_dependency_ambiguity(raw_ambiguity)
            if ambiguity is not None:
                row["dependency_ambiguity"] = ambiguity
            if isinstance(raw_ambiguity, (list, tuple, dict)):
                truncated = truncated or len(raw_ambiguity) > COMPONENT_EVALUABILITY_MAX_LIST_ITEMS
        if row:
            rows.append(row)
    return rows, truncated


def _normalize_restore_stages(value: Any) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
    if isinstance(value, dict):
        source_truncated = len(value) > COMPONENT_EVALUABILITY_MAX_RESTORE_STAGES
        raw_stages = []
        for index, (stage, entry) in enumerate(value.items()):
            if index >= COMPONENT_EVALUABILITY_MAX_RESTORE_STAGES * 2:
                break
            if isinstance(entry, dict):
                raw_stages.append({"stage": stage, **entry})
            if len(raw_stages) > COMPONENT_EVALUABILITY_MAX_RESTORE_STAGES:
                break
    elif isinstance(value, (list, tuple)):
        source_truncated = len(value) > COMPONENT_EVALUABILITY_MAX_RESTORE_STAGES
        raw_stages = value
    else:
        return None, False

    stages: List[Dict[str, Any]] = []
    truncated = source_truncated
    for raw_stage in raw_stages[:COMPONENT_EVALUABILITY_MAX_RESTORE_STAGES]:
        if not isinstance(raw_stage, dict):
            continue
        stage: Dict[str, Any] = {}
        stage_name = _compact_component_text(
            raw_stage.get("stage"), max_chars=COMPONENT_EVALUABILITY_MAX_NAME_CHARS
        )
        if stage_name is not None:
            stage["stage"] = stage_name
        for key in ("validity_present", "historical_world_state_exact"):
            if key in raw_stage:
                optional_bool = _normalize_optional_bool(raw_stage.get(key))
                if optional_bool is not None:
                    stage[key] = optional_bool
        for key in ("boundary_missing_components", "rollout_missing_components"):
            if key in raw_stage:
                components, components_truncated = _normalize_component_list(raw_stage.get(key))
                if components is not None:
                    stage[key] = components
                truncated = truncated or components_truncated
        if stage:
            stages.append(stage)
    return stages, truncated


def normalize_component_evaluability(value: Any) -> Optional[Dict[str, Any]]:
    """Preserve the producer's component-validity schema without large debug payloads."""
    if value is None:
        return None
    if not isinstance(value, dict):
        return {"malformed": True, "input_type": type(value).__name__}

    normalized: Dict[str, Any] = {}
    malformed_fields: List[str] = []
    truncated = False

    bool_keys = (
        "evaluable",
        "aggregation_eligible",
        "model_failure_eligible",
        "boundary_evaluable",
        "boundary_aggregation_eligible",
        "rollout_evaluable",
    )
    for key in bool_keys:
        if key not in value:
            continue
        optional_bool = _normalize_optional_bool(value.get(key))
        if optional_bool is None:
            malformed_fields.append(key)
        else:
            normalized[key] = optional_bool

    for key in ("result_type", "boundary_result_type", "rollout_result_type"):
        if key not in value:
            continue
        text = _compact_component_text(value.get(key), max_chars=COMPONENT_EVALUABILITY_MAX_REASON_CHARS)
        if text is None:
            malformed_fields.append(key)
        else:
            normalized[key] = text

    component_list_keys = (
        "boundary_required_components",
        "boundary_missing_components",
        "rollout_required_components",
        "rollout_missing_components",
        "required_components",
        "missing_components",
    )
    for key in component_list_keys:
        if key not in value:
            continue
        components, list_truncated = _normalize_component_list(value.get(key))
        if components is None:
            malformed_fields.append(key)
        else:
            normalized[key] = components
        truncated = truncated or list_truncated

    if "dependency_ambiguities" in value:
        dependency_ambiguities = _normalize_dependency_ambiguity(value.get("dependency_ambiguities"))
        if dependency_ambiguities is None:
            malformed_fields.append("dependency_ambiguities")
        else:
            normalized["dependency_ambiguities"] = dependency_ambiguities
        raw_ambiguities = value.get("dependency_ambiguities")
        if isinstance(raw_ambiguities, (list, tuple)):
            truncated = truncated or len(raw_ambiguities) > COMPONENT_EVALUABILITY_MAX_LIST_ITEMS
        elif isinstance(raw_ambiguities, dict):
            truncated = truncated or len(raw_ambiguities) > COMPONENT_EVALUABILITY_MAX_LIST_ITEMS

    if "dependency_rows" in value:
        dependency_rows, rows_truncated = _normalize_dependency_rows(value.get("dependency_rows"))
        if dependency_rows is None:
            malformed_fields.append("dependency_rows")
        else:
            normalized["dependency_rows"] = dependency_rows
        truncated = truncated or rows_truncated

    if "restore_stages" in value:
        restore_stages, stages_truncated = _normalize_restore_stages(value.get("restore_stages"))
        if restore_stages is None:
            malformed_fields.append("restore_stages")
        else:
            normalized["restore_stages"] = restore_stages
        truncated = truncated or stages_truncated

    if malformed_fields:
        normalized["malformed_fields"] = sorted(set(malformed_fields))
    if truncated:
        normalized["truncated"] = True
    if not normalized:
        normalized["malformed"] = True
    return normalized


def is_early_metric_activation_review_needed(row: Dict[str, Any]) -> bool:
    if row.get("result_type") != "predicate_satisfied" or not bool(row.get("success")):
        return False
    if bool(row.get("start_all_satisfied")):
        return False

    early_hits = _safe_int(row.get("early_predicate_satisfied_steps"))
    if early_hits is not None and early_hits > 0:
        return True

    first_hit = _safe_int(row.get("first_predicate_satisfied_step"))
    min_steps = _safe_int(row.get("min_success_steps"))
    return first_hit is not None and min_steps is not None and min_steps > 0 and first_hit < min_steps


def has_meaningful_policy_caused_transition(row: Dict[str, Any]) -> bool:
    if row.get("result_type") != "predicate_satisfied" or not bool(row.get("success")):
        return False
    if bool(row.get("start_all_satisfied")):
        return False
    if bool(row.get("early_metric_activation_review_needed")) or is_early_metric_activation_review_needed(row):
        return False

    first_hit = _safe_int(row.get("first_predicate_satisfied_step"))
    min_steps = _safe_int(row.get("min_success_steps"))
    if first_hit is None or min_steps is None or min_steps <= 0:
        return False
    return first_hit >= min_steps


def _restore_entry_rawdata_failed(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    debug = entry.get("debug")
    rawdata = debug.get("rawdata") if isinstance(debug, dict) else None
    return bool(isinstance(rawdata, dict) and rawdata.get("attempted") and not rawdata.get("restored"))


def _restore_entry_rawdata_fallback(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    debug = entry.get("debug")
    if not isinstance(debug, dict):
        return False
    rawdata = debug.get("rawdata")
    return bool(
        isinstance(rawdata, dict)
        and rawdata.get("attempted")
        and not rawdata.get("restored")
        and debug.get("selected_method") in {"cache", "robot"}
    )


def _restore_entry_selected_method(entry: Any) -> Optional[str]:
    if not isinstance(entry, dict):
        return None
    debug = entry.get("debug")
    if isinstance(debug, dict) and debug.get("selected_method") is not None:
        return str(debug.get("selected_method"))
    method = entry.get("method")
    return None if method is None else str(method)


def is_env_task_success_grasp_release_review_needed(row: Dict[str, Any]) -> bool:
    return (
        row.get("result_type") == RESULT_ENV_TASK_SUCCESS_BEFORE_SEGMENT_SUCCESS
        and row.get("metric_family") == "grasp_release"
        and bool(row.get("attemptable", False))
    )


def q(text: Any) -> str:
    return shlex.quote(str(text))


def load_task_mapping(task_mapping_path: Path = TASK_MAPPING_PATH) -> Dict[str, Dict[str, Any]]:
    with task_mapping_path.open() as f:
        return json.load(f)


def build_server_identity(*, out_dir: Path, worker_rank: int, task_name: str, port: int) -> Dict[str, str]:
    task_mapping = load_task_mapping()
    task_prompt = str(task_mapping[task_name]["task"])
    server_run_id = f"{out_dir.name}:worker{worker_rank:03d}:{task_name}:p{int(port)}"
    server_token = hashlib.sha256(server_run_id.encode("utf-8")).hexdigest()
    return {
        "server_run_id": server_run_id,
        "server_token": server_token,
        "task_name": task_name,
        "task_prompt_sha256": hashlib.sha256(task_prompt.encode("utf-8")).hexdigest(),
    }


def parse_csv_ints(text: str) -> List[int]:
    values: List[int] = []
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def parse_csv_strings(text: str) -> List[str]:
    return [chunk.strip().lower() for chunk in text.split(",") if chunk.strip()]


def flatten_skill_desc(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip().lower()
        return [s] if s else []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            out.extend(flatten_skill_desc(item))
        return out
    s = str(value).strip().lower()
    return [s] if s else []


def load_registry(registry_path: Path) -> Dict[str, Dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("skill_metric_registry", registry_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load registry from {registry_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.SKILL_METRIC_REGISTRY)


def parse_frame_duration(frame_duration: Any) -> Optional[Tuple[int, int]]:
    try:
        if isinstance(frame_duration, (list, tuple)) and len(frame_duration) == 2:
            return int(frame_duration[0]), int(frame_duration[1])
        if isinstance(frame_duration, str):
            import ast

            parsed = ast.literal_eval(frame_duration)
            if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                return int(parsed[0]), int(parsed[1])
    except Exception:
        return None
    return None


def get_dynamic_max_steps(frame_duration: Any, fallback: int, cap: int = 0) -> int:
    parsed = parse_frame_duration(frame_duration)
    if parsed is None:
        steps = fallback
    else:
        start, end = parsed
        duration = end - start
        if duration <= 0:
            steps = fallback
        else:
            steps = duration * 2
    if cap and cap > 0:
        steps = min(steps, cap)
    return steps


def resolve_checkpoint_dir(path: Path) -> Path:
    path = path.resolve()
    if (path / "params").is_dir() or (path / "model.safetensors").exists():
        return path
    step_dirs = []
    for child in path.iterdir():
        if child.is_dir() and child.name.isdigit():
            if (child / "params").is_dir() or (child / "model.safetensors").exists():
                step_dirs.append(child)
    if step_dirs:
        return sorted(step_dirs, key=lambda p: int(p.name))[-1]
    return path


def wait_for_server(port: int, timeout_s: int, expected_identity: Optional[Dict[str, str]] = None) -> None:
    start = time.time()
    last_log = start
    health_url = f"http://127.0.0.1:{port}/healthz"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.time() - start < timeout_s:
        try:
            req = urllib.request.Request(health_url, headers={"Connection": "close"})
            with opener.open(req, timeout=2.0) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                if 200 <= int(status) < 300:
                    if expected_identity:
                        payload = json.loads(resp.read().decode("utf-8"))
                        mismatches = [
                            f"{key}: expected={expected!r}, actual={payload.get(key)!r}"
                            for key, expected in expected_identity.items()
                            if expected is not None and payload.get(key) != expected
                        ]
                        if mismatches:
                            raise RuntimeError(
                                f"unexpected server identity at {health_url}: " + "; ".join(mismatches)
                            )
                    return
        except (urllib.error.URLError, ConnectionError, OSError, socket.timeout, ValueError):
            now = time.time()
            # Avoid silent long waits in worker logs.
            if now - last_log >= 30:
                print(f"[worker] waiting for server healthz {health_url}... elapsed={int(now-start)}s", flush=True)
                last_log = now
            time.sleep(1.0)
        else:
            # Non-2xx HTTP response: keep waiting.
            now = time.time()
            if now - last_log >= 30:
                print(f"[worker] server healthz not ready yet at {health_url}... elapsed={int(now-start)}s", flush=True)
                last_log = now
            time.sleep(1.0)
    raise TimeoutError(f"server healthz not ready at {health_url} after {timeout_s}s")


def is_port_free(port: int) -> bool:
    """Best-effort check that a TCP port is available for bind.

    Note: this is inherently racy; callers should still verify the spawned server stays alive.
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", int(port)))
        except OSError:
            return False
        return True


def find_free_port(preferred: int, *, stride: int = 1, max_tries: int = 50) -> int:
    """Find an available port near `preferred`.

    `stride` is used to keep port selections disjoint across workers (e.g. stride=gpus_per_node).
    """

    preferred = int(preferred)
    stride = max(1, int(stride))
    for attempt in range(max(1, int(max_tries))):
        candidate = preferred + attempt * stride
        if is_port_free(candidate):
            return candidate
    raise RuntimeError(f"no free port found near {preferred} (stride={stride}, max_tries={max_tries})")


def wait_for_port_free(port: int, *, timeout_s: float = 30.0) -> bool:
    """Wait briefly for a just-stopped server to release its listening port."""

    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if is_port_free(port):
            return True
        time.sleep(0.5)
    return is_port_free(port)


def wait_for_server_proc(
    *,
    proc: subprocess.Popen[str],
    port: int,
    timeout_s: int,
    log_file: Optional[Path] = None,
    expected_identity: Optional[Dict[str, str]] = None,
) -> None:
    """Wait for server health and fail fast if the process exits."""

    start = time.time()
    last_exc: Optional[BaseException] = None
    while time.time() - start < timeout_s:
        if proc.poll() is not None:
            tail = tail_text(log_file) if log_file is not None else ""
            raise RuntimeError(
                f"server process exited before becoming healthy (code={proc.returncode}). "
                f"log: {log_file}\n{tail}"
            )
        try:
            wait_for_server(port, timeout_s=10, expected_identity=expected_identity)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(1.0)
    tail = tail_text(log_file) if log_file is not None else ""
    raise TimeoutError(
        f"server not ready on port {port} after {timeout_s}s. last_error={last_exc}. log: {log_file}\n{tail}"
    )


def wait_for_path(path: Path, timeout_s: int) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        if path.exists():
            return
        time.sleep(2.0)
    raise TimeoutError(f"timed out waiting for {path}")


def tail_text(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return f"[launcher] log file does not exist yet: {path}"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return f"[launcher] failed to read log tail from {path}: {exc}"
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(["..."] + lines[-max_lines:])


def stop_process(proc: Optional[subprocess.Popen[str]]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def stop_stale_persistent_servers(out_dir: Path, worker_rank: int, *, timeout_s: float = 5.0) -> None:
    """Best-effort cleanup for policy servers launched by a persistent worker.

    ``start_server`` intentionally creates a new process group for the OpenPI
    policy server. If the launcher watchdog kills a wedged worker process group,
    the child server can survive as an orphan and keep its port / GPU memory.
    Limit cleanup to this run and worker rank by matching the structured
    ``--server-run-id`` prefix.
    """

    prefix = f"{out_dir.name}:worker{int(worker_rank):03d}:"
    matches: List[int] = []
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return

    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, args_text = stripped.partition(" ")
        if "scripts/serve_b1k.py" not in args_text or "--server-run-id=" not in args_text:
            continue
        if f"--server-run-id={prefix}" not in args_text:
            continue
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        matches.append(pid)

    for pid in matches:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                continue

    deadline = time.time() + max(0.0, timeout_s)
    while matches and time.time() < deadline:
        alive: List[int] = []
        for pid in matches:
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except ProcessLookupError:
                pass
            except PermissionError:
                alive.append(pid)
        if not alive:
            return
        matches = alive
        time.sleep(0.2)

    for pid in matches:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue
        except OSError:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def load_jsonl_rows(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


PERSISTENT_WORKER_DONE_EVENTS = {"segment_done", "segment_resume_hit", "segment_skipped_done"}


def persistent_worker_status_path(out_dir: Path, worker_rank: int) -> Path:
    return out_dir / "worker_status" / f"persistent_worker_{int(worker_rank):03d}.jsonl"


def summarize_persistent_worker_status(status_path: Path) -> Dict[str, Any]:
    rows = load_jsonl_rows([status_path])
    latest_row = rows[-1] if rows else {}
    latest_heartbeat = next((row for row in reversed(rows) if row.get("event") == "heartbeat"), None)
    done_count = sum(int(row.get("event") in PERSISTENT_WORKER_DONE_EVENTS) for row in rows)
    summary = {
        "status_path": str(status_path),
        "exists": status_path.exists(),
        "row_count": len(rows),
        "done_count": done_count,
        "latest_event": latest_row.get("event"),
        "latest_ts": latest_row.get("ts"),
        "latest_heartbeat_ts": latest_heartbeat.get("ts") if latest_heartbeat else None,
        "state": None,
        "phase_elapsed_s": None,
        "job_key": None,
        "task_name": None,
        "demo_id": None,
        "skill_idx": None,
    }
    source = latest_heartbeat or latest_row
    if source:
        summary.update(
            {
                "state": source.get("state"),
                "phase_elapsed_s": source.get("phase_elapsed_s"),
                "job_key": source.get("job_key"),
                "task_name": source.get("task_name") or source.get("loaded_task"),
                "demo_id": source.get("demo_id"),
                "skill_idx": source.get("skill_idx"),
            }
        )
    return summary


def persistent_worker_restart_reason(
    summary: Dict[str, Any],
    *,
    now: float,
    heartbeat_timeout_s: float,
    task_reload_timeout_s: float,
    segment_timeout_s: float,
) -> Optional[str]:
    last_ts = summary.get("latest_heartbeat_ts") or summary.get("latest_ts")
    if last_ts is None:
        return None

    age_s = max(0.0, now - float(last_ts))
    if age_s > heartbeat_timeout_s:
        return f"heartbeat_stale:{age_s:.1f}s"

    state = str(summary.get("state") or "")
    phase_elapsed_s = summary.get("phase_elapsed_s")
    try:
        phase_elapsed_s = None if phase_elapsed_s is None else float(phase_elapsed_s)
    except (TypeError, ValueError):
        phase_elapsed_s = None
    if phase_elapsed_s is None:
        return None
    if state == "task_loading" and phase_elapsed_s > task_reload_timeout_s:
        return f"task_loading_timeout:{phase_elapsed_s:.1f}s"
    if state == "segment_running" and phase_elapsed_s > segment_timeout_s:
        return f"segment_timeout:{phase_elapsed_s:.1f}s"
    return None


def build_job_key(sample: Dict[str, Any]) -> str:
    return f"{sample['skill']}|{sample['task_name']}|{sample['demo_id']}|{int(sample['skill_idx']):03d}"


def collect_skill_jobs(
    *,
    registry: Dict[str, Dict[str, Any]],
    demo_data_path: Path,
    skills_filter: Sequence[str],
    max_samples_per_skill: int,
    max_samples_per_skill_task: int,
    max_total_jobs: int,
) -> List[Dict[str, Any]]:
    targets = set(registry)
    if skills_filter:
        wanted = set(skills_filter)
        missing = sorted(wanted - targets)
        if missing:
            raise RuntimeError(f"unknown skills requested: {missing}")
        targets = wanted

    jobs: List[Dict[str, Any]] = []
    per_skill = Counter()
    per_skill_task = Counter()
    ann_root = demo_data_path / "annotations"
    for ann_path in sorted(ann_root.glob("task-*/episode_*.json")):
        with ann_path.open() as f:
            ann = json.load(f)
        task_name = str(ann.get("task_name", "")).strip().replace(" ", "_")
        instance_id = ann.get("instance_id")
        for seg in ann.get("skill_annotation", []) or []:
            desc = " ".join(flatten_skill_desc(seg.get("skill_description")))
            if desc not in targets:
                continue
            if max_samples_per_skill > 0 and per_skill[desc] >= max_samples_per_skill:
                continue
            skill_task_key = (desc, task_name)
            if max_samples_per_skill_task > 0 and per_skill_task[skill_task_key] >= max_samples_per_skill_task:
                continue

            demo_id = ann_path.stem.replace("episode_", "")
            sample = {
                "skill": desc,
                "task_dir": ann_path.parent.name,
                "task_name": task_name,
                "ann_path": str(ann_path),
                "demo_id": demo_id,
                "instance_id": instance_id,
                "skill_idx": int(seg.get("skill_idx", 0)),
                "frame_duration": seg.get("frame_duration"),
                "object_id": seg.get("object_id"),
                "manipulating_object_id": seg.get("manipulating_object_id"),
                "spatial_prefix": seg.get("spatial_prefix"),
            }
            sample["job_key"] = build_job_key(sample)
            jobs.append(sample)
            per_skill[desc] += 1
            per_skill_task[skill_task_key] += 1

            if max_samples_per_skill > 0 and all(per_skill[skill] >= max_samples_per_skill for skill in targets):
                return sorted(jobs, key=lambda row: (row["skill"], row["task_name"], row["demo_id"], row["skill_idx"]))
            if max_total_jobs > 0 and len(jobs) >= max_total_jobs:
                return sorted(jobs, key=lambda row: (row["skill"], row["task_name"], row["demo_id"], row["skill_idx"]))
    return sorted(jobs, key=lambda row: (row["skill"], row["task_name"], row["demo_id"], row["skill_idx"]))


def build_worker_assignments(jobs: List[Dict[str, Any]], total_workers: int) -> List[List[Dict[str, Any]]]:
    if total_workers <= 0:
        raise ValueError("total_workers must be positive")
    worker_jobs: List[List[Dict[str, Any]]] = [[] for _ in range(total_workers)]
    if not jobs:
        return worker_jobs

    target_chunk = max(1, math.ceil(len(jobs) / total_workers))
    jobs_by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        jobs_by_task[job["task_name"]].append(job)

    task_chunks: List[List[Dict[str, Any]]] = []
    for task_jobs in [jobs_by_task[name] for name in sorted(jobs_by_task)]:
        task_jobs = sorted(task_jobs, key=lambda row: (row["skill"], row["demo_id"], row["skill_idx"]))
        if len(task_jobs) > max(1, int(target_chunk * 1.5)):
            num_slices = math.ceil(len(task_jobs) / target_chunk)
        else:
            num_slices = 1
        slice_size = math.ceil(len(task_jobs) / num_slices)
        for idx in range(num_slices):
            chunk = task_jobs[idx * slice_size : (idx + 1) * slice_size]
            if chunk:
                task_chunks.append(chunk)

    loads = [0 for _ in range(total_workers)]
    for chunk in sorted(task_chunks, key=len, reverse=True):
        worker_idx = min(range(total_workers), key=lambda idx: (loads[idx], idx))
        worker_jobs[worker_idx].extend(chunk)
        loads[worker_idx] += len(chunk)

    for idx in range(total_workers):
        worker_jobs[idx] = sorted(
            worker_jobs[idx], key=lambda row: (row["task_name"], row["skill"], row["demo_id"], row["skill_idx"])
        )
    return worker_jobs


def render_planned_coverage_md(summary: Dict[str, Any]) -> str:
    lines = ["# Planned Multinode Skill Sweep", ""]
    lines.append(f"- out_dir: `{summary['out_dir']}`")
    lines.append(f"- total_jobs: `{summary['total_jobs']}`")
    lines.append(f"- unique_skills: `{summary['unique_skills']}`")
    lines.append(f"- unique_tasks: `{summary['unique_tasks']}`")
    lines.append(f"- unique_demos: `{summary['unique_demos']}`")
    lines.append(f"- total_workers: `{summary['total_workers']}`")
    lines.append("")
    lines.append("| Skill | Tasks | Demos | Segments |")
    lines.append("|---|---:|---:|---:|")
    for row in summary["skill_rows"]:
        lines.append(f"| {row['skill']} | {row['task_count']} | {row['demo_count']} | {row['segment_count']} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_manifest(args: argparse.Namespace, jobs: List[Dict[str, Any]]) -> None:
    total_workers = args.num_nodes * args.gpus_per_node
    worker_jobs = build_worker_assignments(jobs, total_workers)
    jobs_dir = args.out_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    worker_rows = []
    for worker_rank, worker_job_list in enumerate(worker_jobs):
        worker_path = jobs_dir / f"worker_{worker_rank:03d}.json"
        json_dump(worker_path, worker_job_list)
        worker_rows.append(
            {
                "worker_rank": worker_rank,
                "job_count": len(worker_job_list),
                "task_count": len({row["task_name"] for row in worker_job_list}),
                "skills": len({row["skill"] for row in worker_job_list}),
                "job_file": str(worker_path),
            }
        )

    grouped = defaultdict(list)
    for job in jobs:
        grouped[job["skill"]].append(job)
    skill_rows = []
    for skill, skill_jobs in sorted(grouped.items()):
        skill_rows.append(
            {
                "skill": skill,
                "task_count": len({row["task_name"] for row in skill_jobs}),
                "demo_count": len({row["demo_id"] for row in skill_jobs}),
                "segment_count": len(skill_jobs),
            }
        )

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_dir": str(args.out_dir),
        "ckpt_dir": str(args.ckpt_dir),
        "resolved_ckpt_dir": str(resolve_checkpoint_dir(args.ckpt_dir)),
        "config_name": args.config_name,
        "policy_backend": args.policy_backend,
        "num_nodes": args.num_nodes,
        "gpus_per_node": args.gpus_per_node,
        "total_workers": total_workers,
        "max_steps_fallback": args.max_steps,
        "dry_run": args.dry_run,
        "write_video": args.write_video,
        "segment_predicate_dump_trace": args.segment_predicate_dump_trace,
        "skills_filter": list(args.skills_filter),
        "max_samples_per_skill": args.max_samples_per_skill,
        "max_samples_per_skill_task": args.max_samples_per_skill_task,
        "max_total_jobs": args.max_total_jobs,
        "jobs": jobs,
        "workers": worker_rows,
    }
    json_dump(args.out_dir / "manifest.json", manifest)
    write_csv(
        args.out_dir / "worker_plan.csv",
        worker_rows,
        fieldnames=["worker_rank", "job_count", "task_count", "skills", "job_file"],
    )

    planned_summary = {
        "out_dir": str(args.out_dir),
        "total_jobs": len(jobs),
        "unique_skills": len(skill_rows),
        "unique_tasks": len({row["task_name"] for row in jobs}),
        "unique_demos": len({row["demo_id"] for row in jobs}),
        "total_workers": total_workers,
        "skill_rows": skill_rows,
    }
    json_dump(args.out_dir / "planned_coverage.json", planned_summary)
    write_csv(
        args.out_dir / "planned_skill_coverage.csv",
        skill_rows,
        fieldnames=["skill", "task_count", "demo_count", "segment_count"],
    )
    (args.out_dir / "planned_skill_coverage.md").write_text(render_planned_coverage_md(planned_summary))


def start_server(
    *,
    task_name: str,
    port: int,
    gpu_id: int,
    ckpt_dir: Path,
    config_name: str,
    policy_backend: str,
    server_run_id: str,
    server_token: str,
    openpi_env: str,
    behavior_dir: Path,
    out_dir: Path,
) -> tuple[subprocess.Popen[str], Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"server_{task_name}_gpu{gpu_id}_p{port}.log"
    cmd = f"""
set -euo pipefail
source {q(CONDA_SH)}
conda activate {q(openpi_env)}
cd {q(REPO_ROOT)}
export CUDA_VISIBLE_DEVICES={q(gpu_id)}
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONPATH={q(str(REPO_ROOT / 'src'))}:{q(str(behavior_dir / 'joylo'))}:{q(str(behavior_dir / 'OmniGibson'))}:{q(str(behavior_dir / 'bddl3'))}${{PYTHONPATH:+:$PYTHONPATH}}
python scripts/serve_b1k.py --task_name={q(task_name)} --control_mode=receeding_horizon --max_len=32 --port={q(port)} --server-run-id={q(server_run_id)} --server-token={q(server_token)} --policy-backend={q(policy_backend)} policy:checkpoint --policy.config={q(config_name)} --policy.dir={q(ckpt_dir)}
"""
    with log_file.open("w") as f:
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    return proc, log_file


def load_metrics_row(metrics_path: Path, sample: Dict[str, Any], runtime_ok: bool, returncode: int, segment_log: Path) -> Dict[str, Any]:
    with metrics_path.open() as f:
        metrics = json.load(f)
    if not isinstance(metrics, dict):
        raise ValueError(f"metrics payload must be an object: {metrics_path}")
    rollout = metrics.get("rollout", {}) if isinstance(metrics.get("rollout"), dict) else {}
    predicate_debug = metrics.get("predicate_debug", {}) if isinstance(metrics.get("predicate_debug"), dict) else {}
    restore = metrics.get("restore", {}) if isinstance(metrics.get("restore"), dict) else {}
    component_evaluability = normalize_component_evaluability(metrics.get("component_evaluability"))
    restore_start = restore.get("start") if isinstance(restore.get("start"), dict) else {}
    restore_end = restore.get("end") if isinstance(restore.get("end"), dict) else {}
    restore_rollout_start = restore.get("rollout_start") if isinstance(restore.get("rollout_start"), dict) else {}
    env_terminal_debug = rollout.get("env_terminal_debug")
    env_termination_reason = rollout.get("env_termination_reason")
    if env_termination_reason is None and isinstance(env_terminal_debug, dict):
        env_termination_reason = env_terminal_debug.get("termination_reason")
    row = {
        "job_key": sample["job_key"],
        "skill": sample["skill"],
        "task_name": sample["task_name"],
        "demo_id": sample["demo_id"],
        "instance_id": sample.get("instance_id"),
        "skill_idx": sample["skill_idx"],
        "frame_duration": sample.get("frame_duration"),
        "dynamic_max_steps": get_dynamic_max_steps(sample.get("frame_duration"), fallback=0),
        "runtime_ok": runtime_ok,
        "returncode": returncode,
        "success": metrics.get("success"),
        "result_type": metrics.get("result_type"),
        "aggregation_eligible": _normalize_optional_bool(metrics.get("aggregation_eligible")),
        "boundary_evaluation_eligible": _normalize_optional_bool(metrics.get("boundary_evaluation_eligible")),
        "rollout_evaluation_eligible": _normalize_optional_bool(metrics.get("rollout_evaluation_eligible")),
        "model_evaluated": _normalize_optional_bool(metrics.get("model_evaluated")),
        "model_failure_eligible": _normalize_optional_bool(metrics.get("model_failure_eligible")),
        "component_evaluability": component_evaluability,
        "component_evaluability_json": json.dumps(component_evaluability, ensure_ascii=False, sort_keys=True)
        if component_evaluability is not None
        else None,
        "metric_family": predicate_debug.get("metric_family"),
        "start_all_satisfied": predicate_debug.get("start_all_satisfied"),
        "short_proxy_success": predicate_debug.get("short_proxy_success"),
        "likely_proxy_false_positive": predicate_debug.get("likely_proxy_false_positive"),
        "transfer_pose_proxy_success_unconfirmed": predicate_debug.get(
            "transfer_pose_proxy_success_unconfirmed"
        ),
        "contact_effect_proxy_success_unconfirmed": predicate_debug.get(
            "contact_effect_proxy_success_unconfirmed"
        ),
        "articulation_close_success_unconfirmed": predicate_debug.get(
            "articulation_close_success_unconfirmed"
        ),
        "articulation_close_proxy_success_unconfirmed": predicate_debug.get(
            "articulation_close_proxy_success_unconfirmed"
        ),
        "geometry_base_facing_success_unconfirmed": predicate_debug.get(
            "geometry_base_facing_success_unconfirmed"
        ),
        "relation_transfer_proxy_success_unconfirmed": predicate_debug.get(
            "relation_transfer_proxy_success_unconfirmed"
        ),
        "orientation_proxy_success_unconfirmed": predicate_debug.get(
            "orientation_proxy_success_unconfirmed"
        ),
        "articulation_open_proxy_success_unconfirmed": predicate_debug.get(
            "articulation_open_proxy_success_unconfirmed"
        ),
        "articulation_open_success_unconfirmed": predicate_debug.get(
            "articulation_open_success_unconfirmed"
        ),
        "can_meat_lid_success_unconfirmed": predicate_debug.get("can_meat_lid_success_unconfirmed"),
        "metrics_short_video_problem": predicate_debug.get("short_video_problem"),
        "short_success_required_step": predicate_debug.get("short_success_required_step"),
        "min_success_steps": predicate_debug.get("min_success_steps"),
        "first_predicate_satisfied_step": predicate_debug.get("first_predicate_satisfied_step"),
        "early_predicate_satisfied_steps": predicate_debug.get("early_predicate_satisfied_steps"),
        "final_step": rollout.get("final_step"),
        "rollout_attempted": _normalize_optional_bool(rollout.get("rollout_attempted")),
        "termination_reason": rollout.get("termination_reason"),
        "env_termination_reason": env_termination_reason,
        "env_done_success": rollout.get("env_done_success"),
        "rollout_terminated": rollout.get("env_terminated_seen", rollout.get("terminated")),
        "rollout_truncated": rollout.get("truncated"),
        "rollout_last_terminated": rollout.get("last_terminated"),
        "env_terminated_seen": rollout.get("env_terminated_seen", rollout.get("terminated")),
        "env_done_success_seen": rollout.get("env_done_success_seen", rollout.get("env_done_success")),
        "first_env_terminated_step": rollout.get("first_env_terminated_step"),
        "first_env_done_success_step": rollout.get("first_env_done_success_step"),
        "env_task_success_before_segment_success": rollout.get("env_task_success_before_segment_success"),
        "env_terminal_debug": env_terminal_debug,
        "env_terminal_debug_json": json.dumps(env_terminal_debug, ensure_ascii=False, sort_keys=True)
        if env_terminal_debug is not None
        else None,
        "rollout_start_all_satisfied": rollout.get("start_all_satisfied"),
        "rollout_require_unsatisfied_at_start": rollout.get("require_unsatisfied_at_start"),
        "rollout_max_steps": rollout.get("max_steps"),
        "restore_start_method": _restore_entry_selected_method(restore_start),
        "restore_end_method": _restore_entry_selected_method(restore_end),
        "restore_rollout_start_method": _restore_entry_selected_method(restore_rollout_start),
        "rawdata_restore_start_failed": _restore_entry_rawdata_failed(restore_start),
        "rawdata_restore_end_failed": _restore_entry_rawdata_failed(restore_end),
        "rawdata_restore_rollout_start_failed": _restore_entry_rawdata_failed(restore_rollout_start),
        "rawdata_restore_start_fallback": _restore_entry_rawdata_fallback(restore_start),
        "rawdata_restore_end_fallback": _restore_entry_rawdata_fallback(restore_end),
        "rawdata_restore_rollout_start_fallback": _restore_entry_rawdata_fallback(restore_rollout_start),
        "metrics_path": str(metrics_path),
        "segment_log": str(segment_log),
    }
    row["rawdata_restore_failure_count"] = sum(
        int(row[key])
        for key in (
            "rawdata_restore_start_failed",
            "rawdata_restore_end_failed",
            "rawdata_restore_rollout_start_failed",
        )
    )
    row["rawdata_restore_fallback_count"] = sum(
        int(row[key])
        for key in (
            "rawdata_restore_start_fallback",
            "rawdata_restore_end_fallback",
            "rawdata_restore_rollout_start_fallback",
        )
    )
    row["rawdata_restore_failed"] = row["rawdata_restore_failure_count"] > 0
    row["rawdata_restore_fallback"] = row["rawdata_restore_fallback_count"] > 0
    row["short_video_problem"] = is_short_video_problem_row(row)
    row["transfer_pose_proxy_success_unconfirmed"] = is_transfer_pose_proxy_success_unconfirmed(row)
    row["contact_effect_proxy_success_unconfirmed"] = is_contact_effect_proxy_success_unconfirmed(row)
    row["articulation_close_success_unconfirmed"] = is_articulation_close_success_unconfirmed(row)
    row["articulation_close_proxy_success_unconfirmed"] = is_articulation_close_proxy_success_unconfirmed(row)
    row["geometry_base_facing_success_unconfirmed"] = is_geometry_base_facing_success_unconfirmed(row)
    row["relation_transfer_proxy_success_unconfirmed"] = is_relation_transfer_proxy_success_unconfirmed(row)
    row["orientation_proxy_success_unconfirmed"] = is_orientation_proxy_success_unconfirmed(row)
    row["articulation_open_proxy_success_unconfirmed"] = is_articulation_open_proxy_success_unconfirmed(row)
    row["articulation_open_success_unconfirmed"] = is_articulation_open_success_unconfirmed(row)
    row["can_meat_lid_success_unconfirmed"] = is_can_meat_lid_success_unconfirmed(row)
    row["early_metric_activation_review_needed"] = is_early_metric_activation_review_needed(row)
    row["meaningful_policy_caused_transition"] = has_meaningful_policy_caused_transition(row)
    return row


def run_segment_eval(
    *,
    sample: Dict[str, Any],
    port: int,
    gpu_id: int,
    behavior_env: str,
    behavior_dir: Path,
    demo_data_path: Path,
    rawdata_path: Path,
    out_dir: Path,
    default_max_steps: int,
    max_dynamic_steps_cap: int,
    dry_run: bool,
    write_video: bool,
    partial_scene_load: bool,
    skip_intermediate_obs_in_chunk: bool,
    segment_predicate_dump_trace: bool,
    expected_task_prompt_sha256: str,
    expected_server_run_id: str,
    expected_server_token: str,
) -> Dict[str, Any]:
    task_name = sample["task_name"]
    demo_id = sample["demo_id"]
    skill_idx = int(sample["skill_idx"])
    dynamic_max_steps = get_dynamic_max_steps(
        sample.get("frame_duration"),
        fallback=default_max_steps,
        cap=max(0, int(max_dynamic_steps_cap or 0)),
    )

    skill_out = out_dir / "raw" / task_name / f"demo_{demo_id}" / f"skill_{skill_idx:03d}"
    skill_out.mkdir(parents=True, exist_ok=True)
    segment_log = skill_out / "segment_eval.log"
    # OmniGibson/Isaac appdata must stay on local disk (/tmp). A run-scoped cache forces every
    # recheck to redownload / rebuild Isaac extensions, which dominates these short segment evals.
    # Default to a stable per-GPU cache so reruns reuse warmed artifacts while different workers
    # remain isolated. Allow env override for debugging or more aggressive isolation.
    og_appdata_base = Path(os.environ.get("OMNIGIBSON_APPDATA_PATH_BASE", "/tmp/omnigibson-appdata"))
    og_user = os.environ.get("USER", "user")
    og_appdata_scope = os.environ.get("OMNIGIBSON_APPDATA_SCOPE", "gpu")
    if og_appdata_scope == "gpu":
        og_appdata = og_appdata_base / og_user / f"gpu{gpu_id}"
    elif og_appdata_scope == "gpu_port":
        og_appdata = og_appdata_base / og_user / f"gpu{gpu_id}_p{port}"
    elif og_appdata_scope == "run_gpu_port":
        og_appdata = og_appdata_base / og_user / out_dir.name / f"gpu{gpu_id}_p{port}"
    else:
        raise RuntimeError(
            "invalid OMNIGIBSON_APPDATA_SCOPE; expected one of: gpu, gpu_port, run_gpu_port"
        )
    og_appdata.mkdir(parents=True, exist_ok=True)

    existing_metrics = sorted((skill_out / "metrics").glob("*.json"))
    if existing_metrics:
        row = load_metrics_row(existing_metrics[0], sample, True, 0, segment_log)
        row["resume_hit"] = True
        row["dynamic_max_steps"] = dynamic_max_steps
        return row

    cmd = f"""
set -euo pipefail
source {q(CONDA_SH)}
conda activate {q(behavior_env)}
cd {q(behavior_dir)}
export PYTHONUNBUFFERED=1
export PYTHONPATH={q(str(behavior_dir / 'joylo'))}:{q(str(behavior_dir / 'OmniGibson'))}:{q(str(behavior_dir / 'bddl3'))}${{PYTHONPATH:+:$PYTHONPATH}}
export NO_PROXY="localhost,127.0.0.1,::1${{NO_PROXY:+,$NO_PROXY}}"
export no_proxy="localhost,127.0.0.1,::1${{no_proxy:+,$no_proxy}}"
export CUDA_VISIBLE_DEVICES={q(gpu_id)}
unset OMNIGIBSON_GPU_ID
export OMNIGIBSON_DATA_PATH={q(str(behavior_dir / 'datasets'))}
export OMNIGIBSON_APPDATA_PATH={q(str(og_appdata))}
export MPLBACKEND="${{MPLBACKEND:-Agg}}"
export TORCHDYNAMO_DISABLE="${{TORCHDYNAMO_DISABLE:-1}}"
export TORCHINDUCTOR_DISABLE="${{TORCHINDUCTOR_DISABLE:-1}}"
export OMNIGIBSON_HEADLESS=true
export OMNIGIBSON_DISABLE_EXTENSION_REGISTRY=0
export OMNIGIBSON_DISABLE_DRIVER_VERSION_CHECK=1
python OmniGibson/omnigibson/learning/eval_segment.py \
  policy=websocket \
  task.name={q(task_name)} \
  demo_data_path={q(demo_data_path)} \
  rawdata_path={q(rawdata_path)} \
  segment_level=skill \
  segment_idx={q(skill_idx)} \
  success_mode=segment_predicates \
  grounding_topk=3 \
  dry_run={q(str(dry_run).lower())} \
  log_path={q(skill_out)} \
  demo_id={q(demo_id)} \
  headless=true \
  write_video={q(str(write_video).lower())} \
  segment_max_steps={q(dynamic_max_steps)} \
  model.host=127.0.0.1 \
  model.port={q(port)} \
  model.expected_task_name={q(task_name)} \
  model.expected_task_prompt_sha256={q(expected_task_prompt_sha256)} \
  model.expected_server_run_id={q(expected_server_run_id)} \
  model.expected_server_token={q(expected_server_token)} \
  env_wrapper._target_=omnigibson.learning.wrappers.RGBLowResWrapper \
  partial_scene_load={q(str(partial_scene_load).lower())} \
  skip_intermediate_obs_in_chunk={q(str(skip_intermediate_obs_in_chunk).lower())} \
  segment_predicate_window_mode=consecutive \
  segment_predicate_min_consecutive=3 \
  segment_predicate_dump_trace={q(str(segment_predicate_dump_trace).lower())}
"""
    with segment_log.open("w") as f:
        proc = subprocess.run(["bash", "-lc", cmd], stdout=f, stderr=subprocess.STDOUT, text=True)

    metrics_glob = sorted((skill_out / "metrics").glob("*.json"))
    result: Dict[str, Any] = {
        "job_key": sample["job_key"],
        "skill": sample["skill"],
        "task_name": task_name,
        "demo_id": demo_id,
        "instance_id": sample.get("instance_id"),
        "skill_idx": skill_idx,
        "frame_duration": sample.get("frame_duration"),
        "dynamic_max_steps": dynamic_max_steps,
        "runtime_ok": proc.returncode == 0 and len(metrics_glob) > 0,
        "returncode": proc.returncode,
        "metrics_path": str(metrics_glob[0]) if metrics_glob else None,
        "segment_log": str(segment_log),
    }
    if metrics_glob:
        result.update(load_metrics_row(metrics_glob[0], sample, result["runtime_ok"], proc.returncode, segment_log))
        result["dynamic_max_steps"] = dynamic_max_steps
    return result


def run_worker(args: argparse.Namespace) -> int:
    manifest_path = args.out_dir / "manifest.json"
    wait_for_path(manifest_path, args.prepare_timeout)
    worker_jobs_path = args.out_dir / "jobs" / f"worker_{args.worker_rank:03d}.json"
    if not worker_jobs_path.exists():
        raise RuntimeError(f"missing worker job file: {worker_jobs_path}")
    with worker_jobs_path.open() as f:
        worker_jobs = json.load(f)

    results_path = args.out_dir / "worker_results" / f"worker_{args.worker_rank:03d}.jsonl"
    done_keys = set()
    if args.resume and results_path.exists():
        for row in load_jsonl_rows([results_path]):
            done_keys.add(row["job_key"])

    pending_jobs = [job for job in worker_jobs if job["job_key"] not in done_keys]
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for job in pending_jobs:
        grouped[job["task_name"]].append(job)

    ckpt_dir = resolve_checkpoint_dir(args.ckpt_dir)
    status = {
        "worker_rank": args.worker_rank,
        "node_rank": args.node_rank,
        "gpu_id": args.gpu_id,
        "port": args.port,
        "planned_jobs": len(worker_jobs),
        "pending_jobs": len(pending_jobs),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_dump(args.out_dir / "worker_status" / f"worker_{args.worker_rank:03d}.started.json", status)

    total_runtime_ok = 0
    total_success = 0
    stagger_applied = False
    for task_name in sorted(grouped):
        server_proc: Optional[subprocess.Popen[str]] = None
        server_log: Optional[Path] = None
        task_port = args.port
        try:
            if not stagger_applied and args.server_start_stagger_s > 0 and args.worker_rank >= 0:
                # Stagger heavy model init/JIT to avoid 8-way compile storms on one node.
                delay = args.server_start_stagger_s * int(args.worker_rank)
                if delay > 0:
                    print(f"[worker {args.worker_rank:03d}] staggering server start by {delay}s", flush=True)
                    time.sleep(delay)
                stagger_applied = True
            # Avoid silently connecting to a stale server from another run.
            # If the preferred port is already in use, hop by `gpus_per_node` to keep worker ports disjoint.
            task_port = find_free_port(args.port, stride=args.gpus_per_node, max_tries=200)
            server_identity = build_server_identity(
                out_dir=args.out_dir,
                worker_rank=args.worker_rank,
                task_name=task_name,
                port=task_port,
            )
            server_proc, server_log = start_server(
                task_name=task_name,
                port=task_port,
                gpu_id=args.gpu_id,
                ckpt_dir=ckpt_dir,
                config_name=args.config_name,
                policy_backend=args.policy_backend,
                server_run_id=server_identity["server_run_id"],
                server_token=server_identity["server_token"],
                openpi_env=args.openpi_env,
                behavior_dir=args.behavior_dir,
                out_dir=args.out_dir / "server_logs",
            )
            print(
                f"[worker {args.worker_rank:03d}] started server for task={task_name} gpu={args.gpu_id} port={task_port}",
                flush=True,
            )
            wait_for_server_proc(
                proc=server_proc,
                port=task_port,
                timeout_s=args.server_ready_timeout,
                log_file=server_log,
                expected_identity=server_identity,
            )
            print(f"[worker {args.worker_rank:03d}] server ready on port {task_port}", flush=True)
            for sample in grouped[task_name]:
                print(
                    f"[worker {args.worker_rank:03d}] {sample['skill']} | {sample['task_name']} | "
                    f"demo={sample['demo_id']} | skill_idx={sample['skill_idx']}"
                )
                row = run_segment_eval(
                    sample=sample,
                    port=task_port,
                    gpu_id=args.gpu_id,
                    behavior_env=args.behavior_env,
                    behavior_dir=args.behavior_dir,
                    demo_data_path=args.demo_data_path,
                    rawdata_path=args.rawdata_path,
                    out_dir=args.out_dir,
                    default_max_steps=args.max_steps,
                    max_dynamic_steps_cap=args.max_dynamic_steps_cap,
                    dry_run=args.dry_run,
                    write_video=args.write_video,
                    partial_scene_load=args.partial_scene_load,
                    skip_intermediate_obs_in_chunk=args.skip_intermediate_obs_in_chunk,
                    segment_predicate_dump_trace=args.segment_predicate_dump_trace,
                    expected_task_prompt_sha256=server_identity["task_prompt_sha256"],
                    expected_server_run_id=server_identity["server_run_id"],
                    expected_server_token=server_identity["server_token"],
                )
                row["worker_rank"] = args.worker_rank
                row["node_rank"] = args.node_rank
                row["gpu_id"] = args.gpu_id
                append_jsonl(results_path, row)
                total_runtime_ok += int(bool(row.get("runtime_ok")))
                total_success += int(bool(row.get("success")))
        finally:
            stop_process(server_proc)
            wait_for_port_free(task_port)

    final_status = {
        **status,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed_jobs": len(worker_jobs),
        "runtime_ok": total_runtime_ok,
        "success": total_success,
    }
    json_dump(args.out_dir / "worker_status" / f"worker_{args.worker_rank:03d}.done.json", final_status)
    return 0


def classify_result_row(row: Dict[str, Any]) -> Dict[str, bool]:
    runtime_ok = bool(row.get("runtime_ok"))
    result_type = row.get("result_type")
    state_invalid_restore_components = result_type == RESULT_STATE_INVALID_RESTORE_COMPONENTS
    state_invalid_rollout_components = result_type == RESULT_STATE_INVALID_ROLLOUT_COMPONENTS
    spec_invalid_component_dependency = result_type == RESULT_SPEC_INVALID_COMPONENT_DEPENDENCY
    component_invalid = result_type in COMPONENT_INVALID_RESULT_TYPES
    runtime_pass = runtime_ok and not component_invalid
    pre_satisfied_start = result_type == RESULT_PRE_SATISFIED_START
    short_proxy_success = result_type in {RESULT_SHORT_PROXY_SUCCESS, RESULT_LIKELY_PROXY_FALSE_POSITIVE} or bool(
        row.get("short_proxy_success")
    )
    likely_proxy_false_positive = result_type == RESULT_LIKELY_PROXY_FALSE_POSITIVE or bool(
        row.get("likely_proxy_false_positive")
    )
    transfer_pose_proxy_success_unconfirmed = bool(
        row.get("transfer_pose_proxy_success_unconfirmed")
    ) or is_transfer_pose_proxy_success_unconfirmed(row)
    contact_effect_proxy_success_unconfirmed = bool(
        row.get("contact_effect_proxy_success_unconfirmed")
    ) or is_contact_effect_proxy_success_unconfirmed(row)
    articulation_close_success_unconfirmed = bool(
        row.get("articulation_close_success_unconfirmed")
    ) or is_articulation_close_success_unconfirmed(row)
    articulation_close_proxy_success_unconfirmed = bool(
        row.get("articulation_close_proxy_success_unconfirmed")
    ) or is_articulation_close_proxy_success_unconfirmed(row)
    geometry_base_facing_success_unconfirmed = bool(
        row.get("geometry_base_facing_success_unconfirmed")
    ) or is_geometry_base_facing_success_unconfirmed(row)
    relation_transfer_proxy_success_unconfirmed = bool(
        row.get("relation_transfer_proxy_success_unconfirmed")
    ) or is_relation_transfer_proxy_success_unconfirmed(row)
    orientation_proxy_success_unconfirmed = bool(
        row.get("orientation_proxy_success_unconfirmed")
    ) or is_orientation_proxy_success_unconfirmed(row)
    articulation_open_proxy_success_unconfirmed = bool(
        row.get("articulation_open_proxy_success_unconfirmed")
    ) or is_articulation_open_proxy_success_unconfirmed(row)
    articulation_open_success_unconfirmed = bool(
        row.get("articulation_open_success_unconfirmed")
    ) or is_articulation_open_success_unconfirmed(row)
    can_meat_lid_success_unconfirmed = bool(
        row.get("can_meat_lid_success_unconfirmed")
    ) or is_can_meat_lid_success_unconfirmed(row)
    short_video_problem = (
        result_type == RESULT_SHORT_VIDEO_PROBLEM
        or bool(row.get("short_video_problem"))
        or bool(row.get("metrics_short_video_problem"))
        or is_short_video_problem_row(row)
    )
    early_metric_activation_review_needed = bool(row.get("early_metric_activation_review_needed")) or is_early_metric_activation_review_needed(
        row
    )
    metric_invalid_missing_object = result_type == RESULT_METRIC_INVALID_MISSING_OBJECT
    metric_invalid = str(result_type or "").startswith(METRIC_INVALID_PREFIX)
    rawdata_restore_failed = bool(row.get("rawdata_restore_failed"))
    rawdata_restore_fallback = bool(row.get("rawdata_restore_fallback"))
    aggregation_eligible = _normalize_optional_bool(row.get("aggregation_eligible"))
    boundary_evaluation_eligible = _normalize_optional_bool(row.get("boundary_evaluation_eligible"))
    rollout_evaluation_eligible = _normalize_optional_bool(row.get("rollout_evaluation_eligible"))
    model_evaluated = _normalize_optional_bool(row.get("model_evaluated"))
    model_failure_eligible = _normalize_optional_bool(row.get("model_failure_eligible"))
    rollout_attempted = _normalize_optional_bool(row.get("rollout_attempted"))
    if rollout_attempted is None:
        policy_attempted = result_type in ATTEMPTABLE_RESULT_TYPES
    else:
        policy_attempted = rollout_attempted and result_type in ATTEMPTABLE_RESULT_TYPES
    if model_evaluated is False:
        policy_attempted = False
    producer_eligible = all(
        field is not False
        for field in (
            aggregation_eligible,
            boundary_evaluation_eligible,
            rollout_evaluation_eligible,
            model_failure_eligible,
        )
    )
    attemptable = (
        runtime_pass
        and policy_attempted
        and producer_eligible
        and not component_invalid
        and not pre_satisfied_start
        and not metric_invalid
    )
    policy_success_attemptable = attemptable and bool(row.get("success")) and result_type == "predicate_satisfied"
    policy_success_clean_attemptable = (
        policy_success_attemptable
        and not short_proxy_success
        and not short_video_problem
        and not transfer_pose_proxy_success_unconfirmed
        and not contact_effect_proxy_success_unconfirmed
        and not articulation_close_success_unconfirmed
        and not articulation_close_proxy_success_unconfirmed
        and not geometry_base_facing_success_unconfirmed
        and not relation_transfer_proxy_success_unconfirmed
        and not orientation_proxy_success_unconfirmed
        and not articulation_open_proxy_success_unconfirmed
        and not articulation_open_success_unconfirmed
        and not can_meat_lid_success_unconfirmed
        and not early_metric_activation_review_needed
    )
    meaningful_policy_caused_transition = (
        attemptable
        and (bool(row.get("meaningful_policy_caused_transition")) or has_meaningful_policy_caused_transition(row))
    )
    timeout = attemptable and result_type == "timeout"
    truncated = attemptable and result_type == "truncated"
    env_terminated_metric_unsatisfied = attemptable and result_type == "env_terminated" and not bool(row.get("success"))
    env_task_success_before_segment_success = (
        attemptable and result_type == RESULT_ENV_TASK_SUCCESS_BEFORE_SEGMENT_SUCCESS and not bool(row.get("success"))
    )
    env_task_success_grasp_release_review_needed = runtime_pass and is_env_task_success_grasp_release_review_needed(
        {**row, "attemptable": attemptable}
    )
    rawdata_restore_review_needed = runtime_pass and (rawdata_restore_failed or rawdata_restore_fallback)
    policy_failure_attemptable = attemptable and not policy_success_attemptable
    metric_unsatisfied_attemptable = policy_failure_attemptable
    other_metric_unsatisfied = (
        metric_unsatisfied_attemptable
        and not timeout
        and not truncated
        and not env_terminated_metric_unsatisfied
        and not env_task_success_before_segment_success
        and not likely_proxy_false_positive
    )
    classified_model_result = runtime_pass and not component_invalid
    return {
        "runtime_pass": runtime_pass,
        "runtime_fail": not runtime_pass and not component_invalid,
        "state_invalid_restore_components": state_invalid_restore_components,
        "state_invalid_rollout_components": state_invalid_rollout_components,
        "spec_invalid_component_dependency": spec_invalid_component_dependency,
        "component_invalid": component_invalid,
        "pre_satisfied_start": runtime_pass and pre_satisfied_start,
        "metric_invalid": runtime_pass and metric_invalid,
        "metric_invalid_missing_object": runtime_pass and metric_invalid_missing_object,
        "rawdata_restore_failed": runtime_pass and rawdata_restore_failed,
        "rawdata_restore_fallback": runtime_pass and rawdata_restore_fallback,
        "rawdata_restore_review_needed": rawdata_restore_review_needed,
        "attemptable": attemptable,
        "policy_attempted": attemptable,
        "policy_success_attemptable": policy_success_attemptable,
        "policy_failure_attemptable": policy_failure_attemptable,
        "policy_success_clean_attemptable": policy_success_clean_attemptable,
        "short_video_problem": classified_model_result and short_video_problem,
        "early_metric_activation_review_needed": classified_model_result
        and early_metric_activation_review_needed,
        "short_proxy_success": classified_model_result and short_proxy_success,
        "likely_proxy_false_positive": classified_model_result and likely_proxy_false_positive,
        "transfer_pose_proxy_success_unconfirmed": classified_model_result
        and transfer_pose_proxy_success_unconfirmed,
        "contact_effect_proxy_success_unconfirmed": classified_model_result
        and contact_effect_proxy_success_unconfirmed,
        "articulation_close_success_unconfirmed": classified_model_result
        and articulation_close_success_unconfirmed,
        "articulation_close_proxy_success_unconfirmed": classified_model_result
        and articulation_close_proxy_success_unconfirmed,
        "geometry_base_facing_success_unconfirmed": classified_model_result
        and geometry_base_facing_success_unconfirmed,
        "relation_transfer_proxy_success_unconfirmed": classified_model_result
        and relation_transfer_proxy_success_unconfirmed,
        "orientation_proxy_success_unconfirmed": classified_model_result
        and orientation_proxy_success_unconfirmed,
        "articulation_open_proxy_success_unconfirmed": classified_model_result
        and articulation_open_proxy_success_unconfirmed,
        "articulation_open_success_unconfirmed": classified_model_result
        and articulation_open_success_unconfirmed,
        "can_meat_lid_success_unconfirmed": classified_model_result and can_meat_lid_success_unconfirmed,
        "meaningful_policy_caused_transition": classified_model_result
        and meaningful_policy_caused_transition,
        "timeout": timeout,
        "truncated": truncated,
        "env_terminated_metric_unsatisfied": env_terminated_metric_unsatisfied,
        "env_task_success_before_segment_success": env_task_success_before_segment_success,
        "env_task_success_grasp_release_review_needed": env_task_success_grasp_release_review_needed,
        "metric_unsatisfied_attemptable": metric_unsatisfied_attemptable,
        "other_metric_unsatisfied": other_metric_unsatisfied,
    }


def summarize_result_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    classes = [classify_result_row(row) for row in rows]
    runtime_pass = sum(int(c["runtime_pass"]) for c in classes)
    runtime_fail = sum(int(c["runtime_fail"]) for c in classes)
    state_invalid_restore_components = sum(int(c["state_invalid_restore_components"]) for c in classes)
    state_invalid_rollout_components = sum(int(c["state_invalid_rollout_components"]) for c in classes)
    spec_invalid_component_dependency = sum(int(c["spec_invalid_component_dependency"]) for c in classes)
    component_invalid = sum(int(c["component_invalid"]) for c in classes)
    runtime_pass_rate_denominator = total - component_invalid
    pre_satisfied_start = sum(int(c["pre_satisfied_start"]) for c in classes)
    metric_invalid = sum(int(c["metric_invalid"]) for c in classes)
    metric_invalid_missing_object = sum(int(c["metric_invalid_missing_object"]) for c in classes)
    rawdata_restore_failed = sum(int(c["rawdata_restore_failed"]) for c in classes)
    rawdata_restore_fallback = sum(int(c["rawdata_restore_fallback"]) for c in classes)
    rawdata_restore_review_needed = sum(int(c["rawdata_restore_review_needed"]) for c in classes)
    attemptable = sum(int(c["attemptable"]) for c in classes)
    policy_success_attemptable = sum(int(c["policy_success_attemptable"]) for c in classes)
    policy_failure_attemptable = sum(int(c["policy_failure_attemptable"]) for c in classes)
    policy_success_clean_attemptable = sum(int(c["policy_success_clean_attemptable"]) for c in classes)
    short_video_problem = sum(int(c["short_video_problem"]) for c in classes)
    early_metric_activation_review_needed = sum(int(c["early_metric_activation_review_needed"]) for c in classes)
    short_proxy_success = sum(int(c["short_proxy_success"]) for c in classes)
    likely_proxy_false_positive = sum(int(c["likely_proxy_false_positive"]) for c in classes)
    transfer_pose_proxy_success_unconfirmed = sum(
        int(c["transfer_pose_proxy_success_unconfirmed"]) for c in classes
    )
    contact_effect_proxy_success_unconfirmed = sum(
        int(c["contact_effect_proxy_success_unconfirmed"]) for c in classes
    )
    articulation_close_success_unconfirmed = sum(
        int(c["articulation_close_success_unconfirmed"]) for c in classes
    )
    articulation_close_proxy_success_unconfirmed = sum(
        int(c["articulation_close_proxy_success_unconfirmed"]) for c in classes
    )
    geometry_base_facing_success_unconfirmed = sum(
        int(c["geometry_base_facing_success_unconfirmed"]) for c in classes
    )
    relation_transfer_proxy_success_unconfirmed = sum(
        int(c["relation_transfer_proxy_success_unconfirmed"]) for c in classes
    )
    orientation_proxy_success_unconfirmed = sum(
        int(c["orientation_proxy_success_unconfirmed"]) for c in classes
    )
    articulation_open_proxy_success_unconfirmed = sum(
        int(c["articulation_open_proxy_success_unconfirmed"]) for c in classes
    )
    articulation_open_success_unconfirmed = sum(
        int(c["articulation_open_success_unconfirmed"]) for c in classes
    )
    can_meat_lid_success_unconfirmed = sum(
        int(c["can_meat_lid_success_unconfirmed"]) for c in classes
    )
    meaningful_policy_caused_transition = sum(int(c["meaningful_policy_caused_transition"]) for c in classes)
    metric_unsatisfied_attemptable = sum(int(c["metric_unsatisfied_attemptable"]) for c in classes)
    timeout = sum(int(c["timeout"]) for c in classes)
    truncated = sum(int(c["truncated"]) for c in classes)
    env_unsat = sum(int(c["env_terminated_metric_unsatisfied"]) for c in classes)
    env_task_success_before_segment_success = sum(
        int(c["env_task_success_before_segment_success"]) for c in classes
    )
    env_task_success_grasp_release_review_needed = sum(
        int(c["env_task_success_grasp_release_review_needed"]) for c in classes
    )
    other_unsat = sum(int(c["other_metric_unsatisfied"]) for c in classes)
    raw_completed_denominator = runtime_pass_rate_denominator
    success_raw = sum(
        int(bool(row.get("success")))
        for row, classification in zip(rows, classes)
        if not classification["component_invalid"]
    )
    env_done_success_count = sum(int(row.get("env_done_success") is True) for row in rows)
    rollout_terminated_count = sum(int(row.get("rollout_terminated") is True) for row in rows)
    rollout_truncated_flag_count = sum(int(row.get("rollout_truncated") is True) for row in rows)
    termination_reason_counts = dict(
        sorted(
            Counter(
                str(row.get("termination_reason")) for row in rows if row.get("termination_reason") is not None
            ).items()
        )
    )
    env_termination_reason_counts = dict(
        sorted(
            Counter(
                str(row.get("env_termination_reason"))
                for row in rows
                if row.get("env_termination_reason") is not None
            ).items()
        )
    )
    return {
        "segment_count": total,
        "runtime_pass_count": runtime_pass,
        "runtime_fail_count": runtime_fail,
        "runtime_pass_rate": runtime_pass / runtime_pass_rate_denominator if runtime_pass_rate_denominator else 0.0,
        "runtime_pass_rate_denominator": runtime_pass_rate_denominator,
        "runtime_pass_rate_excluded_component_invalid_count": component_invalid,
        "state_invalid_restore_components_count": state_invalid_restore_components,
        "state_invalid_rollout_components_count": state_invalid_rollout_components,
        "spec_invalid_component_dependency_count": spec_invalid_component_dependency,
        "component_invalid_count": component_invalid,
        "pre_satisfied_start_count": pre_satisfied_start,
        "metric_invalid_count": metric_invalid,
        "metric_invalid_missing_object_count": metric_invalid_missing_object,
        "rawdata_restore_failed_count": rawdata_restore_failed,
        "rawdata_restore_fallback_count": rawdata_restore_fallback,
        "rawdata_restore_review_needed_count": rawdata_restore_review_needed,
        "attemptable_segment_count": attemptable,
        "policy_success_attemptable_count": policy_success_attemptable,
        "policy_success_attemptable_rate": policy_success_attemptable / attemptable if attemptable else 0.0,
        "policy_failure_attemptable_count": policy_failure_attemptable,
        "policy_failure_attemptable_rate": policy_failure_attemptable / attemptable if attemptable else 0.0,
        "policy_success_clean_attemptable_count": policy_success_clean_attemptable,
        "policy_success_clean_attemptable_rate": policy_success_clean_attemptable / attemptable if attemptable else 0.0,
        "short_video_problem_count": short_video_problem,
        "early_metric_activation_review_needed_count": early_metric_activation_review_needed,
        "short_proxy_success_count": short_proxy_success,
        "likely_proxy_false_positive_count": likely_proxy_false_positive,
        "transfer_pose_proxy_success_unconfirmed_count": transfer_pose_proxy_success_unconfirmed,
        "contact_effect_proxy_success_unconfirmed_count": contact_effect_proxy_success_unconfirmed,
        "articulation_close_success_unconfirmed_count": articulation_close_success_unconfirmed,
        "articulation_close_proxy_success_unconfirmed_count": articulation_close_proxy_success_unconfirmed,
        "geometry_base_facing_success_unconfirmed_count": geometry_base_facing_success_unconfirmed,
        "relation_transfer_proxy_success_unconfirmed_count": relation_transfer_proxy_success_unconfirmed,
        "orientation_proxy_success_unconfirmed_count": orientation_proxy_success_unconfirmed,
        "articulation_open_proxy_success_unconfirmed_count": articulation_open_proxy_success_unconfirmed,
        "articulation_open_success_unconfirmed_count": articulation_open_success_unconfirmed,
        "can_meat_lid_success_unconfirmed_count": can_meat_lid_success_unconfirmed,
        "meaningful_policy_caused_transition_count": meaningful_policy_caused_transition,
        "metric_unsatisfied_attemptable_count": metric_unsatisfied_attemptable,
        "timeout_count": timeout,
        "truncated_count": truncated,
        "env_terminated_metric_unsatisfied_count": env_unsat,
        "env_task_success_before_segment_success_count": env_task_success_before_segment_success,
        "env_task_success_grasp_release_review_needed_count": env_task_success_grasp_release_review_needed,
        "other_metric_unsatisfied_count": other_unsat,
        "success_count_raw": success_raw,
        "success_rate_raw_completed": success_raw / raw_completed_denominator if raw_completed_denominator else 0.0,
        "success_rate_raw_completed_denominator": raw_completed_denominator,
        "success_rate_raw_completed_excluded_component_invalid_count": component_invalid,
        "predicate_satisfied_count": sum(1 for row in rows if row.get("result_type") == "predicate_satisfied"),
        "env_done_success_count": env_done_success_count,
        "rollout_terminated_count": rollout_terminated_count,
        "rollout_truncated_flag_count": rollout_truncated_flag_count,
        "termination_reason_counts": termination_reason_counts,
        "termination_reason_counts_json": json.dumps(termination_reason_counts, ensure_ascii=False, sort_keys=True),
        "env_termination_reason_counts": env_termination_reason_counts,
        "env_termination_reason_counts_json": json.dumps(env_termination_reason_counts, ensure_ascii=False, sort_keys=True),
        # Backwards-compatible aliases. For old rows the denominator is unchanged;
        # known component-invalid rows are explicitly excluded rather than shown as failures.
        "success_count": success_raw,
        "success_rate": success_raw / raw_completed_denominator if raw_completed_denominator else 0.0,
        "success_rate_denominator": raw_completed_denominator,
    }


def build_metric_family_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    family_grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family = str(row.get("metric_family") or "unknown")
        family_grouped[family].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for metric_family, family_rows in sorted(family_grouped.items()):
        family_summary = summarize_result_rows(family_rows)
        summary_rows.append(
            {
                "metric_family": metric_family,
                **family_summary,
                "review_needed_success_count": (
                    family_summary["policy_success_attemptable_count"]
                    - family_summary["policy_success_clean_attemptable_count"]
                ),
            }
        )
    return summary_rows


def render_summary_md(summary: Dict[str, Any]) -> str:
    lines = ["# Multinode Skill Segment Sweep Summary", ""]
    lines.append(f"- out_dir: `{summary['out_dir']}`")
    lines.append(f"- planned_jobs: `{summary['planned_jobs']}`")
    lines.append(f"- completed_jobs: `{summary['completed_jobs']}`")
    lines.append(f"- runtime_pass: `{summary['runtime_pass']}`")
    lines.append(f"- runtime_fail: `{summary['runtime_fail']}`")
    lines.append(
        f"- runtime_pass_rate_completed: `{summary.get('runtime_pass_rate_completed', 0.0):.4f}` "
        f"/ denominator `{summary.get('runtime_pass_rate_completed_denominator', 0)}` "
        f"(component-invalid excluded: "
        f"`{summary.get('runtime_pass_rate_completed_excluded_component_invalid_count', 0)}`)"
    )
    lines.append(
        f"- runtime_pass_rate_planned: `{summary.get('runtime_pass_rate_planned', 0.0):.4f}` "
        f"/ denominator `{summary.get('runtime_pass_rate_planned_denominator', 0)}` "
        f"(component-invalid excluded: "
        f"`{summary.get('runtime_pass_rate_planned_excluded_component_invalid_count', 0)}`)"
    )
    lines.append(f"- state_invalid_restore_components: `{summary.get('state_invalid_restore_components', 0)}`")
    lines.append(f"- state_invalid_rollout_components: `{summary.get('state_invalid_rollout_components', 0)}`")
    lines.append(f"- spec_invalid_component_dependency: `{summary.get('spec_invalid_component_dependency', 0)}`")
    lines.append(f"- component_invalid: `{summary.get('component_invalid', 0)}`")
    lines.append(f"- pre_satisfied_start: `{summary['pre_satisfied_start']}`")
    lines.append(f"- metric_invalid: `{summary.get('metric_invalid', 0)}`")
    lines.append(f"- metric_invalid_missing_object: `{summary.get('metric_invalid_missing_object', 0)}`")
    lines.append(f"- rawdata_restore_failed: `{summary.get('rawdata_restore_failed', 0)}`")
    lines.append(f"- rawdata_restore_fallback: `{summary.get('rawdata_restore_fallback', 0)}`")
    lines.append(f"- rawdata_restore_review_needed: `{summary.get('rawdata_restore_review_needed', 0)}`")
    lines.append(f"- attemptable_segments: `{summary['attemptable_segments']}`")
    lines.append(f"- policy_success_attemptable: `{summary['policy_success_attemptable']}`")
    lines.append(f"- policy_success_attemptable_rate: `{summary['policy_success_attemptable_rate']:.4f}`")
    lines.append(f"- policy_failure_attemptable: `{summary.get('policy_failure_attemptable', 0)}`")
    lines.append(f"- policy_failure_attemptable_rate: `{summary.get('policy_failure_attemptable_rate', 0.0):.4f}`")
    lines.append(f"- policy_success_clean_attemptable: `{summary.get('policy_success_clean_attemptable', 0)}`")
    lines.append(
        f"- policy_success_clean_attemptable_rate: "
        f"`{summary.get('policy_success_clean_attemptable_rate', 0.0):.4f}`"
    )
    lines.append(f"- short_video_problem: `{summary.get('short_video_problem', 0)}`")
    lines.append(
        f"- early_metric_activation_review_needed: `{summary.get('early_metric_activation_review_needed', 0)}`"
    )
    lines.append(f"- short_proxy_success: `{summary.get('short_proxy_success', 0)}`")
    lines.append(f"- likely_proxy_false_positive: `{summary.get('likely_proxy_false_positive', 0)}`")
    lines.append(
        f"- transfer_pose_proxy_success_unconfirmed: `{summary.get('transfer_pose_proxy_success_unconfirmed', 0)}`"
    )
    lines.append(
        f"- contact_effect_proxy_success_unconfirmed: `{summary.get('contact_effect_proxy_success_unconfirmed', 0)}`"
    )
    lines.append(
        f"- articulation_close_success_unconfirmed: `{summary.get('articulation_close_success_unconfirmed', 0)}`"
    )
    lines.append(
        f"- articulation_close_proxy_success_unconfirmed: `{summary.get('articulation_close_proxy_success_unconfirmed', 0)}`"
    )
    lines.append(
        f"- geometry_base_facing_success_unconfirmed: `{summary.get('geometry_base_facing_success_unconfirmed', 0)}`"
    )
    lines.append(
        f"- relation_transfer_proxy_success_unconfirmed: `{summary.get('relation_transfer_proxy_success_unconfirmed', 0)}`"
    )
    lines.append(
        f"- orientation_proxy_success_unconfirmed: `{summary.get('orientation_proxy_success_unconfirmed', 0)}`"
    )
    lines.append(
        f"- articulation_open_proxy_success_unconfirmed: "
        f"`{summary.get('articulation_open_proxy_success_unconfirmed', 0)}`"
    )
    lines.append(
        f"- articulation_open_success_unconfirmed: "
        f"`{summary.get('articulation_open_success_unconfirmed', 0)}`"
    )
    lines.append(f"- can_meat_lid_success_unconfirmed: `{summary.get('can_meat_lid_success_unconfirmed', 0)}`")
    lines.append(
        f"- meaningful_policy_caused_transition: `{summary.get('meaningful_policy_caused_transition', 0)}`"
    )
    lines.append(f"- metric_unsatisfied_attemptable: `{summary['metric_unsatisfied_attemptable']}`")
    lines.append(f"- timeout: `{summary['timeout']}`")
    lines.append(f"- truncated: `{summary['truncated']}`")
    lines.append(f"- env_terminated_metric_unsatisfied: `{summary['env_terminated_metric_unsatisfied']}`")
    lines.append(
        f"- env_task_success_before_segment_success: `{summary.get('env_task_success_before_segment_success', 0)}`"
    )
    lines.append(
        f"- env_task_success_grasp_release_review_needed: `{summary.get('env_task_success_grasp_release_review_needed', 0)}`"
    )
    lines.append(f"- env_done_success_count: `{summary.get('env_done_success_count', 0)}`")
    lines.append(f"- rollout_terminated_count: `{summary.get('rollout_terminated_count', 0)}`")
    lines.append(f"- rollout_truncated_flag_count: `{summary.get('rollout_truncated_flag_count', 0)}`")
    lines.append(f"- policy_success_raw: `{summary['policy_success_raw']}`")
    lines.append(
        f"- policy_success_raw_completed_rate: `{summary.get('policy_success_raw_completed_rate', 0.0):.4f}` "
        f"/ denominator `{summary.get('policy_success_raw_completed_denominator', 0)}` "
        f"(component-invalid excluded: "
        f"`{summary.get('policy_success_raw_completed_excluded_component_invalid', 0)}`)"
    )
    lines.append(f"- missing_jobs: `{summary['missing_jobs']}`")
    lines.append("")
    lines.append(
        "| Skill | Tasks | Demos | Segments | Runtime Pass / Eligible | Attemptable | "
        "Policy Success / Attemptable | Pre-satisfied | Component-invalid | Timeout | Truncated | "
        "Env-terminated Unsat |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.extend(
        (
            f"| {row['skill']} | {row['task_count']} | {row['demo_count']} | {row['segment_count']} | "
            f"{row['runtime_pass_count']} / {row['runtime_pass_rate_denominator']} "
            f"({row['runtime_pass_rate']:.4f}) | {row['attemptable_segment_count']} | "
            f"{row['policy_success_attemptable_count']} / {row['attemptable_segment_count']} "
            f"({row['policy_success_attemptable_rate']:.4f}) | {row['pre_satisfied_start_count']} | "
            f"{row['component_invalid_count']} | {row['timeout_count']} | {row['truncated_count']} | "
            f"{row['env_terminated_metric_unsatisfied_count']} |"
        )
        for row in summary["skill_summary"]
    )
    lines.append("")
    if summary.get("metric_family_summary"):
        lines.append(
            "| Metric Family | Segments | Runtime Pass / Eligible | Component-invalid | Attemptable | "
            "Raw Success | Clean Success | Review-needed Success | Timeout | Truncated |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        lines.extend(
            (
                f"| {row['metric_family']} | {row['segment_count']} | "
                f"{row['runtime_pass_count']} / {row['runtime_pass_rate_denominator']} "
                f"({row['runtime_pass_rate']:.4f}) | {row['component_invalid_count']} | "
                f"{row['attemptable_segment_count']} | {row['policy_success_attemptable_count']} | "
                f"{row['policy_success_clean_attemptable_count']} | "
                f"{row['review_needed_success_count']} | {row['timeout_count']} | {row['truncated_count']} |"
            )
            for row in summary["metric_family_summary"]
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def merge_results(args: argparse.Namespace) -> int:
    manifest_path = args.out_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    planned_jobs = manifest.get("jobs", [])
    planned_by_key = {row["job_key"]: row for row in planned_jobs}

    result_rows = load_jsonl_rows(
        sorted(
            list((args.out_dir / "worker_results").glob("worker_*.jsonl"))
            + list((args.out_dir / "worker_results").glob("persistent_worker_*.jsonl"))
        )
    )
    deduped: Dict[str, Dict[str, Any]] = {}
    for row in result_rows:
        deduped[row["job_key"]] = row
    rows = [deduped[key] for key in sorted(deduped)]
    missing_keys = sorted(set(planned_by_key) - set(deduped))
    for row in rows:
        row.update(classify_result_row(row))

    skill_grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    skill_task_grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        skill_grouped[row["skill"]].append(row)
        skill_task_grouped[(row["skill"], row["task_name"])].append(row)

    skill_summary = []
    for skill, skill_rows in sorted(skill_grouped.items()):
        skill_summary.append(
            {
                "skill": skill,
                "task_count": len({row["task_name"] for row in skill_rows}),
                "demo_count": len({row["demo_id"] for row in skill_rows}),
                **summarize_result_rows(skill_rows),
            }
        )

    skill_task_summary = []
    for (skill, task_name), task_rows in sorted(skill_task_grouped.items()):
        skill_task_summary.append(
            {
                "skill": skill,
                "task_name": task_name,
                "demo_count": len({row["demo_id"] for row in task_rows}),
                **summarize_result_rows(task_rows),
            }
        )

    metric_family_summary = build_metric_family_summary(rows)

    top_summary = summarize_result_rows(rows)
    runtime_pass_rate_planned_denominator = max(0, len(planned_jobs) - top_summary["component_invalid_count"])
    result_type_counts = dict(sorted(Counter(str(row.get("result_type")) for row in rows).items()))
    summary = {
        "schema_version": 2,
        "out_dir": str(args.out_dir),
        "planned_jobs": len(planned_jobs),
        "completed_jobs": len(rows),
        "missing_jobs": len(missing_keys),
        "raw_result_rows": len(result_rows),
        "deduped_result_rows": len(rows),
        "runtime_pass": top_summary["runtime_pass_count"],
        "runtime_fail": top_summary["runtime_fail_count"],
        "runtime_pass_rate_completed": top_summary["runtime_pass_rate"],
        "runtime_pass_rate_completed_denominator": top_summary["runtime_pass_rate_denominator"],
        "runtime_pass_rate_completed_excluded_component_invalid_count": top_summary[
            "runtime_pass_rate_excluded_component_invalid_count"
        ],
        "runtime_pass_rate_planned": (
            top_summary["runtime_pass_count"] / runtime_pass_rate_planned_denominator
            if runtime_pass_rate_planned_denominator
            else 0.0
        ),
        "runtime_pass_rate_planned_denominator": runtime_pass_rate_planned_denominator,
        "runtime_pass_rate_planned_excluded_component_invalid_count": top_summary["component_invalid_count"],
        "state_invalid_restore_components": top_summary["state_invalid_restore_components_count"],
        "state_invalid_rollout_components": top_summary["state_invalid_rollout_components_count"],
        "spec_invalid_component_dependency": top_summary["spec_invalid_component_dependency_count"],
        "component_invalid": top_summary["component_invalid_count"],
        "pre_satisfied_start": top_summary["pre_satisfied_start_count"],
        "metric_invalid": top_summary["metric_invalid_count"],
        "metric_invalid_missing_object": top_summary["metric_invalid_missing_object_count"],
        "rawdata_restore_failed": top_summary["rawdata_restore_failed_count"],
        "rawdata_restore_fallback": top_summary["rawdata_restore_fallback_count"],
        "rawdata_restore_review_needed": top_summary["rawdata_restore_review_needed_count"],
        "attemptable_segments": top_summary["attemptable_segment_count"],
        "policy_success_attemptable": top_summary["policy_success_attemptable_count"],
        "policy_success_attemptable_rate": top_summary["policy_success_attemptable_rate"],
        "policy_failure_attemptable": top_summary["policy_failure_attemptable_count"],
        "policy_failure_attemptable_rate": top_summary["policy_failure_attemptable_rate"],
        "policy_success_clean_attemptable": top_summary["policy_success_clean_attemptable_count"],
        "policy_success_clean_attemptable_rate": top_summary["policy_success_clean_attemptable_rate"],
        "short_video_problem": top_summary["short_video_problem_count"],
        "early_metric_activation_review_needed": top_summary["early_metric_activation_review_needed_count"],
        "short_proxy_success": top_summary["short_proxy_success_count"],
        "likely_proxy_false_positive": top_summary["likely_proxy_false_positive_count"],
        "transfer_pose_proxy_success_unconfirmed": top_summary[
            "transfer_pose_proxy_success_unconfirmed_count"
        ],
        "contact_effect_proxy_success_unconfirmed": top_summary[
            "contact_effect_proxy_success_unconfirmed_count"
        ],
        "articulation_close_success_unconfirmed": top_summary[
            "articulation_close_success_unconfirmed_count"
        ],
        "articulation_close_proxy_success_unconfirmed": top_summary[
            "articulation_close_proxy_success_unconfirmed_count"
        ],
        "geometry_base_facing_success_unconfirmed": top_summary[
            "geometry_base_facing_success_unconfirmed_count"
        ],
        "relation_transfer_proxy_success_unconfirmed": top_summary[
            "relation_transfer_proxy_success_unconfirmed_count"
        ],
        "orientation_proxy_success_unconfirmed": top_summary[
            "orientation_proxy_success_unconfirmed_count"
        ],
        "articulation_open_proxy_success_unconfirmed": top_summary[
            "articulation_open_proxy_success_unconfirmed_count"
        ],
        "articulation_open_success_unconfirmed": top_summary[
            "articulation_open_success_unconfirmed_count"
        ],
        "can_meat_lid_success_unconfirmed": top_summary["can_meat_lid_success_unconfirmed_count"],
        "meaningful_policy_caused_transition": top_summary["meaningful_policy_caused_transition_count"],
        "metric_unsatisfied_attemptable": top_summary["metric_unsatisfied_attemptable_count"],
        "timeout": top_summary["timeout_count"],
        "truncated": top_summary["truncated_count"],
        "env_terminated_metric_unsatisfied": top_summary["env_terminated_metric_unsatisfied_count"],
        "env_task_success_before_segment_success": top_summary["env_task_success_before_segment_success_count"],
        "env_task_success_grasp_release_review_needed": top_summary[
            "env_task_success_grasp_release_review_needed_count"
        ],
        "other_metric_unsatisfied": top_summary["other_metric_unsatisfied_count"],
        "env_done_success_count": top_summary["env_done_success_count"],
        "rollout_terminated_count": top_summary["rollout_terminated_count"],
        "rollout_truncated_flag_count": top_summary["rollout_truncated_flag_count"],
        "termination_reason_counts": top_summary["termination_reason_counts"],
        "env_termination_reason_counts": top_summary["env_termination_reason_counts"],
        "policy_success_raw": top_summary["success_count_raw"],
        "policy_success_raw_completed_rate": top_summary["success_rate_raw_completed"],
        "policy_success_raw_completed_denominator": top_summary["success_rate_raw_completed_denominator"],
        "policy_success_raw_completed_excluded_component_invalid": top_summary[
            "success_rate_raw_completed_excluded_component_invalid_count"
        ],
        "policy_success": top_summary["success_count_raw"],
        "result_type_counts": result_type_counts,
        "missing_job_keys": missing_keys,
        "skill_summary": skill_summary,
        "skill_task_summary": skill_task_summary,
        "metric_family_summary": metric_family_summary,
    }

    json_dump(args.out_dir / "multinode_skill_results.json", rows)
    write_csv(
        args.out_dir / "multinode_skill_results.csv",
        rows,
        fieldnames=[
            "job_key",
            "skill",
            "task_name",
            "demo_id",
            "instance_id",
            "skill_idx",
            "frame_duration",
            "dynamic_max_steps",
            "runtime_ok",
            "success",
            "result_type",
            "aggregation_eligible",
            "boundary_evaluation_eligible",
            "rollout_evaluation_eligible",
            "model_evaluated",
            "model_failure_eligible",
            "component_evaluability_json",
            "state_invalid_restore_components",
            "state_invalid_rollout_components",
            "spec_invalid_component_dependency",
            "component_invalid",
            "attemptable",
            "policy_attempted",
            "policy_success_attemptable",
            "policy_failure_attemptable",
            "policy_success_clean_attemptable",
            "metric_family",
            "start_all_satisfied",
            "short_video_problem",
            "metrics_short_video_problem",
            "early_metric_activation_review_needed",
            "short_proxy_success",
            "likely_proxy_false_positive",
            "transfer_pose_proxy_success_unconfirmed",
            "contact_effect_proxy_success_unconfirmed",
            "articulation_close_success_unconfirmed",
            "articulation_close_proxy_success_unconfirmed",
            "geometry_base_facing_success_unconfirmed",
            "relation_transfer_proxy_success_unconfirmed",
            "orientation_proxy_success_unconfirmed",
            "articulation_open_proxy_success_unconfirmed",
            "articulation_open_success_unconfirmed",
            "can_meat_lid_success_unconfirmed",
            "short_success_required_step",
            "min_success_steps",
            "first_predicate_satisfied_step",
            "early_predicate_satisfied_steps",
            "meaningful_policy_caused_transition",
            "final_step",
            "rollout_attempted",
            "termination_reason",
            "env_termination_reason",
            "env_done_success",
            "rollout_terminated",
            "rollout_truncated",
            "rollout_last_terminated",
            "env_terminated_seen",
            "env_done_success_seen",
            "first_env_terminated_step",
            "first_env_done_success_step",
            "env_task_success_before_segment_success",
            "env_task_success_grasp_release_review_needed",
            "restore_start_method",
            "restore_end_method",
            "restore_rollout_start_method",
            "rawdata_restore_start_failed",
            "rawdata_restore_end_failed",
            "rawdata_restore_rollout_start_failed",
            "rawdata_restore_start_fallback",
            "rawdata_restore_end_fallback",
            "rawdata_restore_rollout_start_fallback",
            "rawdata_restore_failure_count",
            "rawdata_restore_fallback_count",
            "rawdata_restore_failed",
            "rawdata_restore_fallback",
            "env_terminal_debug_json",
            "rollout_start_all_satisfied",
            "rollout_require_unsatisfied_at_start",
            "rollout_max_steps",
            "returncode",
            "metrics_path",
            "segment_log",
            "worker_rank",
            "node_rank",
            "gpu_id",
        ],
    )
    json_dump(args.out_dir / "multinode_skill_summary.json", summary)
    write_csv(
        args.out_dir / "multinode_skill_summary.csv",
        skill_summary,
        fieldnames=[
            "skill",
            "task_count",
            "demo_count",
            "segment_count",
            "runtime_pass_count",
            "runtime_fail_count",
            "runtime_pass_rate",
            "runtime_pass_rate_denominator",
            "runtime_pass_rate_excluded_component_invalid_count",
            "state_invalid_restore_components_count",
            "state_invalid_rollout_components_count",
            "spec_invalid_component_dependency_count",
            "component_invalid_count",
            "pre_satisfied_start_count",
            "metric_invalid_count",
            "metric_invalid_missing_object_count",
            "rawdata_restore_failed_count",
            "rawdata_restore_fallback_count",
            "rawdata_restore_review_needed_count",
            "attemptable_segment_count",
            "policy_success_attemptable_count",
            "policy_success_attemptable_rate",
            "policy_failure_attemptable_count",
            "policy_failure_attemptable_rate",
            "policy_success_clean_attemptable_count",
            "policy_success_clean_attemptable_rate",
            "short_video_problem_count",
            "early_metric_activation_review_needed_count",
            "short_proxy_success_count",
            "likely_proxy_false_positive_count",
            "transfer_pose_proxy_success_unconfirmed_count",
            "contact_effect_proxy_success_unconfirmed_count",
            "articulation_close_success_unconfirmed_count",
            "articulation_close_proxy_success_unconfirmed_count",
            "geometry_base_facing_success_unconfirmed_count",
            "relation_transfer_proxy_success_unconfirmed_count",
            "orientation_proxy_success_unconfirmed_count",
            "articulation_open_proxy_success_unconfirmed_count",
            "articulation_open_success_unconfirmed_count",
            "can_meat_lid_success_unconfirmed_count",
            "meaningful_policy_caused_transition_count",
            "metric_unsatisfied_attemptable_count",
            "timeout_count",
            "truncated_count",
            "env_terminated_metric_unsatisfied_count",
            "env_task_success_before_segment_success_count",
            "env_task_success_grasp_release_review_needed_count",
            "other_metric_unsatisfied_count",
            "success_count_raw",
            "success_rate_raw_completed",
            "success_rate_raw_completed_denominator",
            "success_rate_raw_completed_excluded_component_invalid_count",
            "success_count",
            "success_rate",
            "success_rate_denominator",
            "predicate_satisfied_count",
            "env_done_success_count",
            "rollout_terminated_count",
            "rollout_truncated_flag_count",
            "termination_reason_counts_json",
            "env_termination_reason_counts_json",
        ],
    )
    write_csv(
        args.out_dir / "multinode_skill_task_summary.csv",
        skill_task_summary,
        fieldnames=[
            "skill",
            "task_name",
            "demo_count",
            "segment_count",
            "runtime_pass_count",
            "runtime_fail_count",
            "runtime_pass_rate",
            "runtime_pass_rate_denominator",
            "runtime_pass_rate_excluded_component_invalid_count",
            "state_invalid_restore_components_count",
            "state_invalid_rollout_components_count",
            "spec_invalid_component_dependency_count",
            "component_invalid_count",
            "pre_satisfied_start_count",
            "metric_invalid_count",
            "metric_invalid_missing_object_count",
            "rawdata_restore_failed_count",
            "rawdata_restore_fallback_count",
            "rawdata_restore_review_needed_count",
            "attemptable_segment_count",
            "policy_success_attemptable_count",
            "policy_success_attemptable_rate",
            "policy_failure_attemptable_count",
            "policy_failure_attemptable_rate",
            "policy_success_clean_attemptable_count",
            "policy_success_clean_attemptable_rate",
            "short_video_problem_count",
            "early_metric_activation_review_needed_count",
            "short_proxy_success_count",
            "likely_proxy_false_positive_count",
            "transfer_pose_proxy_success_unconfirmed_count",
            "contact_effect_proxy_success_unconfirmed_count",
            "articulation_close_success_unconfirmed_count",
            "articulation_close_proxy_success_unconfirmed_count",
            "geometry_base_facing_success_unconfirmed_count",
            "relation_transfer_proxy_success_unconfirmed_count",
            "orientation_proxy_success_unconfirmed_count",
            "articulation_open_proxy_success_unconfirmed_count",
            "articulation_open_success_unconfirmed_count",
            "can_meat_lid_success_unconfirmed_count",
            "meaningful_policy_caused_transition_count",
            "metric_unsatisfied_attemptable_count",
            "timeout_count",
            "truncated_count",
            "env_terminated_metric_unsatisfied_count",
            "env_task_success_before_segment_success_count",
            "env_task_success_grasp_release_review_needed_count",
            "other_metric_unsatisfied_count",
            "success_count_raw",
            "success_rate_raw_completed",
            "success_rate_raw_completed_denominator",
            "success_rate_raw_completed_excluded_component_invalid_count",
            "success_count",
            "success_rate",
            "success_rate_denominator",
            "predicate_satisfied_count",
            "env_done_success_count",
            "rollout_terminated_count",
            "rollout_truncated_flag_count",
            "termination_reason_counts_json",
            "env_termination_reason_counts_json",
        ],
    )
    json_dump(args.out_dir / "metric_family_summary.json", metric_family_summary)
    write_csv(
        args.out_dir / "metric_family_summary.csv",
        metric_family_summary,
        fieldnames=[
            "metric_family",
            "segment_count",
            "runtime_pass_count",
            "runtime_fail_count",
            "runtime_pass_rate",
            "runtime_pass_rate_denominator",
            "runtime_pass_rate_excluded_component_invalid_count",
            "state_invalid_restore_components_count",
            "state_invalid_rollout_components_count",
            "spec_invalid_component_dependency_count",
            "component_invalid_count",
            "rawdata_restore_failed_count",
            "rawdata_restore_fallback_count",
            "rawdata_restore_review_needed_count",
            "attemptable_segment_count",
            "policy_success_attemptable_count",
            "policy_failure_attemptable_count",
            "policy_success_clean_attemptable_count",
            "review_needed_success_count",
            "short_video_problem_count",
            "early_metric_activation_review_needed_count",
            "short_proxy_success_count",
            "likely_proxy_false_positive_count",
            "transfer_pose_proxy_success_unconfirmed_count",
            "contact_effect_proxy_success_unconfirmed_count",
            "articulation_close_success_unconfirmed_count",
            "articulation_close_proxy_success_unconfirmed_count",
            "geometry_base_facing_success_unconfirmed_count",
            "relation_transfer_proxy_success_unconfirmed_count",
            "orientation_proxy_success_unconfirmed_count",
            "articulation_open_proxy_success_unconfirmed_count",
            "articulation_open_success_unconfirmed_count",
            "can_meat_lid_success_unconfirmed_count",
            "meaningful_policy_caused_transition_count",
            "metric_unsatisfied_attemptable_count",
            "timeout_count",
            "truncated_count",
            "env_terminated_metric_unsatisfied_count",
            "env_task_success_before_segment_success_count",
            "env_task_success_grasp_release_review_needed_count",
            "other_metric_unsatisfied_count",
            "success_count_raw",
            "success_rate_raw_completed",
            "success_rate_raw_completed_denominator",
            "success_rate_raw_completed_excluded_component_invalid_count",
        ],
    )
    (args.out_dir / "multinode_skill_summary.md").write_text(render_summary_md(summary))
    print(json.dumps(summary, indent=2))
    return 0


def materialize_persistent_jobs(args: argparse.Namespace, jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Write per-GPU JSONL job queues for persistent workers.

    Reuses :func:`build_worker_assignments` so every contiguous run of segments
    on a worker shares ``task_name`` (task-affinity scheduling). Honours
    ``--resume`` by skipping any ``job_key`` whose ``metrics/*.json`` already
    exists. Each segment is augmented with a precomputed ``dynamic_max_steps``
    so the worker does not need to reparse ``frame_duration``.

    Returns a list of per-worker plan dicts (rank, gpu_id, jobs_path, count).
    """
    total_workers = args.num_nodes * args.gpus_per_node
    worker_jobs = build_worker_assignments(jobs, total_workers)
    jobs_dir = args.out_dir / "worker_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    plans: List[Dict[str, Any]] = []
    for local_rank, gpu_id in enumerate(args.local_gpu_ids):
        worker_rank = args.node_rank * args.gpus_per_node + local_rank
        jobs_path = jobs_dir / f"persistent_worker_{worker_rank:03d}.jobs.jsonl"
        # Truncate any stale queue from a previous launch so we cannot
        # accidentally replay shutdown lines from yesterday.
        if jobs_path.exists():
            jobs_path.unlink()
        jobs_path.parent.mkdir(parents=True, exist_ok=True)
        jobs_path.touch()

        assignments = worker_jobs[worker_rank]
        written = 0
        skipped_resume = 0
        for sample in assignments:
            if args.resume:
                metrics_dir = (
                    args.out_dir
                    / "raw"
                    / sample["task_name"]
                    / f"demo_{sample['demo_id']}"
                    / f"skill_{int(sample['skill_idx']):03d}"
                    / "metrics"
                )
                if metrics_dir.exists() and any(metrics_dir.glob("*.json")):
                    skipped_resume += 1
                    continue
            payload = dict(sample)
            payload["dynamic_max_steps"] = get_dynamic_max_steps(
                sample.get("frame_duration"),
                fallback=int(args.max_steps),
                cap=max(0, int(args.max_dynamic_steps_cap or 0)),
            )
            append_jsonl(jobs_path, {"action": "assign", "sample": payload})
            written += 1

        plans.append(
            {
                "worker_rank": worker_rank,
                "gpu_id": int(gpu_id),
                "jobs_path": str(jobs_path),
                "assigned": len(assignments),
                "skipped_resume": skipped_resume,
                "written": written,
            }
        )

    write_csv(
        args.out_dir / "persistent_worker_plan.csv",
        plans,
        fieldnames=["worker_rank", "gpu_id", "jobs_path", "assigned", "skipped_resume", "written"],
    )
    return plans


def _persistent_worker_env(args: argparse.Namespace, gpu_id: int, port: int) -> Dict[str, str]:
    """Build the env dict for a persistent worker subprocess (mirrors run_segment_eval)."""
    env = os.environ.copy()
    og_appdata_base = Path(env.get("OMNIGIBSON_APPDATA_PATH_BASE", "/tmp/omnigibson-appdata"))
    og_user = env.get("USER", "user")
    og_appdata_scope = env.get("OMNIGIBSON_APPDATA_SCOPE", "gpu")
    if og_appdata_scope == "gpu":
        og_appdata = og_appdata_base / og_user / f"gpu{gpu_id}"
    elif og_appdata_scope == "gpu_port":
        og_appdata = og_appdata_base / og_user / f"gpu{gpu_id}_p{port}"
    elif og_appdata_scope == "run_gpu_port":
        og_appdata = og_appdata_base / og_user / args.out_dir.name / f"gpu{gpu_id}_p{port}"
    else:
        raise RuntimeError(
            "invalid OMNIGIBSON_APPDATA_SCOPE; expected one of: gpu, gpu_port, run_gpu_port"
        )
    og_appdata.mkdir(parents=True, exist_ok=True)

    pythonpath = ":".join(
        [
            str(REPO_ROOT / "src"),
            str(args.behavior_dir / "joylo"),
            str(args.behavior_dir / "OmniGibson"),
            str(args.behavior_dir / "bddl3"),
        ]
    )
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + ":" + env["PYTHONPATH"]

    no_proxy = env.get("NO_PROXY", "")
    no_proxy_chunks = ["localhost", "127.0.0.1", "::1"]
    if no_proxy:
        no_proxy_chunks.append(no_proxy)
    no_proxy_value = ",".join(no_proxy_chunks)

    overrides = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": pythonpath,
        "NO_PROXY": no_proxy_value,
        "no_proxy": no_proxy_value,
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "OMNIGIBSON_DATA_PATH": str(args.behavior_dir / "datasets"),
        "OMNIGIBSON_APPDATA_PATH": str(og_appdata),
        "MPLBACKEND": env.get("MPLBACKEND", "Agg"),
        "TORCHDYNAMO_DISABLE": env.get("TORCHDYNAMO_DISABLE", "1"),
        "TORCHINDUCTOR_DISABLE": env.get("TORCHINDUCTOR_DISABLE", "1"),
        "OMNIGIBSON_HEADLESS": "true",
        "OMNIGIBSON_DISABLE_EXTENSION_REGISTRY": "0",
        "OMNIGIBSON_DISABLE_DRIVER_VERSION_CHECK": "1",
        # Persistent workers should avoid RGB sensor render-product
        # reconfiguration by default. That path is a known source of slow or
        # unstable task loads in long-lived Isaac processes.
        "PERSISTENT_EVAL_DISABLE_SENSOR_RECONFIG": env.get("PERSISTENT_EVAL_DISABLE_SENSOR_RECONFIG", "1"),
        # Keep persistent workers on the safe headless viewer path by default.
        # Letting eval_segment re-enable the viewer camera / full UI has been a
        # recurring source of long task loads and GPU / property-window crashes.
        "PERSISTENT_EVAL_FORCE_SAFE_VIEWER_FLAGS": env.get("PERSISTENT_EVAL_FORCE_SAFE_VIEWER_FLAGS", "1"),
        # Some tasks can still wedge inside VisionSensor warmup renders during
        # env construction even after disabling RGB sensor reconfiguration and
        # clamping viewer flags. Skip those eager warmup renders by default for
        # persistent workers; the simulator can still render later during normal
        # rollout / observation paths after the env finishes loading.
        "PERSISTENT_EVAL_SKIP_VISION_SENSOR_WARMUP": env.get("PERSISTENT_EVAL_SKIP_VISION_SENSOR_WARMUP", "1"),
        "PERSISTENT_WORKER_HEARTBEAT_S": str(args.persistent_worker_heartbeat_s),
        "PERSISTENT_WORKER_TASK_RELOAD_TIMEOUT_S": str(args.persistent_worker_task_reload_timeout),
        "PERSISTENT_WORKER_SEGMENT_TIMEOUT_S": str(args.persistent_worker_segment_timeout),
    }
    env.pop("OMNIGIBSON_GPU_ID", None)
    env.update(overrides)
    return env


def _spawn_persistent_worker(
    args: argparse.Namespace,
    *,
    worker_rank: int,
    gpu_id: int,
    jobs_path: Path,
) -> Tuple[subprocess.Popen[str], Path, Path]:
    local_rank = worker_rank - args.node_rank * args.gpus_per_node
    port = args.port_base + local_rank
    worker_log = args.out_dir / "launcher_logs" / f"persistent_worker{worker_rank:03d}.log"
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    cmd = f"""
set -euo pipefail
source {q(CONDA_SH)}
conda activate {q(args.behavior_env)}
cd {q(REPO_ROOT)}
python -u {q(str(args.persistent_worker_script))} \\
  --out-dir {q(args.out_dir)} \\
  --worker-rank {q(worker_rank)} \\
  --gpu-id {q(gpu_id)} \\
  --port-base {q(port)} \\
  --gpus-per-node {q(args.gpus_per_node)} \\
  --max-steps {q(args.max_steps)} \\
  --max-dynamic-steps-cap {q(args.max_dynamic_steps_cap)} \\
  --server-ready-timeout {q(args.server_ready_timeout)} \\
  --openpi-env {q(args.openpi_env)} \\
  --behavior-env {q(args.behavior_env)} \\
  --config-name {q(args.config_name)} \\
  --policy-backend {q(args.policy_backend)} \\
  --ckpt-dir {q(args.ckpt_dir)} \\
  --behavior-dir {q(args.behavior_dir)} \\
  --demo-data-path {q(args.demo_data_path)} \\
  --rawdata-path {q(args.rawdata_path)} \\
  --launcher-pid {q(os.getpid())}{' --dry-run' if args.dry_run else ''}{' --write-video' if args.write_video else ''}{' --partial-scene-load' if args.partial_scene_load else ''}{' --render-viewer-camera' if args.render_viewer_camera else ''}{' --gui-viewport-only' if args.gui_viewport_only else ''}{' --skip-intermediate-obs-in-chunk' if args.skip_intermediate_obs_in_chunk else ''} --env-wrapper-target {q(args.env_wrapper_target)}{' --segment-predicate-dump-trace' if args.segment_predicate_dump_trace else ''}{' --resume' if args.resume else ''}
"""
    env = _persistent_worker_env(args, gpu_id=gpu_id, port=port)
    with worker_log.open("a") as f:
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=env,
        )
    return proc, worker_log, jobs_path


def run_persistent_worker_inplace(args: argparse.Namespace) -> int:
    """Re-exec ``persistent_skill_eval_worker.py`` so ``--mode persistent-worker`` is a thin wrapper."""
    if args.worker_rank < 0 or args.gpu_id < 0:
        raise RuntimeError("persistent-worker mode requires --worker-rank and --gpu-id")
    cmd = [
        sys.executable,
        "-u",
        str(args.persistent_worker_script),
        "--out-dir",
        str(args.out_dir),
        "--worker-rank",
        str(args.worker_rank),
        "--gpu-id",
        str(args.gpu_id),
        "--port-base",
        str(args.port_base),
        "--gpus-per-node",
        str(args.gpus_per_node),
        "--max-steps",
        str(args.max_steps),
        "--max-dynamic-steps-cap",
        str(args.max_dynamic_steps_cap),
        "--server-ready-timeout",
        str(args.server_ready_timeout),
        "--openpi-env",
        args.openpi_env,
        "--behavior-env",
        args.behavior_env,
        "--config-name",
        args.config_name,
        "--policy-backend",
        args.policy_backend,
        "--ckpt-dir",
        str(args.ckpt_dir),
        "--behavior-dir",
        str(args.behavior_dir),
        "--demo-data-path",
        str(args.demo_data_path),
        "--rawdata-path",
        str(args.rawdata_path),
        "--launcher-pid",
        str(int(os.environ.get("PERSISTENT_LAUNCHER_PID", os.getppid()))),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.write_video:
        cmd.append("--write-video")
    if args.partial_scene_load:
        cmd.append("--partial-scene-load")
    if args.render_viewer_camera:
        cmd.append("--render-viewer-camera")
    if args.gui_viewport_only:
        cmd.append("--gui-viewport-only")
    if args.skip_intermediate_obs_in_chunk:
        cmd.append("--skip-intermediate-obs-in-chunk")
    cmd.extend(["--env-wrapper-target", args.env_wrapper_target])
    if args.segment_predicate_dump_trace:
        cmd.append("--segment-predicate-dump-trace")
    if args.resume:
        cmd.append("--resume")
    os.execv(cmd[0], cmd)
    return 0  # unreachable


def launch_node_persistent(args: argparse.Namespace) -> int:
    """Spawn 8 persistent per-GPU workers and feed them via task-affinity JSONL queues."""
    manifest_path = args.out_dir / "manifest.json"
    if args.node_rank == 0 and (args.rebuild_manifest or not manifest_path.exists()):
        registry = load_registry(args.behavior_dir / REGISTRY_REL_PATH)
        jobs = collect_skill_jobs(
            registry=registry,
            demo_data_path=args.demo_data_path,
            skills_filter=args.skills_filter,
            max_samples_per_skill=args.max_samples_per_skill,
            max_samples_per_skill_task=args.max_samples_per_skill_task,
            max_total_jobs=args.max_total_jobs,
        )
        write_manifest(args, jobs)
    else:
        wait_for_path(manifest_path, args.prepare_timeout)

    if args.mode == "prepare":
        return 0

    manifest = json.loads(manifest_path.read_text())
    plans = materialize_persistent_jobs(args, manifest.get("jobs", []))
    print(
        f"[launcher-persistent] materialized jobs: "
        + ", ".join(f"r{p['worker_rank']:03d}={p['written']}" for p in plans),
        flush=True,
    )

    children: Dict[int, Dict[str, Any]] = {}
    try:
        for plan in plans:
            proc, worker_log, jobs_path = _spawn_persistent_worker(
                args,
                worker_rank=int(plan["worker_rank"]),
                gpu_id=int(plan["gpu_id"]),
                jobs_path=Path(plan["jobs_path"]),
            )
            worker_rank = int(plan["worker_rank"])
            children[worker_rank] = {
                "proc": proc,
                "worker_rank": worker_rank,
                "gpu_id": int(plan["gpu_id"]),
                "log_path": worker_log,
                "jobs_path": jobs_path,
                "status_path": persistent_worker_status_path(args.out_dir, worker_rank),
                "written": int(plan["written"]),
                "shutdown_appended": False,
                "restart_count": 0,
                "last_restart_at": 0.0,
            }

        heartbeat_timeout_s = max(
            float(args.persistent_worker_heartbeat_s) * 3.0,
            float(args.persistent_worker_heartbeat_s) + 30.0,
        )
        shutdown_sent = False
        while True:
            if shutdown_sent and all(child["proc"].poll() is not None for child in children.values()):
                break

            now = time.time()
            if not shutdown_sent:
                all_done = True
                for rank, child in children.items():
                    summary = summarize_persistent_worker_status(child["status_path"])
                    child["status_summary"] = summary
                    done_count = int(summary.get("done_count", 0))
                    if done_count < int(child["written"]):
                        all_done = False

                    proc = child["proc"]
                    restart_reason = persistent_worker_restart_reason(
                        summary,
                        now=now,
                        heartbeat_timeout_s=heartbeat_timeout_s,
                        task_reload_timeout_s=float(args.persistent_worker_task_reload_timeout),
                        segment_timeout_s=float(args.persistent_worker_segment_timeout),
                    )
                    if (
                        proc.poll() is None
                        and restart_reason
                        and now - float(child.get("last_restart_at", 0.0)) >= heartbeat_timeout_s
                    ):
                        print(
                            f"[launcher-persistent] worker {rank:03d} stalled ({restart_reason}); restarting. "
                            f"status={summary}",
                            flush=True,
                        )
                        stop_process(proc)
                        stop_stale_persistent_servers(args.out_dir, rank)
                        proc, worker_log, jobs_path = _spawn_persistent_worker(
                            args,
                            worker_rank=rank,
                            gpu_id=int(child["gpu_id"]),
                            jobs_path=Path(child["jobs_path"]),
                        )
                        child.update(
                            {
                                "proc": proc,
                                "log_path": worker_log,
                                "jobs_path": jobs_path,
                                "restart_count": int(child["restart_count"]) + 1,
                                "last_restart_at": now,
                            }
                        )
                        continue

                    if proc.poll() is not None and done_count < int(child["written"]):
                        print(
                            f"[launcher-persistent] worker {rank:03d} exited early with code {proc.returncode}; "
                            f"restarting. log: {child['log_path']}",
                            flush=True,
                        )
                        print(tail_text(Path(child["log_path"])))
                        stop_stale_persistent_servers(args.out_dir, rank)
                        proc, worker_log, jobs_path = _spawn_persistent_worker(
                            args,
                            worker_rank=rank,
                            gpu_id=int(child["gpu_id"]),
                            jobs_path=Path(child["jobs_path"]),
                        )
                        child.update(
                            {
                                "proc": proc,
                                "log_path": worker_log,
                                "jobs_path": jobs_path,
                                "restart_count": int(child["restart_count"]) + 1,
                                "last_restart_at": now,
                            }
                        )
                        all_done = False

                if all_done:
                    for child in children.values():
                        if not child["shutdown_appended"]:
                            append_jsonl(Path(child["jobs_path"]), {"action": "shutdown"})
                            child["shutdown_appended"] = True
                    shutdown_sent = True
            time.sleep(2.0)

        # Drain final return codes.
        any_failed = False
        for rank, child in children.items():
            proc = child["proc"]
            log_path = Path(child["log_path"])
            code = proc.poll() if proc.poll() is not None else proc.wait(timeout=args.persistent_worker_shutdown_timeout)
            if code != 0:
                any_failed = True
                print(
                    f"[launcher-persistent] worker {rank:03d} final exit code {code}; log: {log_path}",
                    flush=True,
                )
                print(tail_text(log_path))
        return 1 if any_failed else 0
    finally:
        # SIGTERM/SIGKILL anyone still alive past the deadline.
        for rank, child in children.items():
            stop_process(child.get("proc"))
            stop_stale_persistent_servers(args.out_dir, rank)



def launch_node(args: argparse.Namespace) -> int:
    manifest_path = args.out_dir / "manifest.json"
    if args.node_rank == 0 and (args.rebuild_manifest or not manifest_path.exists()):
        registry = load_registry(args.behavior_dir / REGISTRY_REL_PATH)
        jobs = collect_skill_jobs(
            registry=registry,
            demo_data_path=args.demo_data_path,
            skills_filter=args.skills_filter,
            max_samples_per_skill=args.max_samples_per_skill,
            max_samples_per_skill_task=args.max_samples_per_skill_task,
            max_total_jobs=args.max_total_jobs,
        )
        write_manifest(args, jobs)
    else:
        wait_for_path(manifest_path, args.prepare_timeout)

    if args.mode == "prepare":
        return 0

    local_gpu_ids = args.local_gpu_ids
    children: List[Tuple[subprocess.Popen[str], int, Path]] = []
    try:
        for local_rank, gpu_id in enumerate(local_gpu_ids):
            worker_rank = args.node_rank * args.gpus_per_node + local_rank
            port = args.port_base + local_rank
            worker_log = args.out_dir / "launcher_logs" / f"node{args.node_rank:02d}_worker{worker_rank:03d}.log"
            worker_log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--mode",
                "worker",
                "--out-dir",
                str(args.out_dir),
                "--node-rank",
                str(args.node_rank),
                "--num-nodes",
                str(args.num_nodes),
                "--gpus-per-node",
                str(args.gpus_per_node),
                "--port-base",
                str(args.port_base),
                "--max-steps",
                str(args.max_steps),
                "--server-ready-timeout",
                str(args.server_ready_timeout),
                "--prepare-timeout",
                str(args.prepare_timeout),
                "--openpi-env",
                args.openpi_env,
                "--behavior-env",
                args.behavior_env,
                "--config-name",
                args.config_name,
                "--policy-backend",
                args.policy_backend,
                "--ckpt-dir",
                str(args.ckpt_dir),
                "--behavior-dir",
                str(args.behavior_dir),
                "--demo-data-path",
                str(args.demo_data_path),
                "--rawdata-path",
                str(args.rawdata_path),
                "--worker-rank",
                str(worker_rank),
                "--gpu-id",
                str(gpu_id),
                "--port",
                str(port),
                "--max-dynamic-steps-cap",
                str(args.max_dynamic_steps_cap),
            ]
            if args.server_start_stagger_s > 0:
                cmd.extend(["--server-start-stagger-s", str(args.server_start_stagger_s)])
            if args.dry_run:
                cmd.append("--dry-run")
            if args.write_video:
                cmd.append("--write-video")
            if args.partial_scene_load:
                cmd.append("--partial-scene-load")
            if args.render_viewer_camera:
                cmd.append("--render-viewer-camera")
            if args.gui_viewport_only:
                cmd.append("--gui-viewport-only")
            if args.skip_intermediate_obs_in_chunk:
                cmd.append("--skip-intermediate-obs-in-chunk")
            cmd.extend(["--env-wrapper-target", args.env_wrapper_target])
            if args.segment_predicate_dump_trace:
                cmd.append("--segment-predicate-dump-trace")
            if args.resume:
                cmd.append("--resume")
            with worker_log.open("w") as f:
                proc = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            children.append((proc, worker_rank, worker_log))

        remaining = list(children)
        while remaining:
            next_remaining: List[Tuple[subprocess.Popen[str], int, Path]] = []
            for proc, worker_rank, worker_log in remaining:
                code = proc.poll()
                if code is None:
                    next_remaining.append((proc, worker_rank, worker_log))
                    continue
                if code != 0:
                    print(
                        f"[launcher] worker {worker_rank:03d} exited with code {code}. "
                        f"log: {worker_log}"
                    )
                    print(f"[launcher] ===== worker {worker_rank:03d} log tail begin =====")
                    print(tail_text(worker_log))
                    print(f"[launcher] ===== worker {worker_rank:03d} log tail end =====")
                    return code
            if not next_remaining:
                return 0
            time.sleep(2.0)
            remaining = next_remaining
        return 0
    finally:
        for proc, _, _ in children:
            stop_process(proc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multinode segment/skill sweep for BEHAVIOR-1K skill success evaluation."
    )
    parser.add_argument(
        "--mode",
        choices=["launch", "prepare", "merge", "worker", "launch-persistent", "persistent-worker"],
        default="launch",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "segment_eval_runs" / f"skill_multinode_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--node-rank", type=int, default=int(os.environ.get("NODE_RANK", "0")))
    parser.add_argument("--num-nodes", type=int, default=int(os.environ.get("NUM_NODES", "1")))
    parser.add_argument("--gpus-per-node", type=int, default=int(os.environ.get("GPUS_PER_NODE", "8")))
    parser.add_argument("--local-gpu-ids", default=os.environ.get("LOCAL_GPU_IDS", ""))
    parser.add_argument(
        "--port-base",
        type=int,
        default=0,
        help="base port for per-GPU workers; 0 picks a random safe base automatically",
    )
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument(
        "--max-dynamic-steps-cap",
        type=int,
        default=0,
        help="optional upper bound for duration-derived segment_max_steps; 0 disables the cap",
    )
    parser.add_argument("--server-ready-timeout", type=int, default=1800)
    parser.add_argument(
        "--server-start-stagger-s",
        type=int,
        default=0,
        help="optional stagger (seconds) multiplied by worker_rank before starting the task server; helps avoid JIT storms",
    )
    parser.add_argument("--prepare-timeout", type=int, default=3600)
    parser.add_argument("--openpi-env", default="openpi-comet-nas")
    parser.add_argument("--behavior-env", default="behavior")
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--policy-backend", choices=["auto", "torch", "jax"], default="auto")
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT)
    parser.add_argument(
        "--behavior-dir",
        type=Path,
        default=BEHAVIOR_DIR_DEFAULT,
        help=(
            "BEHAVIOR-1K repo root; this launcher derives OMNIGIBSON_DATA_PATH as <behavior_dir>/datasets. "
            f"For SSD asset placement and recommended defaults, see {PERF_OPT_DOC_HINT}"
        ),
    )
    parser.add_argument("--demo-data-path", type=Path, default=DEMO_DATA_PATH_DEFAULT)
    parser.add_argument("--rawdata-path", type=Path, default=RAWDATA_PATH_DEFAULT)
    parser.add_argument("--skills", default="", help="comma-separated subset of skill names")
    parser.add_argument("--max-samples-per-skill", type=int, default=0, help="0 means use all matched segments")
    parser.add_argument(
        "--max-samples-per-skill-task",
        type=int,
        default=0,
        help="optional cap per (skill, task) pair; 0 means unlimited",
    )
    parser.add_argument("--max-total-jobs", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--dry-run", action="store_true", help="pass dry_run=true to eval_segment.py")
    parser.add_argument("--write-video", action="store_true")
    parser.add_argument(
        "--partial-scene-load",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "whether to enable slim_only / task-relevant room loading; defaults to off. "
            f"See {PERF_OPT_DOC_HINT} for when this is worth enabling"
        ),
    )
    parser.add_argument("--render-viewer-camera", action="store_true")
    parser.add_argument("--gui-viewport-only", action="store_true")
    parser.add_argument(
        "--skip-intermediate-obs-in-chunk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "whether to enable slim_skipobs / chunk-internal obs skipping; defaults to on. "
            f"See {PERF_OPT_DOC_HINT} for recommended launcher defaults"
        ),
    )
    parser.add_argument("--env-wrapper-target", default="omnigibson.learning.wrappers.RGBLowResWrapper")
    parser.add_argument("--segment-predicate-dump-trace", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument(
        "--allow-tmp-out-dir",
        action="store_true",
        help="allow writing outputs under /tmp; not recommended for formal evaluation",
    )
    parser.add_argument("--worker-rank", type=int, default=-1)
    parser.add_argument("--gpu-id", type=int, default=-1)
    parser.add_argument("--port", type=int, default=-1)
    # Persistent-worker mode knobs.
    parser.add_argument(
        "--persistent-worker-script",
        type=Path,
        default=Path(__file__).resolve().parent / "persistent_skill_eval_worker.py",
        help="path to the per-GPU persistent worker entrypoint",
    )
    parser.add_argument(
        "--persistent-worker-shutdown-timeout",
        type=int,
        default=900,
        help="seconds to wait after appending {action: shutdown} before SIGTERM/SIGKILL",
    )
    parser.add_argument(
        "--persistent-worker-heartbeat-s",
        type=float,
        default=float(os.environ.get("PERSISTENT_WORKER_HEARTBEAT_S", "60")),
        help="expected persistent-worker heartbeat interval in seconds",
    )
    parser.add_argument(
        "--persistent-worker-task-reload-timeout",
        type=int,
        default=int(os.environ.get("PERSISTENT_WORKER_TASK_RELOAD_TIMEOUT_S", "1800")),
        help="watchdog timeout for task loading / server startup before restarting a worker",
    )
    parser.add_argument(
        "--persistent-worker-segment-timeout",
        type=int,
        default=int(os.environ.get("PERSISTENT_WORKER_SEGMENT_TIMEOUT_S", "5400")),
        help="watchdog timeout for a single persistent segment run before restarting a worker",
    )
    args = parser.parse_args()

    args.out_dir = args.out_dir.resolve()
    args.ckpt_dir = args.ckpt_dir.resolve()
    args.behavior_dir = args.behavior_dir.resolve()
    args.demo_data_path = args.demo_data_path.resolve()
    args.rawdata_path = args.rawdata_path.resolve()
    args.skills_filter = parse_csv_strings(args.skills)
    if args.local_gpu_ids.strip():
        args.local_gpu_ids = parse_csv_ints(args.local_gpu_ids)
    else:
        args.local_gpu_ids = list(range(args.gpus_per_node))
    if len(args.local_gpu_ids) != args.gpus_per_node:
        raise RuntimeError("LOCAL_GPU_IDS count must match --gpus-per-node")

    # Pick a random, non-privileged port base by default to reduce collisions with stale runs.
    # Keep enough headroom for retry hopping (find_free_port with stride=gpus_per_node).
    if int(args.port_base) <= 0:
        rng = random.SystemRandom()
        args.port_base = rng.randint(20000, 60000)
    if str(args.out_dir).startswith("/tmp") and not args.allow_tmp_out_dir:
        raise RuntimeError(
            "refusing to write formal evaluation outputs under /tmp. "
            "Please use a persistent path under the repo, or pass --allow-tmp-out-dir for temporary smoke tests."
        )
    return args


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "merge":
        return merge_results(args)
    if args.mode == "worker":
        if args.worker_rank < 0 or args.gpu_id < 0 or args.port < 0:
            raise RuntimeError("worker mode requires --worker-rank, --gpu-id and --port")
        return run_worker(args)
    if args.mode == "persistent-worker":
        # Thin wrapper that re-exec's the dedicated persistent worker entrypoint so
        # the launcher binary is the single CLI users learn.
        return run_persistent_worker_inplace(args)
    if args.mode == "launch-persistent":
        return launch_node_persistent(args)
    return launch_node(args)


if __name__ == "__main__":
    sys.exit(main())
