#!/usr/bin/env python3
"""Persistent per-GPU skill-eval worker.

This worker keeps a single OmniGibson / Isaac Sim process alive across many
segment rollouts so the dominant 600-700 s pre-rollout setup is paid once per
task (and reused across all segments of that task on this GPU).

Lifecycle (per worker, per GPU):
  1. Apply OmniGibson `gm` flags BEFORE importing OmniGibsonEnv.
  2. Import the BEHAVIOR-1K segment evaluator.
  3. Watch a JSONL job queue at ``worker_jobs/persistent_worker_{rank}.jobs.jsonl``.
  4. For each ``{"action": "assign", "sample": {...}}`` line:
       - if the assigned ``task_name`` differs from the loaded task, tear the
         current scene/policy server down and (re)build the evaluator;
       - start (or reuse) a ``serve_b1k.py`` policy server with a fresh
         ``server_run_id`` / ``server_token`` and wait for health;
       - reconfigure the evaluator for this segment and run it via
         :func:`omnigibson.learning.eval_segment.run_segment_on_env`;
       - emit the same row schema as
         :func:`run_skill_metric_multinode_sweep.load_metrics_row` to
         ``worker_results/persistent_worker_{rank}.jsonl``.
  5. ``{"action": "shutdown"}`` (or launcher PID disappearing) triggers a clean
     exit; an unhandled exception or a configurable cap on segments-since-boot
     triggers a soft restart via ``os.execv`` so leaked Isaac state is purged.

The launcher (``run_skill_metric_multinode_sweep.py --mode persistent-worker``)
is the only supported producer of jobs files; this script intentionally does
not parse the manifest itself.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Import the launcher's helpers (already used by --mode worker) so the row
# schema, server identity / health helpers, and dynamic-step formula stay in
# one place.
import run_skill_metric_multinode_sweep as launcher  # noqa: E402

logger = logging.getLogger("persistent_skill_eval_worker")

DEFAULT_HEARTBEAT_S = 60.0
DEFAULT_MAX_SEGMENTS_BEFORE_RESTART = 64
DEFAULT_TASK_RELOAD_TIMEOUT_S = 1800
DEFAULT_WATCHDOG_POLL_S = 5.0
_GM_FLAGS_APPLIED = False


def _setup_logging(rank: int) -> None:
    fmt = f"%(asctime)s [persistent_worker {rank:03d}] %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, force=True)


def _ensure_isaac_env() -> None:
    """Populate Isaac Sim env vars that may be lost across ``os.execv``.

    A first-time OmniGibson launch can succeed without explicit Isaac env vars
    if Isaac Sim was imported through the active environment, but a later
    ``execv`` puts the worker back at process start.  In that path
    OmniGibson's launcher reads ``ISAAC_PATH`` / ``EXP_PATH`` directly, so make
    the inferred Isaac install paths explicit before any evaluator construction
    / soft restart.
    """
    def _set_from_isaac_path(isaac_path: Path) -> bool:
        apps_path = isaac_path / "apps"
        if not ((isaac_path / "VERSION").exists() and apps_path.exists() and (isaac_path / "exts").exists()):
            return False
        os.environ.setdefault("ISAAC_PATH", str(isaac_path))
        os.environ.setdefault("EXP_PATH", str(apps_path))
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        logger.info("Using Isaac env ISAAC_PATH=%s EXP_PATH=%s", os.environ["ISAAC_PATH"], os.environ["EXP_PATH"])
        return True

    existing_isaac_path = os.environ.get("ISAAC_PATH")
    if existing_isaac_path and _set_from_isaac_path(Path(existing_isaac_path)):
        return

    candidates: List[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        envs_dir = Path(conda_prefix).resolve().parent
        candidates.extend(envs_dir.glob("*/lib/python*/site-packages/isaacsim"))
    candidates.extend(Path(sys.executable).resolve().parents[1].glob("*/lib/python*/site-packages/isaacsim"))

    for candidate in candidates:
        if _set_from_isaac_path(candidate):
            return


def _reexec_in_behavior_env(args: argparse.Namespace, argv: List[str], *, force: bool = False) -> None:
    """Re-exec this worker under the requested BEHAVIOR / OmniGibson env.

    The policy server intentionally runs under the OpenPI env, but the
    simulator-side persistent worker must run under ``--behavior-env``.  This
    guard also makes soft restarts robust when the worker was launched through
    a wrapper or a manual retry from the wrong conda env.
    """
    behavior_env = str(args.behavior_env)
    current_env = os.environ.get("CONDA_DEFAULT_ENV")
    current_prefix_name = Path(os.environ.get("CONDA_PREFIX", "")).name
    if not force and behavior_env in {current_env, current_prefix_name}:
        return

    cmd = " && ".join(
        [
            f"source {shlex.quote(str(launcher.CONDA_SH))}",
            f"conda activate {shlex.quote(behavior_env)}",
            f"cd {shlex.quote(str(REPO_ROOT))}",
            "exec python -u " + " ".join(shlex.quote(str(part)) for part in argv),
        ]
    )
    os.execv("/bin/bash", ["bash", "-lc", cmd])


def _apply_gm_flags() -> None:
    """Apply OmniGibson rendering / physics flags BEFORE importing the env.

    These match the headless-eval knobs documented in AGENTS.md and the
    persistent-skill-eval design doc.
    """
    global _GM_FLAGS_APPLIED
    if _GM_FLAGS_APPLIED:
        return

    from omnigibson.macros import gm

    with gm.unlocked():
        gm.HEADLESS = True
        if hasattr(gm, "GUI_VIEWPORT_ONLY"):
            gm.GUI_VIEWPORT_ONLY = True
        if hasattr(gm, "RENDER_VIEWER_CAMERA"):
            gm.RENDER_VIEWER_CAMERA = False
        if hasattr(gm, "ENABLE_FLATCACHE"):
            gm.ENABLE_FLATCACHE = True
        if hasattr(gm, "USE_GPU_DYNAMICS"):
            gm.USE_GPU_DYNAMICS = False

    _GM_FLAGS_APPLIED = True


class PersistentWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rank: int = int(args.worker_rank)
        self.gpu_id: int = int(args.gpu_id)
        self.out_dir: Path = Path(args.out_dir).resolve()

        self.jobs_path: Path = self.out_dir / "worker_jobs" / f"persistent_worker_{self.rank:03d}.jobs.jsonl"
        self.results_path: Path = self.out_dir / "worker_results" / f"persistent_worker_{self.rank:03d}.jsonl"
        self.status_path: Path = self.out_dir / "worker_status" / f"persistent_worker_{self.rank:03d}.jsonl"
        for path in (self.jobs_path.parent, self.results_path.parent, self.status_path.parent):
            path.mkdir(parents=True, exist_ok=True)

        self.heartbeat_s: float = float(
            os.environ.get("PERSISTENT_WORKER_HEARTBEAT_S", DEFAULT_HEARTBEAT_S)
        )
        self.max_segments_before_restart: int = int(
            os.environ.get(
                "PERSISTENT_WORKER_MAX_SEGMENTS_BEFORE_RESTART",
                DEFAULT_MAX_SEGMENTS_BEFORE_RESTART,
            )
        )
        self.task_reload_timeout_s: int = int(
            os.environ.get(
                "PERSISTENT_WORKER_TASK_RELOAD_TIMEOUT_S",
                DEFAULT_TASK_RELOAD_TIMEOUT_S,
            )
        )

        self.launcher_pid: Optional[int] = (
            int(args.launcher_pid) if args.launcher_pid and int(args.launcher_pid) > 0 else None
        )

        self._segments_since_boot = 0
        self._last_heartbeat = 0.0

        self._evaluator = None  # type: ignore[assignment]
        self._evaluator_ctx = None  # context manager (with ... as evaluator) entered manually
        self._loaded_task_name: Optional[str] = None
        self._server_proc: Optional[subprocess.Popen[str]] = None
        self._server_log: Optional[Path] = None
        self._server_port: Optional[int] = None
        self._server_identity: Optional[Dict[str, str]] = None

        self._done_keys: set[str] = set()
        if args.resume and self.results_path.exists():
            for row in launcher.load_jsonl_rows([self.results_path]):
                key = row.get("job_key")
                if key:
                    self._done_keys.add(key)

        self._original_argv = list(sys.argv)
        self._original_executable = sys.executable

    # ------------------------------------------------------------------ status

    def _emit_status(self, event: str, **fields: Any) -> None:
        row = {
            "ts": time.time(),
            "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rank": self.rank,
            "gpu_id": self.gpu_id,
            "pid": os.getpid(),
            "event": event,
            **fields,
        }
        try:
            launcher.append_jsonl(self.status_path, row)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to append status event %s: %s", event, exc)

    def _maybe_heartbeat(self, **fields: Any) -> None:
        now = time.time()
        if now - self._last_heartbeat < self.heartbeat_s:
            return
        self._last_heartbeat = now
        self._emit_status(
            "heartbeat",
            loaded_task=self._loaded_task_name,
            segments_since_boot=self._segments_since_boot,
            **fields,
        )

    # ----------------------------------------------------------- queue tailing

    def _read_jobs_jsonl(self) -> List[Dict[str, Any]]:
        if not self.jobs_path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with self.jobs_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed jobs line: %s (%s)", line[:200], exc)
        return rows

    def _drain_jobs(self, *, cursor: int) -> tuple[List[Dict[str, Any]], int]:
        rows = self._read_jobs_jsonl()
        if cursor >= len(rows):
            return [], cursor
        return rows[cursor:], len(rows)

    # -------------------------------------------------------- launcher watchdog

    def _launcher_alive(self) -> bool:
        if self.launcher_pid is None:
            return True
        try:
            os.kill(self.launcher_pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    # ------------------------------------------------------- policy server mgmt

    def _start_server(self, task_name: str) -> None:
        port = launcher.find_free_port(
            int(self.args.port_base),
            stride=int(self.args.gpus_per_node),
            max_tries=200,
        )
        identity = launcher.build_server_identity(
            out_dir=self.out_dir,
            worker_rank=self.rank,
            task_name=task_name,
            port=port,
        )
        proc, log_file = launcher.start_server(
            task_name=task_name,
            port=port,
            gpu_id=self.gpu_id,
            ckpt_dir=launcher.resolve_checkpoint_dir(Path(self.args.ckpt_dir)),
            config_name=self.args.config_name,
            policy_backend=self.args.policy_backend,
            server_run_id=identity["server_run_id"],
            server_token=identity["server_token"],
            openpi_env=self.args.openpi_env,
            behavior_dir=Path(self.args.behavior_dir),
            out_dir=self.out_dir / "server_logs",
        )
        self._server_proc = proc
        self._server_log = log_file
        self._server_port = port
        self._server_identity = identity
        self._emit_status(
            "server_started",
            task_name=task_name,
            port=port,
            server_run_id=identity["server_run_id"],
            log=str(log_file),
        )
        launcher.wait_for_server_proc(
            proc=proc,
            port=port,
            timeout_s=int(self.args.server_ready_timeout),
            log_file=log_file,
            expected_identity=identity,
        )
        self._emit_status("server_ready", task_name=task_name, port=port)

    def _stop_server(self) -> None:
        if self._server_proc is None:
            return
        try:
            launcher.stop_process(self._server_proc)
        finally:
            self._emit_status(
                "server_stopped",
                task_name=self._loaded_task_name,
                port=self._server_port,
            )
            if self._server_port is not None:
                launcher.wait_for_port_free(self._server_port)
            self._server_proc = None
            self._server_log = None
            self._server_port = None
            self._server_identity = None

    # ------------------------------------------------------------- task switch

    def _build_eval_cfg(self, task_name: str, sample: Dict[str, Any]) -> Any:
        """Compose the Hydra config the evaluator constructor expects."""
        from inspect import getsourcefile

        import hydra
        from omegaconf import OmegaConf

        from omnigibson.learning.eval_segment import (  # noqa: F401  (registers resolvers)
            register_omegaconf_resolvers,
        )

        register_omegaconf_resolvers()

        # Locate the Hydra config dir shipped with eval_segment.py.
        from omnigibson.learning import eval_segment as eval_segment_mod

        eval_segment_src = getsourcefile(eval_segment_mod) or eval_segment_mod.__file__
        config_dir = f"{Path(eval_segment_src).parents[0]}/configs"

        log_path = Path(str(sample["log_path"]))

        overrides = [
            "policy=websocket",
            f"task.name={task_name}",
            f"demo_data_path={self.args.demo_data_path}",
            f"rawdata_path={self.args.rawdata_path}",
            "segment_level=skill",
            f"segment_idx={int(sample['skill_idx'])}",
            "success_mode=segment_predicates",
            "grounding_topk=3",
            f"dry_run={'true' if self.args.dry_run else 'false'}",
            f"log_path={log_path}",
            f"demo_id={sample['demo_id']}",
            "headless=true",
            f"write_video={'true' if self.args.write_video else 'false'}",
            f"segment_max_steps={int(sample['dynamic_max_steps'])}",
            "model.host=127.0.0.1",
            f"model.port={int(self._server_port)}",
            f"model.expected_task_name={task_name}",
            f"model.expected_task_prompt_sha256={self._server_identity['task_prompt_sha256']}",
            f"model.expected_server_run_id={self._server_identity['server_run_id']}",
            f"model.expected_server_token={self._server_identity['server_token']}",
            "env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper",
            "partial_scene_load=true",
            "segment_predicate_window_mode=consecutive",
            "segment_predicate_min_consecutive=3",
            f"segment_predicate_dump_trace={'true' if self.args.segment_predicate_dump_trace else 'false'}",
        ]
        with hydra.initialize_config_dir(config_dir, version_base="1.1"):
            cfg = hydra.compose("eval_segment_config.yaml", overrides=overrides)
        OmegaConf.resolve(cfg)
        OmegaConf.set_struct(cfg, False)
        return cfg

    def _unload_evaluator(self) -> None:
        if self._evaluator is None and self._evaluator_ctx is None:
            return
        try:
            if self._evaluator_ctx is not None:
                self._evaluator_ctx.__exit__(None, None, None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Evaluator __exit__ raised: %s", exc)
        finally:
            self._emit_status("task_unloaded", task_name=self._loaded_task_name)
            self._evaluator = None
            self._evaluator_ctx = None
            self._loaded_task_name = None

    def _ensure_task_loaded(self, task_name: str, sample: Dict[str, Any]) -> None:
        if self._loaded_task_name == task_name and self._evaluator is not None:
            return

        if self._loaded_task_name is not None:
            self._emit_status("task_switching", from_task=self._loaded_task_name, to_task=task_name)
            # ``SubTaskEvaluator.__exit__`` tears down the global OmniGibson /
            # Isaac Sim application, which can terminate the worker process
            # before the next task is loaded.  Treat task changes as a clean
            # process restart instead; with ``--resume`` the restarted worker
            # skips already-written result rows and continues at the first job
            # for the new task.
            self._soft_restart(reason=f"task_switch:{self._loaded_task_name}->{task_name}")

        # Apply gm flags before importing the evaluator (no-op on subsequent calls).
        _apply_gm_flags()

        self._start_server(task_name)

        cfg = self._build_eval_cfg(task_name, sample)

        if os.environ.get("PERSISTENT_EVAL_DISABLE_SENSOR_RECONFIG", "0").lower() in {"1", "true", "yes"}:
            # Keep a lightweight pass-through wrapper instead of setting
            # env_wrapper=None. Some Hydra / OmegaConf paths still try to
            # instantiate a null wrapper and return None, which makes
            # Evaluator.self.env None during task load. The base
            # EnvironmentWrapper avoids RGBWrapper's render-product mutation
            # while preserving the expected wrapper interface.
            cfg.env_wrapper._target_ = "omnigibson.envs.env_wrapper.EnvironmentWrapper"

        # Import lazily so gm flags above are applied first.
        from omnigibson.learning.eval_subtask_reset import SubTaskEvaluator

        evaluator_ctx = SubTaskEvaluator(cfg)
        evaluator = evaluator_ctx.__enter__()
        self._evaluator_ctx = evaluator_ctx
        self._evaluator = evaluator
        self._loaded_task_name = task_name
        self._emit_status("task_loaded", task_name=task_name)

    # --------------------------------------------------------- per-segment run

    def _resume_hit_row(
        self,
        sample: Dict[str, Any],
        metrics_path: Path,
        segment_log: Path,
        dynamic_max_steps: int,
    ) -> Dict[str, Any]:
        row = launcher.load_metrics_row(metrics_path, sample, True, 0, segment_log)
        row["resume_hit"] = True
        row["dynamic_max_steps"] = dynamic_max_steps
        row["worker_rank"] = self.rank
        row["gpu_id"] = self.gpu_id
        row["worker_kind"] = "persistent"
        return row

    def _run_assignment(self, sample: Dict[str, Any]) -> None:
        task_name = str(sample["task_name"])
        demo_id = str(sample["demo_id"])
        skill_idx = int(sample["skill_idx"])
        dynamic_max_steps = int(
            sample.get(
                "dynamic_max_steps",
                launcher.get_dynamic_max_steps(
                    sample.get("frame_duration"),
                    fallback=int(self.args.max_steps),
                    cap=int(self.args.max_dynamic_steps_cap),
                ),
            )
        )
        sample = dict(sample)
        sample["dynamic_max_steps"] = dynamic_max_steps

        skill_out = self.out_dir / "raw" / task_name / f"demo_{demo_id}" / f"skill_{skill_idx:03d}"
        skill_out.mkdir(parents=True, exist_ok=True)
        segment_log = skill_out / "segment_eval.log"
        sample["log_path"] = str(skill_out)

        existing_metrics = sorted((skill_out / "metrics").glob("*.json"))
        if existing_metrics:
            row = self._resume_hit_row(sample, existing_metrics[0], segment_log, dynamic_max_steps)
            launcher.append_jsonl(self.results_path, row)
            self._emit_status(
                "segment_resume_hit",
                job_key=sample.get("job_key"),
                task_name=task_name,
                demo_id=demo_id,
                skill_idx=skill_idx,
            )
            return

        try:
            self._ensure_task_loaded(task_name, sample)
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            logger.exception(
                "Failed while loading task %s for demo=%s skill_idx=%s",
                task_name,
                demo_id,
                skill_idx,
            )
            self._emit_status(
                "task_load_error",
                job_key=sample.get("job_key"),
                task_name=task_name,
                demo_id=demo_id,
                skill_idx=skill_idx,
                traceback=tb,
            )
            raise

        from omnigibson.learning.eval_segment import run_segment_on_env

        t0 = time.time()
        runtime_ok = False
        returncode = 0
        metrics_path: Optional[Path] = None
        try:
            self._emit_status(
                "segment_start",
                job_key=sample.get("job_key"),
                task_name=task_name,
                demo_id=demo_id,
                skill_idx=skill_idx,
                dynamic_max_steps=dynamic_max_steps,
            )
            run_segment_on_env(self._evaluator, sample, write_metrics=True)
            metrics_glob = sorted((skill_out / "metrics").glob("*.json"))
            if metrics_glob:
                metrics_path = metrics_glob[0]
                runtime_ok = True
            else:
                returncode = 1
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            returncode = 99
            self._emit_status(
                "segment_error",
                job_key=sample.get("job_key"),
                task_name=task_name,
                demo_id=demo_id,
                skill_idx=skill_idx,
                traceback=tb,
            )
            # Re-raise so the outer loop triggers a soft-restart with a clean Isaac.
            raise
        finally:
            elapsed = time.time() - t0

            if metrics_path is not None:
                row = launcher.load_metrics_row(metrics_path, sample, runtime_ok, returncode, segment_log)
                row["dynamic_max_steps"] = dynamic_max_steps
            else:
                row = {
                    "job_key": sample.get("job_key"),
                    "skill": sample.get("skill"),
                    "task_name": task_name,
                    "demo_id": demo_id,
                    "instance_id": sample.get("instance_id"),
                    "skill_idx": skill_idx,
                    "frame_duration": sample.get("frame_duration"),
                    "dynamic_max_steps": dynamic_max_steps,
                    "runtime_ok": runtime_ok,
                    "returncode": returncode,
                    "metrics_path": None,
                    "segment_log": str(segment_log),
                }
            row["worker_rank"] = self.rank
            row["gpu_id"] = self.gpu_id
            row["worker_kind"] = "persistent"
            row["segment_runtime_s"] = elapsed
            launcher.append_jsonl(self.results_path, row)
            self._segments_since_boot += 1
            self._emit_status(
                "segment_done",
                job_key=sample.get("job_key"),
                task_name=task_name,
                demo_id=demo_id,
                skill_idx=skill_idx,
                runtime_ok=runtime_ok,
                returncode=returncode,
                success=row.get("success"),
                result_type=row.get("result_type"),
                segment_runtime_s=elapsed,
                segments_since_boot=self._segments_since_boot,
            )

    # ----------------------------------------------------------------- restart

    def _soft_restart(self, reason: str) -> None:
        self._emit_status("restart", reason=reason, segments_since_boot=self._segments_since_boot)
        # Do not call ``_unload_evaluator`` here.  Exiting the evaluator shuts
        # down the singleton SimulationApp and can end the interpreter before
        # ``execv`` runs, causing the launcher to believe the worker completed
        # successfully while leaving the remaining queue unprocessed.  ``execv``
        # replaces this process image, so the simulator state is discarded by
        # the OS without relying on OmniGibson's Python-level shutdown path.
        try:
            self._stop_server()
        except Exception:  # noqa: BLE001
            pass
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        _reexec_in_behavior_env(self.args, self._original_argv, force=True)

    # ----------------------------------------------------------------- main loop

    def run(self) -> int:
        self._emit_status(
            "started",
            jobs_path=str(self.jobs_path),
            results_path=str(self.results_path),
            launcher_pid=self.launcher_pid,
            max_segments_before_restart=self.max_segments_before_restart,
        )

        cursor = 0
        # Skip past any already-processed assign lines tracked in done_keys.
        # We still iterate over the queue to discover shutdown / new work.
        idle_poll_s = 1.0
        while True:
            if not self._launcher_alive():
                self._emit_status("launcher_gone", launcher_pid=self.launcher_pid)
                break

            new_rows, cursor = self._drain_jobs(cursor=cursor)
            self._maybe_heartbeat(state="idle" if not new_rows else "busy")

            if not new_rows:
                time.sleep(idle_poll_s)
                continue

            for entry in new_rows:
                action = str(entry.get("action", "")).lower()
                if action == "shutdown":
                    self._emit_status("shutdown_requested")
                    self._unload_evaluator()
                    self._stop_server()
                    self._emit_status("heartbeat", state="shutdown")
                    return 0

                if action != "assign":
                    self._emit_status("unknown_action", action=action)
                    continue

                sample = entry.get("sample") or {}
                if not sample:
                    self._emit_status("empty_sample")
                    continue

                job_key = sample.get("job_key")
                if job_key and job_key in self._done_keys:
                    self._emit_status("segment_skipped_done", job_key=job_key)
                    continue

                try:
                    self._run_assignment(sample)
                except Exception:  # noqa: BLE001 -- task_load_error / segment_error already logged above
                    logger.exception("Unhandled assignment failure for job_key=%s", job_key)
                    self._emit_status(
                        "assignment_exception",
                        job_key=job_key,
                        task_name=sample.get("task_name"),
                        demo_id=sample.get("demo_id"),
                        skill_idx=sample.get("skill_idx"),
                        traceback=traceback.format_exc(),
                    )
                    self._soft_restart(reason="segment_exception")

                if job_key:
                    self._done_keys.add(job_key)

                if self._segments_since_boot >= self.max_segments_before_restart:
                    self._soft_restart(reason="max_segments_cap")

        self._unload_evaluator()
        self._stop_server()
        return 0


# --------------------------------------------------------------------- argparse


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent per-GPU skill-eval worker")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--worker-rank", required=True, type=int)
    parser.add_argument("--gpu-id", required=True, type=int)
    parser.add_argument("--port-base", required=True, type=int)
    parser.add_argument("--gpus-per-node", default=int(os.environ.get("GPUS_PER_NODE", "8")), type=int)
    parser.add_argument("--max-steps", default=120, type=int)
    parser.add_argument("--max-dynamic-steps-cap", default=0, type=int)
    parser.add_argument("--server-ready-timeout", default=1800, type=int)
    parser.add_argument("--openpi-env", default="openpi-comet-nas")
    parser.add_argument("--behavior-env", default="behavior")
    parser.add_argument("--config-name", default=launcher.DEFAULT_CONFIG)
    parser.add_argument("--policy-backend", choices=["auto", "torch", "jax"], default="auto")
    parser.add_argument("--ckpt-dir", required=True, type=Path)
    parser.add_argument("--behavior-dir", default=launcher.BEHAVIOR_DIR_DEFAULT, type=Path)
    parser.add_argument("--demo-data-path", default=launcher.DEMO_DATA_PATH_DEFAULT, type=Path)
    parser.add_argument("--rawdata-path", default=launcher.RAWDATA_PATH_DEFAULT, type=Path)
    parser.add_argument("--launcher-pid", default=0, type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-video", action="store_true")
    parser.add_argument("--segment-predicate-dump-trace", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.worker_rank)
    _reexec_in_behavior_env(args, list(sys.argv), force=False)
    _ensure_isaac_env()

    # Ignore SIGPIPE so a closed launcher pipe never crashes us mid-segment.
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    worker = PersistentWorker(args)
    try:
        return worker.run()
    except KeyboardInterrupt:
        worker._emit_status("interrupted")
        worker._unload_evaluator()
        worker._stop_server()
        return 130


if __name__ == "__main__":
    sys.exit(main())
