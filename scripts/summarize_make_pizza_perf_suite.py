#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ELAPSED_RE = re.compile(r"\[(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--select-phase", type=str, default="")
    parser.add_argument("--select-top", type=int, default=0)
    parser.add_argument("--ranking-metric", type=str, default="")
    parser.add_argument("--print-groups", action="store_true")
    return parser.parse_args()


def parse_elapsed_s(line: str) -> float | None:
    line = ANSI_RE.sub("", line)
    match = ELAPSED_RE.search(line)
    if not match:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(value: str | None) -> int | None:
    parsed = parse_float(value)
    return int(parsed) if parsed is not None else None


def summarize_timing(run_dir: Path) -> dict[str, float | int | None]:
    timing_files = sorted(run_dir.glob("**/timing.csv"))
    if not timing_files:
        return {
            "timing_rows": 0,
            "env_step_ms_mean": None,
            "env_step_ms_p50": None,
            "env_step_ms_p95": None,
            "policy_ms_mean": None,
            "rtt_ms_mean": None,
        }

    env_vals: list[float] = []
    policy_vals: list[float] = []
    rtt_vals: list[float] = []
    total_rows = 0

    for timing_path in timing_files:
        with timing_path.open(newline="") as f:
            reader = csv.DictReader(f)
            per_key_counts: dict[tuple[str, str], int] = defaultdict(int)
            buffered_rows: list[dict[str, str]] = list(reader)
        total_rows += len(buffered_rows)

        for row in buffered_rows:
            key = (row.get("episode", ""), row.get("instance_id", ""))
            per_key_counts[key] += 1
            if per_key_counts[key] <= 20:
                continue
            env = parse_float(row.get("env_step_ms"))
            policy = parse_float(row.get("policy_ms"))
            rtt = parse_float(row.get("rtt_ms"))
            if env is not None:
                env_vals.append(env)
            if policy is not None:
                policy_vals.append(policy)
            if rtt is not None:
                rtt_vals.append(rtt)

    return {
        "timing_rows": total_rows,
        "env_step_ms_mean": mean(env_vals) if env_vals else None,
        "env_step_ms_p50": percentile(env_vals, 0.50),
        "env_step_ms_p95": percentile(env_vals, 0.95),
        "policy_ms_mean": mean(policy_vals) if policy_vals else None,
        "rtt_ms_mean": mean(rtt_vals) if rtt_vals else None,
    }


def summarize_logs(run_dir: Path) -> dict[str, float | int | str | None]:
    eval_logs = sorted(run_dir.glob("eval_gpu*_p*.log"))
    if not eval_logs:
        return {
            "init_to_start_eval_s": None,
            "max_eval_log_elapsed_s": None,
            "finished_instances": 0,
            "log_status": "missing_eval_logs",
        }

    start_eval_times: list[float] = []
    max_elapsed_s: float | None = None
    finished_instances = 0

    for log_path in eval_logs:
        for raw_line in log_path.read_text(errors="ignore").splitlines():
            line = ANSI_RE.sub("", raw_line)
            elapsed_s = parse_elapsed_s(line)
            if elapsed_s is not None:
                max_elapsed_s = max(elapsed_s, max_elapsed_s or elapsed_s)
            if "Starting evaluation..." in line and elapsed_s is not None:
                start_eval_times.append(elapsed_s)
            if "Evaluation finished at step" in line:
                finished_instances += 1

    return {
        "init_to_start_eval_s": min(start_eval_times) if start_eval_times else None,
        "max_eval_log_elapsed_s": max_elapsed_s,
        "finished_instances": finished_instances,
        "log_status": "ok",
    }


def summarize_metrics(run_dir: Path) -> dict[str, float | int | None]:
    metric_files = sorted(run_dir.glob("**/metrics/*.json"))
    q_scores: list[float] = []
    agent_base: list[float] = []

    for metric_path in metric_files:
        payload = json.loads(metric_path.read_text())
        q = parse_float(payload.get("q_score", {}).get("final"))
        base = parse_float(payload.get("agent_distance", {}).get("base"))
        if q is not None:
            q_scores.append(q)
        if base is not None:
            agent_base.append(base)

    return {
        "metrics_count": len(metric_files),
        "success_count_q_gt_0": sum(1 for q in q_scores if q > 0),
        "avg_q_score": mean(q_scores) if q_scores else None,
        "avg_agent_distance_base": mean(agent_base) if agent_base else None,
    }


def build_status(row: dict[str, Any]) -> str:
    if row.get("status") == "skipped":
        return "skipped"
    exit_code = parse_int(row.get("exit_code"))
    # Some runs finish and write metrics, but still end up with an exit code 141
    # (commonly SIGPIPE / wrapper termination). Treat this as OK if the evaluator
    # reached the normal "finished" marker so perf stats remain usable.
    if exit_code not in (0, None):
        finished_instances = parse_int(row.get("finished_instances")) or 0
        if exit_code == 141 and finished_instances > 0:
            exit_code = 0
        else:
            return "failed"
    if parse_int(row.get("metrics_count")) in (None, 0):
        return "no_metrics"
    return "ok"


def add_phase_deltas(rows: list[dict[str, Any]]) -> None:
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_phase[row["phase"]].append(row)

    for phase_rows in by_phase.values():
        baseline = next((r for r in phase_rows if "baseline_current" in r["group"]), None)
        if baseline is None:
            continue
        for key in ("wall_clock_s", "env_step_ms_mean", "init_to_start_eval_s"):
            baseline_value = parse_float(baseline.get(key))
            if baseline_value is None or baseline_value <= 0:
                continue
            for row in phase_rows:
                cur = parse_float(row.get(key))
                if cur is None:
                    continue
                row[f"{key}_improve_pct"] = (baseline_value - cur) / baseline_value * 100.0


def read_suite(suite_dir: Path) -> list[dict[str, Any]]:
    manifest_path = suite_dir / "experiments_manifest.tsv"
    rows = load_manifest(manifest_path)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        timing = summarize_timing(run_dir) if run_dir.exists() else {}
        logs = summarize_logs(run_dir) if run_dir.exists() else {}
        metrics = summarize_metrics(run_dir) if run_dir.exists() else {}
        launch_epoch_s = parse_float(row.get("launch_epoch_s"))
        end_epoch_s = parse_float(row.get("end_epoch_s"))
        wall_clock_s = None
        if launch_epoch_s is not None and end_epoch_s is not None and end_epoch_s >= launch_epoch_s:
            wall_clock_s = end_epoch_s - launch_epoch_s
        merged: dict[str, Any] = dict(row)
        merged.update(timing)
        merged.update(logs)
        merged.update(metrics)
        merged["wall_clock_s"] = wall_clock_s
        merged["computed_status"] = build_status(merged)
        enriched.append(merged)
    add_phase_deltas(enriched)
    return enriched


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "phase",
        "group",
        "computed_status",
        "run_dir",
        "exit_code",
        "wall_clock_s",
        "wall_clock_s_improve_pct",
        "init_to_start_eval_s",
        "init_to_start_eval_s_improve_pct",
        "timing_rows",
        "env_step_ms_mean",
        "env_step_ms_p50",
        "env_step_ms_p95",
        "env_step_ms_mean_improve_pct",
        "policy_ms_mean",
        "rtt_ms_mean",
        "finished_instances",
        "metrics_count",
        "success_count_q_gt_0",
        "avg_q_score",
        "avg_agent_distance_base",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    parsed = parse_float(str(value))
    if parsed is not None and "." in str(value):
        return f"{parsed:.{digits}f}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# make_pizza perf suite summary",
        "",
        "| phase | group | status | wall_clock_s | wall_improve_% | init_s | env_mean_ms | env_improve_% | metrics | avg_q | avg_agent_base |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {phase} | {group} | {computed_status} | {wall} | {wall_imp} | {init_s} | {env} | {env_imp} | {metrics} | {avg_q} | {agent} |".format(
                phase=row["phase"],
                group=row["group"],
                computed_status=row["computed_status"],
                wall=_fmt(parse_float(row.get("wall_clock_s"))),
                wall_imp=_fmt(parse_float(row.get("wall_clock_s_improve_pct"))),
                init_s=_fmt(parse_float(row.get("init_to_start_eval_s"))),
                env=_fmt(parse_float(row.get("env_step_ms_mean"))),
                env_imp=_fmt(parse_float(row.get("env_step_ms_mean_improve_pct"))),
                metrics=_fmt(parse_int(row.get("metrics_count"))),
                avg_q=_fmt(parse_float(row.get("avg_q_score"))),
                agent=_fmt(parse_float(row.get("avg_agent_distance_base"))),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _baseline_for_phase(rows: list[dict[str, Any]], phase: str) -> dict[str, Any] | None:
    return next((row for row in rows if row["phase"] == phase and "baseline_current" in row["group"]), None)


def select_rows(rows: list[dict[str, Any]], phase: str, top_k: int, ranking_metric: str) -> list[dict[str, Any]]:
    filtered = [row for row in rows if row["phase"] == phase and row["computed_status"] == "ok"]
    if phase == "phase_a_smoke_filter" and not ranking_metric:
        baseline = _baseline_for_phase(rows, phase)
        baseline_wall = parse_float(baseline.get("wall_clock_s")) if baseline else None
        baseline_env = parse_float(baseline.get("env_step_ms_mean")) if baseline else None

        gated: list[dict[str, Any]] = []
        for row in filtered:
            wall = parse_float(row.get("wall_clock_s"))
            env = parse_float(row.get("env_step_ms_mean"))
            if wall is None or env is None:
                continue
            if baseline_wall is not None and wall > baseline_wall * 1.02:
                continue
            if baseline_env is not None and env > baseline_env * 1.02:
                continue
            gated.append(row)

        filtered = gated
        filtered.sort(
            key=lambda row: (
                parse_float(row.get("wall_clock_s")) or float("inf"),
                parse_float(row.get("env_step_ms_mean")) or float("inf"),
            )
        )
        return filtered[:top_k]
    elif not ranking_metric:
        ranking_metric = "wall_clock_s"
    filtered = [row for row in filtered if parse_float(row.get(ranking_metric)) is not None]
    filtered.sort(key=lambda row: parse_float(row.get(ranking_metric)) or float("inf"))
    return filtered[:top_k]


def main() -> None:
    args = parse_args()
    rows = read_suite(args.suite_dir)

    csv_out = args.csv_out or (args.suite_dir / "suite_summary.csv")
    md_out = args.md_out or (args.suite_dir / "suite_summary.md")
    write_csv(rows, csv_out)
    write_markdown(rows, md_out)

    if args.select_phase and args.select_top > 0:
        selected = select_rows(rows, args.select_phase, args.select_top, args.ranking_metric)
        if args.print_groups:
            for row in selected:
                print(row["group"])
        else:
            print(json.dumps(selected, indent=2))
        return

    print(f"Wrote CSV summary to {csv_out}")
    print(f"Wrote Markdown summary to {md_out}")


if __name__ == "__main__":
    main()
