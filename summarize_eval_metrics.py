#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def summarize(run_dir: Path) -> dict[str, Any]:
    metric_files = sorted(run_dir.glob("**/metrics/*.json"))

    per_task: dict[str, list[float]] = {}
    for fp in metric_files:
        # Layout:
        #   <run_dir>/<task>/eval_gpuX_pYYYY/metrics/<task>_<instance_id>_<episode>.json
        try:
            rel = fp.relative_to(run_dir)
            task = rel.parts[0]
        except Exception:
            # Fallback: best-effort parsing from filename
            task = fp.stem.split("_")[0]

        data = _read_json(fp)
        q = float(data.get("q_score", {}).get("final", 0.0))
        per_task.setdefault(task, []).append(q)

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "num_metric_files": len(metric_files),
        "tasks": {},
    }

    all_q: list[float] = []
    for task, qs in sorted(per_task.items()):
        qs_sorted = sorted(qs)
        n = len(qs_sorted)
        succ = sum(1 for q in qs_sorted if q > 0)
        avg = (sum(qs_sorted) / n) if n else 0.0
        all_q.extend(qs_sorted)
        summary["tasks"][task] = {
            "n": n,
            "success_count_q_gt_0": succ,
            "avg_q": avg,
            "min_q": qs_sorted[0] if n else None,
            "max_q": qs_sorted[-1] if n else None,
        }

    if all_q:
        summary["overall"] = {
            "n": len(all_q),
            "success_count_q_gt_0": sum(1 for q in all_q if q > 0),
            "avg_q": sum(all_q) / len(all_q),
            "min_q": min(all_q),
            "max_q": max(all_q),
        }
    else:
        summary["overall"] = {
            "n": 0,
            "success_count_q_gt_0": 0,
            "avg_q": 0.0,
            "min_q": None,
            "max_q": None,
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize BEHAVIOR evaluation results by scanning "
            "<run_dir>/**/metrics/*.json and aggregating q_score.final"
        )
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="eval_logs/pi05-b1kpt50-cs32/easy8_full_afterfix_20260119_023021",
        help="Run directory produced by scripts/run_b1k_eval_parallel_tasks.sh",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write full summary JSON.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    summary = summarize(run_dir=run_dir)

    print(f"Found {summary['num_metric_files']} metrics json files under {run_dir}")
    for task, s in summary["tasks"].items():
        n = s["n"]
        if n == 0:
            continue
        print(
            f"{task:24s} n={n:2d}  "
            f"success(q>0)={s['success_count_q_gt_0']:2d}  "
            f"avg_q={s['avg_q']:.3f}  "
            f"min={s['min_q']:.3f}  max={s['max_q']:.3f}"
        )

    overall = summary["overall"]
    if overall["n"] > 0:
        print("-" * 80)
        print(
            f"OVERALL: n={overall['n']}  "
            f"success(q>0)={overall['success_count_q_gt_0']}  "
            f"avg_q={overall['avg_q']:.3f}"
        )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Wrote JSON summary to {out_path}")


if __name__ == "__main__":
    main()