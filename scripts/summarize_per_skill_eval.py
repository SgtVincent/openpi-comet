"""Summarize per-skill evaluation outputs.

This script is intentionally lightweight (stdlib-only). It consumes the merged outputs
produced by `scripts/run_skill_metric_multinode_sweep.py --mode merge`, typically:

- `multinode_skill_summary.csv`
- `multinode_skill_results.csv`
- (optional) `multinode_skill_task_summary.csv`

and prints a human-readable summary (Markdown by default).

Design goals (Spec Coding style):
- Deterministic, auditable output: every number comes from CSV rows.
- Works for both "1 sample per skill" and "N samples per skill" runs.
- Makes it easy to answer: "which skills are good" vs "which are bad", with evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SkillSummaryRow:
    skill: str
    segment_count: int
    success_count: int
    success_rate: float
    runtime_pass_count: int
    runtime_pass_rate: float
    timeout_count: int | None
    predicate_satisfied_count: int | None


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_int(v: str | None, default: int | None = None) -> int | None:
    if v is None or v == "":
        return default
    return int(v)


def _to_float(v: str | None, default: float | None = None) -> float | None:
    if v is None or v == "":
        return default
    return float(v)


def load_skill_summary(path: Path) -> list[SkillSummaryRow]:
    rows = _read_csv_dicts(path)
    out: list[SkillSummaryRow] = []
    for r in rows:
        out.append(
            SkillSummaryRow(
                skill=(r.get("skill") or "").strip(),
                segment_count=int(r["segment_count"]),
                success_count=int(r["success_count"]),
                success_rate=float(r["success_rate"]),
                runtime_pass_count=int(r["runtime_pass_count"]),
                runtime_pass_rate=float(r["runtime_pass_rate"]),
                timeout_count=_to_int(r.get("timeout_count"), default=None),
                predicate_satisfied_count=_to_int(r.get("predicate_satisfied_count"), default=None),
            )
        )
    # Stable ordering for deterministic output
    out.sort(key=lambda x: x.skill)
    return out


def load_result_rows(path: Path) -> list[dict[str, str]]:
    return _read_csv_dicts(path)


def _group_result_types(results: list[dict[str, str]]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = {}
    for r in results:
        skill = (r.get("skill") or "").strip()
        rt = (r.get("result_type") or "").strip() or "<empty>"
        counts.setdefault(skill, Counter())[rt] += 1
    return counts


def _pick_examples(results: list[dict[str, str]], max_per_skill: int) -> dict[str, list[dict[str, str]]]:
    by_skill: dict[str, list[dict[str, str]]] = {}
    for r in results:
        skill = (r.get("skill") or "").strip()
        arr = by_skill.setdefault(skill, [])
        if len(arr) >= max_per_skill:
            continue
        arr.append(r)
    return by_skill


def render_markdown(
    run_dir: Path,
    summary_rows: list[SkillSummaryRow],
    result_type_counts: dict[str, Counter[str]],
    examples: dict[str, list[dict[str, str]]],
    good_threshold: float,
    top_k: int,
) -> str:
    total_segments = sum(r.segment_count for r in summary_rows)
    total_success = sum(r.success_count for r in summary_rows)
    unique_skills = len(summary_rows)

    good = [r for r in summary_rows if r.success_rate >= good_threshold]
    bad = [r for r in summary_rows if r.success_rate < good_threshold]

    # Sort: best first for good; worst first for bad.
    good.sort(key=lambda r: (r.success_rate, r.success_count, -r.segment_count, r.skill), reverse=True)
    bad.sort(key=lambda r: (r.success_rate, r.success_count, r.segment_count, r.skill))

    lines: list[str] = []
    lines.append(f"# Per-skill Eval Summary")
    lines.append("")
    lines.append(f"- run_dir: `{run_dir}`")
    lines.append(f"- skills: `{unique_skills}`")
    lines.append(f"- segments: `{total_segments}`")
    lines.append(f"- success: `{total_success}/{total_segments}`")
    lines.append(f"- good_threshold(success_rate): `{good_threshold}`")
    lines.append("")

    def _rt_summary(skill: str) -> str:
        c = result_type_counts.get(skill)
        if not c:
            return ""
        top = ", ".join([f"{k}:{v}" for k, v in c.most_common(3)])
        return top

    def _example_str(skill: str) -> str:
        ex = examples.get(skill) or []
        if not ex:
            return ""
        parts: list[str] = []
        for r in ex:
            parts.append(
                f"{r.get('task_name','')}/{r.get('demo_id','')}#skill_{str(r.get('skill_idx','')).zfill(3)}"
            )
        return "; ".join(parts)

    # Good table
    lines.append("## Good Skills")
    lines.append("")
    lines.append("| skill | success_rate | success/segments | top result_type | example segment |")
    lines.append("| --- | --- | --- | --- | --- |")
    for r in good[:top_k]:
        rt = _rt_summary(r.skill)
        ex = _example_str(r.skill)
        lines.append(
            f"| {r.skill} | {r.success_rate:.3f} | {r.success_count}/{r.segment_count} | {rt} | {ex} |"
        )
    if len(good) > top_k:
        lines.append("")
        lines.append(f"(showing top {top_k} / {len(good)} good skills)")
    lines.append("")

    # Bad table
    lines.append("## Bad Skills")
    lines.append("")
    lines.append("| skill | success_rate | success/segments | top result_type | example segment |")
    lines.append("| --- | --- | --- | --- | --- |")
    for r in bad[:top_k]:
        rt = _rt_summary(r.skill)
        ex = _example_str(r.skill)
        lines.append(
            f"| {r.skill} | {r.success_rate:.3f} | {r.success_count}/{r.segment_count} | {rt} | {ex} |"
        )
    if len(bad) > top_k:
        lines.append("")
        lines.append(f"(showing top {top_k} / {len(bad)} bad skills)")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- This summary is based on `multinode_skill_summary.csv` + `multinode_skill_results.csv` only; it does not infer anything outside these outputs."
    )
    lines.append(
        "- If `segment_count` is 1 per skill, success_rate is binary and results are highly sensitive to which representative segment was sampled."
    )
    return "\n".join(lines) + "\n"


def render_json(
    run_dir: Path,
    summary_rows: list[SkillSummaryRow],
    result_type_counts: dict[str, Counter[str]],
    good_threshold: float,
) -> dict[str, Any]:
    total_segments = sum(r.segment_count for r in summary_rows)
    total_success = sum(r.success_count for r in summary_rows)
    good = [r for r in summary_rows if r.success_rate >= good_threshold]
    bad = [r for r in summary_rows if r.success_rate < good_threshold]

    def _row(r: SkillSummaryRow) -> dict[str, Any]:
        return {
            "skill": r.skill,
            "segment_count": r.segment_count,
            "success_count": r.success_count,
            "success_rate": r.success_rate,
            "runtime_pass_count": r.runtime_pass_count,
            "runtime_pass_rate": r.runtime_pass_rate,
            "result_type_counts": dict(result_type_counts.get(r.skill, Counter())),
        }

    return {
        "run_dir": str(run_dir),
        "skills": len(summary_rows),
        "segments": total_segments,
        "success": {"count": total_success, "total": total_segments, "rate": (total_success / total_segments) if total_segments else 0.0},
        "good_threshold": good_threshold,
        "good_skills": [_row(r) for r in sorted(good, key=lambda x: (-x.success_rate, x.skill))],
        "bad_skills": [_row(r) for r in sorted(bad, key=lambda x: (x.success_rate, x.skill))],
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Summarize per-skill eval outputs (merged run dir).")
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing multinode_skill_summary.csv and multinode_skill_results.csv",
    )
    p.add_argument(
        "--good-threshold",
        type=float,
        default=0.5,
        help="Classify skills with success_rate >= threshold as good (default: 0.5)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="How many skills to show in each list (default: 50)",
    )
    p.add_argument(
        "--format",
        choices=["md", "json"],
        default="md",
        help="Output format (default: md)",
    )
    p.add_argument(
        "--examples-per-skill",
        type=int,
        default=1,
        help="How many example segments to print per skill (default: 1)",
    )
    args = p.parse_args()

    run_dir: Path = args.run_dir
    summary_csv = run_dir / "multinode_skill_summary.csv"
    results_csv = run_dir / "multinode_skill_results.csv"
    if not summary_csv.exists():
        raise SystemExit(f"Missing required file: {summary_csv}")
    if not results_csv.exists():
        raise SystemExit(f"Missing required file: {results_csv}")

    summary_rows = load_skill_summary(summary_csv)
    results = load_result_rows(results_csv)
    rt_counts = _group_result_types(results)
    examples = _pick_examples(results, max_per_skill=max(0, args.examples_per_skill))

    if args.format == "json":
        payload = render_json(run_dir, summary_rows, rt_counts, good_threshold=args.good_threshold)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(
        render_markdown(
            run_dir,
            summary_rows,
            rt_counts,
            examples,
            good_threshold=args.good_threshold,
            top_k=args.top_k,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

