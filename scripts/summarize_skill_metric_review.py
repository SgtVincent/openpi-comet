#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def normalize_label(text: str) -> str:
    return str(text or "").strip().lower()


def load_manifests(paths: Iterable[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        rows.extend(read_manifest(path))
    return rows


def bool_from_text(text: str) -> bool:
    return normalize_label(text) in {"1", "true", "yes"}


def review_outcome(row: Dict[str, str]) -> Tuple[str, str]:
    pred = "pass" if bool_from_text(row.get("success", "")) else "fail"
    human = normalize_label(row.get("human_judgement", ""))
    if human not in {"pass", "fail"}:
        return pred, "unreviewed"
    if pred == human:
        return pred, "agree"
    if pred == "pass" and human == "fail":
        return pred, "false_positive"
    return pred, "false_negative"


def safe_rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def render_report(summary: Dict[str, Any], by_skill: List[Dict[str, Any]], by_issue: List[Dict[str, Any]]) -> str:
    lines = ["# Skill Metric Review Report", ""]
    lines.append(f"- manifests: `{summary['manifest_count']}`")
    lines.append(f"- total rows: `{summary['total_rows']}`")
    lines.append(f"- reviewed rows: `{summary['reviewed_rows']}`")
    lines.append(f"- agreement rate: `{summary['agreement_rate']:.4f}`")
    lines.append(f"- false_positive: `{summary['false_positive']}`")
    lines.append(f"- false_negative: `{summary['false_negative']}`")
    lines.append(f"- uncertain_or_unreviewed: `{summary['uncertain_or_unreviewed']}`")
    lines.append("")
    lines.append("| Skill | Reviewed | Agreement | False Positive | False Negative |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in by_skill:
        lines.append(
            f"| {row['skill']} | {row['reviewed_rows']} | {row['agreement_rate']:.4f} | "
            f"{row['false_positive']} | {row['false_negative']} |"
        )
    lines.append("")
    lines.append("| Issue Bucket | Count |")
    lines.append("|---|---:|")
    for row in by_issue:
        lines.append(f"| {row['issue_bucket']} | {row['count']} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize manual review results for skill metric validation.")
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Path to review_manifest.csv; repeatable",
    )
    args = parser.parse_args()

    manifest_paths = [Path(path).resolve() for path in args.manifest]
    rows = load_manifests(manifest_paths)
    if not rows:
        raise RuntimeError("No review rows loaded from manifests.")

    reviewed_rows = 0
    agreement = 0
    false_positive = 0
    false_negative = 0
    uncertain_or_unreviewed = 0
    by_skill_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    issue_buckets: Counter[str] = Counter()
    detailed_rows: List[Dict[str, Any]] = []

    for row in rows:
        predicted, outcome = review_outcome(row)
        human = normalize_label(row.get("human_judgement", ""))
        issue_bucket = normalize_label(row.get("issue_bucket", "")) or "unspecified"
        if human in {"pass", "fail"}:
            reviewed_rows += 1
            agreement += int(outcome == "agree")
            false_positive += int(outcome == "false_positive")
            false_negative += int(outcome == "false_negative")
            by_skill_counts[row["skill"]]["reviewed_rows"] += 1
            by_skill_counts[row["skill"]][outcome] += 1
            issue_buckets[issue_bucket] += int(outcome != "agree")
        else:
            uncertain_or_unreviewed += 1

        detailed_rows.append(
            {
                **row,
                "predicted_label": predicted,
                "review_outcome": outcome,
            }
        )

    by_skill = []
    for skill, counts in sorted(by_skill_counts.items()):
        by_skill.append(
            {
                "skill": skill,
                "reviewed_rows": counts["reviewed_rows"],
                "agree_count": counts["agree"],
                "agreement_rate": safe_rate(counts["agree"], counts["reviewed_rows"]),
                "false_positive": counts["false_positive"],
                "false_negative": counts["false_negative"],
            }
        )

    by_issue = [
        {"issue_bucket": bucket, "count": count}
        for bucket, count in issue_buckets.most_common()
    ]

    summary = {
        "manifest_count": len(manifest_paths),
        "total_rows": len(rows),
        "reviewed_rows": reviewed_rows,
        "agreement_count": agreement,
        "agreement_rate": safe_rate(agreement, reviewed_rows),
        "false_positive": false_positive,
        "false_negative": false_negative,
        "uncertain_or_unreviewed": uncertain_or_unreviewed,
    }

    output_dir = manifest_paths[0].parent
    write_csv(output_dir / "review_summary.csv", [summary], list(summary.keys()))
    write_csv(
        output_dir / "review_by_skill.csv",
        by_skill,
        ["skill", "reviewed_rows", "agree_count", "agreement_rate", "false_positive", "false_negative"],
    )
    write_csv(output_dir / "review_by_issue_bucket.csv", by_issue, ["issue_bucket", "count"])
    write_csv(
        output_dir / "review_detailed.csv",
        detailed_rows,
        list(detailed_rows[0].keys()),
    )
    with (output_dir / "review_report.md").open("w") as f:
        f.write(render_report(summary, by_skill, by_issue))
    with (output_dir / "review_summary.json").open("w") as f:
        json.dump({"summary": summary, "by_skill": by_skill, "by_issue_bucket": by_issue}, f, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
