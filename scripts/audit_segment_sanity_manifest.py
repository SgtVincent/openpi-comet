#!/usr/bin/env python3

"""Sample and audit per-segment sanity manifests.

Modes:
- sample: create `audit_manifest.csv` from `sanity_manifest.jsonl`
- summarize: compare `vision_judgement` vs `metric_success` and write `audit_summary.md`

This script does NOT require Isaac/OmniGibson runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def normalize_judgement(x: Any) -> str:
    s = str(x or "").strip().lower()
    if s in ("", "na", "n/a"):
        return ""
    if s in ("success", "pass", "true", "1", "yes", "y", "成功"):
        return "success"
    if s in ("fail", "failure", "false", "0", "no", "n", "失败"):
        return "fail"
    if s in ("unclear", "unknown", "maybe", "不确定", "无法判断"):
        return "unclear"
    return s


def parse_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("", "none", "null"):
        return None
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    # tolerate already-boolean JSON
    if isinstance(x, bool):
        return x
    return None


def bucket_key(row: Dict[str, Any]) -> str:
    return f"{row.get('result_type')}|{int(bool(row.get('metric_success')))}"


def stratified_select(rows: List[Dict[str, Any]], sample_limit: int, holdout_limit: int, seed: int) -> List[Dict[str, Any]]:
    if sample_limit <= 0 and holdout_limit <= 0:
        for r in rows:
            r["review_split"] = "discovery"
        return rows

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[bucket_key(r)].append(r)
    for b in buckets.values():
        b.sort(key=lambda item: (str(item.get("task_name")), str(item.get("demo_id")), int(item.get("segment_idx") or 0)))

    rng = random.Random(seed)
    ordered = sorted(buckets)
    selected: List[Dict[str, Any]] = []
    while len(selected) < max(sample_limit, 0):
        progressed = False
        for b in ordered:
            if not buckets[b]:
                continue
            pick = 0 if len(buckets[b]) == 1 else rng.randrange(len(buckets[b]))
            selected.append(buckets[b].pop(pick))
            progressed = True
            if len(selected) >= sample_limit:
                break
        if not progressed:
            break
    for r in selected:
        r["review_split"] = "discovery"

    holdout: List[Dict[str, Any]] = []
    while len(holdout) < max(holdout_limit, 0):
        progressed = False
        for b in ordered:
            if not buckets[b]:
                continue
            holdout.append(buckets[b].pop(0))
            progressed = True
            if len(holdout) >= holdout_limit:
                break
        if not progressed:
            break
    for r in holdout:
        r["review_split"] = "holdout"

    return selected + holdout


def mode_sample(manifest_path: Path, out_csv: Path, samples_per_skill: int, holdout_per_skill: int, seed: int) -> int:
    rows = read_jsonl(manifest_path)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[str(r.get("skill") or "")].append(r)

    selected: List[Dict[str, Any]] = []
    for skill, skill_rows in sorted(grouped.items()):
        selected.extend(stratified_select(skill_rows, samples_per_skill, holdout_per_skill, seed))

    # normalize field set & add annotation fields
    out_rows: List[Dict[str, Any]] = []
    for r in selected:
        out_rows.append(
            {
                "review_split": r.get("review_split", ""),
                "run_dir": r.get("run_dir", ""),
                "task_name": r.get("task_name", ""),
                "demo_id": r.get("demo_id", ""),
                "segment_idx": r.get("segment_idx", ""),
                "skill": r.get("skill", ""),
                "metric_success": r.get("metric_success", ""),
                "result_type": r.get("result_type", ""),
                "metric_family": r.get("metric_family", ""),
                "metrics_path": r.get("metrics_path", ""),
                "start_obs_rgb": r.get("start_obs_rgb", ""),
                "end_obs_rgb": r.get("end_obs_rgb", ""),
                "final_obs_rgb": r.get("final_obs_rgb", ""),
                "video_path": r.get("video_path", ""),
                "vision_judgement": "",
                "vision_reason": "",
                "issue_bucket": "",
            }
        )

    fieldnames = list(out_rows[0].keys()) if out_rows else [
        "review_split",
        "run_dir",
        "task_name",
        "demo_id",
        "segment_idx",
        "skill",
        "metric_success",
        "result_type",
        "metric_family",
        "metrics_path",
        "start_obs_rgb",
        "end_obs_rgb",
        "final_obs_rgb",
        "video_path",
        "vision_judgement",
        "vision_reason",
        "issue_bucket",
    ]
    write_csv(out_csv, out_rows, fieldnames)
    print(json.dumps({"selected": len(out_rows), "skills": len(grouped), "out": str(out_csv)}, ensure_ascii=False, indent=2))
    return 0


def mode_summarize(audit_csv: Path, out_md: Path, top_k: int) -> int:
    rows = read_csv(audit_csv)
    mismatches: List[Dict[str, Any]] = []
    by_bucket = Counter()
    by_issue = Counter()

    for r in rows:
        metric_success = parse_bool(r.get("metric_success"))
        vj = normalize_judgement(r.get("vision_judgement"))
        if metric_success is None or vj in ("", "unclear"):
            continue
        vision_success = True if vj == "success" else False if vj == "fail" else None
        if vision_success is None:
            continue
        if vision_success != metric_success:
            mismatches.append(r)
            bucket = f"{r.get('skill')}|{r.get('result_type')}|metric={int(metric_success)}|vision={vj}"
            by_bucket[bucket] += 1
            issue = f"{r.get('skill')}|{r.get('issue_bucket') or 'unbucketed'}"
            by_issue[issue] += 1

    lines: List[str] = []
    lines.append(f"# Audit Summary\n")
    lines.append(f"- audit_csv: `{audit_csv}`\n")
    lines.append(f"- total_rows: {len(rows)}\n")
    lines.append(f"- mismatches: {len(mismatches)}\n")

    lines.append("\n## Top mismatch buckets\n")
    for key, cnt in by_bucket.most_common(20):
        lines.append(f"- {cnt}: {key}")

    lines.append("\n## Top issue buckets\n")
    for key, cnt in by_issue.most_common(20):
        lines.append(f"- {cnt}: {key}")

    lines.append("\n## Examples\n")
    for r in mismatches[: max(0, top_k)]:
        lines.append(
            "- "
            + json.dumps(
                {
                    "skill": r.get("skill"),
                    "result_type": r.get("result_type"),
                    "metric_success": r.get("metric_success"),
                    "vision_judgement": r.get("vision_judgement"),
                    "issue_bucket": r.get("issue_bucket"),
                    "start_obs_rgb": r.get("start_obs_rgb"),
                    "final_obs_rgb": r.get("final_obs_rgb"),
                    "metrics_path": r.get("metrics_path"),
                },
                ensure_ascii=False,
            )
        )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"mismatches": len(mismatches), "out": str(out_md)}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample and summarize sanity manifests for multimodal metric audit.")
    parser.add_argument("--mode", choices=["sample", "summarize", "llm"], required=True)
    parser.add_argument("--manifest", type=str, help="Path to sanity_manifest.jsonl (for mode=sample)")
    parser.add_argument("--audit-csv", type=str, help="Path to audit_manifest.csv (for mode=summarize)")
    parser.add_argument("--out", type=str, default="", help="Output path (csv for sample, md for summarize)")
    parser.add_argument("--samples-per-skill", type=int, default=8)
    parser.add_argument("--holdout-per-skill", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-llm", type=int, default=0, help="Max rows to pre-annotate in mode=llm (0 disables)")
    args = parser.parse_args()

    if args.mode == "sample":
        if not args.manifest:
            raise SystemExit("--manifest is required for mode=sample")
        manifest_path = Path(args.manifest).resolve()
        out_csv = Path(args.out).resolve() if args.out else manifest_path.parent / "audit_manifest.csv"
        return mode_sample(manifest_path, out_csv, args.samples_per_skill, args.holdout_per_skill, args.seed)

    if args.mode == "summarize":
        if not args.audit_csv:
            raise SystemExit("--audit-csv is required for mode=summarize")
        audit_csv = Path(args.audit_csv).resolve()
        out_md = Path(args.out).resolve() if args.out else audit_csv.parent / "audit_summary.md"
        return mode_summarize(audit_csv, out_md, args.top_k)

    if args.mode == "llm":
        if not args.audit_csv:
            raise SystemExit("--audit-csv is required for mode=llm")
        audit_csv = Path(args.audit_csv).resolve()
        out_csv = Path(args.out).resolve() if args.out else audit_csv.parent / "audit_manifest.llm.csv"
        rows = read_csv(audit_csv)
        api_key = ("OPENAI_API_KEY" in __import__("os").environ) or ("OPENAI_API_TOKEN" in __import__("os").environ)
        annotated = 0
        max_llm = max(0, int(args.max_llm))
        for r in rows:
            if max_llm and annotated >= max_llm:
                break
            if normalize_judgement(r.get("vision_judgement")):
                continue
            r["vision_judgement"] = "unclear"
            r["vision_reason"] = "llm_disabled_no_api_key" if not api_key else "llm_preannotation_not_implemented_in_cli"
            annotated += 1
        fieldnames = list(rows[0].keys()) if rows else []
        if fieldnames:
            write_csv(out_csv, rows, fieldnames)
        print(json.dumps({"annotated": annotated, "out": str(out_csv)}, ensure_ascii=False, indent=2))
        return 0

    raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
