import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("full_run_dir")
    parser.add_argument("segment_run_dir")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def load_episode_table(full_run_dir: Path):
    rows = []
    by_instance = {}
    for metric_path in sorted(full_run_dir.glob("eval_gpu*_p*/metrics/*.json")):
        try:
            data = json.loads(metric_path.read_text())
        except Exception:
            continue
        parts = metric_path.stem.split("_")
        if len(parts) < 3:
            continue
        try:
            instance_id = int(parts[-2])
            rollout_id = int(parts[-1])
        except ValueError:
            continue
        q_final = data.get("q_score", {}).get("final", 0.0)
        row = {
            "instance_id": instance_id,
            "rollout_id": rollout_id,
            "episode_success": bool(q_final and q_final > 0),
            "episode_q_score_final": q_final,
            "episode_metric_path": str(metric_path),
        }
        rows.append(row)
        by_instance.setdefault(instance_id, row)
    return rows, by_instance


def load_segment_table(segment_run_dir: Path):
    rows = []
    manifest_path = segment_run_dir / "segment_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest_by_key = {}
    for job in manifest.get("jobs", []):
        manifest_by_key[(str(job["demo_id"]), int(job["segment_idx"]))] = job

    for metric_path in sorted(segment_run_dir.glob("raw/**/metrics/*.json")):
        try:
            data = json.loads(metric_path.read_text())
        except Exception:
            continue
        demo_id = str(data.get("demo_id"))
        segment_idx = int(data.get("segment_idx"))
        key = (demo_id, segment_idx)
        manifest_row = manifest_by_key.get(key, {})
        q_delta = data.get("q_score", {}).get("delta")
        rollout = data.get("rollout", {})
        row = {
            "demo_id": demo_id,
            "instance_id": int(data.get("instance_id")),
            "segment_idx": segment_idx,
            "segment_level": data.get("segment_level"),
            "segment_desc": data.get("segment_desc", manifest_row.get("segment_desc", "unknown")),
            "success": bool(data.get("success")),
            "result_type": data.get("result_type"),
            "q_score_delta": q_delta,
            "rollout_final_step": rollout.get("final_step"),
            "rollout_best_progress": rollout.get("best_progress"),
            "rollout_final_progress": rollout.get("final_progress"),
            "metric_path": str(metric_path),
            "frame_duration": data.get("frame_duration", manifest_row.get("frame_duration")),
            "episode_success_from_manifest": manifest_row.get("episode_success"),
            "episode_q_from_manifest": manifest_row.get("episode_q_score_final"),
        }
        rows.append(row)
    return rows


def safe_rate(num, den):
    if den == 0:
        return None
    return num / den


def make_skill_summary(joined_rows):
    grouped = defaultdict(list)
    for row in joined_rows:
        grouped[row["segment_desc"]].append(row)

    summary = []
    for skill_desc, rows in sorted(grouped.items()):
        n_segments = len(rows)
        n_success = sum(1 for r in rows if r["success"])
        n_fail = sum(1 for r in rows if not r["success"])
        matched_fail_rows = [r for r in rows if (not r["success"]) and (r["episode_success"] in (True, False))]
        n_fail_and_episode_fail = sum(1 for r in matched_fail_rows if r["episode_success"] is False)
        summary.append(
            {
                "skill_desc": skill_desc,
                "n_segments": n_segments,
                "skill_success_rate": safe_rate(n_success, n_segments),
                "skill_fail_count": n_fail,
                "matched_skill_fail_count": len(matched_fail_rows),
                "episode_fail_when_skill_fail_rate": safe_rate(n_fail_and_episode_fail, len(matched_fail_rows)),
            }
        )
    return summary


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(summary_json):
    lines = []
    lines.append("# Skill Success vs Episode Failure Summary")
    lines.append("")
    lines.append(f"- full_run_dir: `{summary_json['full_run_dir']}`")
    lines.append(f"- segment_run_dir: `{summary_json['segment_run_dir']}`")
    lines.append(f"- episode_rows: {len(summary_json['episode_table'])}")
    lines.append(f"- segment_rows: {len(summary_json['segment_table'])}")
    lines.append("")
    lines.append("| skill_desc | n_segments | skill_success_rate | skill_fail_count | episode_fail_when_skill_fail_rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in summary_json["skill_summary"]:
        def fmt(v):
            if v is None:
                return "NA"
            if isinstance(v, float) and not math.isnan(v):
                return f"{v:.4f}"
            return str(v)
        lines.append(
            f"| {row['skill_desc']} | {row['n_segments']} | {fmt(row['skill_success_rate'])} | {row['skill_fail_count']} | {fmt(row['episode_fail_when_skill_fail_rate'])} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    full_run_dir = Path(args.full_run_dir).resolve()
    segment_run_dir = Path(args.segment_run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else segment_run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_table, episode_by_instance = load_episode_table(full_run_dir)
    segment_table = load_segment_table(segment_run_dir)

    joined_table = []
    for row in segment_table:
        episode = episode_by_instance.get(row["instance_id"])
        joined = dict(row)
        joined["episode_success"] = episode["episode_success"] if episode else row["episode_success_from_manifest"]
        joined["episode_q_score_final"] = episode["episode_q_score_final"] if episode else row["episode_q_from_manifest"]
        joined_table.append(joined)

    skill_summary = make_skill_summary(joined_table)
    summary_json = {
        "full_run_dir": str(full_run_dir),
        "segment_run_dir": str(segment_run_dir),
        "matched_episode_rows": sum(1 for row in joined_table if row["episode_success"] in (True, False)),
        "episode_table": episode_table,
        "segment_table": segment_table,
        "joined_table": joined_table,
        "skill_summary": skill_summary,
    }

    json_path = output_dir / "summary_skill_segment.json"
    csv_path = output_dir / "summary_skill_segment.csv"
    md_path = output_dir / "summary_skill_segment.md"
    joined_csv_path = output_dir / "summary_skill_segment_joined.csv"

    json_path.write_text(json.dumps(summary_json, indent=2))
    write_csv(csv_path, skill_summary)
    write_csv(joined_csv_path, joined_table)
    md_path.write_text(render_markdown(summary_json))

    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "joined_csv": str(joined_csv_path), "md": str(md_path), "skill_summary_count": len(skill_summary)}, indent=2))


if __name__ == "__main__":
    main()
