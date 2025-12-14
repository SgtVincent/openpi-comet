#!/usr/bin/env python3
import argparse
import glob
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple


def load_task_mapping(tasks_jsonl_path: str) -> Dict[str, int]:
    task_name_to_index: Dict[str, int] = {}
    with open(tasks_jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            task_name_to_index[data["task_name"]] = int(data["task_index"])
    return task_name_to_index


def resolve_tasks_jsonl_path(path_like: str) -> str:
    if os.path.isabs(path_like):
        return path_like
    # Try CWD first
    if os.path.exists(path_like):
        return path_like
    # Try relative to workspace root (5 levels up from this script: pipelines/reasoning/behavior/rollout/script.py)
    # Original was 4 levels.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))))
    candidate = os.path.join(workspace_root, path_like)
    return candidate


def read_task_name_from_exp(exp_dir: str) -> Optional[str]:
    """
    Parse task_name from cfg*.out file located directly under exp_dir.
    Expect a line like: task_name=turning_on_radio
    """
    cfg_files = sorted(glob.glob(os.path.join(exp_dir, "cfg*.out")))
    if not cfg_files:
        return None
    for cfg_file in cfg_files:
        try:
            with open(cfg_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("task_name="):
                        return line.split("=", 1)[1].strip()
        except Exception:
            continue
    return None


def is_valid_rollout_run(run_dir: str) -> bool:
    """
    Define validity as presence of non-empty state_action.npz.
    """
    npz_path = os.path.join(run_dir, "state_action.npz")
    return os.path.isfile(npz_path) and os.path.getsize(npz_path) > 0


def find_run_dirs(root_dir: str) -> List[Tuple[str, str, str, str]]:
    """
    Return list of tuples:
      (date_dir_name, exp_name, run_name, run_dir_abs_path)
    Supports two input styles:
      - root_dir = /mnt/sdb/rollouts  (contains date subdirs)
      - root_dir = /mnt/sdb/rollouts/2025-11-11 (a specific date dir)
    """
    runs: List[Tuple[str, str, str, str]] = []

    # Pattern A: root/date/exp/rollouts/run
    pattern_a = os.path.join(root_dir, "*", "*", "rollouts", "*")
    # Pattern B: (root is a date) root/exp/rollouts/run
    pattern_b = os.path.join(root_dir, "*", "rollouts", "*")

    # Prefer A; if empty, fall back to B
    candidates = glob.glob(pattern_a)
    use_pattern_b = False
    if not candidates:
        candidates = glob.glob(pattern_b)
        use_pattern_b = True

    for run_dir in sorted(candidates):
        if not os.path.isdir(run_dir):
            continue
        if use_pattern_b:
            # root_dir = date, structure: root/exp/rollouts/run
            run_name = os.path.basename(run_dir)
            exp_dir = os.path.dirname(os.path.dirname(run_dir))  # root/exp
            exp_name = os.path.basename(exp_dir)
            date_dir_name = os.path.basename(os.path.abspath(root_dir))
        else:
            # root/date/exp/rollouts/run
            run_name = os.path.basename(run_dir)
            exp_dir = os.path.dirname(os.path.dirname(run_dir))  # root/date/exp
            exp_name = os.path.basename(exp_dir)
            date_dir = os.path.dirname(exp_dir)  # root/date
            date_dir_name = os.path.basename(date_dir)

        runs.append((date_dir_name, exp_name, run_name, os.path.abspath(run_dir)))

    return runs


def collect_entries(root_dir: str) -> Iterable[Tuple[str, str, str, str]]:
    """
    Yield valid (date_dir_name, exp_name, run_name, run_dir_abs_path)
    skipping empty rollout_run folders.
    """
    for date_dir_name, exp_name, run_name, run_dir in find_run_dirs(root_dir):
        if is_valid_rollout_run(run_dir):
            yield date_dir_name, exp_name, run_name, run_dir


def build_exp_to_task_map(entries: Iterable[Tuple[str, str, str, str]]) -> Dict[str, str]:
    """
    Build a mapping from exp_dir_abs -> task_name using cfg*.out.
    entries provide run_dir; exp_dir = parent of 'rollouts'.
    """
    exp_to_task: Dict[str, str] = {}
    seen_exp_dirs: set = set()
    for _, exp_name, _, run_dir in entries:
        exp_dir = os.path.dirname(os.path.dirname(run_dir))  # .../<exp>
        if exp_dir in seen_exp_dirs:
            continue
        seen_exp_dirs.add(exp_dir)
        task_name = read_task_name_from_exp(exp_dir)
        if task_name is not None:
            exp_to_task[exp_dir] = task_name
    return exp_to_task


def find_max_indices_by_base_from_rollouts(
    output_dir: str,
    exclude_file: Optional[str],
    filename_prefix: str = "rollouts",
) -> Dict[int, int]:
    """
    Scan existing JSONL files in output_dir whose basenames start with `filename_prefix`
    and compute per-base (first 4 digits of episode_id) max current_idx (last 4 digits).
    This is a simpler and robust way to avoid collisions across multiple runs/files.
    """
    base_to_max_idx: Dict[int, int] = {}
    if not os.path.isdir(output_dir):
        return base_to_max_idx

    jsonl_pattern = os.path.join(output_dir, "*.jsonl")
    for jsonl_file in glob.glob(jsonl_pattern):
        if exclude_file and os.path.abspath(jsonl_file) == os.path.abspath(exclude_file):
            continue
        base_name = os.path.basename(jsonl_file)
        if not base_name.startswith(filename_prefix):
            continue
        try:
            with open(jsonl_file, "r") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    parts = line.rsplit(" ", 1)
                    if len(parts) != 2:
                        continue
                    _rel_path, episode_id_str = parts
                    # Episode id must be numeric with at least 4 digits
                    if not episode_id_str.isdigit() or len(episode_id_str) < 4:
                        continue
                    try:
                        episode_id_int = int(episode_id_str)
                    except ValueError:
                        continue
                    base = episode_id_int // 10000
                    current_idx = episode_id_int % 10000
                    prev = base_to_max_idx.get(base)
                    base_to_max_idx[base] = current_idx if prev is None else max(prev, current_idx)
        except Exception:
            continue

    return base_to_max_idx


def main():
    parser = argparse.ArgumentParser(description="Dump rollout entries to RFT dataset with episode IDs to JSONL")
    parser.add_argument("root_dir", help="Rollouts root directory to scan")
    parser.add_argument("output_jsonl", help="Output RFT dataset JSONL path")
    parser.add_argument(
        "--tasks_jsonl",
        default="2025-challenge-demos/meta/tasks.jsonl",
        help="Path to BEHAVIOR tasks.jsonl file (default: 2025-challenge-demos/meta/tasks.jsonl)",
    )
    args = parser.parse_args()

    tasks_jsonl_path = args.tasks_jsonl
    if not os.path.exists(tasks_jsonl_path):
        raise FileNotFoundError(f"Tasks JSONL file not found: {tasks_jsonl_path}")

    task_name_to_index = load_task_mapping(tasks_jsonl_path)
    
    # Materialize entries once to reuse
    entries = list(collect_entries(args.root_dir))
    if not entries:
        print(f"No valid rollout runs found under {args.root_dir}")
        # Still create empty file
        os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
        with open(args.output_jsonl, "w"):
            pass
        return

    # Build exp->task cache
    exp_to_task = build_exp_to_task_map(entries)

    # Prepare output dir and find existing per-task max indices to avoid overlaps
    output_dir = os.path.dirname(os.path.abspath(args.output_jsonl))
    os.makedirs(output_dir, exist_ok=True)
    # Global per-base maxima across JSONLs (excluding target file)
    base_max_indices = find_max_indices_by_base_from_rollouts(
        output_dir=output_dir,
        exclude_file=os.path.abspath(args.output_jsonl),
        filename_prefix="rollouts",
    )

    # Load existing file if present; preserve its lines exactly and do not change episode IDs
    existing_lines: List[str] = []
    existing_rels: set = set()
    # Also incorporate existing file's episode IDs into the per-base maxima
    if os.path.exists(args.output_jsonl):
        with open(args.output_jsonl, "r") as f_in:
            for raw in f_in:
                line = raw.rstrip("\n")
                if not line.strip():
                    existing_lines.append(raw)
                    continue
                parts = line.rsplit(" ", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    rel = parts[0]
                    episode_id_str = parts[1]
                    try:
                        episode_id_int = int(episode_id_str)
                        base = episode_id_int // 10000
                        current_idx = episode_id_int % 10000
                        prev = base_max_indices.get(base)
                        base_max_indices[base] = current_idx if prev is None else max(prev, current_idx)
                    except ValueError:
                        pass
                    existing_rels.add(rel)
                else:
                    # Line without episode id; treat whole line as rel to avoid duplication
                    rel_only = line.strip()
                    existing_rels.add(rel_only)
                # Preserve original line (including newline)
                existing_lines.append(raw)

    # Build a quick index from rel -> (exp_dir, task_name, task_index)
    rel_to_exp_dir: Dict[str, str] = {}
    rel_to_task_name: Dict[str, Optional[str]] = {}
    rel_to_task_index: Dict[str, Optional[int]] = {}
    for date_dir_name, exp_name, run_name, run_dir in entries:
        rel = f"{date_dir_name}/{exp_name}/{run_name}"
        if rel in rel_to_exp_dir:
            continue
        exp_dir = os.path.dirname(os.path.dirname(run_dir))
        rel_to_exp_dir[rel] = exp_dir
        task_name = exp_to_task.get(exp_dir)
        rel_to_task_name[rel] = task_name
        if task_name is None:
            rel_to_task_index[rel] = None
        else:
            rel_to_task_index[rel] = task_name_to_index.get(task_name)

    # Determine which rels are new (not already present in the existing file)
    all_rels_now = sorted(rel_to_exp_dir.keys())
    new_rels = [rel for rel in all_rels_now if rel not in existing_rels]

    # Append new lines with incremented episode IDs per base
    appended = 0
    new_lines: List[str] = []
    for rel in new_rels:
        task_index = rel_to_task_index.get(rel)
        if task_index is None:
            # Unknown task; write rel without episode id
            new_lines.append(rel + "\n")
            appended += 1
            continue
        base = 150 + int(task_index)
        # Next index after global maxima
        next_idx = base_max_indices.get(base, -1) + 1
        
        # Check for overflow
        if next_idx > 9999:
            task_name = rel_to_task_name.get(rel, "unknown")
            print(f"WARNING: Episode index overflow for task '{task_name}' (base={base}). "
                  f"Reached {next_idx}, max is 9999. Skipping entry: {rel}")
            continue
        
        base_max_indices[base] = next_idx
        episode_id = f"{base:04d}{next_idx:04d}"
        new_lines.append(f"{rel} {episode_id}\n")
        appended += 1

    # Write back: existing lines first (unchanged), then newly appended lines
    with open(args.output_jsonl, "w") as f_out:
        for raw in existing_lines:
            f_out.write(raw)
        for raw in new_lines:
            f_out.write(raw)

    total_written = len(existing_lines) + len(new_lines)
    print(f"Kept {len(existing_lines)} existing lines, appended {len(new_lines)} new lines. Total now {total_written}. -> {args.output_jsonl}")


if __name__ == "__main__":
    main()

