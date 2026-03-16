import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

from behavior.learning.datas.dataset_utils import convert_episode_annotation_to_subtasks


_TPL_PATH: str | None = None
_MAP_PATH: str | None = None
_JOINER: str | None = None
_OUT_ROOT: str | None = None
_OVERWRITE: bool = False


def _pool_init(template_path: str, mapping_path: str, joiner: str, out_root: str, overwrite: bool) -> None:
    global _TPL_PATH, _MAP_PATH, _JOINER, _OUT_ROOT, _OVERWRITE
    _TPL_PATH = template_path
    _MAP_PATH = mapping_path
    _JOINER = joiner
    _OUT_ROOT = out_root
    _OVERWRITE = overwrite


def _episode_index_from_path(p: Path) -> int | None:
    if p.stem.startswith("episode_") and p.stem[8:].isdigit():
        return int(p.stem[8:])
    return None


def _convert_one(in_path_str: str) -> tuple[str, bool, str]:
    in_path = Path(in_path_str)
    task_dir = in_path.parent.name
    out_dir = Path(_OUT_ROOT) / task_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / in_path.name
    if out_path.exists() and not _OVERWRITE:
        return (in_path_str, False, "exists")

    with open(in_path, "r", encoding="utf-8") as f:
        episode_ann = json.load(f)
    out_obj = convert_episode_annotation_to_subtasks(
        episode_ann=episode_ann,
        episode_index=_episode_index_from_path(in_path),
        subtask_template_path=str(_TPL_PATH),
        subtask_object_name_mapping_path=str(_MAP_PATH),
        subtask_joiner=str(_JOINER),
    )
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, out_path)
    return (in_path_str, True, "ok")


def _parse_task_indices(s: str | None) -> set[int] | None:
    if s is None or not str(s).strip():
        return None
    out: set[int] = set()
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a.strip())
            b = int(b.strip())
            for x in range(min(a, b), max(a, b) + 1):
                out.add(x)
        else:
            out.add(int(part))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--subtask-joiner", type=str, default=" then ")
    parser.add_argument("--subtask-template-path", type=str, default=None)
    parser.add_argument("--subtask-object-name-mapping-path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    ann_root = dataset_root / "annotations"
    out_root = Path(args.output_root) if args.output_root is not None else (dataset_root / "subtasks")
    out_root.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    tpl_path = (
        Path(args.subtask_template_path)
        if args.subtask_template_path is not None
        else (repo_root / "src/behavior/learning/datas/b1k_subtask_phrase_templates.json")
    )
    map_path = (
        Path(args.subtask_object_name_mapping_path)
        if args.subtask_object_name_mapping_path is not None
        else (repo_root / "src/behavior/learning/datas/b1k_object_id_name_mapping.json")
    )

    allowed_tasks = _parse_task_indices(args.tasks)

    inputs: list[str] = []
    for d in sorted(ann_root.glob("task-*")):
        if not d.is_dir():
            continue
        if allowed_tasks is not None:
            try:
                tid = int(d.name.split("-", 1)[1])
            except Exception:
                continue
            if tid not in allowed_tasks:
                continue
        for p in sorted(d.glob("episode_*.json")):
            inputs.append(str(p))
            if args.limit is not None and len(inputs) >= int(args.limit):
                break
        if args.limit is not None and len(inputs) >= int(args.limit):
            break

    with Pool(
        processes=max(1, int(args.num_workers)),
        initializer=_pool_init,
        initargs=(str(tpl_path), str(map_path), str(args.subtask_joiner), str(out_root), bool(args.overwrite)),
    ) as pool:
        ok = 0
        skipped = 0
        failed = 0
        for in_path, wrote, status in pool.imap_unordered(_convert_one, inputs, chunksize=32):
            if status == "ok":
                ok += 1
            elif status == "exists":
                skipped += 1
            else:
                failed += 1
        print(json.dumps({"total": len(inputs), "ok": ok, "skipped": skipped, "failed": failed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
