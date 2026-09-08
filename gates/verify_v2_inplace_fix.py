#!/usr/bin/env python3
"""证明 v2 的原地修复「只动了该动的，且只动了元数据」。

为什么需要它
------------
数据侧对 v2 做的是**原地修改**：83 个 episode 的 `meta_data` + manifest。原地修改
会把修复前的状态抹掉，所以修复前先落了一份快照（`v2_prefix_snapshot/`）：
全部 10,000 个文件的 md5、那 83 个文件的整份原文、以及 manifest 原件。

gate 的 `--baseline-report` 给的是**聚合量**的差异（token 分布、chunk 计数……）。
聚合量相同只能说明「总量没变」，不能排除「A 文件多了 B 文件少了」。这里做的是
**逐文件、逐字段**的对照，比聚合强得多：

1. 变化的文件集合必须**恰好等于**那 83 个（外加 manifest）—— 多一个少一个都算失败
2. 那 83 个里，`memory_annotation` 必须**逐字节不变** —— 区间结构一根手指都不许动
3. 变化只允许出现在 `meta_data`，且只允许 `valid_duration` / `task_duration`
4. 修复后必须满足 `valid_duration[1] == memory_annotation[-1].frame_duration[1]`

用法：verify_v2_inplace_fix.py [--data-root PATH] [--snapshot PATH] [--out report.json]
退出码 0 = 全部成立，1 = 有不符，3 = 快照或数据不可用（NOT-MEASURED）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Any

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

DEFAULT_ROOT = Path("/mnt/bn/behavior-data-hl/chenjunting/data/2025-challenge-demos/derived/"
                    "fixed_compact_memory_annotations_v2")
DEFAULT_SNAP = Path(__file__).resolve().parent / "v2_prefix_snapshot"
ALLOWED_META_KEYS = {"valid_duration", "task_duration"}


def _md5(p: Path) -> tuple[str, str]:
    return hashlib.md5(p.read_bytes()).hexdigest(), str(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAP)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    snap, root = args.snapshot, args.data_root
    before_md5_path = snap / "v2_before_all.md5"
    if not before_md5_path.exists() or before_md5_path.stat().st_size == 0:
        print(json.dumps({"verdict": "CANNOT-ASSESS",
                          "why": f"快照台账不可用: {before_md5_path}"}, ensure_ascii=False))
        return 3
    before = {}
    for line in before_md5_path.read_text().splitlines():
        if line.strip():
            h, rel = line.split("  ", 1)
            before[rel] = h
    expected_changed = {ln.split("\t")[0] for ln in
                        (snap / "v2_before_mismatch_list.txt").read_text().splitlines() if ln.strip()}

    files = sorted(root.glob("task-*/episode_*.json"))
    if len(files) != len(before):
        print(json.dumps({"verdict": "CANNOT-ASSESS",
                          "why": f"文件数变了: 现在 {len(files)}，快照 {len(before)}"},
                         ensure_ascii=False))
        return 3

    with Pool(args.jobs) as pool:
        now = {str(Path(p).relative_to(root)): h
               for h, p in pool.imap_unordered(_md5, files, chunksize=50)}

    changed = sorted(k for k in now if now[k] != before.get(k))
    unchanged = len(now) - len(changed)

    rows: list[dict[str, Any]] = []
    ok = True

    def check(name: str, passed: bool, measured: Any, why: str = "") -> None:
        nonlocal ok
        ok &= passed
        rows.append({"check": name, "passed": bool(passed), "measured": measured, "why": why})

    check("changed_set_equals_expected_83",
          set(changed) == expected_changed,
          {"changed": len(changed), "expected": len(expected_changed),
           "unexpected_changed": sorted(set(changed) - expected_changed)[:10],
           "expected_but_unchanged": sorted(expected_changed - set(changed))[:10]},
          "多改一个或漏改一个都算失败；聚合量相同掩盖不了这个")
    check("untouched_files_byte_identical", unchanged == len(before) - len(expected_changed),
          {"unchanged": unchanged, "expected_unchanged": len(before) - len(expected_changed)})

    # 逐字段对照那 83 个
    ann_changed, meta_key_violations, still_mismatch, other_top_keys = [], [], [], []
    for rel in sorted(expected_changed):
        b = json.loads((snap / "before_files" / rel.replace("/", "__")).read_text())
        a = json.loads((root / rel).read_text())
        if json.dumps(b["memory_annotation"], sort_keys=True) != \
           json.dumps(a["memory_annotation"], sort_keys=True):
            ann_changed.append(rel)
        for k in set(b) | set(a):
            if k == "meta_data":
                continue
            if json.dumps(b.get(k), sort_keys=True) != json.dumps(a.get(k), sort_keys=True):
                other_top_keys.append(f"{rel}:{k}")
        mb, ma_ = b.get("meta_data") or {}, a.get("meta_data") or {}
        for k in set(mb) | set(ma_):
            if mb.get(k) != ma_.get(k) and k not in ALLOWED_META_KEYS:
                meta_key_violations.append(f"{rel}:{k}")
        vd = ma_.get("valid_duration")
        last_end = a["memory_annotation"][-1]["frame_duration"][1]
        if not vd or int(vd[1]) != int(last_end):
            still_mismatch.append(f"{rel}: valid_duration={vd} last_end={last_end}")

    check("memory_annotation_byte_identical", not ann_changed,
          {"changed": len(ann_changed), "examples": ann_changed[:5]},
          "区间结构一根手指都不许动 —— 这是「只改元数据」最硬的判据")
    check("only_meta_data_top_level_changed", not other_top_keys,
          {"violations": len(other_top_keys), "examples": other_top_keys[:5]})
    check("only_allowed_meta_keys_changed", not meta_key_violations,
          {"violations": len(meta_key_violations), "examples": meta_key_violations[:5]},
          f"只允许 {sorted(ALLOWED_META_KEYS)}")
    check("valid_duration_now_matches_last_end", not still_mismatch,
          {"still_mismatched": len(still_mismatch), "examples": still_mismatch[:5]},
          "修复的目标本身")

    mf = root / "manifest.json"
    mf_before = (snap / "v2_before_manifest.md5").read_text().split("  ")[0]
    mf_now = hashlib.md5(mf.read_bytes()).hexdigest()
    check("manifest_changed", mf_before != mf_now,
          {"before": mf_before[:12], "now": mf_now[:12]},
          "manifest 应当也被更新；没变说明只改了文件没改台账")

    report = {"verdict": "PASS" if ok else "FAIL", "data_root": str(root),
              "files": len(now), "changed": len(changed), "unchanged": unchanged,
              "changed_files": changed, "checks": rows}
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for r in rows:
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['check']}: "
              f"{json.dumps(r['measured'], ensure_ascii=False)}"
              + (f"   ({r['why']})" if r["why"] else ""))
    print(f"\nINPLACE_FIX_VERDICT: {report['verdict']}  "
          f"changed={len(changed)} unchanged={unchanged}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
