#!/usr/bin/env python3
"""独立于 gate 的一次性统计，给三向验证提供「事先写下的预期条数」。

刻意**不**导入 openpi：直接用 SentencePiece 复刻 `tokenizer.py:503/533` 的清洗
口径。这样预期值与 gate 的实现互相独立 —— 拿 gate 自己的输出当预期没有信息量。
（清洗口径一致性由 gate 的三向验证结果本身检验：若两边口径不同，条数就对不上。）

用法：_expected_counts.py <filelist> <frames_root> <out.json>
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import sentencepiece

TOKENIZER_MODEL = "/home/tiger/.cache/openpi/big_vision/paligemma_tokenizer.model"
SPEC = (("Memory", "fixed_compact_memory"), ("Primitive", "current_primitive"),
        ("Skill", "current_skill"), ("Next skill", "next_skill"),
        ("Next primitive", "next_primitive"))
FALLBACK = ("pick up from + place on next to", "pick up from + place in next to",
            "pick up from + chop + place on next to", "pick up from + place on",
            "pick up from + place in", "pick up from + chop",
            "pick up from + pour", "unknown primitive")
FIELDS = ("current_primitive", "next_primitive")
CHUNK = 250

_SP = None
_STATE = None
_LENS: dict[int, int] = {}


def _init(lens):
    global _SP, _STATE, _LENS
    _LENS = lens
    _SP = sentencepiece.SentencePieceProcessor(model_file=TOKENIZER_MODEL)
    bins = np.linspace(-1, 1, 257)[:-1]
    best = None
    for cand in (np.full(32, -5.0), np.full(32, -1.0), np.zeros(32), np.full(32, 5.0)):
        s = " ".join(map(str, np.digitize(cand, bins=bins) - 1))
        n = len(_SP.encode(s))
        if best is None or n > best[0]:
            best = (n, s)
    _STATE = best[1]


def _clean(t: str) -> str:
    return t.strip().replace("_", " ").replace("\n", " ")


def _one(ps: str):
    p = Path(ps)
    d = json.loads(p.read_text(encoding="utf-8"))
    ma = d["memory_annotation"]
    task = d.get("task_name") or ""
    L = _LENS[int(p.stem.split("_")[1])]
    tl, pl, prl = [], [], []
    drops = {f: 0 for f in FALLBACK}
    incomplete = 0
    for r in ma:
        t = "\n".join(f"{lab}: {r[f]}" for lab, f in SPEC)
        tl.append(2 + len(_SP.encode(_clean(t))))
        # 生产口径 tokenize_memory -> MemoryTextCodec.encode(text.strip())：
        # 既不把换行压成空格，**也不做下划线 -> 空格** —— 与另一条路径的清洗是两套。
        # 所以这里独立按 .strip() 编码，而不是给文档口径 +4。
        prl.append(2 + len(_SP.encode(t.strip())))
        pre = (f"Task: {_clean(task)}, State: {_STATE};\n"
               f"Previous memory: {_clean(r['previous_fixed_compact_memory'])}")
        pl.append(len(_SP.encode(pre, add_bos=True)))
        blob = " | ".join(str(r.get(f) or "") for f in FIELDS)
        if " + " in blob or "unknown primitive" in blob:
            if not any(x in blob for x in FALLBACK):
                incomplete += 1
            for dp in FALLBACK:
                if not any(x in blob for x in FALLBACK if x != dp):
                    drops[dp] += 1
    cs = ma[0]["frame_duration"][0]
    ce = min(ma[-1]["frame_duration"][1], L)
    nch = -(-L // CHUNK)
    dead = sum(1 for c0 in range(0, L, CHUNK) if min(c0 + CHUNK, L) <= cs or c0 >= ce)
    return tl, pl, prl, nch, dead, drops, incomplete, len(ma)


def main() -> int:
    filelist, frames_root, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    files = [x for x in filelist.read_text().split() if x.strip()]
    lens: dict[int, int] = {}
    with (frames_root / "meta" / "episodes.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                lens[int(r["episode_index"])] = int(r["length"])
    if not files:
        raise SystemExit("CANNOT-ASSESS: 文件清单为空")

    TL: list[int] = []
    PL: list[int] = []
    PRL: list[int] = []
    NCH = DEAD = INC = NIV = 0
    DR = {f: 0 for f in FALLBACK}
    with Pool(4, initializer=_init, initargs=(lens,)) as pool:
        for tl, pl, prl, nch, dead, dr, inc, niv in pool.imap_unordered(_one, files, chunksize=20):
            TL += tl
            PL += pl
            PRL += prl
            NCH += nch
            DEAD += dead
            INC += inc
            NIV += niv
            for k, v in dr.items():
                DR[k] += v
    TL.sort()
    PL.sort()
    PRL.sort()
    res = {"files": len(files), "intervals": NIV,
           "target_max_doc_path": TL[-1], "target_max_production": PRL[-1],
           "delta_two_paths_distinct_values": sorted({b - a for a, b in zip(TL, PRL)})[:8],
           "prompt_max": PL[-1],
           "chunks": NCH, "dead_chunks": DEAD, "live_chunks": NCH - DEAD,
           "fallback_taxonomy_incomplete": INC,
           "drop_one_phrase_exposes": DR}
    for t in (128, 160, 192, 256, 100, 90, 83, 70):
        res[f"target_over_{t}"] = sum(1 for x in PRL if x > t)
        res[f"target_over_docpath_{t}"] = sum(1 for x in TL if x > t)
    for t in (320, 512, 1024, 200, 180, 162):
        res[f"prompt_over_{t}"] = sum(1 for x in PL if x > t)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
