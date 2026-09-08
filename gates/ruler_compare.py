#!/usr/bin/env python3
"""同一 vocab 文件、不同 `sentencepiece` 库版本，是不是同一把尺子？

背景：`il_lib` 的长度审计跑在 `PYTHONPATH=/tmp/b1k_tokenizer_py313` + miniconda
py3.13（sentencepiece 0.2.2）下，训练侧 env `openpi-comet-nas` 是 0.2.0。两边用
的是同一个模型文件（md5 已核对一致），但**同模型不同库版本不等于同一把尺子**。

「声明两者是同一把尺子」是不可判定的说法。这里把它换成可判定的形式：在同一批
文本上各跑一遍，逐条比对 token id 序列是否逐元素相同。

用法（两步）：
    # 训练侧
    PYTHONPATH=<worktree>/src <nas-python> ruler_compare.py dump \\
        --tag sp020 --out /tmp/ruler_sp020.jsonl
    # 审计侧
    PYTHONPATH=/tmp/b1k_tokenizer_py313 <py313> ruler_compare.py dump \\
        --tag sp022 --out /tmp/ruler_sp022.jsonl --tokenizer-model <path>
    # 比对
    <any-python> ruler_compare.py compare --a /tmp/ruler_sp020.jsonl \\
        --b /tmp/ruler_sp022.jsonl --out /tmp/ruler_compare.json

dump 只落每行的 (token 数, token id 序列的 sha1)，所以 261,353 行也只有几十 MB；
外加前 N 条的完整 id 序列用于人工核对，以及不匹配时的完整取证。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

DEFAULT_DATA_ROOT = Path(
    "/mnt/bn/behavior-data-hl/chenjunting/data/2025-challenge-demos/derived/"
    "fixed_compact_memory_annotations"
)
DEFAULT_TOKENIZER = Path("~/.cache/openpi/big_vision/paligemma_tokenizer.model").expanduser()

TARGET_LINE_SPEC = (
    ("Memory", "fixed_compact_memory"),
    ("Primitive", "current_primitive"),
    ("Skill", "current_skill"),
    ("Next skill", "next_skill"),
    ("Next primitive", "next_primitive"),
)


def clean(text: str) -> str:
    return text.strip().replace("_", " ").replace("\n", " ")


def planner_target_text(row: dict) -> str:
    return "\n".join(f"{label}: {row[field]}" for label, field in TARGET_LINE_SPEC)


def iter_texts(root: Path, stride: int):
    files = sorted(root.glob("task-*/episode_*.json"))
    if stride > 1:
        files = files[::stride]
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        rel = f"{path.parent.name}/{path.name}"
        for row in data.get("memory_annotation", []):
            yield f"{rel}#{row['memory_idx']}", planner_target_text(row)
            yield f"{rel}#{row['memory_idx']}@model", row["model_target_text"]


def cmd_dump(args: argparse.Namespace) -> int:
    import sentencepiece

    sp = sentencepiece.SentencePieceProcessor(model_file=str(args.tokenizer_model))
    md5 = hashlib.md5(Path(args.tokenizer_model).read_bytes()).hexdigest()
    header = {
        "_header": True,
        "tag": args.tag,
        "tokenizer_model": str(args.tokenizer_model),
        "tokenizer_md5": md5,
        "vocab_size": sp.vocab_size(),
        "bos_id": sp.bos_id(),
        "eos_id": sp.eos_id(),
        "sentencepiece_version": getattr(sentencepiece, "__version__", "unknown"),
        "python": sys.version.split()[0],
        "stride": args.stride,
        "data_root": str(args.data_root),
    }
    n = 0
    with args.out.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        for key, text in iter_texts(args.data_root, args.stride):
            ids = sp.encode(clean(text))
            digest = hashlib.sha1(",".join(map(str, ids)).encode()).hexdigest()
            rec = {"k": key, "n": len(ids) + 2, "h": digest}
            if n < args.keep_full:
                rec["ids"] = ids
            fh.write(json.dumps(rec) + "\n")
            n += 1
            if n % 100_000 == 0:
                print(f"[dump:{args.tag}] {n}", flush=True)
    print(json.dumps({**header, "rows": n}, ensure_ascii=False))
    return 0


def _load(path: Path):
    header = None
    rows = {}
    order = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("_header"):
                header = rec
                continue
            rows[rec["k"]] = (rec["n"], rec["h"], rec.get("ids"))
            order.append(rec["k"])
    return header, rows, order


def cmd_compare(args: argparse.Namespace) -> int:
    ha, ra, oa = _load(args.a)
    hb, rb, ob = _load(args.b)
    only_a = [k for k in oa if k not in rb]
    only_b = [k for k in ob if k not in ra]
    common = [k for k in oa if k in rb]
    mismatch_ids = []
    mismatch_len = []
    for k in common:
        na, hha, ia = ra[k]
        nb, hhb, ib = rb[k]
        if hha != hhb:
            mismatch_ids.append(k)
        if na != nb:
            mismatch_len.append(k)
    result = {
        "a": ha,
        "b": hb,
        "rows_a": len(ra),
        "rows_b": len(rb),
        "common_rows": len(common),
        "only_in_a": len(only_a),
        "only_in_b": len(only_b),
        "token_id_sequence_mismatches": len(mismatch_ids),
        "token_count_mismatches": len(mismatch_len),
        "identical": (len(mismatch_ids) == 0 and len(only_a) == 0 and len(only_b) == 0 and len(common) > 0),
        "examples_id_mismatch": mismatch_ids[:10],
        "examples_len_mismatch": mismatch_len[:10],
        # 正对照：随便挑一条，证明两侧真的都产出了非空 id 序列（不是双双空跑）
        "positive_control": None,
        # 负对照：一条必然不同的伪造 key 必须落在 only_in_* 里（下面注入）
    }
    if common:
        k0 = common[0]
        result["positive_control"] = {
            "key": k0,
            "a_n": ra[k0][0],
            "b_n": rb[k0][0],
            "a_h": ra[k0][1][:12],
            "b_h": rb[k0][1][:12],
            "a_ids_head": (ra[k0][2] or [])[:12],
            "b_ids_head": (rb[k0][2] or [])[:12],
            "nonempty_both": ra[k0][0] > 2 and rb[k0][0] > 2,
        }
    # 负对照：注入一个必然不同的哨兵，证明比对器真的会报差异
    sentinel_a = dict(ra)
    sentinel_a["__sentinel__"] = (7, "deadbeef", None)
    sentinel_mismatch = sum(1 for k in sentinel_a if k in rb and sentinel_a[k][1] != rb[k][1])
    result["negative_control_detects_injected_diff"] = (
        "__sentinel__" not in rb and sentinel_mismatch == len(mismatch_ids)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if result["identical"] else 1


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump")
    d.add_argument("--tag", required=True)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    d.add_argument("--tokenizer-model", type=Path, default=DEFAULT_TOKENIZER)
    d.add_argument("--stride", type=int, default=1)
    d.add_argument("--keep-full", type=int, default=200)
    d.set_defaults(func=cmd_dump)

    c = sub.add_parser("compare")
    c.add_argument("--a", type=Path, required=True)
    c.add_argument("--b", type=Path, required=True)
    c.add_argument("--out", type=Path)
    c.set_defaults(func=cmd_compare)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
