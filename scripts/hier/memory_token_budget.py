#!/usr/bin/env python3
"""Memory token-budget check (design section 8, P1 item 1's second half).

Reports P50/P90/P99 memory-text token length against the *actual*
``subtask_max_len`` in use, plus the growth slope that decides whether the
current headroom survives the full corpus.

Why this exists as a script rather than a one-off measurement
------------------------------------------------------------
Overflow does not fail loudly.  ``tokenize_memory`` truncates, and truncation
removes the **last** fields -- ``Next skill`` and ``Next primitive``.  That reads
as "the model is bad at predicting next-skill" rather than as a format bug, so
the budget has to be checked before a run, not diagnosed after one.

The current preview corpus fits, but only just, and the margin is consumed by a
measurable mechanism: the ``Memory:`` field accumulates completed occurrences as
an episode progresses (design section 7 lists this as a risk).  So this must be
re-run on the real training corpus, not assumed from the preview.

Usage::

    PYTHONPATH=<worktree>/src python scripts/hier/memory_token_budget.py \\
        --jsonl <path.jsonl> [--field model_target_text] [--subtask-max-len 128]

Exits non-zero when the P99 headroom ratio is below ``--min-headroom``, so it can
gate a launch script.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _load(path: str, limit: int | None) -> list[dict]:
    rows = []
    with open(path) as handle:
        for i, line in enumerate(handle):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="memory training samples JSONL")
    ap.add_argument("--field", default="model_target_text", help="field holding the memory text")
    ap.add_argument("--subtask-max-len", type=int, default=None, help="defaults to Pi05SubtaskConfig")
    ap.add_argument("--min-headroom", type=float, default=1.5, help="required subtask_max_len / P99 ratio")
    ap.add_argument("--max-rows", type=int, default=None, help="cap rows read (for a quick check)")
    ap.add_argument(
        "--scheme",
        default=None,
        choices=["plain_text", "reserved_slot"],
        help="label scheme; defaults to the module default (plain text)",
    )
    args = ap.parse_args()

    from openpi.models import memory_text as mem
    from openpi.models import tokenizer as tokenizer_mod
    from openpi.models.pi05_subtask_config import Pi05SubtaskConfig

    cfg = Pi05SubtaskConfig()
    limit = args.subtask_max_len or cfg.subtask_max_len
    scheme = mem.LabelScheme(args.scheme) if args.scheme else mem.DEFAULT_LABEL_SCHEME
    tok = tokenizer_mod.SubtaskTokenizer(prompt_max_len=cfg.max_token_len, subtask_max_len=limit)
    codec = tok.memory_codec(scheme)

    rows = _load(args.jsonl, args.max_rows)
    if not rows:
        print(f"FAIL: no rows read from {args.jsonl}", file=sys.stderr)
        return 2
    if args.field not in rows[0]:
        print(f"FAIL: field {args.field!r} not in record; keys={sorted(rows[0])}", file=sys.stderr)
        return 2

    lengths, occurrences, progress = [], [], []
    for row in rows:
        text = str(row[args.field]).split("Action Query:")[0].strip()
        # validate=False: this is a measurement, not a gate on text well-formedness.
        lengths.append(len(codec.encode(text, validate=False)) + 2)  # +BOS/EOS
        occurrences.append(text.splitlines()[0].count("(") if text else 0)
        dur = row.get("episode_duration")
        idx = row.get("frame_index")
        progress.append(idx / dur if dur else float("nan"))

    arr = np.asarray(lengths)
    p50, p90, p99 = (float(np.percentile(arr, p)) for p in (50, 90, 99))
    headroom = limit / p99 if p99 else float("inf")
    over = int((arr > limit).sum())

    print(f"rows={len(rows)}  field={args.field}  scheme={scheme.value}  subtask_max_len={limit}")
    print(f"  P50={p50:.1f}  P90={p90:.1f}  P99={p99:.1f}  max={int(arr.max())}")
    print(f"  P99 headroom = {headroom:.2f}x   over-limit = {over}/{len(arr)} ({100 * over / len(arr):.2f}%)")

    prog = np.asarray(progress, dtype=float)
    finite = np.isfinite(prog)
    if finite.sum() > 2:
        print(f"  corr(length, episode progress) = {float(np.corrcoef(arr[finite], prog[finite])[0, 1]):.3f}")
    occ = np.asarray(occurrences)
    if occ.max() > occ.min():
        print(f"  corr(length, #occurrences)     = {float(np.corrcoef(arr, occ)[0, 1]):.3f}")
        print("  length by #occurrences in the Memory line (this is what eats the headroom):")
        for k in sorted(set(occ.tolist())):
            sel = occ == k
            print(f"    occurrences={k:<3d} n={int(sel.sum()):<6d} mean={arr[sel].mean():6.1f} max={int(arr[sel].max()):4d}")

    if over:
        print(
            f"\nFAIL: {over} sample(s) exceed subtask_max_len={limit}. Truncation drops the trailing "
            "fields (Next skill / Next primitive) and corrupts the CE target.",
            file=sys.stderr,
        )
        return 1
    if headroom < args.min_headroom:
        print(
            f"\nFAIL: P99 headroom {headroom:.2f}x is below the required {args.min_headroom:.2f}x. "
            "Nothing overflows yet, but the Memory field grows with completed occurrences, so raise "
            "subtask_max_len or bound the completed-occurrence list before training.",
            file=sys.stderr,
        )
        return 1
    print(f"\nOK: P99 headroom {headroom:.2f}x >= {args.min_headroom:.2f}x and nothing exceeds the limit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
