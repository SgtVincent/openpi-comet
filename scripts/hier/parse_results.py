#!/usr/bin/env python3
"""Re-derive pytest counts from per-file logs.

Written as a separate parser because the first inline shell version required a
non-digit before the count and therefore scored every log whose summary line
STARTS with the number (e.g. "22 passed in 3.15s") as zero.  Deriving the
before/after numbers with one parser keeps them comparable.
"""
import re
import sys
from pathlib import Path

SUMMARY = re.compile(r"^=*\s*(?:\d+ (?:passed|failed|error|skipped|xfailed|deselected|warning)s?[,\s].*|.*\bin \d+\.\d+s.*)$")
FIELD = re.compile(r"(\d+) (passed|failed|errors?|skipped)")


def parse(log: Path):
    text = log.read_text(errors="replace")
    best = {}
    for line in text.splitlines():
        if " in " not in line and "passed" not in line and "failed" not in line and "error" not in line:
            continue
        fields = dict((m.group(2), int(m.group(1))) for m in FIELD.finditer(line))
        if fields:
            best = fields  # last summary line wins
    norm = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
    for k, v in best.items():
        norm["error" if k.startswith("error") else k] = v
    return norm


def main(outdir: str):
    d = Path(outdir)
    rows = []
    for log in sorted(d.glob("*.log")):
        rows.append((log.name, parse(log)))
    tot = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
    for name, r in rows:
        for k in tot:
            tot[k] += r[k]
        if r["failed"] or r["error"]:
            print(f"  NONZERO {name}: {r}")
    print(f"FILES={len(rows)} passed={tot['passed']} failed={tot['failed']} "
          f"errors={tot['error']} skipped={tot['skipped']}")


if __name__ == "__main__":
    main(sys.argv[1])
