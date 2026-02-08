import argparse
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Hit:
    line_no: int
    line: str
    kind: str


DEFAULT_PATTERNS = [
    ("fatal", r"SIGSEGV|Signal 11|Segmentation fault|exitcode\s*:\s*-11|ChildFailedError"),
    ("cuda", r"CUDA error|CUDNN|cuDNN|illegal memory access|device-side assert|cublas|curand"),
    ("nccl", r"NCCL\s+(WARN|ERROR)|ProcessGroupNCCL|ncclComm|Connection closed"),
    ("dataloader", r"DataLoader|worker.*(exited|killed)|BrokenPipeError|resource_tracker: There appear to be"),
    ("oom", r"out of memory|CUDA out of memory|Killed process|OOM"),
    ("nan_inf", r"\bnan\b|\binf\b|overflow|underflow"),
    ("timeout_dump", r"Timeout\s+\(0:30:00\)!"),
    ("exception", r"Traceback \(most recent call last\):|RuntimeError:|ValueError:|AssertionError:"),
]


def _read_lines(path: Path, since_line: int) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            if i < since_line:
                continue
            lines.append(line.rstrip("\n"))
    return lines


def _compile_patterns(patterns: list[tuple[str, str]]) -> list[tuple[str, re.Pattern[str]]]:
    compiled = []
    for kind, pat in patterns:
        compiled.append((kind, re.compile(pat)))
    return compiled


def _guess_rank(line: str) -> str | None:
    m = re.search(r"\[(rank\d+)\]", line)
    if m:
        return m.group(1)
    m = re.search(r"^\[(default\d+)\]:", line)
    if m:
        return m.group(1)
    return None


def _extract_step(line: str) -> int | None:
    m = re.search(r"\bstep=(\d+)\b", line)
    if m:
        return int(m.group(1))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", type=str)
    ap.add_argument("--since-line", type=int, default=1)
    ap.add_argument("--context", type=int, default=60)
    ap.add_argument("--topk", type=int, default=80)
    ap.add_argument(
        "--patterns",
        type=str,
        default="",
        help="Optional additional regex patterns joined by |. Example: 'Downloading data|snapshot_download'",
    )
    args = ap.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"ERROR: log not found: {log_path}", file=sys.stderr)
        return 2

    patterns = list(DEFAULT_PATTERNS)
    if args.patterns.strip():
        patterns.append(("custom", args.patterns.strip()))
    compiled = _compile_patterns(patterns)

    lines = _read_lines(log_path, args.since_line)
    if not lines:
        print("No lines read (empty file or since-line beyond EOF).")
        return 0

    hits: list[Hit] = []
    by_kind = Counter()
    by_rank = Counter()
    last_step: int | None = None
    step_samples: dict[str, int] = defaultdict(int)

    for idx0, line in enumerate(lines):
        line_no = idx0 + args.since_line
        step = _extract_step(line)
        if step is not None:
            last_step = step
            r = _guess_rank(line) or "unknown"
            step_samples[r] = max(step_samples[r], step)

        for kind, pat in compiled:
            if pat.search(line):
                hits.append(Hit(line_no=line_no, line=line, kind=kind))
                by_kind[kind] += 1
                r = _guess_rank(line)
                if r:
                    by_rank[r] += 1
                break

    print(f"Log: {log_path}")
    print(f"Lines analyzed: {len(lines)} (since line {args.since_line})")
    if last_step is not None:
        print(f"Last observed step: {last_step}")
    if step_samples:
        top_steps = ", ".join(f"{k}={v}" for k, v in sorted(step_samples.items()))
        print(f"Max step by stream: {top_steps}")
    print()

    if by_kind:
        print("Hit counts by kind:")
        for k, v in by_kind.most_common():
            print(f"  - {k}: {v}")
        print()

    if by_rank:
        print("Hit counts by rank/stream (best effort):")
        for k, v in by_rank.most_common(16):
            print(f"  - {k}: {v}")
        print()

    if not hits:
        print("No hits matched. Try adding --patterns or reducing --since-line.")
        return 0

    topk = min(args.topk, len(hits))
    print(f"Top {topk} hits (with +/- {args.context} lines context):")
    print()

    line_index = {i + args.since_line: line for i, line in enumerate(lines)}
    for h in hits[-topk:]:
        start = max(args.since_line, h.line_no - args.context)
        end = min(args.since_line + len(lines) - 1, h.line_no + args.context)
        print(f"=== [{h.kind}] L{h.line_no} (context L{start}-L{end}) ===")
        for ln in range(start, end + 1):
            prefix = ">>" if ln == h.line_no else "  "
            text = line_index.get(ln, "")
            print(f"{prefix} {ln}: {text}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

