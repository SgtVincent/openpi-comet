import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Hit:
    line_no: int
    kind: str
    line: str


DEFAULT_PATTERNS = [
    ("py_traceback", r"Traceback \(most recent call last\):"),
    ("kit_error", r"\[Error\]"),
    ("py_stderr", r"\[py stderr\]"),
    ("segfault", r"Fatal Python error: Segmentation fault|Segmentation fault|Signal 11|SIGSEGV"),
    ("driver_rtx", r"verifyDriverVersion|unsupported NVIDIA graphics driver|HydraEngine rtx failed creating scene renderer"),
    ("display", r"failed to open the default display|carb\.windowing-glfw\.plugin"),
    ("extension", r"No module named 'omni\.[^']+'|Failed parsing execute string|extension.*failed|Failed to startup plugin"),
    ("rendering", r"viewer_camera|cam\.get_obs|syntheticdata|Replicator|bbox_2d|bbox_3d|Image\.fromarray"),
    ("dtype_bridge", r"expected np\.ndarray|Cannot interpret 'dtype|Cannot handle this data type|from_numpy|apply_transform"),
    ("shutdown", r"unloading all plugins|removePath called on non-existent path|Could not find category .* for removal"),
    ("scene", r"Simulation App Startup Complete|Imported scene 0\.|Simulation App Starting"),
    ("exception", r"(TypeError|RuntimeError|ValueError|AssertionError|ModuleNotFoundError|AttributeError):"),
]


def _read_lines(source: str, since_line: int) -> list[str]:
    if source == "-":
        raw = sys.stdin.read().splitlines()
        return [line for idx, line in enumerate(raw, start=1) if idx >= since_line]
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(source)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for idx, line in enumerate(f, start=1) if idx >= since_line]


def _compile_patterns(patterns: list[tuple[str, str]]) -> list[tuple[str, re.Pattern[str]]]:
    return [(kind, re.compile(pattern)) for kind, pattern in patterns]


def _terminal_exception(line: str) -> bool:
    return bool(
        re.search(
            r"(TypeError|RuntimeError|ValueError|AssertionError|ModuleNotFoundError|AttributeError):|Fatal Python error:",
            line,
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", type=str, help="Path to log file, or '-' to read from stdin")
    ap.add_argument("--since-line", type=int, default=1)
    ap.add_argument("--context", type=int, default=40)
    ap.add_argument("--topk", type=int, default=60)
    ap.add_argument("--patterns", type=str, default="")
    args = ap.parse_args()

    patterns = list(DEFAULT_PATTERNS)
    if args.patterns.strip():
        patterns.append(("custom", args.patterns.strip()))
    compiled = _compile_patterns(patterns)

    try:
        lines = _read_lines(args.log_path, args.since_line)
    except FileNotFoundError:
        print(f"ERROR: log not found: {args.log_path}", file=sys.stderr)
        return 2

    if not lines:
        print("No lines read.")
        return 0

    hits: list[Hit] = []
    counts = Counter()
    terminal_exceptions: list[tuple[int, str]] = []
    line_index = {idx + args.since_line: line for idx, line in enumerate(lines)}

    for idx0, line in enumerate(lines):
        line_no = idx0 + args.since_line
        if _terminal_exception(line):
            terminal_exceptions.append((line_no, line))
        for kind, pattern in compiled:
            if pattern.search(line):
                hits.append(Hit(line_no=line_no, kind=kind, line=line))
                counts[kind] += 1
                break

    print(f"Lines analyzed: {len(lines)} (since line {args.since_line})")
    if counts:
        print("Hit counts:")
        for kind, count in counts.most_common():
            print(f"  - {kind}: {count}")
        print()

    if terminal_exceptions:
        last_line_no, last_exc = terminal_exceptions[-1]
        print(f"Last terminal exception: L{last_line_no}: {last_exc}")
        print()

    if not hits:
        print("No known patterns matched.")
        return 0

    topk = min(args.topk, len(hits))
    print(f"Top {topk} hits with +/- {args.context} lines context:")
    print()
    for hit in hits[-topk:]:
        start = max(args.since_line, hit.line_no - args.context)
        end = min(args.since_line + len(lines) - 1, hit.line_no + args.context)
        print(f"=== [{hit.kind}] L{hit.line_no} (context L{start}-L{end}) ===")
        for line_no in range(start, end + 1):
            prefix = ">>" if line_no == hit.line_no else "  "
            print(f"{prefix} {line_no}: {line_index.get(line_no, '')}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
