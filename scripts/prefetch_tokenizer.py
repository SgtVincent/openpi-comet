from __future__ import annotations

import argparse
import os
import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--openpi-data-home",
        type=str,
        default=None,
        help="If set, overrides OPENPI_DATA_HOME for this run.",
    )
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.openpi_data_home is not None:
        os.environ["OPENPI_DATA_HOME"] = str(args.openpi_data_home)
    os.environ.setdefault("OPENPI_DOWNLOAD_TIMEOUT_S", str(float(args.timeout_s)))

    from openpi.shared import download

    path = download.maybe_download(
        "gs://big_vision/paligemma_tokenizer.model",
        force_download=bool(args.force),
        timeout_s=float(args.timeout_s),
        gs={"token": "anon"},
    )
    print(str(path))


if __name__ == "__main__":
    main()

