#!/usr/bin/env bash
# Per-FILE pytest execution.
#
# Why: the container is capped at 32 GiB (/sys/fs/cgroup/memory.max) and a
# single pytest process that imports every heavy torch model module gets
# OOM-killed (exit 137).  Running one file per process keeps peak RSS bounded
# and makes the BEFORE/AFTER counts comparable and reproducible.
#
# Excluded (identically before and after) for reasons that pre-date this branch:
#   numba/NumPy incompat: scripts/test_b1k_openpi.py, tests/test_lean_b1k_stride12.py
#   openpi-client editable install still pinned to the MAIN tree ("import file
#   mismatch"): packages/openpi-client/src/openpi_client/{image_tools,msgpack_numpy}_test.py
set -uo pipefail
WT="/mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet-hier-moma"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:?usage: run_suite_isolated.sh <output-dir>}"
mkdir -p "${OUT}"
: > "${OUT}/per_file.tsv"

mapfile -t FILES < <(cd "${WT}" && find src scripts packages tests -name 'test_*.py' -o -name '*_test.py' \
  | grep -v -e '^scripts/test_b1k_openpi.py$' \
            -e '^tests/test_lean_b1k_stride12.py$' \
            -e 'openpi_client/image_tools_test.py$' \
            -e 'openpi_client/msgpack_numpy_test.py$' | sort)

for f in "${FILES[@]}"; do
  log="${OUT}/$(echo "$f" | tr '/' '_').log"
  ( cd "${WT}" && "${HERE}/run_tests.sh" -q -p no:cacheprovider "$f" ) > "${log}" 2>&1
  rc=$?
  printf 'ran %s (rc=%s)\n' "$f" "$rc"
done

# Counts are derived by a single shared parser so BEFORE and AFTER are comparable.
"/mnt/bn/behavior-data-hl/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python" \
  "${HERE}/parse_results.py" "${OUT}" | tee "${OUT}/TOTAL.txt"
