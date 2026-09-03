#!/usr/bin/env bash
# Reproducible regression-suite definition used for the BEFORE/AFTER comparison.
#
# Four modules are excluded because they fail at COLLECTION time for reasons
# that pre-date this branch and are unrelated to it.  They are excluded
# identically in the before and after runs so the counts are comparable.
#
#   numba/NumPy incompatibility ("Numba needs NumPy 2.3 or less. Got NumPy 2.4"):
#     - scripts/test_b1k_openpi.py
#     - tests/test_lean_b1k_stride12.py
#   openpi-client is a SECOND editable install still pinned to the MAIN tree, so
#   pytest reports "import file mismatch" against openpi-comet/packages/...:
#     - packages/openpi-client/src/openpi_client/image_tools_test.py
#     - packages/openpi-client/src/openpi_client/msgpack_numpy_test.py
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/run_tests.sh" \
  --ignore=scripts/test_b1k_openpi.py \
  --ignore=tests/test_lean_b1k_stride12.py \
  --ignore=packages/openpi-client/src/openpi_client/image_tools_test.py \
  --ignore=packages/openpi-client/src/openpi_client/msgpack_numpy_test.py \
  "$@"
