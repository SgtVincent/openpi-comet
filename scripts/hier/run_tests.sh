#!/usr/bin/env bash
# Wrapper that PINS openpi to THIS worktree.
# openpi is an editable install hardcoded to the MAIN tree; without this
# PYTHONPATH override every test silently exercises the main tree's old code.
set -uo pipefail
WT="/mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet-hier-moma"
PY="/mnt/bn/behavior-data-hl/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python"
export PYTHONPATH="${WT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false

# Fail loudly (not silently) if openpi resolved to any other tree.
RESOLVED="$("${PY}" -c 'import openpi;print(openpi.__file__)' 2>/dev/null)"
EXPECTED="${WT}/src/openpi/__init__.py"
echo "OPENPI_RESOLVED=${RESOLVED}"
if [ "${RESOLVED}" != "${EXPECTED}" ]; then
  echo "FATAL: openpi resolved to '${RESOLVED}', expected '${EXPECTED}'" >&2
  exit 97
fi
exec "${PY}" -m pytest "$@"
