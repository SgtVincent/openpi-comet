#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); L="$ROOT/scripts/run_pi05_moma_memory_short_1x2.sh"; T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
# Rewrite only immutable constants in a private launcher copy; production has no test-mode branch.
LT="$T/launcher.sh"; cp "$L" "$LT"
R="$T/repo"; mkdir "$R"; git -C "$R" init -q; echo x >"$R/x"; git -C "$R" add x; git -C "$R" -c user.name=t -c user.email=t@t commit -qm x; P=$(git -C "$R" rev-parse HEAD)
W="$T/w"; mkdir "$W"; printf weight >"$W/model.safetensors"; WS=$(stat -c%s "$W/model.safetensors"); WH=$(sha256sum "$W/model.safetensors"|awk '{print $1}')
V="$T/v2"; mkdir -p "$V"; printf '{}' >"$V/manifest.json"; VM=$(md5sum "$V/manifest.json"|awk '{print $1}'); D="$T/data"; mkdir -p "$D/derived"; ln -s "$V" "$D/derived/fixed_compact_memory_annotations"
N="$T/nvidia-smi"; printf '#!/bin/sh\nprintf "0\\n1\\n"\n' >"$N"; chmod +x "$N"
python - "$LT" "$W" "$WS" "$WH" "$V" "$VM" "$T/runs" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1]); s=p.read_text(); vals=sys.argv[2:]
s=s.replace('/mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet/checkpoints/pi05_base_pytorch',vals[0]).replace('7233650408',vals[1]).replace('62cffa633517d0ad8672933c04ae2d8b4758630feab98de88efc692f9a4a0fad',vals[2]).replace('/mnt/bn/behavior-data-hl/chenjunting/data/2025-challenge-demos/derived/fixed_compact_memory_annotations_v2',vals[3]).replace('50726ae508ef5c48e0d48d2679b6cd64',vals[4]).replace('/mnt/bn/behavior-data-hl/chenjunting/repo/moma_handoff_20260907/short_train_stage2/runs',vals[5]).replace('nvidia-smi --query-gpu=index','"'+str(pathlib.Path(sys.argv[1]).parent/'nvidia-smi')+'" --query-gpu=index')
p.write_text(s)
PY
base=(env CONFIG_NAME=pi05_moma_memory_b1k-k1-short NUM_STEPS=10 REPO_ROOT="$R" OPENPI_BEHAVIOR_DATASET_ROOT="$D" MOMAVLA_PIN_COMMIT="$P" ARNOLD_WORKER_NUM=1 ARNOLD_WORKER_GPU=2)
mkdir -p "$T/runs/ok"; set +e; timeout 2 "${base[@]}" RUN_ROOT="$T/runs/ok" bash "$LT" >/tmp/mshort.$$ 2>&1; rc=$?; set -e; [[ $rc -eq 124 ]]; [[ -s "$T/runs/ok/launch_manifest.json" ]]; [[ -s "$T/runs/ok/heartbeat" || -s "$T/runs/ok/status.json" ]]
bad(){ want=$1; shift; set +e; "$@" >/tmp/mshort.bad.$$ 2>&1; rc=$?; set -e; [[ $rc -eq 2 ]]; grep -q "REFUSE $want" /tmp/mshort.bad.$$; }
mkdir "$T/runs/a"; bad config env CONFIG_NAME=bad NUM_STEPS=10 RUN_ROOT="$T/runs/a" REPO_ROOT="$R" OPENPI_BEHAVIOR_DATASET_ROOT="$D" MOMAVLA_PIN_COMMIT="$P" ARNOLD_WORKER_NUM=1 ARNOLD_WORKER_GPU=2 bash "$LT"
mkdir "$T/runs/b"; bad topology "${base[@]}" ARNOLD_WORKER_GPU=8 RUN_ROOT="$T/runs/b" bash "$LT"
mkdir "$T/runs/c"; bad pin_format "${base[@]}" MOMAVLA_PIN_COMMIT=bad RUN_ROOT="$T/runs/c" bash "$LT"
mkdir "$T/runs/d"; bad run_root_nonempty "${base[@]}" RUN_ROOT="$T/runs/d" bash -c 'echo x >"$RUN_ROOT/x"; exec bash "$1"' _ "$LT"
mkdir "$T/runs/e"; bad override "${base[@]}" KEEPALIVE_DISABLE=1 RUN_ROOT="$T/runs/e" bash "$LT"
mkdir "$T/runs/f"; bad positional_args "${base[@]}" RUN_ROOT="$T/runs/f" bash "$LT" extra
echo SHORT_LAUNCH_TESTS_PASS
