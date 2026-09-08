#!/usr/bin/env bash
# Fail-closed single-node 2×H20 launcher for paired BF16 MoMA short runs.
set -u -o pipefail
[[ $# -eq 0 ]] || { echo "REFUSE positional_args" >&2; exit 2; }
CONFIG_NAME="${CONFIG_NAME:?}"; RUN_ROOT="${RUN_ROOT:?}"; REPO_ROOT="${REPO_ROOT:?}"; NUM_STEPS="${NUM_STEPS:?}"
DATA_ROOT="${OPENPI_BEHAVIOR_DATASET_ROOT:?}"; PIN="${MOMAVLA_PIN_COMMIT:?}"
NAS_PREFIX="/mnt/bn/behavior-data-hl/chenjunting/repo/moma_handoff_20260907/short_train_stage2/runs"
WEIGHT_DIR="/mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet/checkpoints/pi05_base_pytorch"
WEIGHT_SIZE=7233650408; WEIGHT_SHA=62cffa633517d0ad8672933c04ae2d8b4758630feab98de88efc692f9a4a0fad
V2_REAL=/mnt/bn/behavior-data-hl/chenjunting/data/2025-challenge-demos/derived/fixed_compact_memory_annotations_v2
V2_MD5=50726ae508ef5c48e0d48d2679b6cd64
PY=/mnt/bn/behavior-data-hl/chenjunting/miniconda3/envs/openpi-comet-nas/bin/python
OVERLAY=/mnt/bn/navigation-hl/mlx/users/chenjunting/h20_fastce/pyoverlay
case "$CONFIG_NAME" in pi05_moma_memory_b1k-k1-short|pi05_moma_memory_b1k-mix-c-short);; *) echo "REFUSE config" >&2; exit 2;; esac
[[ "$NUM_STEPS" == 10 || "$NUM_STEPS" == 100 ]] || { echo "REFUSE steps" >&2; exit 2; }
[[ "$PIN" =~ ^[0-9a-f]{40}$ ]] || { echo "REFUSE pin_format" >&2; exit 2; }
for bad in SHORT_LAUNCH_TEST_MODE SHORT_LAUNCH_DRY_RUN TRAIN_COMMAND LAUNCHER PREFLIGHT_TEST_MODE WEIGHT_PREFLIGHT_ENABLE KEEPALIVE_DISABLE KEEPALIVE_ON_SUCCESS EXPECTED_GPUS_PER_NODE STRICT_GPU_COUNT STOP_OCCUPIERS_FILE OCCUPY_RUNTIME_DIR OCCUPIER_PYTHON OCCUPIER_DRY_RUN; do [[ -z "${!bad:-}" ]] || { echo "REFUSE override=$bad" >&2; exit 2; }; done
[[ -n "${ARNOLD_WORKER_NUM:-}" && -n "${ARNOLD_WORKER_GPU:-}" ]] || { echo "REFUSE topology_unset" >&2; exit 2; }
[[ "$ARNOLD_WORKER_NUM" == 1 && "$ARNOLD_WORKER_GPU" == 2 ]] || { echo "REFUSE topology" >&2; exit 2; }
[[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -eq 2 ]] || { echo "REFUSE visible_gpus" >&2; exit 2; }
HEAD=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null); [[ "$HEAD" == "$PIN" ]] || { echo "REFUSE head" >&2; exit 2; }
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || { echo "REFUSE dirty" >&2; exit 2; }
[[ -d "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || { echo "REFUSE run_root_type" >&2; exit 2; }
RUN_REAL=$(realpath "$RUN_ROOT"); case "$RUN_REAL/" in "$NAS_PREFIX"/*) ;; *) echo "REFUSE run_root_prefix" >&2; exit 2;; esac
[[ -z "$(ls -A "$RUN_REAL")" ]] || { echo "REFUSE run_root_nonempty" >&2; exit 2; }
LINK="$DATA_ROOT/derived/fixed_compact_memory_annotations"
[[ "$(realpath "$LINK")" == "$V2_REAL" ]] || { echo "REFUSE data_realpath" >&2; exit 2; }
[[ "$(md5sum "$LINK/manifest.json"|awk '{print $1}')" == "$V2_MD5" ]] || { echo "REFUSE data_manifest" >&2; exit 2; }
[[ "$(stat -c %s "$WEIGHT_DIR/model.safetensors")" == "$WEIGHT_SIZE" ]] || { echo "REFUSE weight_size" >&2; exit 2; }
[[ "$(sha256sum "$WEIGHT_DIR/model.safetensors"|awk '{print $1}')" == "$WEIGHT_SHA" ]] || { echo "REFUSE weight_hash" >&2; exit 2; }
export PYTHONPATH="$OVERLAY:$REPO_ROOT/src" OPENPI_BEHAVIOR_DATASET_ROOT="$DATA_ROOT"
export OPENPI_LOSS_FINITE_CHECK=1 OPENPI_MAX_CONSECUTIVE_NONFINITE_LOSSES=1 OPENPI_MAX_CONSECUTIVE_SKIPPED_UPDATES=1 OPENPI_PERSISTENT_WORKERS=0
cat > "$RUN_REAL/launch_manifest.json" <<EOF
{"config":"$CONFIG_NAME","runtime_precision":"bfloat16","code_commit":"$HEAD","weight_size":$WEIGHT_SIZE,"weight_sha256":"$WEIGHT_SHA","dataset_realpath":"$V2_REAL","dataset_manifest_md5":"$V2_MD5","world_size":2,"num_steps":$NUM_STEPS,"warmup_steps":20,"decay_steps":200}
EOF
finish(){ rc=$1; phase=$2; printf '{"rc":%s,"phase":"%s","completed_at":"%s"}\n' "$rc" "$phase" "$(date -u +%FT%TZ)" > "$RUN_REAL/status.json"; occupy "$rc"; }
occupy(){ original_rc=$1; marker="__MOMA_SHORT_OCCUPY_${ARNOLD_TRIAL_ID:-manual}"; pids=(); for gpu in 0 1; do CUDA_VISIBLE_DEVICES=$gpu "$PY" -c "import torch,time; x=torch.ones((1024,1024),device='cuda'); print('$marker gpu=$gpu',flush=True); exec('while True:\n x=torch.mm(x,x); x/=x.abs().max().clamp_min(1); time.sleep(0.1)')" >>"$RUN_REAL/occupier_gpu${gpu}.log" 2>&1 & pids+=("$!"); done; printf '%s\n' "${pids[@]}" > "$RUN_REAL/occupiers.pid"; : > "$RUN_REAL/heartbeat"; while :; do date -u +%FT%TZ > "$RUN_REAL/heartbeat"; [[ -e "$RUN_REAL/STOP" ]] && { kill "${pids[@]}" 2>/dev/null; wait "${pids[@]}" 2>/dev/null; exit "$original_rc"; }; sleep 10; done; }
set +e
"$PY" "$REPO_ROOT/scripts/hier/preflight_weight_load.py" --config "$CONFIG_NAME" --weight-dir "$WEIGHT_DIR" --load-mode stream --expect-size "$WEIGHT_SIZE" --expect-hash "$WEIGHT_SHA" --require-openpi-under "$REPO_ROOT" --json "$RUN_REAL/weight_gate.json" >"$RUN_REAL/weight_gate.log" 2>&1
gate_rc=$?; set -e
pass_count=$(awk '$1=="WEIGHT_LOAD_GATE_VERDICT" && $2=="PASS" {n++} END {print n+0}' "$RUN_REAL/weight_gate.log")
[[ $gate_rc -eq 0 && $pass_count -eq 1 ]] || finish 2 weight_gate
set +e
torchrun --standalone --nnodes=1 --nproc-per-node=2 "$REPO_ROOT/scripts/train_accelerate.py" "$CONFIG_NAME" --pytorch-weight-path "$WEIGHT_DIR" --pytorch-training-precision bfloat16 --accelerate-mixed-precision bf16 --batch-size-per-gpu 1 --batch-size 32 --expected-global-batch 32 --gradient-accumulation-steps 16 --num-workers 1 --num-train-steps "$NUM_STEPS" --lr-schedule.warmup-steps 20 --lr-schedule.decay-steps 200 --log-interval 10 --save-interval "$NUM_STEPS" --no-wandb-enabled --checkpoint-base-dir "$RUN_REAL/checkpoints" --log-base-dir "$RUN_REAL/logs" --assets-base-dir "$RUN_REAL/assets" > >(tee "$RUN_REAL/console.log") 2>&1
rc=${PIPESTATUS[0]}; set -e
finish "$rc" trainer
