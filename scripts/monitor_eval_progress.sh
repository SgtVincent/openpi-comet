#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

RUN_DIR="${1:-}"
INTERVAL="${INTERVAL:-20}"
ONCE="${ONCE:-0}"
EXPECTED_TOTAL="${EXPECTED_TOTAL:-10}"
ERROR_PATTERN='Traceback \(most recent call last\)|Fatal Python error|Segmentation fault|core dumped|EnvironmentLocationNotFound|No device could be created|Failed to create any GPU devices|activeGpu index|FileNotFoundError|One or more evaluator processes failed'

if [[ -z "$RUN_DIR" ]]; then
  latest="$(find "$REPO_ROOT/eval_logs" -maxdepth 1 -mindepth 1 -type d -name 'parallel_*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | awk '{print $2}')"
  if [[ -z "$latest" ]]; then
    echo "[Error] No parallel eval run found under $REPO_ROOT/eval_logs"
    exit 1
  fi
  RUN_DIR="$latest"
fi

if [[ ! -d "$RUN_DIR" ]]; then
  echo "[Error] Run directory not found: $RUN_DIR"
  exit 1
fi

contains_any_error() {
  local file="$1"
  grep -Eqi "$ERROR_PATTERN" "$file"
}

is_done_log() {
  local file="$1"
  grep -Eqi "All evaluators completed successfully|Evaluation finished at step|Saved video to" "$file"
}

extract_assigned_total() {
  local dir="$1"
  local raw
  raw="$(grep -hoE 'eval_instance_ids=\[[0-9, ]+\]' "$dir"/eval_gpu*_p*.log 2>/dev/null || true)"
  if [[ -z "$raw" ]]; then
    echo "$EXPECTED_TOTAL"
    return
  fi
  echo "$raw" | sed -E 's/.*\[([^]]+)\].*/\1/' | tr ',' '\n' | tr -d ' ' | grep -E '^[0-9]+$' | sort -u | wc -l
}

while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  assigned_total="$(extract_assigned_total "$RUN_DIR")"

  echo "========================================="
  echo "Eval Progress Monitor"
  echo "========================================="
  echo "Time:           $ts"
  echo "Run directory:  $RUN_DIR"
  echo "Assigned IDs:   $assigned_total"
  echo "Expected total: $EXPECTED_TOTAL"
  echo

  mapfile -t eval_logs < <(find "$RUN_DIR" -maxdepth 1 -type f -name 'eval_gpu*_p*.log' | sort)
  mapfile -t server_logs < <(find "$RUN_DIR" -maxdepth 1 -type f -name 'server_gpu*_p*.log' | sort)

  if (( ${#eval_logs[@]} == 0 )); then
    echo "[Warn] No evaluator logs found yet."
  fi

  done_count=0
  fail_count=0
  running_count=0
  metrics_total=0

  echo "--- Evaluator Status ---"
  for log in "${eval_logs[@]:-}"; do
    base="$(basename "$log")"
    eval_dir_name="${base%.log}"
    eval_dir="$RUN_DIR/$eval_dir_name"
    metrics_count=0

    if [[ -d "$eval_dir/metrics" ]]; then
      metrics_count="$(find "$eval_dir/metrics" -type f -name '*.json' | wc -l)"
    fi

    metrics_total=$((metrics_total + metrics_count))

    if is_done_log "$log"; then
      status="DONE"
      done_count=$((done_count + 1))
    elif contains_any_error "$log"; then
      status="FAILED"
      fail_count=$((fail_count + 1))
    else
      status="RUNNING"
      running_count=$((running_count + 1))
    fi

    printf "%-32s status=%-8s metrics=%s\n" "$base" "$status" "$metrics_count"
  done

  echo
  echo "--- Server Status ---"
  for log in "${server_logs[@]:-}"; do
    base="$(basename "$log")"
    if contains_any_error "$log"; then
      status="FAILED"
    else
      status="OK"
    fi
    printf "%-32s status=%s\n" "$base" "$status"
  done

  echo
  echo "--- Summary ---"
  echo "Evaluator logs: ${#eval_logs[@]}"
  echo "Done:           $done_count"
  echo "Running:        $running_count"
  echo "Failed:         $fail_count"
  echo "Metrics total:  $metrics_total"

  if (( fail_count > 0 )); then
    echo
    echo "--- Recent Error Snippets ---"
    for log in "${eval_logs[@]:-}" "${server_logs[@]:-}"; do
      if [[ -f "$log" ]] && contains_any_error "$log"; then
        echo "[$(basename "$log")]"
        grep -Ein "$ERROR_PATTERN" "$log" | tail -n 5 || true
      fi
    done
  fi

  if [[ "$ONCE" == "1" ]]; then
    break
  fi

  echo
  echo "(refresh in ${INTERVAL}s; Ctrl+C to stop)"
  sleep "$INTERVAL"
  echo

done
