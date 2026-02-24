#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/check_empty_checkpoints.sh [--root <checkpoints_dir>] [--remove] [--min-age <duration>]

默认只列出“疑似失败”的 exp_name（其 checkpoints 目录下看不到任何 ckpt 标记文件），不会删除。

判定为“有 ckpt”的条件（满足其一即可）：
  - 目录内（<=4 层）存在 _CHECKPOINT_METADATA
  - 目录内（<=4 层）存在 params/_METADATA
  - 目录内（<=4 层）存在 *.safetensors / *.pt / *.pth / *.ckpt / *.bin

--remove 会同时删除：
  <root>/<exp_name>
  <root>/console_logs/<exp_name>   (如果存在)
  <root>/console_logs/<exp_name>.log / <exp_name>.node*.log (如果存在)
  <root>/torchrun_logs/<exp_name>  (如果存在)

--min-age <duration> 如果 exp 目录“最近修改时间”距离当前小于该值，则跳过（不列出，也不删除）。
  duration 支持：Ns / Nm / Nh / Nd 或纯数字(秒)。例如：3d / 72h / 0
  默认：3d
EOF
}

ROOT="./checkpoints"
REMOVE=0
MIN_AGE="3d"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="${2:-}"
      shift 2
      ;;
    --remove)
      REMOVE=1
      shift
      ;;
    --min-age)
      MIN_AGE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${ROOT}" ]]; then
  echo "--root requires a value" >&2
  exit 2
fi

if [[ ! -d "${ROOT}" ]]; then
  echo "Checkpoints dir not found: ${ROOT}" >&2
  exit 1
fi

ROOT_ABS="$(cd "${ROOT}" && pwd)"

parse_duration_seconds() {
  local v="$1"
  if [[ -z "${v}" ]]; then
    echo "empty duration" >&2
    return 2
  fi
  if [[ "${v}" =~ ^[0-9]+$ ]]; then
    printf "%s\n" "${v}"
    return 0
  fi
  if [[ "${v}" =~ ^([0-9]+)([smhdSMHD])$ ]]; then
    local n="${BASH_REMATCH[1]}"
    local u="${BASH_REMATCH[2]}"
    case "${u}" in
      s|S) printf "%s\n" "${n}" ;;
      m|M) printf "%s\n" "$((n * 60))" ;;
      h|H) printf "%s\n" "$((n * 3600))" ;;
      d|D) printf "%s\n" "$((n * 86400))" ;;
    esac
    return 0
  fi
  echo "invalid duration: ${v} (expected like 3d/72h/0)" >&2
  return 2
}

MIN_AGE_S="$(parse_duration_seconds "${MIN_AGE}")"
NOW_S="$(date +%s)"

has_ckpt() {
  local d="$1"
  local hit
  hit="$(
    find "${d}" -maxdepth 4 -type f \( \
      -name '_CHECKPOINT_METADATA' -o \
      -path '*/params/_METADATA' -o \
      -name 'model.safetensors' -o \
      -name '*.safetensors' -o \
      -name '*.pt' -o \
      -name '*.pth' -o \
      -name '*.ckpt' -o \
      -name '*.bin' \
    \) -print -quit 2>/dev/null || true
  )"
  [[ -n "${hit}" ]]
}

declare -A candidates=()

add_candidate() {
  local name="$1"
  [[ -z "${name}" ]] && return 0
  case "${name}" in
    _exp_name_sync|torchrun_logs|console_logs|hf_home|hf_datasets_cache|openpi_comet)
      return 0
      ;;
  esac
  candidates["${name}"]=1
}

newest_mtime_for_exp() {
  local exp_name="$1"
  local newest_mtime=0
  local p
  local p_mtime

  for p in "${ROOT_ABS}/${exp_name}" "${ROOT_ABS}/torchrun_logs/${exp_name}" "${ROOT_ABS}/console_logs/${exp_name}"; do
    if [[ -e "${p}" ]]; then
      p_mtime="$(stat -c %Y -- "${p}" 2>/dev/null || true)"
      if [[ -n "${p_mtime}" && "${p_mtime}" =~ ^[0-9]+$ && "${p_mtime}" -gt "${newest_mtime}" ]]; then
        newest_mtime="${p_mtime}"
      fi
    fi
  done

  if [[ -d "${ROOT_ABS}/console_logs" ]]; then
    while IFS= read -r -d '' p; do
      p_mtime="$(stat -c %Y -- "${p}" 2>/dev/null || true)"
      if [[ -n "${p_mtime}" && "${p_mtime}" =~ ^[0-9]+$ && "${p_mtime}" -gt "${newest_mtime}" ]]; then
        newest_mtime="${p_mtime}"
      fi
    done < <(
      find "${ROOT_ABS}/console_logs" -maxdepth 1 -type f \( \
        -name "${exp_name}.log" -o \
        -name "${exp_name}.node*.log" \
      \) -print0 2>/dev/null || true
    )
  fi

  printf "%s\n" "${newest_mtime}"
}

remove_exp_artifacts() {
  local exp_name="$1"
  local target

  for target in "${ROOT_ABS}/${exp_name}" "${ROOT_ABS}/torchrun_logs/${exp_name}" "${ROOT_ABS}/console_logs/${exp_name}"; do
    if [[ -e "${target}" ]]; then
      rm -rf -- "${target}"
      echo "removed: ${target}"
    fi
  done

  if [[ -d "${ROOT_ABS}/console_logs" ]]; then
    while IFS= read -r -d '' target; do
      rm -f -- "${target}"
      echo "removed: ${target}"
    done < <(
      find "${ROOT_ABS}/console_logs" -maxdepth 1 -type f \( \
        -name "${exp_name}.log" -o \
        -name "${exp_name}.node*.log" \
      \) -print0 2>/dev/null || true
    )
  fi
}

while IFS= read -r -d '' exp_dir; do
  add_candidate "$(basename "${exp_dir}")"
done < <(find "${ROOT_ABS}" -mindepth 1 -maxdepth 1 -type d -print0)

if [[ -d "${ROOT_ABS}/torchrun_logs" ]]; then
  while IFS= read -r -d '' d; do
    add_candidate "$(basename "${d}")"
  done < <(find "${ROOT_ABS}/torchrun_logs" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null || true)
fi

if [[ -d "${ROOT_ABS}/console_logs" ]]; then
  while IFS= read -r -d '' d; do
    add_candidate "$(basename "${d}")"
  done < <(find "${ROOT_ABS}/console_logs" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null || true)

  while IFS= read -r -d '' f; do
    name="$(basename "${f}")"
    if [[ "${name}" =~ ^(.+)\.node[0-9]+\.log$ ]]; then
      add_candidate "${BASH_REMATCH[1]}"
    elif [[ "${name}" == *.log ]]; then
      add_candidate "${name%.log}"
    fi
  done < <(find "${ROOT_ABS}/console_logs" -maxdepth 1 -type f -name '*.log' -print0 2>/dev/null || true)
fi

failed=()

for exp_name in "${!candidates[@]}"; do
  if [[ "${exp_name}" == *"/"* || "${exp_name}" == *".."* ]]; then
    continue
  fi

  exp_dir="${ROOT_ABS}/${exp_name}"
  if [[ -d "${exp_dir}" ]]; then
    if has_ckpt "${exp_dir}"; then
      continue
    fi
  fi

  if [[ "${MIN_AGE_S}" -gt 0 ]]; then
    newest_mtime="$(newest_mtime_for_exp "${exp_name}")"
    if [[ -n "${newest_mtime}" && "${newest_mtime}" =~ ^[0-9]+$ && "${newest_mtime}" -gt 0 ]]; then
      age_s="$((NOW_S - newest_mtime))"
      if [[ "${age_s}" -lt "${MIN_AGE_S}" ]]; then
        continue
      fi
    fi
  fi

  failed+=("${exp_name}")
done

if ((${#failed[@]} == 0)); then
  echo "No empty checkpoints found under: ${ROOT_ABS}"
  exit 0
fi

printf "%s\n" "${failed[@]}" | sort
echo "count: ${#failed[@]}"

if [[ "${REMOVE}" == "1" ]]; then
  for exp_name in "${failed[@]}"; do
    if [[ "${exp_name}" == *"/"* || "${exp_name}" == *".."* ]]; then
      echo "skip suspicious exp_name: ${exp_name}" >&2
      continue
    fi
    remove_exp_artifacts "${exp_name}"
  done
fi
