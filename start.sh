#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ROOT}/app/.env"
CHAT_SESSION="tianshu-chat"
SEG_SESSION="tianshu-segformer"
LOG_DIR="${ROOT}/runtime/logs"
RUN_ID="${RUN_ID:-$(date +%H%M%S)-$$}"
COMMAND="${1:-start}"
MODEL_MANIFEST="${MODEL_MANIFEST:-${ROOT}/models/integrity-manifest.json}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

CHAT_PORT="${PORT:-8000}"
SEG_PORT="${SEGFORMER_PORT:-8010}"
CHAT_HEALTH_URL="http://127.0.0.1:${CHAT_PORT}/api/health"
SEG_HEALTH_URL="http://127.0.0.1:${SEG_PORT}/health"

resolve_python() {
  local candidate="$1"
  local label="$2"
  if [[ "${candidate}" == */* ]]; then
    if [[ "${candidate}" != /* ]]; then
      candidate="${ROOT}/${candidate#./}"
    fi
    [[ -x "${candidate}" ]] || { echo "${label}不存在：${candidate}" >&2; return 1; }
    printf '%s\n' "${candidate}"
  else
    command -v "${candidate}" 2>/dev/null || { echo "找不到${label}：${candidate}" >&2; return 1; }
  fi
}

resolve_root_path() {
  local candidate="$1"
  if [[ "${candidate}" != /* ]]; then
    candidate="${ROOT}/${candidate#./}"
  fi
  printf '%s\n' "${candidate}"
}

require_file() {
  local path="$1"
  local label="$2"
  [[ -f "${path}" ]] || { echo "缺少${label}：${path}" >&2; return 1; }
}

require_min_size() {
  local path="$1"
  local minimum_bytes="$2"
  local label="$3"
  local actual_bytes
  require_file "${path}" "${label}"
  actual_bytes="$(wc -c <"${path}")"
  if (( actual_bytes < minimum_bytes )); then
    echo "${label}文件异常小：${path}（${actual_bytes} bytes）" >&2
    return 1
  fi
}

validate_scalar_config() {
  local value
  for value in "${CHAT_PORT}" "${SEG_PORT}"; do
    if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( value < 1 || value > 65535 )); then
      echo "端口必须是1-65535之间的整数：${value}" >&2
      return 1
    fi
  done
  if [[ "${CHAT_PORT}" == "${SEG_PORT}" ]]; then
    echo "聊天端口与分割端口不能相同：${CHAT_PORT}" >&2
    return 1
  fi
  if [[ "${DOWNSAMPLE_MODE:-16x}" != "4x" && "${DOWNSAMPLE_MODE:-16x}" != "16x" ]]; then
    echo "DOWNSAMPLE_MODE仅支持4x或16x。" >&2
    return 1
  fi
  if [[ ! "${MAX_SLICE_NUMS:-9}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_SLICE_NUMS必须是正整数。" >&2
    return 1
  fi
}

preflight() {
  local model_path adapter_path frontend_dist seg_config seg_checkpoint
  model_path="$(resolve_root_path "${MODEL_PATH:-models/terrascope}")"
  frontend_dist="${ROOT}/app/frontend/dist"

  validate_scalar_config

  require_file "${model_path}/config.json" "MiniCPM配置"
  if [[ -f "${model_path}/model.safetensors" ]]; then
    require_min_size "${model_path}/model.safetensors" 100000000 "MiniCPM权重"
  elif [[ ! -f "${model_path}/model.safetensors.index.json" ]]; then
    echo "缺少MiniCPM权重：${model_path}/model.safetensors（或分片索引）" >&2
    return 1
  fi
  if [[ -n "${MODEL_ADAPTER_PATH:-}" ]]; then
    adapter_path="$(resolve_root_path "${MODEL_ADAPTER_PATH}")"
    require_file "${adapter_path}/adapter_config.json" "LoRA配置"
    require_min_size "${adapter_path}/adapter_model.safetensors" 1000000 "LoRA权重"
  fi
  require_file "${frontend_dist}/index.html" "前端静态文件"

  seg_config="$(resolve_root_path "${SEGFORMER_CONFIG:-models/segmentation/config.py}")"
  seg_checkpoint="$(resolve_root_path "${SEGFORMER_CHECKPOINT:-models/segmentation/checkpoint.pth}")"
  require_file "${seg_config}" "SegFormer配置"
  require_min_size "${seg_checkpoint}" 1000000 "SegFormer权重"

  if [[ "${SEGMENTATION_ENDPOINT:-http://127.0.0.1:${SEG_PORT}/segment}" == http://127.0.0.1:* ]] &&
    [[ "${SEGMENTATION_ENDPOINT:-http://127.0.0.1:${SEG_PORT}/segment}" != "http://127.0.0.1:${SEG_PORT}/segment" ]]; then
    echo "SEGMENTATION_ENDPOINT与SEGFORMER_PORT不一致。" >&2
    return 1
  fi

  "${PYTHON_BIN}" "${ROOT}/scripts/model_integrity.py" verify \
    --root "${ROOT}" --manifest "${MODEL_MANIFEST}"
}

health() {
  local name="$1"
  local url="$2"
  if response="$(curl -fsS "${url}" 2>/dev/null)"; then
    echo "${name}：已就绪"
    return 0
  fi
  echo "${name}：未就绪"
  return 1
}

wait_for_health() {
  local name="$1"
  local url="$2"
  local attempts="$3"
  local expected="${4:-}"
  for _ in $(seq 1 "${attempts}"); do
    if response="$(curl -fsS "${url}" 2>/dev/null)"; then
      if [[ -z "${expected}" ]] || grep -Fq "${expected}" <<<"${response}"; then
        echo "${name}：已就绪"
        return 0
      fi
    fi
    sleep 1
  done
  echo "${name}启动超时，请查看 runtime/logs/。" >&2
  return 1
}

run_chat_server() {
  local app_root="${ROOT}/app"
  PYTHON_BIN="$(resolve_python "${PYTHON_BIN:-python3}" "MiniCPM Python")"
  DRIVER_LIB="${DRIVER_LIB:-}"
  if [[ -n "${DRIVER_LIB}" ]] && [[ -d "${DRIVER_LIB}" ]]; then
    export LD_LIBRARY_PATH="${DRIVER_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi

  export DOWNSAMPLE_MODE="${DOWNSAMPLE_MODE:-16x}"
  export MAX_SLICE_NUMS="${MAX_SLICE_NUMS:-9}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${ROOT}/runtime/pycache}"
  mkdir -p "${PYTHONPYCACHEPREFIX}"

  cd "${app_root}/backend"
  exec "${PYTHON_BIN}" -m uvicorn app.main:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --loop asyncio
}

if [[ "${COMMAND}" == "_chat" ]]; then
  run_chat_server
fi

if [[ "${COMMAND}" == "status" ]]; then
  health "聊天服务" "${CHAT_HEALTH_URL}" || true
  health "分割服务" "${SEG_HEALTH_URL}" || true
  exit 0
fi

if [[ "${COMMAND}" == "manifest" ]]; then
  [[ "$#" == "1" ]] || { echo "用法：bash start.sh manifest" >&2; exit 2; }
  PYTHON_BIN="$(resolve_python "${PYTHON_BIN:-python3}" "MiniCPM Python")"
  "${PYTHON_BIN}" "${ROOT}/scripts/model_integrity.py" generate \
    --root "${ROOT}" --manifest "${MODEL_MANIFEST}"
  exit 0
fi

if [[ "${COMMAND}" == "doctor" ]]; then
  [[ "$#" == "1" ]] || { echo "用法：bash start.sh doctor" >&2; exit 2; }
  command -v tmux >/dev/null 2>&1 || { echo "缺少tmux。" >&2; exit 1; }
  command -v curl >/dev/null 2>&1 || { echo "缺少curl。" >&2; exit 1; }
  mkdir -p "${ROOT}/runtime/pycache"
  PYTHON_BIN="$(resolve_python "${PYTHON_BIN:-python3}" "MiniCPM Python")"
  SEGFORMER_PYTHON="$(resolve_python "${SEGFORMER_PYTHON:-${PYTHON_BIN}}" "SegFormer Python")"
  preflight
  "${PYTHON_BIN}" -c 'import fastapi, PIL, swift, torch, transformers, uvicorn'
  "${SEGFORMER_PYTHON}" -c 'import mmcv, mmengine, mmseg, numpy, PIL, torch'
  bash -n "${ROOT}/start.sh" "${ROOT}/stop.sh" \
    "${ROOT}/env/install.sh" "${ROOT}/services/segmentation/start.sh"
  "${PYTHON_BIN}" -m py_compile \
    "${ROOT}/scripts/model_integrity.py" "${ROOT}/scripts/daily_log.py"
  (
    cd "${ROOT}/app/backend"
    "${PYTHON_BIN}" -c 'from app.config import Settings; Settings()'
    PYTHONPYCACHEPREFIX="${ROOT}/runtime/pycache" \
      "${PYTHON_BIN}" -m unittest discover -v
    PYTHONPYCACHEPREFIX="${ROOT}/runtime/pycache" \
      "${PYTHON_BIN}" -m compileall -q -x '(^|/)\._' \
        app "${ROOT}/services/segmentation"
  )
  (
    cd "${ROOT}/scripts"
    PYTHONPYCACHEPREFIX="${ROOT}/runtime/pycache" \
      "${PYTHON_BIN}" -m unittest discover -v
  )
  (
    cd "${ROOT}"
    PYTHONPATH="${ROOT}/services" \
      PYTHONPYCACHEPREFIX="${ROOT}/runtime/pycache" \
      "${SEGFORMER_PYTHON}" -m unittest discover -v \
        -s services/segmentation -p 'test_*.py'
  )
  "${PYTHON_BIN}" - "${ROOT}/app/frontend" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

frontend = Path(sys.argv[1])
index = frontend / "dist" / "index.html"
html = index.read_text(encoding="utf-8")
assets = re.findall(r'(?:src|href)="(/assets/[^"]+)"', html)
if not assets:
    raise SystemExit("前端index.html没有引用构建产物")
missing = [asset for asset in assets if not (index.parent / asset.lstrip("/")).is_file()]
if missing:
    raise SystemExit(f"前端构建产物缺失：{', '.join(missing)}")
manifest_path = index.parent / "build-info.json"
if not manifest_path.is_file():
    raise SystemExit("缺少前端构建清单：dist/build-info.json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
stale = []
for name, expected in manifest.get("sources", {}).items():
    path = frontend / name
    if not path.is_file():
        stale.append(name)
        continue
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        stale.append(name)
if stale:
    raise SystemExit("前端源码与dist不一致，请重新构建：" + ", ".join(stale))
PY
  echo "工程化检查通过"
  exit 0
fi

if [[ "${COMMAND}" != "start" ]]; then
  echo "用法：bash start.sh [start|status|doctor|manifest]" >&2
  exit 2
fi

command -v tmux >/dev/null 2>&1 || { echo "缺少tmux。" >&2; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "缺少curl。" >&2; exit 1; }
mkdir -p "${LOG_DIR}"

PYTHON_BIN="$(resolve_python "${PYTHON_BIN:-python3}" "MiniCPM Python")"
SEGFORMER_PYTHON="$(resolve_python "${SEGFORMER_PYTHON:-${PYTHON_BIN}}" "SegFormer Python")"
preflight >/dev/null
SEGFORMER_CONFIG="$(resolve_root_path "${SEGFORMER_CONFIG:-models/segmentation/config.py}")"
SEGFORMER_CHECKPOINT="$(resolve_root_path "${SEGFORMER_CHECKPOINT:-models/segmentation/checkpoint.pth}")"

echo "正在启动天枢·遥感，请稍候……"

if ! "${PYTHON_BIN}" -c 'import swift, uvicorn' >/dev/null 2>&1; then
  echo "MiniCPM Python缺少swift或uvicorn：${PYTHON_BIN}" >&2
  echo "请在app/.env中设置PYTHON_BIN。" >&2
  exit 2
fi
if ! "${SEGFORMER_PYTHON}" -c 'import mmcv, mmengine, mmseg, numpy, PIL, torch' >/dev/null 2>&1; then
  echo "SegFormer Python环境不完整：${SEGFORMER_PYTHON}" >&2
  exit 2
fi

seg_health="$(curl -fsS "${SEG_HEALTH_URL}" 2>/dev/null || true)"
if [[ -n "${seg_health}" ]] && ! grep -Fq '"expert": "segformer_b5_loveda"' <<<"${seg_health}"; then
  echo "端口${SEG_PORT}上的服务与SegFormer专家不匹配，请先停止该进程。" >&2
  exit 1
fi
if [[ -z "${seg_health}" ]]; then
  if ! tmux has-session -t "${SEG_SESSION}" 2>/dev/null; then
    tmux new-session -d -s "${SEG_SESSION}" \
      "cd '${ROOT}' && SEGFORMER_PYTHON='${SEGFORMER_PYTHON}' SEGFORMER_CONFIG='${SEGFORMER_CONFIG:-}' SEGFORMER_CHECKPOINT='${SEGFORMER_CHECKPOINT:-}' SEGFORMER_GPU_ID='${SEGFORMER_GPU_ID:-0}' SEGFORMER_PORT='${SEG_PORT}' bash services/segmentation/start.sh 2>&1 | '${PYTHON_BIN}' scripts/daily_log.py '${LOG_DIR}' segmentation.log '${RUN_ID}'"
  fi
  wait_for_health "分割服务" "${SEG_HEALTH_URL}" 120 '"expert": "segformer_b5_loveda"'
else
  health "分割服务" "${SEG_HEALTH_URL}"
fi

if ! health "聊天服务" "${CHAT_HEALTH_URL}" >/dev/null; then
  if ! tmux has-session -t "${CHAT_SESSION}" 2>/dev/null; then
    tmux new-session -d -s "${CHAT_SESSION}" \
      "cd '${ROOT}' && PYTHON_BIN='${PYTHON_BIN}' bash start.sh _chat 2>&1 | '${PYTHON_BIN}' scripts/daily_log.py '${LOG_DIR}' chat.log '${RUN_ID}'"
  fi
  wait_for_health "聊天服务" "${CHAT_HEALTH_URL}" 120 '"ok":true'
else
  health "聊天服务" "${CHAT_HEALTH_URL}"
fi

echo "访问地址：http://127.0.0.1:${CHAT_PORT}"
