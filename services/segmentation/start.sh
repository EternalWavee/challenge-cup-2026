#!/usr/bin/env bash
set -euo pipefail

FINAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEGFORMER_PYTHON="${SEGFORMER_PYTHON:-python3}"
DEFAULT_DEVICE="cuda:0"
CUDA_DEVICES="${SEGFORMER_GPU_ID:-0}"

if [[ "${SEGFORMER_PYTHON}" == */* ]]; then
  [[ -x "${SEGFORMER_PYTHON}" ]] || { echo "SegFormer Python不存在：${SEGFORMER_PYTHON}" >&2; exit 2; }
else
  command -v "${SEGFORMER_PYTHON}" >/dev/null 2>&1 || { echo "找不到SegFormer Python：${SEGFORMER_PYTHON}" >&2; exit 2; }
fi

export PYTHONPATH="${FINAL_ROOT}/services${PYTHONPATH:+:${PYTHONPATH}}"
export SEGFORMER_CONFIG="${SEGFORMER_CONFIG:-${FINAL_ROOT}/models/segmentation/config.py}"
export SEGFORMER_CHECKPOINT="${SEGFORMER_CHECKPOINT:-${FINAL_ROOT}/models/segmentation/checkpoint.pth}"
export SEGFORMER_OUTPUT_ROOT="${SEGFORMER_OUTPUT_ROOT:-${FINAL_ROOT}/runtime/segmentation}"
export SEGFORMER_DEVICE="${SEGFORMER_DEVICE:-${DEFAULT_DEVICE}}"

[[ -f "${SEGFORMER_CONFIG}" ]] || { echo "缺少配置：${SEGFORMER_CONFIG}" >&2; exit 2; }
[[ -f "${SEGFORMER_CHECKPOINT}" ]] || { echo "缺少权重：${SEGFORMER_CHECKPOINT}" >&2; exit 2; }

cd "${FINAL_ROOT}"
exec env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  SEGFORMER_HOST=127.0.0.1 \
  SEGFORMER_PORT="${SEGFORMER_PORT:-8010}" \
  "${SEGFORMER_PYTHON}" -m segmentation.service
