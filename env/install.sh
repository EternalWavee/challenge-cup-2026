#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ROOT}/env"
PACKAGE_DIR="${ENV_DIR}/packages"
REQUIREMENTS_DIR="${ENV_DIR}/requirements"
CONDA_ENV_DIR="${ENV_DIR}/conda"
ENV_FILE="${ROOT}/app/.env"
MODE="${1:-uv}"

install_archive() {
  local name="$1"
  local archive="${PACKAGE_DIR}/${name}-env.tar.gz"
  local target="${ENV_DIR}/${name}"

  if [[ -x "${target}/bin/python" ]]; then
    echo "${name} 环境已存在，跳过：${target}"
    return
  fi
  if [[ -e "${target}" ]]; then
    echo "目标目录已存在但环境不完整，请先移走后重试：${target}" >&2
    exit 1
  fi
  [[ -f "${archive}" ]] || { echo "缺少环境包：${archive}" >&2; exit 1; }

  echo "正在解压 ${name} 环境……"
  mkdir -p "${target}"
  tar -xzf "${archive}" -C "${target}"
  if [[ -x "${target}/bin/conda-unpack" ]]; then
    PATH="${target}/bin:${PATH}" "${target}/bin/conda-unpack"
  fi
  [[ -x "${target}/bin/python" ]] || { echo "环境包中缺少 bin/python：${name}" >&2; exit 1; }
  "${target}/bin/python" --version
}

create_conda_env() {
  local target="$1"
  local python_version="$2"

  if [[ -x "${target}/bin/python" ]]; then
    echo "Conda 环境已存在，将复用并核验：${target}"
    return
  fi
  if [[ -e "${target}" ]]; then
    echo "目标目录已存在但环境不完整，请先移走后重试：${target}" >&2
    exit 1
  fi

  conda create --yes --prefix "${target}" "python=${python_version}" pip
}

install_with_conda() {
  local main_target="${CONDA_ENV_DIR}/terrascope"
  local seg_target="${CONDA_ENV_DIR}/segmentation"
  local main_python="${main_target}/bin/python"
  local seg_python="${seg_target}/bin/python"

  command -v conda >/dev/null 2>&1 || {
    echo "找不到 conda，请先安装并初始化 Miniforge 或 Miniconda。" >&2
    echo "安装说明：https://docs.conda.io/projects/conda/en/stable/user-guide/install/" >&2
    exit 2
  }

  create_conda_env "${main_target}" "3.11"
  create_conda_env "${seg_target}" "3.10"

  echo "正在安装 TerraScope CUDA 12.8 运行依赖……"
  "${main_python}" -m pip install --disable-pip-version-check \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchvision==0.23.0
  "${main_python}" -m pip install --disable-pip-version-check \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --constraint "${REQUIREMENTS_DIR}/terrascope.constraints.txt" \
    --requirement "${REQUIREMENTS_DIR}/terrascope.txt"
  "${main_python}" -m pip check

  echo "正在安装 SegFormer CUDA 12.1 运行依赖……"
  "${seg_python}" -m pip install --disable-pip-version-check \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.1.0 torchvision==0.16.0
  "${seg_python}" -m pip install --disable-pip-version-check \
    --no-index --no-deps \
    --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html \
    mmcv==2.1.0
  "${seg_python}" -m pip install --disable-pip-version-check \
    --constraint "${REQUIREMENTS_DIR}/segmentation.constraints.txt" \
    --requirement "${REQUIREMENTS_DIR}/segmentation.txt"
  "${seg_python}" -m pip check

  configure_python_paths \
    "./env/conda/terrascope/bin/python" \
    "./env/conda/segmentation/bin/python"
}

configure_python_paths() {
  local terrascope_python="$1"
  local segmentation_python="$2"
  local temporary

  if [[ ! -f "${ENV_FILE}" ]]; then
    cp "${ROOT}/app/.env.example" "${ENV_FILE}"
  fi
  temporary="$(mktemp "${ENV_FILE}.XXXXXX")"
  awk -v main_python="${terrascope_python}" -v seg_python="${segmentation_python}" '
    BEGIN { main_seen = 0; seg_seen = 0 }
    /^PYTHON_BIN=/ { print "PYTHON_BIN=" main_python; main_seen = 1; next }
    /^SEGFORMER_PYTHON=/ { print "SEGFORMER_PYTHON=" seg_python; seg_seen = 1; next }
    { print }
    END {
      if (!main_seen) print "PYTHON_BIN=" main_python
      if (!seg_seen) print "SEGFORMER_PYTHON=" seg_python
    }
  ' "${ENV_FILE}" >"${temporary}"
  mv "${temporary}" "${ENV_FILE}"
  echo "已更新配置：${ENV_FILE}"
}

case "${MODE}" in
  uv)
    command -v uv >/dev/null 2>&1 || {
      echo "找不到 uv，请先安装：https://docs.astral.sh/uv/" >&2
      exit 2
    }
    (cd "${ROOT}" && uv sync --frozen)
    uv sync --project "${ROOT}/services/segmentation" --frozen
    configure_python_paths "./.venv/bin/python" "./services/segmentation/.venv/bin/python"
    ;;
  conda)
    install_with_conda
    ;;
  offline)
    install_archive terrascope
    install_archive segmentation
    configure_python_paths "./env/terrascope/bin/python" "./env/segmentation/bin/python"
    ;;
  *)
    echo "用法：bash env/install.sh [uv|conda|offline]" >&2
    exit 2
    ;;
esac

echo "环境准备完成。请运行 ./start.sh doctor 检查后，再运行 ./start.sh 启动。"
