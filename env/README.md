# Python Environments

TerraScope 与 SegFormer 使用两套隔离的 Python 环境。根目录的 `pyproject.toml` 与 `uv.lock` 描述 TerraScope 推理环境；`services/segmentation/` 下的同名文件描述 SegFormer 推理环境。

环境安装与模型权重下载相互独立。模型文件始终位于 `models/`，本目录只负责 Python 运行环境。

两套环境不能合并：TerraScope 使用 Python 3.11、PyTorch 2.8.0 与 CUDA 12.8；本项目选用的 SegFormer 预编译 MMCV wheel 面向 Python 3.10、PyTorch 2.1.0 与 CUDA 12.1。

| 环境 | Python | PyTorch / torchvision | 主要框架 |
| --- | --- | --- | --- |
| TerraScope | 3.11 | 2.8.0 / 0.23.0，CUDA 12.8 | MS-SWIFT 4.2.2，Transformers 5.7.0 |
| SegFormer | 3.10 | 2.1.0 / 0.16.0，CUDA 12.1 | MMCV 2.1.0，MMEngine 0.10.7，MMSegmentation 1.2.2 |

表中为直接依赖；所有传递依赖的精确版本记录在两份 `uv.lock` 中。Conda 安装使用由锁文件机械导出的 `*.constraints.txt`，避免安装时间不同造成依赖漂移。

仓库提供的 uv 与 Conda 配置面向 Linux x86_64 和 NVIDIA GPU。其他平台需要替换 PyTorch 与 MMCV wheel，并重新验证锁文件。

## Install with uv

使用 [uv](https://docs.astral.sh/uv/) 从锁文件重建环境：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
bash ./env/install.sh uv
```

脚本将创建：

```text
.venv/                              # TerraScope
services/segmentation/.venv/        # SegFormer
```

并自动更新 `app/.env` 中的 `PYTHON_BIN` 与 `SEGFORMER_PYTHON`。其余本机配置保持不变。

## Install with Conda

没有 uv 时也可以使用 Conda。建议在 Linux x86_64 服务器上安装 [Miniforge](https://github.com/conda-forge/miniforge)，或按 [Conda 官方安装说明](https://docs.conda.io/projects/conda/en/stable/user-guide/install/)安装 Miniconda。以 Miniforge 为例：

```bash
curl -L -o /tmp/Miniforge3.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/Miniforge3.sh -b -p "${HOME}/miniforge3"
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
```

随后在仓库根目录运行：

```bash
bash ./env/install.sh conda
```

脚本会分别创建 `env/conda/terrascope/` 与 `env/conda/segmentation/`，从对应 CUDA wheel 源安装 PyTorch 和 MMCV，再依据 `env/requirements/` 中的直接依赖与锁定约束完成安装。安装结束前会执行 `pip check`；两套环境验证通过后才会写入 `app/.env`。

## Install from offline packages

预构建环境包的网盘链接见[项目主页](../README.md#3-python-environments)。下载后将文件放入：

```text
env/packages/terrascope-env.tar.gz
env/packages/segmentation-env.tar.gz
```

先验证文件完整性，再从仓库根目录安装：

```bash
cd env/packages
sha256sum -c SHA256SUMS
cd ../..
bash ./env/install.sh offline
```

脚本会将环境解压到 `env/terrascope/` 与 `env/segmentation/`，执行 `conda-unpack`，并将 `app/.env` 切换到对应解释器。离线环境包依赖操作系统、CPU 架构、GPU 驱动与 CUDA 版本，不能直接用于不兼容的平台。

离线包来自已验证的推理环境，可能包含锁文件之外的兼容工具。需要最小化、可审计的全新环境时，应优先使用 uv；Conda 的 constraints 由同一份 `uv.lock` 导出。

## Verify

无论使用 uv、Conda 还是离线包，启动前均应运行：

```bash
./start.sh doctor
```

该命令检查两套 Python 依赖、模型文件及校验值、前端构建产物和测试套件。
