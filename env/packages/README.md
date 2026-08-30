# Offline Environment Archives

此目录存放 TerraScope 的两套可迁移 Python 环境。压缩包面向 **Linux x86_64 + NVIDIA GPU**，适用于无法访问 Python 软件源的部署环境。模型权重不放在本目录，下载方式见[项目主页](../../README.md#2-model-weights)。

```text
terrascope-env.tar.gz
segmentation-env.tar.gz
```

下载后先校验文件，再从项目根目录安装：

```bash
cd env/packages
sha256sum -c SHA256SUMS
cd ../..
bash ./env/install.sh offline
```

安装脚本会分别解压环境、执行 `conda-unpack` 修复安装前缀，并将 Python 路径写入 `app/.env`。两套环境中的 PyTorch 分别使用 CUDA 12.8 和 CUDA 12.1 构建，部署设备需要安装兼容 CUDA 12.x 的 NVIDIA 驱动。

压缩包与解压后的环境均不进入 Git；`SHA256SUMS` 保留在仓库中，用于验证网盘文件是否完整。
