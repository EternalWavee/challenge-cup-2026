# Segmentation Service

本目录包含 TerraScope Segmentation 的推理适配器与 HTTP 服务。模型权重、配置和模型卡位于 [`models/segmentation/`](../../models/segmentation/)。

服务默认监听 `127.0.0.1:8010`，由仓库根目录的统一入口管理：

```bash
./start.sh doctor
./start.sh
```

## API

| Endpoint | Method | Description |
| --- | --- | --- |
| `/health` | `GET` | 返回模型、设备和类别状态 |
| `/segment` | `POST` | 执行指定类别或全类别语义分割 |

`/segment` 接受本机图片路径、`targeted` 或 `all_classes` 模式及目标类别列表。输出包括标签图、彩色掩膜、叠加图、类别图例和面积占比。

模型可通过 ModelScope 或离线镜像下载到 `models/segmentation/`；两种环境安装方式见 [`env/README.md`](../../env/README.md)。设备、端口和模型路径通过 `app/.env` 配置。该服务默认只监听本机地址，由主 API 调用。
