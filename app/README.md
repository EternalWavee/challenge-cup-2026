# Chat Application

`app` 提供 TerraScope 的 Web 界面与统一 API。用户上传一至两张图像并直接提问，后端负责模型路由、定位结果解析、语义分割调用和会话记录。

模型权重与 Python 环境不由本目录管理。首次运行前，请分别完成项目主页中的 [Model Weights](../README.md#2-model-weights) 与 [Python Environments](../README.md#3-python-environments) 步骤。

## Request Flow

明确的单项分割请求直接进入 SegFormer-B5；其他问题由 TerraScope 回答。TerraScope 返回 `<seg>` 或 `<seg_all>` 时，同样会调用分割模型。普通视觉问答无需任务前缀。

界面提供三档推理模式：`fast` 使用 `16x/9`，`quality` 使用 `4x/9`，`precision` 对用户显式选择的单图定位请求执行 Top-5 两阶段精细定位。双图请求会自动回退到 `quality`，实际执行模式会随响应返回并显示在会话中。

## API

| Endpoint | Method | 作用 |
| --- | --- | --- |
| `/api/health` | `GET` | 服务与模型状态 |
| `/api/route` | `POST` | 查看路由结果，不执行推理 |
| `/api/chat` | `POST` | 完整多模态推理 |
| `/api/visualizations/{id}` | `GET` | 获取定位或分割结果图 |

```bash
curl -X POST http://127.0.0.1:8000/api/route \
  -F 'prompt=请分割图中汽车并且数一下有多少个' \
  -F 'image_count=1'

curl -X POST http://127.0.0.1:8000/api/chat \
  -F 'prompt=请定位图中的小型车辆' \
  -F 'inference_mode=precision' \
  -F 'images=@/path/to/image.png'
```

`/api/chat` 的 `inference_mode` 可取 `fast`、`quality` 或 `precision`，省略时默认为 `quality`。响应中的 `inference.requested_mode` 与 `inference.effective_mode` 分别表示用户选择和实际执行模式。

## Run

从仓库根目录启动：

```bash
./start.sh
./start.sh status
./stop.sh
```

配置位于 `app/.env`，相对路径均以仓库根目录为基准。已构建的前端静态文件位于 `frontend/dist/`，直接运行项目不需要 Node.js。

只有修改前端源码时才需要重新构建：

```bash
cd app/frontend
npm install
npm run build
```

运行产物统一写入仓库根目录的 `runtime/`，不进入版本控制：

- `conversations/<date>/<request-id>/` 保存请求元数据和输入图像；
- `visualizations/<request-id>.png` 供 API 访问，会话目录中的同名内容使用硬链接归档；
- `segmentation/<request-id>/` 保存 SegFormer 中间产物；
- `logs/<date>/<startup-id>/` 独立保存每次启动的聊天与分割日志。

左侧栏会话历史保存在浏览器 IndexedDB 中，因此不同域名或端口之间不共享前端历史；服务端请求归档不受此影响。
