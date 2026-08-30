from __future__ import annotations

import asyncio
import io
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from .config import Settings
from .expert_router import (
    TARGET_LABELS,
    call_segmentation_expert,
    route_segmentation,
    route_segmentation_token,
    route_tasks,
)
from .fault_log import record_fault
from .model_service import (
    MODE_LABELS,
    GroundingBox,
    InferenceMode,
    ModelService,
    extract_response_grounding,
    normalize_inference_mode,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("remote-sensing-chat")

settings = Settings()
model_service = ModelService(settings)


def inference_payload(
    requested: InferenceMode,
    effective: InferenceMode | str,
    details: dict | None = None,
) -> dict:
    if isinstance(effective, InferenceMode):
        effective_value = effective.value
        label = MODE_LABELS[effective]
    else:
        effective_value = effective
        label = "语义分割" if effective == "segmentation" else effective
    return {
        "requested_mode": requested.value,
        "effective_mode": effective_value,
        "label": label,
        "details": details or {},
    }


def log_fault(
    *,
    component: str,
    stage: str,
    error: BaseException,
    request_id: str | None = None,
    recovered: bool = False,
) -> None:
    try:
        record_fault(
            settings.fault_log_dir,
            component=component,
            stage=stage,
            error=error,
            request_id=request_id,
            recovered=recovered,
        )
    except Exception:
        logger.exception("Failed to write structured fault event")


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_tasks: list[asyncio.Task] = []
    if settings.eager_load:
        load_tasks = [asyncio.create_task(model_service.load())]

        def report_failure(task: asyncio.Task) -> None:
            if task.cancelled():
                return
            error = task.exception()
            if error:
                logger.error("Model loading failed: %s", error)
                log_fault(component="minicpm", stage="load", error=error)

        for task in load_tasks:
            task.add_done_callback(report_failure)
    yield
    for task in load_tasks:
        if not task.done():
            task.cancel()


app = FastAPI(
    title="天枢·遥感",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": model_service.status == "ready",
        **model_service.public_state(),
    }


@app.get("/api/config")
async def config() -> dict:
    return {
        "max_images": settings.max_images,
        "max_image_mb": settings.max_image_bytes // (1024 * 1024),
        "max_prompt_chars": settings.max_prompt_chars,
        "supports": [
            "grounding",
            "scene_classification",
            "semantic_segmentation",
            "change_detection",
            "object_counting",
            "image_captioning",
        ],
        "inference_modes": [
            {
                "id": InferenceMode.FAST.value,
                "label": MODE_LABELS[InferenceMode.FAST],
                "downsample_mode": "16x",
                "max_slice_nums": 9,
            },
            {
                "id": InferenceMode.QUALITY.value,
                "label": MODE_LABELS[InferenceMode.QUALITY],
                "downsample_mode": "4x",
                "max_slice_nums": 9,
            },
            {
                "id": InferenceMode.PRECISION.value,
                "label": MODE_LABELS[InferenceMode.PRECISION],
                "protocol": "selector_top5_two_stage_rl_v2",
            },
        ],
    }


@app.post("/api/route")
async def route_preview(
    prompt: str = Form(...),
    image_count: int = Form(1),
) -> dict:
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(422, "请输入问题")
    if len(prompt) > settings.max_prompt_chars:
        raise HTTPException(422, "问题过长")
    if not 1 <= image_count <= settings.max_images:
        raise HTTPException(422, f"图片数量必须为1至{settings.max_images}")
    return route_tasks(prompt, image_count=image_count).as_dict()


async def decode_image(upload: UploadFile) -> Image.Image:
    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(415, f"{upload.filename or '文件'}不是图片")
    data = await upload.read(settings.max_image_bytes + 1)
    if len(data) > settings.max_image_bytes:
        raise HTTPException(
            413,
            f"{upload.filename or '图片'}超过{settings.max_image_bytes // 1024 // 1024}MB",
        )
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(415, f"无法读取{upload.filename or '图片'}") from exc


def save_conversation(
    request_id: str,
    prompt: str,
    images: list[Image.Image],
    answer: str,
    raw_answer: str,
    elapsed_ms: int,
    grounding: list[dict] | None = None,
    visualization: Image.Image | None = None,
    segmentation: dict | None = None,
    routing: dict | None = None,
    inference: dict | None = None,
) -> str:
    day = datetime.now().astimezone().strftime("%Y-%m-%d")
    session_dir = Path(settings.archive_dir) / day / request_id
    session_dir.mkdir(parents=True, exist_ok=False)

    for index, image in enumerate(images, start=1):
        image.save(session_dir / f"input_{index}.png", format="PNG")
    if visualization is not None:
        filename = (
            "segmentation_visualization.png"
            if segmentation
            else "grounding_visualization.png"
        )
        archive_visualization = session_dir / filename
        shared_visualization = (
            Path(settings.visualization_dir) / f"{request_id}.png"
        )
        try:
            archive_visualization.hardlink_to(shared_visualization)
        except OSError:
            visualization.save(archive_visualization, format="PNG")

    metadata = {
        "id": request_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "prompt": prompt,
        "answer": answer,
        "raw_answer": raw_answer,
        "elapsed_ms": elapsed_ms,
        "image_count": len(images),
        "model": model_service.public_state(),
        "grounding": grounding,
        "segmentation": segmentation,
        "routing": routing,
        "inference": inference,
    }
    (session_dir / "conversation.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(session_dir)


def render_grounding_visualization(
    image: Image.Image,
    boxes: list[GroundingBox],
) -> tuple[Image.Image, list[dict]]:
    visualization = image.copy()
    draw = ImageDraw.Draw(visualization)
    font = ImageFont.load_default(size=max(12, min(image.size) // 45))
    line_width = max(3, min(image.size) // 180)
    pixels = []

    for index, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box.coordinates
        px = (
            round(x1 * (image.width - 1) / 1000),
            round(y1 * (image.height - 1) / 1000),
            round(x2 * (image.width - 1) / 1000),
            round(y2 * (image.height - 1) / 1000),
        )
        draw.rectangle(px, outline="#ef3e36", width=line_width)

        marker = str(index)
        marker_box = draw.textbbox((0, 0), marker, font=font)
        marker_width = marker_box[2] - marker_box[0] + line_width * 3
        marker_height = marker_box[3] - marker_box[1] + line_width * 2
        marker_top = max(0, px[1] - marker_height)
        draw.rectangle(
            (px[0], marker_top, px[0] + marker_width, marker_top + marker_height),
            fill="#ef3e36",
        )
        draw.text(
            (px[0] + line_width, marker_top + line_width // 2),
            marker,
            fill="#fffaf7",
            font=font,
        )
        pixels.append(
            {
                "index": index,
                "label": box.label or f"目标 {index}",
                "normalized": list(box.coordinates),
                "pixels": list(px),
            }
        )

    return visualization, pixels


def save_visualization(request_id: str, image: Image.Image) -> Path:
    output_dir = Path(settings.visualization_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{request_id}.png"
    image.save(output_path, format="PNG")
    return output_path


def save_expert_input(request_id: str, image: Image.Image) -> Path:
    output_dir = Path(settings.expert_input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{request_id}.png"
    image.save(output_path, format="PNG")
    return output_path


@app.get("/api/visualizations/{request_id}")
async def visualization(request_id: str) -> FileResponse:
    try:
        safe_id = str(uuid.UUID(request_id))
    except ValueError as exc:
        raise HTTPException(404, "可视化结果不存在") from exc
    path = Path(settings.visualization_dir) / f"{safe_id}.png"
    if not path.is_file():
        raise HTTPException(404, "可视化结果不存在")
    return FileResponse(path, media_type="image/png", filename=f"{safe_id}.png")


async def execute_segmentation(
    *,
    request_id: str,
    prompt: str,
    image: Image.Image,
    route,
    route_source: str,
    started: float,
    raw_answer: str = "",
    routing: dict | None = None,
    requested_mode: InferenceMode = InferenceMode.QUALITY,
) -> dict:
    inference = inference_payload(requested_mode, "segmentation")
    unsupported_labels = [
        TARGET_LABELS.get(target, target) for target in route.unsupported
    ]
    if unsupported_labels:
        answer = "当前分割专家暂不支持：" + "、".join(unsupported_labels) + "。"
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        return {
            "id": request_id,
            "answer": answer,
            "elapsed_ms": elapsed_ms,
            "image_count": 1,
            "model": model_service.public_state(),
            "archive_path": None,
            "grounding": None,
            "routing": routing,
            "segmentation": {
                "status": "unsupported_target",
                "route_source": route_source,
                "mode": route.mode,
                "targets": [],
                "unsupported": list(route.unsupported),
                "visualization_url": None,
            },
            "inference": inference,
        }
    if route.mode == "targeted" and not route.targets:
        raise HTTPException(422, "已识别为分割任务，但无法确定需要分割的类别")

    expert_input = await asyncio.to_thread(save_expert_input, request_id, image)
    try:
        expert_result = await asyncio.to_thread(
            call_segmentation_expert,
            settings.segmentation_endpoint,
            image_path=expert_input,
            route=route,
            request_id=request_id,
            timeout_seconds=settings.expert_timeout_seconds,
        )
    except Exception as exc:
        logger.exception("Segmentation expert failed")
        log_fault(
            component="segformer",
            stage="inference",
            error=exc,
            request_id=request_id,
        )
        raise HTTPException(503, "分割专家暂时不可用，请稍后重试") from exc
    finally:
        expert_input.unlink(missing_ok=True)

    overlay_path = Path(expert_result["artifacts"]["overlay"])
    try:
        with Image.open(overlay_path) as overlay_source:
            segmentation_visualization = overlay_source.convert("RGB").copy()
        await asyncio.to_thread(
            save_visualization,
            request_id,
            segmentation_visualization,
        )
    except Exception as exc:
        logger.exception("Failed to load segmentation visualization")
        log_fault(
            component="segformer",
            stage="visualization",
            error=exc,
            request_id=request_id,
        )
        raise HTTPException(500, "无法读取分割可视化结果") from exc

    target_labels = [
        TARGET_LABELS.get(target, target)
        for target in expert_result.get("targets", [])
    ]
    if route.mode == "all_classes":
        answer = (
            "已完成整幅遥感图像的全类别语义分割，覆盖："
            + "、".join(target_labels)
            + "。"
        )
    else:
        answer = "已完成指定类别的语义分割：" + "、".join(target_labels) + "。"

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    segmentation_payload = {
        "status": "ok",
        "route_source": route_source,
        "mode": route.mode,
        "targets": list(expert_result.get("targets", [])),
        "unsupported": [],
        "area_ratios": expert_result.get("area_ratios", {}),
        "palette": expert_result.get("palette", {}),
        "visualization_url": f"/api/visualizations/{request_id}",
    }
    archive_path = None
    if settings.save_requests:
        try:
            archive_path = await asyncio.to_thread(
                save_conversation,
                request_id,
                prompt,
                [image],
                answer,
                raw_answer or answer,
                elapsed_ms,
                None,
                segmentation_visualization,
                segmentation_payload,
                routing,
                inference,
            )
        except Exception as exc:
            logger.exception("Failed to archive conversation %s", request_id)
            log_fault(
                component="archive",
                stage="write",
                error=exc,
                request_id=request_id,
                recovered=True,
            )

    return {
        "id": request_id,
        "answer": answer,
        "elapsed_ms": elapsed_ms,
        "image_count": 1,
        "model": model_service.public_state(),
        "archive_path": archive_path,
        "grounding": None,
        "routing": routing,
        "segmentation": segmentation_payload,
        "inference": inference,
    }


@app.post("/api/chat")
async def chat(
    prompt: str = Form(...),
    images: list[UploadFile] = File(...),
    inference_mode: str = Form(InferenceMode.QUALITY.value),
) -> dict:
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(422, "请输入问题")
    if len(prompt) > settings.max_prompt_chars:
        raise HTTPException(422, "问题过长")
    if not 1 <= len(images) <= settings.max_images:
        raise HTTPException(422, f"请上传1至{settings.max_images}张图片")
    try:
        requested_mode = normalize_inference_mode(inference_mode)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    request_id = str(uuid.uuid4())
    decoded_images = [await decode_image(upload) for upload in images]
    started = time.perf_counter()
    routing_decision = route_tasks(prompt, image_count=len(decoded_images))
    routing_payload = routing_decision.as_dict()
    logger.info(
        "Request %s routed to %s%s",
        request_id,
        routing_decision.tasks or ("minicpm_fallback",),
        " (compound)" if routing_decision.is_compound else "",
    )

    segmentation_route = None
    if (
        settings.expert_routing_enabled
        and len(decoded_images) == 1
        and routing_decision.tasks == ("semantic_segmentation",)
    ):
        segmentation_route = route_segmentation(prompt)

    if segmentation_route is not None:
        return await execute_segmentation(
            request_id=request_id,
            prompt=prompt,
            image=decoded_images[0],
            route=segmentation_route,
            route_source="rule",
            started=started,
            routing=routing_payload,
            requested_mode=requested_mode,
        )

    if model_service.status == "error":
        error = RuntimeError(model_service.error or "模型加载失败")
        log_fault(
            component="minicpm",
            stage="availability",
            error=error,
            request_id=request_id,
        )
        raise HTTPException(503, "模型暂时不可用，请稍后重试")

    try:
        result = await model_service.infer(
            prompt,
            decoded_images,
            requested_mode,
        )
    except Exception as exc:
        logger.exception("Inference failed")
        log_fault(
            component="minicpm",
            stage="inference",
            error=exc,
            request_id=request_id,
        )
        raise HTTPException(500, "模型推理失败，请稍后重试") from exc

    answer = result.answer
    raw_answer = result.raw_answer

    if (
        settings.expert_routing_enabled
        and len(decoded_images) == 1
        and not routing_decision.is_compound
    ):
        segmentation_route = route_segmentation_token(answer, prompt)
        if segmentation_route is not None:
            return await execute_segmentation(
                request_id=request_id,
                prompt=prompt,
                image=decoded_images[0],
                route=segmentation_route,
                route_source="minicpm",
                started=started,
                raw_answer=raw_answer,
                routing=routing_payload,
                requested_mode=requested_mode,
            )

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    inference = inference_payload(
        result.requested_mode,
        result.effective_mode,
        result.details,
    )

    grounding_payload = None
    grounding_boxes = extract_response_grounding(
        answer,
        raw_answer,
        result.effective_mode,
    )
    grounding_visualization = None
    if grounding_boxes:
        try:
            source_index = 0
            grounding_visualization, rendered_boxes = await asyncio.to_thread(
                render_grounding_visualization,
                decoded_images[source_index],
                grounding_boxes,
            )
            await asyncio.to_thread(
                save_visualization,
                request_id,
                grounding_visualization,
            )
            grounding_payload = {
                "boxes": rendered_boxes,
                "source_image_index": source_index,
                "visualization_url": f"/api/visualizations/{request_id}",
            }
        except Exception as exc:
            logger.exception("Failed to render grounding result %s", request_id)
            log_fault(
                component="grounding",
                stage="visualization",
                error=exc,
                request_id=request_id,
                recovered=True,
            )

    archive_path = None
    if settings.save_requests:
        try:
            archive_path = await asyncio.to_thread(
                save_conversation,
                request_id,
                prompt,
                decoded_images,
                answer,
                raw_answer,
                elapsed_ms,
                grounding_payload["boxes"] if grounding_payload else None,
                grounding_visualization,
                None,
                routing_payload,
                inference,
            )
        except Exception as exc:
            logger.exception("Failed to archive conversation %s", request_id)
            log_fault(
                component="archive",
                stage="write",
                error=exc,
                request_id=request_id,
                recovered=True,
            )

    return {
        "id": request_id,
        "answer": answer,
        "elapsed_ms": elapsed_ms,
        "image_count": len(decoded_images),
        "model": model_service.public_state(),
        "archive_path": archive_path,
        "grounding": grounding_payload,
        "routing": routing_payload,
        "segmentation": None,
        "inference": inference,
    }


frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
