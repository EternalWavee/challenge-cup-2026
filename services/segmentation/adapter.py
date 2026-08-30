"""MMSegmentation adapter with stable JSON and image-artifact outputs."""

from __future__ import annotations

import re
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


EXPECTED_CLASSES = (
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agricultural",
)

FALLBACK_PALETTE = (
    (255, 255, 255),
    (255, 0, 0),
    (255, 255, 0),
    (0, 0, 255),
    (159, 129, 183),
    (0, 255, 0),
    (255, 195, 128),
)


def _safe_request_id(value: str | None) -> str:
    if value and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
        return value
    return uuid.uuid4().hex


def _add_legend(
    image: Image.Image,
    *,
    classes: tuple[str, ...],
    palette: tuple[tuple[int, int, int], ...],
    area_ratios: dict[str, float],
) -> Image.Image:
    """Append a compact, dependency-free legend to a visualization."""
    scale = max(1, min(image.size) // 512)
    row_height = 30 * scale
    margin = 16 * scale
    legend_width = 220 * scale
    canvas = Image.new(
        "RGB",
        (image.width + legend_width, max(image.height, margin * 2 + row_height * len(classes))),
        "white",
    )
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=13 * scale)
    for index, class_name in enumerate(classes):
        top = margin + index * row_height
        color = palette[index]
        draw.rounded_rectangle(
            (image.width + margin, top, image.width + margin + 18 * scale, top + 18 * scale),
            radius=3 * scale,
            fill=color,
            outline=(100, 116, 139),
            width=scale,
        )
        ratio = area_ratios.get(class_name, 0.0) * 100
        draw.text(
            (image.width + margin + 28 * scale, top + scale),
            f"{class_name}  {ratio:.1f}%",
            fill=(30, 41, 59),
            font=font,
        )
    return canvas


class SegFormerExpert:
    """Lazy-load a single SegFormer model and serialize GPU inference."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        checkpoint_path: str | Path,
        output_root: str | Path,
        device: str = "cuda:0",
    ):
        self.config_path = Path(config_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.output_root = Path(output_root).expanduser().resolve()
        self.device = device
        self._model: Any = None
        self._inference_model: Any = None
        self._lock = threading.Lock()
        self.classes = EXPECTED_CLASSES
        self.palette = FALLBACK_PALETTE

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Missing SegFormer config: {self.config_path}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing SegFormer checkpoint: {self.checkpoint_path}"
            )

        from mmseg.apis import inference_model, init_model

        self._model = init_model(
            str(self.config_path),
            str(self.checkpoint_path),
            device=self.device,
        )
        self._inference_model = inference_model

        metadata = getattr(self._model, "dataset_meta", {}) or {}
        classes = tuple(
            str(item).strip().lower()
            for item in (metadata.get("classes") or EXPECTED_CLASSES)
        )
        palette = tuple(tuple(item) for item in metadata.get("palette") or FALLBACK_PALETTE)
        if len(classes) != len(palette):
            raise ValueError(
                f"SegFormer metadata mismatch: {len(classes)} classes, "
                f"{len(palette)} palette entries"
            )
        self.classes = classes
        self.palette = palette

    def _predict_label_map(self, image_path: Path) -> np.ndarray:
        result = self._inference_model(self._model, str(image_path))
        return (
            result.pred_sem_seg.data[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

    def run(
        self,
        *,
        image_path: str | Path,
        mode: str,
        targets: list[str] | tuple[str, ...] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        resolved_image = Path(image_path).expanduser().resolve()
        if not resolved_image.is_file():
            raise FileNotFoundError(f"Missing input image: {resolved_image}")
        if mode not in {"targeted", "all_classes"}:
            raise ValueError("mode must be 'targeted' or 'all_classes'")

        requested_targets = tuple(dict.fromkeys(targets or []))
        if mode == "targeted" and not requested_targets:
            raise ValueError("targeted mode requires at least one target")

        rid = _safe_request_id(request_id)
        request_output = self.output_root / rid

        with self._lock:
            self._ensure_model()
            unsupported = [
                target for target in requested_targets if target not in self.classes
            ]
            if unsupported:
                raise ValueError(
                    "Checkpoint does not support: " + ", ".join(unsupported)
                )

            label_map = self._predict_label_map(resolved_image)

        request_output.mkdir(parents=True, exist_ok=False)
        with Image.open(resolved_image) as source_image:
            source = np.asarray(source_image.convert("RGB"))
        if label_map.shape != source.shape[:2]:
            label_map = np.asarray(
                Image.fromarray(label_map).resize(
                    (source.shape[1], source.shape[0]),
                    resample=Image.Resampling.NEAREST,
                )
            )

        if label_map.size and int(label_map.max()) >= len(self.classes):
            raise ValueError("SegFormer output contains an unknown class index")

        palette = np.asarray(self.palette, dtype=np.uint8)
        full_color_mask = palette[label_map]
        blended = np.clip(
            source.astype(np.float32) * 0.55
            + full_color_mask.astype(np.float32) * 0.45,
            0,
            255,
        ).astype(np.uint8)

        if mode == "all_classes":
            result_targets = self.classes
            visible = np.ones(label_map.shape, dtype=bool)
        else:
            result_targets = requested_targets
            target_ids = [self.classes.index(target) for target in result_targets]
            visible = np.isin(label_map, target_ids)

        color_mask = np.zeros_like(full_color_mask)
        color_mask[visible] = full_color_mask[visible]
        overlay = source.copy()
        overlay[visible] = blended[visible]

        label_path = request_output / "label_map.png"
        color_path = request_output / "color_mask.png"
        overlay_path = request_output / "overlay.png"
        Image.fromarray(label_map).save(label_path)
        Image.fromarray(color_mask).save(color_path)
        overlay_image = Image.fromarray(overlay)

        masks: dict[str, str] = {}
        area_ratios: dict[str, float] = {}
        for target in result_targets:
            class_id = self.classes.index(target)
            binary_mask = label_map == class_id
            area_ratios[target] = round(float(binary_mask.mean()), 8)
            if mode == "targeted":
                mask_path = request_output / f"mask_{target}.png"
                Image.fromarray(binary_mask.astype(np.uint8) * 255).save(mask_path)
                masks[target] = str(mask_path)

        overlay_image = _add_legend(
            overlay_image,
            classes=result_targets,
            palette=tuple(self.palette[self.classes.index(target)] for target in result_targets),
            area_ratios=area_ratios,
        )
        overlay_image.save(overlay_path)

        return {
            "schema_version": "1.0",
            "status": "ok",
            "expert": "segformer_b5_loveda",
            "request_id": rid,
            "mode": mode,
            "targets": list(result_targets),
            "classes": list(self.classes),
            "palette": {
                class_name: list(self.palette[index])
                for index, class_name in enumerate(self.classes)
            },
            "image": {
                "path": str(resolved_image),
                "width": int(source.shape[1]),
                "height": int(source.shape[0]),
            },
            "artifacts": {
                "label_map": str(label_path),
                "color_mask": str(color_path),
                "overlay": str(overlay_path),
                "masks": masks,
            },
            "area_ratios": area_ratios,
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "expert": "segformer_b5_loveda",
            "loaded": self.loaded,
            "device": self.device,
            "config_exists": self.config_path.is_file(),
            "checkpoint_exists": self.checkpoint_path.is_file(),
            "classes": list(self.classes),
        }
