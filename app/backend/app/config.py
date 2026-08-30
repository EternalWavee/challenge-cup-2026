from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


FINAL_ROOT = Path(__file__).resolve().parents[3]

SYSTEM_PROMPT = (
    "你是遥感影像智能解译助手。请根据图像和自然语言问题直接回答，不输出思考过程。"
    "若提供两张图，第一张为时相一，第二张为时相二。指定类别分割只输出<seg>；"
    "整图全类别分割只输出<seg_all>。输出路由词时不得包含其他内容。"
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> str:
    value = Path(os.getenv(name, str(default))).expanduser()
    if not value.is_absolute():
        value = FINAL_ROOT / value
    return str(value.resolve())


def _env_optional_path(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = FINAL_ROOT / path
    return str(path.resolve())


@dataclass(frozen=True)
class Settings:
    model_path: str = _env_path("MODEL_PATH", FINAL_ROOT / "models" / "terrascope")
    adapter_path: str | None = _env_optional_path("MODEL_ADAPTER_PATH")
    precision_selector_path: str = _env_path(
        "PRECISION_SELECTOR_PATH",
        FINAL_ROOT / "models" / "precision-selector",
    )
    precision_top3_policy_path: str = _env_path(
        "PRECISION_TOP3_POLICY_PATH",
        FINAL_ROOT / "models" / "precision-router" / "top3_policy.pt",
    )
    precision_top5_policy_path: str = _env_path(
        "PRECISION_TOP5_POLICY_PATH",
        FINAL_ROOT / "models" / "precision-router" / "top5_residual_policy.pt",
    )
    precision_margin: float = float(os.getenv("PRECISION_CROP_MARGIN", "64"))
    device: str = os.getenv("MODEL_DEVICE", "cuda:0")
    eager_load: bool = _env_bool("MODEL_EAGER_LOAD", True)
    save_requests: bool = _env_bool("SAVE_REQUESTS", True)
    archive_dir: str = _env_path(
        "APP_DATA_DIR", FINAL_ROOT / "runtime" / "conversations"
    )
    visualization_dir: str = _env_path(
        "VISUALIZATION_DIR", FINAL_ROOT / "runtime" / "visualizations"
    )
    expert_input_dir: str = _env_path(
        "EXPERT_INPUT_DIR", FINAL_ROOT / "runtime" / "expert_inputs"
    )
    fault_log_dir: str = _env_path(
        "FAULT_LOG_DIR", FINAL_ROOT / "runtime" / "logs"
    )
    segmentation_endpoint: str = os.getenv(
        "SEGMENTATION_ENDPOINT", "http://127.0.0.1:8010/segment"
    )
    expert_routing_enabled: bool = _env_bool("EXPERT_ROUTING_ENABLED", True)
    expert_timeout_seconds: int = int(os.getenv("EXPERT_TIMEOUT_SECONDS", "300"))
    max_new_tokens: int = int(os.getenv("MODEL_MAX_NEW_TOKENS", "512"))
    temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0"))
    max_images: int = 2
    max_image_bytes: int = 15 * 1024 * 1024
    max_prompt_chars: int = 4000
