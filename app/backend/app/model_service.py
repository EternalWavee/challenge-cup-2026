from __future__ import annotations

import asyncio
import math
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

from .config import SYSTEM_PROMPT, Settings
from .precision_grounding import (
    PrecisionCandidate,
    PrecisionPolicies,
    boundary_penalty,
    map_local_to_global,
    region_bounds,
)


class InferenceMode(str, Enum):
    FAST = "fast"
    QUALITY = "quality"
    PRECISION = "precision"


@dataclass(frozen=True)
class VisualProfile:
    downsample_mode: str
    max_slice_nums: int


VISUAL_PROFILES = {
    InferenceMode.FAST: VisualProfile("16x", 9),
    InferenceMode.QUALITY: VisualProfile("4x", 9),
    InferenceMode.PRECISION: VisualProfile("4x", 16),
}
MODE_LABELS = {
    InferenceMode.FAST: "快速模式",
    InferenceMode.QUALITY: "增强模式",
    InferenceMode.PRECISION: "高精度定位",
}


def normalize_inference_mode(value: str | InferenceMode) -> InferenceMode:
    try:
        return value if isinstance(value, InferenceMode) else InferenceMode(value)
    except ValueError as exc:
        supported = "、".join(mode.value for mode in InferenceMode)
        raise ValueError(f"未知推理模式：{value}。支持：{supported}") from exc


@dataclass
class InferenceResult:
    answer: str
    raw_answer: str
    elapsed_ms: int
    requested_mode: InferenceMode
    effective_mode: InferenceMode
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroundingBox:
    label: str
    coordinates: tuple[int, int, int, int]


THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
GROUNDING_BLOCK = re.compile(
    r"(?:<ref>(.*?)</ref>\s*)?<box>(.*?)</box>",
    re.IGNORECASE | re.DOTALL,
)
BRACKET_BOX = re.compile(
    r"[\[(]\s*(-?\d+(?:\.\d+)?)\s*[, ]+(-?\d+(?:\.\d+)?)\s*[, ]+"
    r"(-?\d+(?:\.\d+)?)\s*[, ]+(-?\d+(?:\.\d+)?)\s*[\])]"
)

REGION_LETTERS = "ABCDEFGHIJKLMNOP"
SELECTOR_SYSTEM_PROMPT = (
    "You are a remote-sensing coarse-region selector. Divide the complete image "
    "into a fixed 4 by 4 grid. Rows run top-to-bottom and columns left-to-right. "
    "The region map is: A=[0,0,250,250], B=[250,0,500,250], "
    "C=[500,0,750,250], D=[750,0,1000,250]; E=[0,250,250,500], "
    "F=[250,250,500,500], G=[500,250,750,500], H=[750,250,1000,500]; "
    "I=[0,500,250,750], J=[250,500,500,750], K=[500,500,750,750], "
    "L=[750,500,1000,750]; M=[0,750,250,1000], N=[250,750,500,1000], "
    "O=[500,750,750,1000], P=[750,750,1000,1000]. Select the region "
    "containing the center of the described target. Answer exactly 'Region ID: X', "
    "replacing X with one letter from A to P. Do not output coordinates, explanations, "
    "reference tags, or bounding boxes."
)
SELECTOR_USER_PROMPT = (
    "Use the fixed 4 by 4 region map from the system instruction. Select the region "
    "containing the center of the described target. Answer exactly: Region ID: X.\n"
    "Description: {description}"
)
LOCAL_GROUNDING_PROMPT = (
    "This is a bounding-box visual-grounding task, not segmentation. The image is a "
    "candidate crop from a larger remote-sensing image. Locate the target described "
    "below if it is visible. Return only <box>x1 y1 x2 y2</box> using integer "
    "coordinates normalized to 0-1000 relative to this crop. If the target is not "
    "visible, return only None. Do not output explanations, <ref>, <seg>, or <seg_all>.\n"
    "Description: {description}"
)


def clean_model_output(raw_answer: str) -> str:
    """Hide model reasoning blocks while preserving the task answer verbatim."""
    cleaned = THINK_BLOCK.sub("", raw_answer).strip()
    return cleaned or raw_answer.strip()


def extract_grounding_boxes(answer: str) -> list[GroundingBox]:
    boxes: list[GroundingBox] = []
    for match in GROUNDING_BLOCK.finditer(answer):
        label = re.sub(r"\s+", " ", (match.group(1) or "")).strip()
        numbers = [
            int(float(value))
            for value in re.findall(r"-?\d+(?:\.\d+)?", match.group(2))
        ]
        for index in range(0, len(numbers) - 3, 4):
            x1, y1, x2, y2 = (
                max(0, min(1000, value)) for value in numbers[index : index + 4]
            )
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            if right > left and bottom > top:
                boxes.append(
                    GroundingBox(
                        label=label,
                        coordinates=(left, top, right, bottom),
                    )
                )
    return boxes


def extract_response_grounding(
    answer: str,
    raw_answer: str,
    effective_mode: InferenceMode,
) -> list[GroundingBox]:
    """Parse the public answer first, with a safe raw-output fallback.

    Precision mode returns coordinates remapped to the complete image, while its raw
    output belongs to a local crop and must never be used as a global fallback.
    """
    boxes = extract_grounding_boxes(answer)
    if boxes or effective_mode is InferenceMode.PRECISION:
        return boxes
    return extract_grounding_boxes(raw_answer)


def extract_single_box(answer: str) -> tuple[int, int, int, int] | None:
    boxes = extract_grounding_boxes(answer)
    if boxes:
        return boxes[0].coordinates
    match = BRACKET_BOX.search(answer)
    if match is None:
        return None
    values = [max(0, min(1000, round(float(value)))) for value in match.groups()]
    left, right = sorted((values[0], values[2]))
    top, bottom = sorted((values[1], values[3]))
    return (left, top, right, bottom) if right > left and bottom > top else None


class ModelService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.status = "idle"
        self.error: str | None = None
        self.engine: Any = None
        self.template: Any = None
        self.model: Any = None
        self.processor: Any = None
        self.torch: Any = None
        self.device: Any = None
        self.precision_policies: PrecisionPolicies | None = None
        self._main_adapter_name: str | None = None
        self._selector_adapter_name: str | None = None
        self._selector_prefix_ids: list[int] = []
        self._selector_label_token_ids: list[int] = []
        self._box_first_token_id: int | None = None
        self._absence_first_token_ids: list[int] = []
        self._load_lock = asyncio.Lock()
        self._infer_lock = asyncio.Lock()

    async def load(self) -> None:
        if self.status == "ready":
            return
        async with self._load_lock:
            if self.status == "ready":
                return
            self.status = "loading"
            self.error = None
            try:
                await asyncio.to_thread(self._load_sync)
            except Exception as exc:
                self.status = "error"
                self.error = str(exc)
                raise
            self.status = "ready"

    def _load_sync(self) -> None:
        model_path = Path(self.settings.model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        import torch
        from peft import PeftModel
        from swift.infer_engine import TransformersEngine
        from swift.model import get_model_processor
        from swift.template import get_template
        from transformers import AutoProcessor

        self.torch = torch
        self.device = torch.device(self.settings.device)
        base, swift_processor = get_model_processor(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map=self.settings.device,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        adapter_path = getattr(self.settings, "adapter_path", None)
        selector_path = getattr(self.settings, "precision_selector_path", None)
        if adapter_path:
            main_path = Path(adapter_path)
            if not main_path.is_dir():
                raise FileNotFoundError(f"Adapter directory not found: {main_path}")
            self._main_adapter_name = "main"
            self.model = PeftModel.from_pretrained(
                base, main_path, adapter_name=self._main_adapter_name
            )
            if selector_path:
                precision_path = Path(selector_path)
                self._require_directory(precision_path, "Precision selector")
                self._selector_adapter_name = "precision_selector"
                self.model.load_adapter(
                    precision_path, adapter_name=self._selector_adapter_name
                )
        elif selector_path:
            precision_path = Path(selector_path)
            self._require_directory(precision_path, "Precision selector")
            self._selector_adapter_name = "precision_selector"
            self.model = PeftModel.from_pretrained(
                base,
                precision_path,
                adapter_name=self._selector_adapter_name,
            )
        else:
            self.model = base

        top3_policy_path = Path(self.settings.precision_top3_policy_path)
        top5_policy_path = Path(self.settings.precision_top5_policy_path)
        if selector_path:
            self._require_file(top3_policy_path, "Top-3 precision policy")
            self._require_file(top5_policy_path, "Top-5 residual policy")
            self.precision_policies = PrecisionPolicies(
                top3_policy_path,
                top5_policy_path,
            )

        self.model = self.model.to(self.device).eval()
        self.template = get_template(
            swift_processor,
            template_type="minicpmv4_6",
            enable_thinking=False,
            add_non_thinking_prefix=True,
        )
        self.engine = TransformersEngine(
            self.model,
            template=self.template,
            max_batch_size=1,
        )
        self._prepare_selector_tokens()

    @staticmethod
    def _require_directory(path: Path, label: str) -> None:
        if not path.is_dir():
            raise FileNotFoundError(f"{label} directory not found: {path}")

    @staticmethod
    def _require_file(path: Path, label: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    def _prepare_selector_tokens(self) -> None:
        tokenizer = self.processor.tokenizer
        self._selector_prefix_ids = tokenizer.encode(
            "Region ID:", add_special_tokens=False
        )
        self._selector_label_token_ids = []
        for label in REGION_LETTERS:
            candidate = tokenizer.encode(f" {label}", add_special_tokens=False)
            answer = tokenizer.encode(f"Region ID: {label}", add_special_tokens=False)
            if len(candidate) != 1 or answer != self._selector_prefix_ids + candidate:
                raise ValueError(f"Unexpected selector tokenization for region {label}")
            self._selector_label_token_ids.append(candidate[0])
        if len(set(self._selector_label_token_ids)) != len(REGION_LETTERS):
            raise ValueError("Selector region token IDs are not unique")

        box_ids = tokenizer.encode("<box>", add_special_tokens=False)
        if box_ids:
            self._box_first_token_id = box_ids[0]
        self._absence_first_token_ids = sorted(
            {
                ids[0]
                for text in ("NONE", "None", "none", "NO", "No", "no")
                if (ids := tokenizer.encode(text, add_special_tokens=False))
            }
        )

    @contextmanager
    def _adapter(self, name: str | None) -> Iterator[None]:
        if not hasattr(self.model, "disable_adapter"):
            if name is not None:
                raise RuntimeError("The loaded model does not support adapters")
            yield
            return
        if name is None:
            with self.model.disable_adapter():
                yield
            return
        self.model.set_adapter(name)
        yield

    def _prepare_inputs(
        self,
        *,
        system_prompt: str,
        prompt: str,
        images: list[Image.Image],
        profile: VisualProfile,
    ) -> Any:
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "downsample_mode": profile.downsample_mode,
                "max_slice_nums": profile.max_slice_nums,
            },
        ).to(self.device)

    def _decode_precision_output(
        self,
        sequences: Any,
        prompt_length: int,
    ) -> tuple[str, Any]:
        """Decode precision-only generations without dropping grounding tokens."""
        new_ids = sequences[:, prompt_length:]
        raw_answer = self.processor.batch_decode(
            new_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return raw_answer, new_ids

    def _generation_options(self, max_new_tokens: int) -> dict[str, Any]:
        temperature = float(self.settings.temperature)
        options: dict[str, Any] = {
            "do_sample": temperature > 0,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
        }
        if temperature > 0:
            options["temperature"] = temperature
        return options

    def _infer_standard_sync(
        self,
        prompt: str,
        images: list[Image.Image],
        mode: InferenceMode,
    ) -> tuple[str, str]:
        from swift.infer_engine import InferRequest, RequestConfig

        profile = VISUAL_PROFILES[mode]
        self.template.downsample_mode = profile.downsample_mode
        self.template.max_slice_nums = profile.max_slice_nums
        request = InferRequest(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"{'<image>' * len(images)}\n{prompt}",
                },
            ],
            images=images,
        )
        request_config = RequestConfig(
            max_tokens=self.settings.max_new_tokens,
            temperature=self.settings.temperature,
            stream=False,
        )
        with self._adapter(self._main_adapter_name), self.torch.inference_mode():
            responses = self.engine.infer(
                [request],
                request_config,
                use_tqdm=False,
            )
        raw_answer = str(responses[0].choices[0].message.content)
        return clean_model_output(raw_answer), raw_answer

    def _select_regions(
        self,
        description: str,
        image: Image.Image,
    ) -> list[tuple[str, float]]:
        if self._selector_adapter_name is None:
            raise RuntimeError("Precision selector adapter is not configured")
        profile = VISUAL_PROFILES[InferenceMode.PRECISION]
        inputs = self._prepare_inputs(
            system_prompt=SELECTOR_SYSTEM_PROMPT,
            prompt=SELECTOR_USER_PROMPT.format(description=description),
            images=[image],
            profile=profile,
        )
        prompt_length = int(inputs["input_ids"].shape[-1])

        def allowed_tokens(_batch_id: int, sequence: Any) -> list[int]:
            generated_count = int(sequence.shape[-1]) - prompt_length
            if generated_count < len(self._selector_prefix_ids):
                return [self._selector_prefix_ids[generated_count]]
            return self._selector_label_token_ids

        with self._adapter(self._selector_adapter_name), self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                downsample_mode=profile.downsample_mode,
                do_sample=False,
                max_new_tokens=len(self._selector_prefix_ids) + 1,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
                prefix_allowed_tokens_fn=allowed_tokens,
            )
        label_tensor = self.torch.tensor(
            self._selector_label_token_ids, device=self.device
        )
        logits = generated.scores[-1][0].float().index_select(0, label_tensor)
        probabilities = self.torch.softmax(logits, dim=0)
        order = self.torch.argsort(logits, descending=True).tolist()
        return [
            (REGION_LETTERS[position], float(probabilities[position].item()))
            for position in order
        ]

    @staticmethod
    def _crop_image(
        image: Image.Image,
        crop: tuple[float, float, float, float],
    ) -> Image.Image:
        left = max(0, math.floor(crop[0] * image.width / 1000.0))
        top = max(0, math.floor(crop[1] * image.height / 1000.0))
        right = min(image.width, math.ceil(crop[2] * image.width / 1000.0))
        bottom = min(image.height, math.ceil(crop[3] * image.height / 1000.0))
        return image.crop((left, top, right, bottom))

    def _local_grounding(
        self,
        description: str,
        crop_image: Image.Image,
    ) -> tuple[
        str,
        tuple[int, int, int, int] | None,
        float,
        dict[str, float | int | None],
    ]:
        profile = VISUAL_PROFILES[InferenceMode.PRECISION]
        inputs = self._prepare_inputs(
            system_prompt=SYSTEM_PROMPT,
            prompt=LOCAL_GROUNDING_PROMPT.format(description=description),
            images=[crop_image],
            profile=profile,
        )
        prompt_length = int(inputs["input_ids"].shape[-1])
        with self._adapter(self._main_adapter_name), self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                downsample_mode=profile.downsample_mode,
                do_sample=False,
                max_new_tokens=64,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        raw_answer, new_ids = self._decode_precision_output(
            generated.sequences, prompt_length
        )
        chosen_log_probabilities: list[float] = []
        for index, scores in enumerate(generated.scores):
            if index >= new_ids.shape[-1]:
                break
            token_id = int(new_ids[0, index].item())
            log_probabilities = self.torch.log_softmax(scores[0].float(), dim=-1)
            chosen_log_probabilities.append(float(log_probabilities[token_id].item()))

        box_log_probability = absence_log_probability = None
        presence_margin = -100.0
        if (
            generated.scores
            and self._box_first_token_id is not None
            and self._absence_first_token_ids
        ):
            first_log_probabilities = self.torch.log_softmax(
                generated.scores[0][0].float(), dim=-1
            )
            box_log_probability = float(
                first_log_probabilities[self._box_first_token_id].item()
            )
            absence_ids = self.torch.tensor(
                self._absence_first_token_ids, device=self.device
            )
            absence_log_probability = float(
                self.torch.logsumexp(
                    first_log_probabilities.index_select(0, absence_ids), dim=0
                ).item()
            )
            presence_margin = box_log_probability - absence_log_probability

        sequence_sum = sum(chosen_log_probabilities)
        confidence: dict[str, float | int | None] = {
            "sequence_logprob_mean": (
                sequence_sum / len(chosen_log_probabilities)
                if chosen_log_probabilities
                else None
            ),
            "sequence_logprob_sum": (
                sequence_sum if chosen_log_probabilities else None
            ),
            "first_chosen_logprob": (
                chosen_log_probabilities[0] if chosen_log_probabilities else None
            ),
            "box_logprob": box_log_probability,
            "absence_logprob_sum": absence_log_probability,
            "output_tokens": int(new_ids.shape[-1]),
        }
        return raw_answer, extract_single_box(raw_answer), presence_margin, confidence

    def _infer_precision_sync(
        self,
        prompt: str,
        image: Image.Image,
    ) -> tuple[str, str, InferenceMode, dict[str, Any]]:
        if self.precision_policies is None:
            raise RuntimeError("Precision policies are not configured")
        ranked_regions = self._select_regions(prompt, image)
        margin = float(self.settings.precision_margin)
        candidates: list[PrecisionCandidate] = []
        for rank, (region, probability) in enumerate(ranked_regions[:5], start=1):
            crop = region_bounds(region, margin)
            crop_image = self._crop_image(image, crop)
            raw_answer, local_box, presence_margin, confidence = self._local_grounding(
                prompt, crop_image
            )
            global_box = (
                map_local_to_global(local_box, crop) if local_box is not None else None
            )
            candidates.append(
                PrecisionCandidate(
                    region=region,
                    rank=rank,
                    selector_probability=probability,
                    crop=crop,
                    local_box=local_box,
                    global_box=global_box,
                    presence_margin=presence_margin,
                    boundary_penalty=boundary_penalty(local_box),
                    inference_confidence=confidence,
                    raw_answer=raw_answer,
                )
            )

        decision = self.precision_policies.choose(candidates)
        details = {
            "protocol": "selector_top5_two_stage_rl_v2",
            "selector_regions": [candidate.region for candidate in candidates],
            "selector_probabilities": [
                round(candidate.selector_probability, 8) for candidate in candidates
            ],
            "margin": margin,
            "top3_action": decision.top3_action,
            "top3_policy_margin": decision.top3_margin,
            "top5_action": decision.top5_action,
            "top5_policy_margin": decision.top5_margin,
            **decision.details,
            "candidates": [
                {
                    "region": candidate.region,
                    "rank": candidate.rank,
                    "crop": [round(value, 3) for value in candidate.crop],
                    "local_box": list(candidate.local_box) if candidate.local_box else None,
                    "global_box": (
                        [round(value, 3) for value in candidate.global_box]
                        if candidate.global_box
                        else None
                    ),
                    "presence_margin": round(candidate.presence_margin, 6),
                    "agreement": round(candidate.agreement, 6),
                    "fusion_score": (
                        round(candidate.fusion_score, 6)
                        if math.isfinite(candidate.fusion_score)
                        else None
                    ),
                }
                for candidate in candidates
            ],
        }
        selected = decision.candidate
        if selected is None or selected.global_box is None:
            answer, raw_answer = self._infer_standard_sync(
                prompt, [image], InferenceMode.QUALITY
            )
            details["fallback"] = "quality_no_valid_local_box"
            return answer, raw_answer, InferenceMode.QUALITY, details

        coordinates = [
            max(0, min(1000, round(value))) for value in selected.global_box
        ]
        answer = f"<ref>目标</ref><box>{' '.join(map(str, coordinates))}</box>"
        return answer, selected.raw_answer, InferenceMode.PRECISION, details

    async def infer(
        self,
        prompt: str,
        images: list[Image.Image],
        mode: str | InferenceMode = InferenceMode.QUALITY,
    ) -> InferenceResult:
        if self.status != "ready":
            await self.load()
        requested_mode = normalize_inference_mode(mode)

        async with self._infer_lock:
            started = time.perf_counter()
            if (
                requested_mode is InferenceMode.PRECISION
                and len(images) == 1
            ):
                answer, raw_answer, effective_mode, details = await asyncio.to_thread(
                    self._infer_precision_sync, prompt, images[0]
                )
            else:
                effective_mode = (
                    InferenceMode.QUALITY
                    if requested_mode is InferenceMode.PRECISION
                    else requested_mode
                )
                answer, raw_answer = await asyncio.to_thread(
                    self._infer_standard_sync,
                    prompt,
                    images,
                    effective_mode,
                )
                details = {}
                if requested_mode is InferenceMode.PRECISION:
                    details["fallback"] = "quality_precision_requires_one_image"
            elapsed_ms = round((time.perf_counter() - started) * 1000)

        return InferenceResult(
            answer=answer,
            raw_answer=raw_answer,
            elapsed_ms=elapsed_ms,
            requested_mode=requested_mode,
            effective_mode=effective_mode,
            details=details,
        )

    def public_state(self) -> dict[str, Any]:
        adapter_path = getattr(self.settings, "adapter_path", None)
        selector_path = getattr(self.settings, "precision_selector_path", None)
        return {
            "status": self.status,
            "error": self.error,
            "model": Path(self.settings.model_path).name,
            "adapter": Path(adapter_path).name if adapter_path else "merged",
            "precision_selector": Path(selector_path).name if selector_path else None,
            "device": self.settings.device,
        }
