from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REGION_LETTERS = "ABCDEFGHIJKLMNOP"
REGION_STEP = 250.0


@dataclass
class PrecisionCandidate:
    region: str
    rank: int
    selector_probability: float
    crop: tuple[float, float, float, float]
    local_box: tuple[int, int, int, int] | None
    global_box: tuple[float, float, float, float] | None
    presence_margin: float
    boundary_penalty: float
    inference_confidence: dict[str, float | int | None]
    raw_answer: str
    agreement: float = 0.0
    fusion_score: float = float("-inf")


@dataclass(frozen=True)
class PrecisionDecision:
    candidate: PrecisionCandidate | None
    top3_action: str
    top3_margin: float | None
    top5_action: str
    top5_margin: float | None
    details: dict[str, Any] = field(default_factory=dict)


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def region_bounds(label: str, margin: float) -> tuple[float, float, float, float]:
    index = REGION_LETTERS.index(label)
    row, column = divmod(index, 4)
    return (
        max(0.0, column * REGION_STEP - margin),
        max(0.0, row * REGION_STEP - margin),
        min(1000.0, (column + 1) * REGION_STEP + margin),
        min(1000.0, (row + 1) * REGION_STEP + margin),
    )


def map_local_to_global(
    local_box: tuple[int, int, int, int],
    crop: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    left, top, right, bottom = crop
    width = right - left
    height = bottom - top
    return tuple(
        clip(value, 0.0, 1000.0)
        for value in (
            left + local_box[0] * width / 1000.0,
            top + local_box[1] * height / 1000.0,
            left + local_box[2] * width / 1000.0,
            top + local_box[3] * height / 1000.0,
        )
    )


def box_iou(
    left: tuple[float, float, float, float] | None,
    right: tuple[float, float, float, float] | None,
) -> float:
    if left is None or right is None:
        return 0.0
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / max(union, 1e-12)


def boundary_penalty(local_box: tuple[int, int, int, int] | None) -> float:
    if local_box is None:
        return 1.0
    clearance = min(
        local_box[0], local_box[1], 1000 - local_box[2], 1000 - local_box[3]
    )
    return clip((60.0 - clearance) / 60.0, 0.0, 1.0)


def ensure_fusion_scores(candidates: list[PrecisionCandidate]) -> None:
    for candidate in candidates:
        agreements = [
            box_iou(candidate.global_box, other.global_box)
            for other in candidates
            if other is not candidate
            and candidate.global_box is not None
            and other.global_box is not None
        ]
        candidate.agreement = max(agreements, default=0.0)
        candidate.fusion_score = (
            math.log(max(candidate.selector_probability, 1e-12))
            + 0.5 * math.tanh(candidate.presence_margin / 4.0)
            + candidate.agreement
            - 0.25 * candidate.boundary_penalty
        )


def fusion_index(candidates: list[PrecisionCandidate]) -> int:
    ensure_fusion_scores(candidates)
    valid_indices = [
        index for index, candidate in enumerate(candidates) if candidate.global_box is not None
    ]
    if not valid_indices:
        return 0
    return max(
        valid_indices,
        key=lambda index: (candidates[index].fusion_score, -candidates[index].rank),
    )


def candidate_features(
    candidate: PrecisionCandidate,
    candidates: list[PrecisionCandidate],
    baseline_index: int,
) -> list[float]:
    local_box = candidate.local_box
    global_box = candidate.global_box
    parsed = float(global_box is not None)
    local_values = (
        [0.0] * 4
        if local_box is None
        else [clip(float(value) / 1000.0, 0.0, 1.0) for value in local_box]
    )
    global_values = (
        [0.0] * 4
        if global_box is None
        else [clip(float(value) / 1000.0, 0.0, 1.0) for value in global_box]
    )
    if global_box is None:
        center_x = center_y = width = height = area = aspect = 0.0
    else:
        width = clip((global_box[2] - global_box[0]) / 1000.0, 0.0, 1.0)
        height = clip((global_box[3] - global_box[1]) / 1000.0, 0.0, 1.0)
        center_x = clip((global_box[0] + global_box[2]) / 2000.0, 0.0, 1.0)
        center_y = clip((global_box[1] + global_box[3]) / 2000.0, 0.0, 1.0)
        area = width * height
        aspect = clip(
            math.log(max(width, 1e-5) / max(height, 1e-5)) / 5.0,
            -1.0,
            1.0,
        )
    probability = clip(candidate.selector_probability, 1e-9, 1.0)
    confidence = candidate.inference_confidence
    agreements = [
        box_iou(global_box, other.global_box)
        for other in candidates
        if other is not candidate
    ]
    rank = candidate.rank
    sequence_mean = confidence.get("sequence_logprob_mean")
    sequence_sum = confidence.get("sequence_logprob_sum")
    first_chosen = confidence.get("first_chosen_logprob")
    box_logprob = confidence.get("box_logprob")
    absence_logprob = confidence.get("absence_logprob_sum")
    output_tokens = confidence.get("output_tokens")
    return [
        probability,
        clip(math.log(probability) / 12.0, -1.0, 0.0),
        math.tanh(candidate.presence_margin / 4.0),
        clip(candidate.presence_margin / 20.0, -1.0, 1.0),
        clip(candidate.boundary_penalty, 0.0, 1.0),
        clip(candidate.agreement, 0.0, 1.0),
        clip(candidate.fusion_score / 12.0, -2.0, 1.0),
        parsed,
        center_x,
        center_y,
        width,
        height,
        area,
        aspect,
        *local_values,
        *global_values,
        float(rank == 1),
        float(rank == 2),
        float(rank == 3),
        float(rank - 1) / 2.0,
        float(rank - 1 == baseline_index),
        max(agreements, default=0.0),
        statistics.fmean(agreements) if agreements else 0.0,
        clip(float(sequence_mean if sequence_mean is not None else -100.0) / 10.0, -2.0, 0.0),
        clip(float(sequence_sum if sequence_sum is not None else -100.0) / 50.0, -2.0, 0.0),
        clip(float(first_chosen if first_chosen is not None else -100.0) / 10.0, -2.0, 0.0),
        clip(float(box_logprob if box_logprob is not None else -100.0) / 20.0, -2.0, 0.0),
        clip(float(absence_logprob if absence_logprob is not None else -100.0) / 20.0, -2.0, 0.0),
        clip(float(output_tokens if output_tokens is not None else 0.0) / 64.0, 0.0, 2.0),
    ]


def residual_features(
    candidate: PrecisionCandidate,
    candidates: list[PrecisionCandidate],
    baseline_index: int,
) -> list[float]:
    baseline = candidates[baseline_index]
    own = candidate_features(candidate, candidates, baseline_index)
    base = candidate_features(baseline, candidates, baseline_index)
    return own + base + [
        clip(candidate.selector_probability - baseline.selector_probability, -1.0, 1.0),
        clip((candidate.presence_margin - baseline.presence_margin) / 20.0, -2.0, 2.0),
        clip(candidate.boundary_penalty - baseline.boundary_penalty, -1.0, 1.0),
        clip(candidate.fusion_score - baseline.fusion_score, -20.0, 20.0) / 10.0,
        box_iou(candidate.global_box, baseline.global_box),
        float(candidate.rank == 4),
        float(candidate.rank == 5),
    ]


class PrecisionPolicies:
    def __init__(self, top3_path: Path, top5_path: Path) -> None:
        import torch
        from torch import nn

        class SharedCandidatePolicy(nn.Module):
            def __init__(self, feature_count: int, hidden: int) -> None:
                super().__init__()
                self.candidate = (
                    nn.Linear(feature_count, 1)
                    if hidden == 0
                    else nn.Sequential(
                        nn.Linear(feature_count, hidden),
                        nn.Tanh(),
                        nn.Linear(hidden, 1),
                    )
                )
                self.rank_bias = nn.Parameter(torch.zeros(3))
                self.register_buffer("keep_logit", torch.tensor(0.0))

            def forward(self, features):
                candidate_logits = self.candidate(features).squeeze(-1) + self.rank_bias
                keep = self.keep_logit.expand(features.shape[0], 1)
                return torch.cat([keep, candidate_logits], dim=1)

        class ResidualPolicy(nn.Module):
            def __init__(self, feature_count: int, hidden: int) -> None:
                super().__init__()
                self.candidate = (
                    nn.Linear(feature_count, 1)
                    if hidden == 0
                    else nn.Sequential(
                        nn.Linear(feature_count, hidden),
                        nn.Tanh(),
                        nn.Linear(hidden, 1),
                    )
                )
                self.rank_bias = nn.Parameter(torch.zeros(2))

            def forward(self, features):
                candidate_logits = self.candidate(features).squeeze(-1) + self.rank_bias
                keep = torch.zeros(
                    (features.shape[0], 1),
                    dtype=features.dtype,
                    device=features.device,
                )
                return torch.cat([keep, candidate_logits], dim=1)

        self.torch = torch
        top3_checkpoint = torch.load(top3_path, map_location="cpu", weights_only=True)
        self.top3 = SharedCandidatePolicy(
            int(top3_checkpoint["feature_count"]),
            int(top3_checkpoint["hidden"]),
        )
        self.top3.load_state_dict(top3_checkpoint["state_dict"])
        self.top3.eval()
        self.top3_threshold = float(top3_checkpoint["threshold"])

        top5_checkpoint = torch.load(top5_path, map_location="cpu", weights_only=True)
        self.top5 = ResidualPolicy(
            int(top5_checkpoint["feature_count"]),
            int(top5_checkpoint["hidden"]),
        )
        self.top5.load_state_dict(top5_checkpoint["state_dict"])
        self.top5.eval()
        self.top5_threshold = float(top5_checkpoint["threshold"])

    def choose(self, candidates: list[PrecisionCandidate]) -> PrecisionDecision:
        if len(candidates) != 5:
            raise ValueError(f"Expected five precision candidates, got {len(candidates)}")

        top3_candidates = candidates[:3]
        top3_baseline = fusion_index(top3_candidates)
        top3_features = self.torch.tensor(
            [[
                candidate_features(candidate, top3_candidates, top3_baseline)
                for candidate in top3_candidates
            ]],
            dtype=self.torch.float32,
        )
        with self.torch.inference_mode():
            top3_logits = self.top3(top3_features)[0]
        top3_candidate_logit, top3_candidate_index = top3_logits[1:].max(dim=0)
        top3_margin = float((top3_candidate_logit - top3_logits[0]).item())
        top3_proposed = int(top3_candidate_index.item())
        top3_selected = (
            top3_proposed
            if top3_margin > self.top3_threshold and top3_proposed != top3_baseline
            else top3_baseline
        )
        top3_action = (
            f"BOX_{top3_selected + 1}"
            if top3_selected != top3_baseline
            else "KEEP_TOP3_FUSION"
        )

        ensure_fusion_scores(candidates)
        top5_features = self.torch.tensor(
            [[
                residual_features(candidate, candidates, top3_selected)
                for candidate in candidates[3:5]
            ]],
            dtype=self.torch.float32,
        )
        with self.torch.inference_mode():
            top5_logits = self.top5(top5_features)[0]
        top5_candidate_logit, top5_candidate_index = top5_logits[1:].max(dim=0)
        top5_margin = float((top5_candidate_logit - top5_logits[0]).item())
        top5_proposed = int(top5_candidate_index.item()) + 3
        top5_selected = (
            top5_proposed if top5_margin > self.top5_threshold else top3_selected
        )
        top5_action = (
            f"BOX_{top5_selected + 1}"
            if top5_selected != top3_selected
            else "KEEP_TOP3_RL"
        )
        selected = candidates[top5_selected]
        if selected.global_box is None:
            parsed = [candidate for candidate in candidates if candidate.global_box is not None]
            selected = max(parsed, key=lambda item: item.fusion_score) if parsed else None
            top5_action = "FALLBACK_VALID_FUSION" if selected else "NO_VALID_BOX"

        return PrecisionDecision(
            candidate=selected,
            top3_action=top3_action,
            top3_margin=top3_margin,
            top5_action=top5_action,
            top5_margin=top5_margin,
            details={
                "top3_threshold": self.top3_threshold,
                "top5_threshold": self.top5_threshold,
                "top3_baseline_rank": top3_baseline + 1,
                "top3_selected_rank": top3_selected + 1,
                "final_selected_rank": selected.rank if selected else None,
            },
        )
