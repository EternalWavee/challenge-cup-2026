from __future__ import annotations

import unittest

from app.precision_grounding import (
    PrecisionCandidate,
    boundary_penalty,
    candidate_features,
    ensure_fusion_scores,
    map_local_to_global,
    region_bounds,
    residual_features,
)


def candidate(rank: int) -> PrecisionCandidate:
    local = (100 + rank, 200, 400, 600)
    crop = region_bounds("ABCDE"[rank - 1], 64)
    return PrecisionCandidate(
        region="ABCDE"[rank - 1],
        rank=rank,
        selector_probability=0.3 / rank,
        crop=crop,
        local_box=local,
        global_box=map_local_to_global(local, crop),
        presence_margin=2.0 - rank,
        boundary_penalty=boundary_penalty(local),
        inference_confidence={
            "sequence_logprob_mean": -0.2,
            "sequence_logprob_sum": -1.0,
            "first_chosen_logprob": -0.1,
            "box_logprob": -0.3,
            "absence_logprob_sum": -1.5,
            "output_tokens": 5,
        },
        raw_answer="<box>100 200 400 600</box>",
    )


class PrecisionGeometryTest(unittest.TestCase):
    def test_expanded_region_and_coordinate_mapping(self) -> None:
        self.assertEqual(region_bounds("A", 64), (0.0, 0.0, 314.0, 314.0))
        self.assertEqual(
            map_local_to_global((0, 0, 1000, 1000), (186.0, 186.0, 564.0, 564.0)),
            (186.0, 186.0, 564.0, 564.0),
        )

    def test_policy_feature_contract_matches_saved_checkpoints(self) -> None:
        candidates = [candidate(rank) for rank in range(1, 6)]
        ensure_fusion_scores(candidates[:3])
        self.assertEqual(len(candidate_features(candidates[0], candidates[:3], 0)), 35)
        ensure_fusion_scores(candidates)
        self.assertEqual(len(residual_features(candidates[3], candidates, 0)), 77)


if __name__ == "__main__":
    unittest.main()
