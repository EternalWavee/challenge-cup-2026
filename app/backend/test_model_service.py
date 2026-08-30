from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.model_service import (
    InferenceMode,
    ModelService,
    clean_model_output,
    extract_grounding_boxes,
    extract_response_grounding,
    extract_single_box,
    normalize_inference_mode,
)


class ModelOutputTest(unittest.TestCase):
    def test_merged_model_reports_no_external_adapter(self) -> None:
        settings = SimpleNamespace(
            model_path="/models/terrascope",
            adapter_path=None,
            device="cuda:0",
        )
        self.assertEqual(ModelService(settings).public_state()["adapter"], "merged")

    def test_thinking_is_hidden_without_changing_answer(self) -> None:
        raw = (
            "<think>internal reasoning</think>\n"
            "<ref>灰色体育场</ref><box>336 301 656 568</box>"
        )
        self.assertEqual(
            clean_model_output(raw),
            "<ref>灰色体育场</ref><box>336 301 656 568</box>",
        )

    def test_grounding_parser_supports_multiple_boxes(self) -> None:
        answer = "<ref>飞机</ref><box>10 20 30 40 50 60 70 80</box>"
        boxes = extract_grounding_boxes(answer)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0].label, "飞机")
        self.assertEqual(boxes[0].coordinates, (10, 20, 30, 40))
        self.assertEqual(boxes[1].coordinates, (50, 60, 70, 80))

    def test_grounding_parser_normalizes_and_clamps_coordinates(self) -> None:
        boxes = extract_grounding_boxes("<box>1100 900 -20 100</box>")
        self.assertEqual(boxes[0].coordinates, (0, 100, 1000, 900))

    def test_grounding_parser_drops_degenerate_boxes(self) -> None:
        self.assertEqual(extract_grounding_boxes("<box>10 20 10 40</box>"), [])
        self.assertEqual(extract_grounding_boxes("普通文字回答"), [])

    def test_local_box_parser_accepts_bracket_format(self) -> None:
        self.assertEqual(extract_single_box("[10, 20, 30, 40]"), (10, 20, 30, 40))

    def test_local_box_parser_accepts_raw_special_token_format(self) -> None:
        self.assertEqual(
            extract_single_box("<box>100 300 160 350</box><|im_end|>"),
            (100, 300, 160, 350),
        )

    def test_standard_grounding_falls_back_to_raw_special_tokens(self) -> None:
        boxes = extract_response_grounding(
            "最上面的船只230 160 500 280",
            "<ref>最上面的船只</ref><box>230 160 500 280</box><|im_end|>",
            InferenceMode.QUALITY,
        )
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].label, "最上面的船只")
        self.assertEqual(boxes[0].coordinates, (230, 160, 500, 280))

    def test_precision_never_parses_local_raw_box_as_global(self) -> None:
        boxes = extract_response_grounding(
            "无法定位目标",
            "<box>100 300 160 350</box><|im_end|>",
            InferenceMode.PRECISION,
        )
        self.assertEqual(boxes, [])

    def test_inference_modes_are_strict(self) -> None:
        self.assertIs(normalize_inference_mode("quality"), InferenceMode.QUALITY)
        with self.assertRaises(ValueError):
            normalize_inference_mode("slow")


if __name__ == "__main__":
    unittest.main()
