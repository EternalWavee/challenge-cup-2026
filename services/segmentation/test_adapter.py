from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from segmentation.adapter import EXPECTED_CLASSES, SegFormerExpert


class SyntheticSegFormerExpert(SegFormerExpert):
    """Exercise post-processing with a deterministic in-memory label map."""

    def _ensure_model(self) -> None:
        self._model = object()

    def _predict_label_map(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as image:
            width, height = image.size
        yy, xx = np.indices((height, width))
        return (
            (xx * len(self.classes) // max(width, 1))
            + (yy * len(self.classes) // max(height, 1))
        ).astype(np.uint8) % len(self.classes)


class SegFormerAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_path = self.root / "input.png"
        Image.new("RGB", (96, 64), (120, 130, 140)).save(self.image_path)
        self.expert = SyntheticSegFormerExpert(
            config_path=self.root / "unused.py",
            checkpoint_path=self.root / "unused.pth",
            output_root=self.root / "outputs",
            device="cpu",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_targeted_mode_masks_only_requested_classes(self) -> None:
        result = self.expert.run(
            image_path=self.image_path,
            mode="targeted",
            targets=["building"],
            request_id="targeted",
        )
        self.assertEqual(result["targets"], ["building"])
        self.assertEqual(tuple(result["palette"]), EXPECTED_CLASSES)
        self.assertEqual(tuple(result["artifacts"]["masks"]), ("building",))
        self.assertTrue(Path(result["artifacts"]["overlay"]).is_file())

    def test_all_classes_mode_returns_full_legend_metadata(self) -> None:
        result = self.expert.run(
            image_path=self.image_path,
            mode="all_classes",
            request_id="all-classes",
        )
        self.assertEqual(tuple(result["targets"]), EXPECTED_CLASSES)
        self.assertEqual(tuple(result["area_ratios"]), EXPECTED_CLASSES)
        self.assertEqual(result["artifacts"]["masks"], {})
        with Image.open(result["artifacts"]["overlay"]) as overlay:
            self.assertGreater(overlay.width, 96)
            self.assertGreaterEqual(overlay.height, 64)


if __name__ == "__main__":
    unittest.main()
