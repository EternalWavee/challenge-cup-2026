from __future__ import annotations

import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from PIL import Image

from app.main import (
    decode_image,
    model_service,
    render_grounding_visualization,
    save_conversation,
)
from app.model_service import GroundingBox


def png_bytes(size: tuple[int, int] = (32, 24)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")
    return buffer.getvalue()


class UploadValidationTest(unittest.TestCase):
    def test_valid_image_is_decoded_as_rgb(self) -> None:
        upload = UploadFile(
            filename="sample.png",
            file=io.BytesIO(png_bytes()),
            headers={"content-type": "image/png"},
        )
        image = asyncio.run(decode_image(upload))
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (32, 24))

    def test_non_image_content_type_is_rejected(self) -> None:
        upload = UploadFile(
            filename="sample.txt",
            file=io.BytesIO(b"not an image"),
            headers={"content-type": "text/plain"},
        )
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(decode_image(upload))
        self.assertEqual(raised.exception.status_code, 415)

    def test_invalid_image_payload_is_rejected(self) -> None:
        upload = UploadFile(
            filename="broken.png",
            file=io.BytesIO(b"broken"),
            headers={"content-type": "image/png"},
        )
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(decode_image(upload))
        self.assertEqual(raised.exception.status_code, 415)


class GroundingVisualizationTest(unittest.TestCase):
    def test_normalized_box_is_converted_to_pixels(self) -> None:
        image = Image.new("RGB", (1001, 501), "white")
        _, boxes = render_grounding_visualization(
            image,
            [GroundingBox("目标", (100, 200, 900, 800))],
        )
        self.assertEqual(boxes[0]["pixels"], [100, 100, 900, 400])
        self.assertEqual(boxes[0]["normalized"], [100, 200, 900, 800])

    def test_conversation_reuses_shared_visualization_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            visualizations = root / "visualizations"
            visualizations.mkdir()
            request_id = "00000000-0000-0000-0000-000000000001"
            image = Image.new("RGB", (16, 16), "white")
            shared = visualizations / f"{request_id}.png"
            image.save(shared, format="PNG")
            test_settings = SimpleNamespace(
                archive_dir=str(root / "conversations"),
                visualization_dir=str(visualizations),
            )

            with (
                patch("app.main.settings", test_settings),
                patch.object(model_service, "public_state", return_value={}),
            ):
                session = Path(
                    save_conversation(
                        request_id,
                        "定位目标",
                        [image],
                        "<ref>目标</ref><box>1 2 3 4</box>",
                        "<ref>目标</ref><box>1 2 3 4</box>",
                        10,
                        visualization=image,
                    )
                )

            archived = session / "grounding_visualization.png"
            self.assertEqual(shared.stat().st_ino, archived.stat().st_ino)


if __name__ == "__main__":
    unittest.main()
