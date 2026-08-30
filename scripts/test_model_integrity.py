from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from model_integrity import REQUIRED_FILES, generate, verify


class ModelIntegrityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "models" / "integrity-manifest.json"
        for index, relative in enumerate(REQUIRED_FILES):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"artifact-{index}".encode())

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_generated_manifest_verifies(self) -> None:
        generate(self.root, self.manifest)
        verify(self.root, self.manifest)

    def test_tampered_artifact_is_rejected(self) -> None:
        generate(self.root, self.manifest)
        (self.root / REQUIRED_FILES[-1]).write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            verify(self.root, self.manifest)


if __name__ == "__main__":
    unittest.main()
