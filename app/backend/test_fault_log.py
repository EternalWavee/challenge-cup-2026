from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from app.fault_log import record_fault


class FaultLogTest(unittest.TestCase):
    def test_fault_is_written_as_single_json_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record_fault(
                directory,
                component="minicpm",
                stage="inference",
                error=RuntimeError("simulated failure"),
                request_id="request-1",
                recovered=True,
            )
            date = datetime.now().astimezone().strftime("%Y-%m-%d")
            path = Path(directory) / date / "faults.jsonl"
            event = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(event["request_id"], "request-1")
        self.assertEqual(event["component"], "minicpm")
        self.assertEqual(event["stage"], "inference")
        self.assertEqual(event["error_type"], "RuntimeError")
        self.assertTrue(event["recovered"])


if __name__ == "__main__":
    unittest.main()
