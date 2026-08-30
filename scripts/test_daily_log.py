from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from daily_log import DailyLogWriter


class DailyLogWriterTest(unittest.TestCase):
    def test_output_rotates_when_date_changes(self) -> None:
        dates = iter((datetime(2026, 8, 13), datetime(2026, 8, 14)))
        with tempfile.TemporaryDirectory() as directory:
            writer = DailyLogWriter(Path(directory), "chat.log", lambda: next(dates))
            writer.write("first\n")
            writer.write("second\n")
            writer.close()
            first = Path(directory) / "2026-08-13" / "chat.log"
            second = Path(directory) / "2026-08-14" / "chat.log"
            self.assertEqual(first.read_text(encoding="utf-8"), "first\n")
            self.assertEqual(second.read_text(encoding="utf-8"), "second\n")

    def test_output_can_be_partitioned_by_startup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = DailyLogWriter(
                Path(directory),
                "chat.log",
                now=lambda: datetime(2026, 8, 30),
                run_id="203000-1234",
            )
            writer.write("ready\n")
            writer.close()
            output = (
                Path(directory)
                / "2026-08-30"
                / "203000-1234"
                / "chat.log"
            )
            self.assertEqual(output.read_text(encoding="utf-8"), "ready\n")


if __name__ == "__main__":
    unittest.main()
