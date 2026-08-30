#!/usr/bin/env python3
"""Write piped service output into date-partitioned log files."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, TextIO


class DailyLogWriter:
    def __init__(
        self,
        root: Path,
        filename: str,
        now: Callable[[], datetime] | None = None,
        run_id: str | None = None,
    ) -> None:
        self.root = root
        self.filename = filename
        self.run_id = run_id
        self.now = now or (lambda: datetime.now().astimezone())
        self._date = ""
        self._stream: TextIO | None = None

    def write(self, text: str) -> None:
        current_date = self.now().strftime("%Y-%m-%d")
        if current_date != self._date:
            self.close()
            directory = self.root / current_date
            if self.run_id:
                directory /= self.run_id
            directory.mkdir(parents=True, exist_ok=True)
            self._stream = (directory / self.filename).open("a", encoding="utf-8")
            self._date = current_date
        assert self._stream is not None
        self._stream.write(text)
        self._stream.flush()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


def main() -> None:
    if len(sys.argv) not in {3, 4}:
        raise SystemExit(
            "用法：daily_log.py <日志根目录> <文件名> [启动批次]"
        )
    writer = DailyLogWriter(
        Path(sys.argv[1]),
        sys.argv[2],
        run_id=sys.argv[3] if len(sys.argv) == 4 else None,
    )
    try:
        for line in sys.stdin:
            writer.write(line)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
