from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path


_write_lock = threading.Lock()


def record_fault(
    log_dir: str,
    *,
    component: str,
    stage: str,
    error: BaseException,
    request_id: str | None = None,
    recovered: bool = False,
) -> None:
    now = datetime.now().astimezone()
    event = {
        "timestamp": now.isoformat(),
        "request_id": request_id,
        "component": component,
        "stage": stage,
        "error_type": type(error).__name__,
        "message": str(error)[:1000],
        "recovered": recovered,
    }
    output = Path(log_dir) / now.strftime("%Y-%m-%d") / "faults.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock, output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
