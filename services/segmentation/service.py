"""Dependency-free local HTTP service for the SegFormer expert."""

from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .adapter import SegFormerExpert


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "models" / "segmentation"

expert = SegFormerExpert(
    config_path=os.environ.get(
        "SEGFORMER_CONFIG",
        str(MODEL_ROOT / "config.py"),
    ),
    checkpoint_path=os.environ.get(
        "SEGFORMER_CHECKPOINT",
        str(MODEL_ROOT / "checkpoint.pth"),
    ),
    output_root=os.environ.get(
        "SEGFORMER_OUTPUT_ROOT",
        str(PROJECT_ROOT / "runtime" / "segmentation"),
    ),
    device=os.environ.get("SEGFORMER_DEVICE", "cuda:0"),
)


class SegFormerHandler(BaseHTTPRequestHandler):
    server_version = "SegFormerExpert/1.0"

    def _send_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/health":
            self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Not found"})
            return
        self._send_json(HTTPStatus.OK, expert.health())

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/segment":
            self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1024 * 1024:
                raise ValueError("Invalid request body size")
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            image_path = request.get("image_path")
            mode = request.get("mode")
            targets = request.get("targets", [])
            if not isinstance(image_path, str) or not image_path:
                raise ValueError("image_path must be a non-empty string")
            if mode not in {"targeted", "all_classes"}:
                raise ValueError("mode must be 'targeted' or 'all_classes'")
            if not isinstance(targets, list) or not all(
                isinstance(target, str) for target in targets
            ):
                raise ValueError("targets must be a list of strings")

            result = expert.run(
                image_path=image_path,
                mode=mode,
                targets=targets,
                request_id=request.get("request_id"),
            )
            self._send_json(HTTPStatus.OK, result)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"detail": str(exc)})
        except Exception as exc:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"detail": str(exc)},
            )

    def log_message(self, message: str, *args: object) -> None:
        print(
            f"{self.client_address[0]} - {message % args}",
            flush=True,
        )


def main() -> None:
    host = os.environ.get("SEGFORMER_HOST", "127.0.0.1")
    port = int(os.environ.get("SEGFORMER_PORT", "8010"))
    server = ThreadingHTTPServer((host, port), SegFormerHandler)
    print(f"SegFormer expert listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
