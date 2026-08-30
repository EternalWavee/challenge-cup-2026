#!/usr/bin/env python3
"""Generate and verify SHA-256 checksums for deployed model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_FILES = (
    "models/terrascope/chat_template.jinja",
    "models/terrascope/config.json",
    "models/terrascope/generation_config.json",
    "models/terrascope/model.safetensors",
    "models/terrascope/preprocessor_config.json",
    "models/terrascope/processor_config.json",
    "models/terrascope/tokenizer.json",
    "models/terrascope/tokenizer_config.json",
    "models/segmentation/config.py",
    "models/segmentation/checkpoint.pth",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(root: Path, manifest_path: Path) -> None:
    files: dict[str, dict[str, int | str]] = {}
    for relative in REQUIRED_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"缺少模型文件：{relative}")
        files[relative] = {"size": path.stat().st_size, "sha256": sha256(path)}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "files": files,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"已生成模型完整性清单：{manifest_path}")


def verify(root: Path, manifest_path: Path) -> None:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少模型完整性清单：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(
        manifest.get("files"), dict
    ):
        raise ValueError("模型完整性清单格式无效")

    entries = manifest["files"]
    for relative in REQUIRED_FILES:
        entry = entries.get(relative)
        if not isinstance(entry, dict):
            raise ValueError(f"清单缺少模型文件：{relative}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"缺少模型文件：{relative}")
        actual_size = path.stat().st_size
        expected_size = entry.get("size")
        if actual_size != expected_size:
            raise ValueError(
                f"模型文件大小不一致：{relative}（期望 {expected_size}，实际 {actual_size}）"
            )
        actual_hash = sha256(path)
        if actual_hash != entry.get("sha256"):
            raise ValueError(f"模型文件SHA-256不一致：{relative}")
    print("模型完整性校验通过")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("generate", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest.resolve()
    try:
        if args.action == "generate":
            generate(root, manifest_path)
        else:
            verify(root, manifest_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError) as exc:
        print(f"模型完整性校验失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
