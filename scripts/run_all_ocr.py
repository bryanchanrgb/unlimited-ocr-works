"""Run every configured OCR provider for one or more raster images."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse batch OCR inputs and provider-specific options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more raster images to process",
    )
    parser.add_argument("--region", help="AWS region for Textract")
    parser.add_argument(
        "--unlimited-mode",
        choices=("gundam", "base"),
        default="gundam",
        help="Unlimited-OCR image mode (default: gundam)",
    )
    parser.add_argument(
        "--paddle-python",
        type=Path,
        help="PaddleOCR environment Python executable",
    )
    parser.add_argument("--paddle-lang", default="en", help="PaddleOCR language")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROCESSED_DIR / "ocr_run_metadata.json",
        help="Batch metadata JSON path (default: data/processed/ocr_run_metadata.json)",
    )
    return parser.parse_args()


def timestamp() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(UTC).isoformat()


def validate_input(path: Path) -> Path:
    """Return a verified absolute raster-image path."""
    input_path = path.expanduser().resolve()
    if not input_path.is_file():
        raise ValueError(f"Input file does not exist: {input_path}")
    if input_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image type: {input_path}")
    return input_path


def run_provider(name: str, command: list[str]) -> dict[str, Any]:
    """Run one provider and capture timing, status, and diagnostic output."""
    started_at = timestamp()
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    return {
        "provider": name,
        "command": command,
        "started_at": started_at,
        "finished_at": timestamp(),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def provider_commands(
    args: argparse.Namespace, input_path: Path
) -> list[tuple[str, list[str]]]:
    """Build the three OCR provider commands for one input image."""
    scripts_dir = PROJECT_ROOT / "scripts"
    textract = [
        sys.executable,
        str(scripts_dir / "run_textract.py"),
        str(input_path),
    ]
    if args.region:
        textract.extend(("--region", args.region))

    unlimited = [
        sys.executable,
        str(scripts_dir / "run_unlimited_ocr.py"),
        "image",
        str(input_path),
        "--mode",
        args.unlimited_mode,
    ]

    paddle = [
        sys.executable,
        str(scripts_dir / "run_paddle_ocr.py"),
        str(input_path),
        "--lang",
        args.paddle_lang,
    ]
    if args.paddle_python:
        paddle.extend(("--python", str(args.paddle_python.expanduser().resolve())))
    return [
        ("textract", textract),
        ("unlimited_ocr", unlimited),
        ("paddle_ocr", paddle),
    ]


def main() -> int:
    """Run all providers, write complete batch metadata, and return aggregate status."""
    args = parse_args()
    try:
        inputs = [validate_input(path) for path in args.inputs]
        metadata_path = args.metadata.expanduser().resolve()
        batch = {
            "started_at": timestamp(),
            "parameters": {
                "region": args.region,
                "unlimited_mode": args.unlimited_mode,
                "paddle_lang": args.paddle_lang,
                "paddle_python": (
                    str(args.paddle_python.expanduser().resolve())
                    if args.paddle_python
                    else None
                ),
            },
            "images": [],
        }
        failures = 0
        for input_path in inputs:
            print(f"Processing {input_path.name}")
            image_run: dict[str, Any] = {
                "input": str(input_path),
                "providers": [],
            }
            for name, command in provider_commands(args, input_path):
                print(f"  Running {name}...")
                result = run_provider(name, command)
                image_run["providers"].append(result)
                failures += result["return_code"] != 0
                print(f"  {name}: {result['runtime_seconds']:.3f}s")
            batch["images"].append(image_run)
        batch["finished_at"] = timestamp()
        batch["success"] = failures == 0
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(batch, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved OCR run metadata to {metadata_path}")
        return 0 if failures == 0 else 1
    except (OSError, ValueError) as error:
        print(f"Batch OCR failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
