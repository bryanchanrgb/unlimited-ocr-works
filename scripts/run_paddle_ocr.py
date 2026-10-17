"""Run PaddleOCR PP-StructureV3 and save its raw document-analysis result."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def default_python() -> Path:
    """Return the PaddleOCR environment interpreter for the current platform."""
    executable = "python.exe" if os.name == "nt" else "python"
    return (
        PROJECT_ROOT
        / ".venv-paddleocr"
        / ("Scripts" if os.name == "nt" else "bin")
        / executable
    )


def parse_args() -> argparse.Namespace:
    """Parse the public PaddleOCR command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to a supported raster image")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path (default: data/processed/<input-name>/paddle_ocr/result.json)",
    )
    parser.add_argument(
        "--python",
        type=Path,
        help="PaddleOCR environment Python executable (default: .venv-paddleocr)",
    )
    parser.add_argument("--lang", default="en", help="PaddleOCR language (default: en)")
    return parser.parse_args()


def parse_worker_args(arguments: list[str]) -> argparse.Namespace:
    """Parse private worker arguments run by the PaddleOCR environment."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("lang")
    return parser.parse_args(arguments)


def validate_input(input_path: Path) -> None:
    """Ensure the source is a supported local image."""
    if not input_path.is_file():
        raise ValueError(f"Input file does not exist: {input_path}")
    if input_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError("Input must be a BMP, JPEG, PNG, TIFF, or WebP image")


def run_worker(args: argparse.Namespace) -> int:
    """Run PP-StructureV3 and persist its complete JSON-serializable result."""
    from paddleocr import PPStructureV3

    pipeline = PPStructureV3(
        lang=args.lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )
    pages: list[dict[str, Any]] = [
        result.json
        for result in pipeline.predict(
            str(args.input),
            use_table_recognition=True,
            use_ocr_results_with_table_cells=True,
        )
    ]
    payload = {
        "source": str(args.input),
        "pipeline": "PPStructureV3",
        "language": args.lang,
        "pages": pages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved PaddleOCR JSON to {args.output}")
    return 0


def main() -> int:
    """Launch PaddleOCR using its dedicated compatible Python environment."""
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        return run_worker(parse_worker_args(sys.argv[2:]))

    args = parse_args()
    input_path = args.input.expanduser().resolve()
    try:
        validate_input(input_path)
        output_path = (
            args.output.expanduser().resolve()
            if args.output
            else PROCESSED_DIR / input_path.stem / "paddle_ocr" / "result.json"
        )
        executable = (
            args.python.expanduser().resolve() if args.python else default_python()
        )
        if not executable.is_file():
            raise ValueError(
                f"PaddleOCR Python executable not found: {executable}. "
                "Create .venv-paddleocr with Python 3.13 and install "
                "paddlepaddle==3.2.2 paddleocr==3.7.0 paddlex[ocr]==3.7.2, "
                "or pass --python."
            )
        completed = subprocess.run(
            [
                str(executable),
                str(Path(__file__).resolve()),
                "--worker",
                str(input_path),
                str(output_path),
                args.lang,
            ],
            check=False,
        )
        return completed.returncode
    except (OSError, ValueError) as error:
        print(f"PaddleOCR failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
