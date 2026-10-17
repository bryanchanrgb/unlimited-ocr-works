"""Run the Universal repo's OCR demo and preserve raw generations and detections."""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = PROJECT_ROOT.parent / "Unlimited-OCR-Universal"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
EVENT_PATTERN = re.compile(
    r"<\|ref\|>(.*?)<\|/ref\|>|<\|det\|>(.*?)<\|/det\|>", re.DOTALL
)


def add_common_arguments(parser: argparse.ArgumentParser, prompt: str) -> None:
    """Add options shared by all OCR input modes."""
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(os.environ.get("UNLIMITED_OCR_REPO", DEFAULT_REPO)),
        help="Unlimited-OCR-Universal checkout (default: sibling git checkout)",
    )
    parser.add_argument(
        "--python",
        type=Path,
        help="Repo environment's Python executable (default: <repo>/.venv)",
    )
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--prompt", default=prompt)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=35)
    parser.add_argument("--ngram-window", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)


def parse_args() -> argparse.Namespace:
    """Parse the public image, multi-image, and PDF command-line forms."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="kind", required=True)

    image_parser = subparsers.add_parser("image", help="Run a single image")
    image_parser.add_argument("image", type=Path)
    image_parser.add_argument("--mode", choices=("gundam", "base"), default="gundam")
    add_common_arguments(image_parser, "<image>document parsing.")
    image_parser.set_defaults(ngram_window=128)

    multi_parser = subparsers.add_parser("multi", help="Run multiple images as pages")
    multi_parser.add_argument("images", nargs="+", type=Path)
    add_common_arguments(multi_parser, "<image>Multi page parsing.")
    multi_parser.set_defaults(ngram_window=1024)

    pdf_parser = subparsers.add_parser("pdf", help="Render and run a PDF")
    pdf_parser.add_argument("pdf", type=Path)
    pdf_parser.add_argument("--dpi", type=int, default=300)
    add_common_arguments(pdf_parser, "<image>Multi page parsing.")
    pdf_parser.set_defaults(ngram_window=1024)

    return parser.parse_args()


def parse_worker_args(arguments: list[str]) -> argparse.Namespace:
    """Parse the private worker arguments used inside the model repository's venv."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("kind", choices=("image", "multi", "pdf"))
    parser.add_argument("repo", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("prompt")
    parser.add_argument("mode")
    parser.add_argument("dpi", type=int)
    parser.add_argument("max_length", type=int)
    parser.add_argument("ngram_size", type=int)
    parser.add_argument("ngram_window", type=int)
    parser.add_argument("temperature", type=float)
    parser.add_argument("inputs", nargs="+", type=Path)
    return parser.parse_args(arguments)


def worker_python(repo: Path, requested_python: Path | None) -> Path:
    """Return the Python executable in the model repo's virtual environment."""
    if requested_python is not None:
        executable = requested_python.expanduser().resolve()
    else:
        relative_executable = (
            Path("Scripts") / "python.exe"
            if os.name == "nt"
            else Path("bin") / "python"
        )
        executable = repo / ".venv" / relative_executable
    if not executable.is_file():
        raise ValueError(
            f"Python executable not found: {executable}. Create the model repo's "
            "environment first or pass --python."
        )
    return executable


def launch_worker(args: argparse.Namespace) -> int:
    """Run this script under the model repository's own Python environment."""
    repo = args.repo.expanduser().resolve()
    if not (repo / "demo" / "run_ocr.py").is_file():
        raise ValueError(f"No demo/run_ocr.py found in model repository: {repo}")
    if not args.prompt.startswith("<image>"):
        raise ValueError("The model prompt must begin with the literal '<image>' token")

    inputs = [path.expanduser().resolve() for path in args.inputs]
    for input_path in inputs:
        if not input_path.is_file():
            raise ValueError(f"Input file does not exist: {input_path}")
    if args.kind == "pdf":
        if len(inputs) != 1 or inputs[0].suffix.lower() != ".pdf":
            raise ValueError("pdf mode requires exactly one PDF input")
    elif any(path.suffix.lower() not in IMAGE_SUFFIXES for path in inputs):
        raise ValueError("image and multi modes accept image files only")

    output = (
        args.output.expanduser().resolve()
        if args.output
        else PROCESSED_DIR / inputs[0].stem / "unlimited_ocr"
    )
    output.mkdir(parents=True, exist_ok=True)
    python = worker_python(repo, args.python)
    worker_arguments = [
        str(python),
        str(Path(__file__).resolve()),
        "--worker",
        args.kind,
        str(repo),
        str(output),
        args.prompt,
        getattr(args, "mode", "base"),
        str(getattr(args, "dpi", 300)),
        str(args.max_length),
        str(args.no_repeat_ngram_size),
        str(args.ngram_window),
        str(args.temperature),
        *(str(path) for path in inputs),
    ]
    completed = subprocess.run(worker_arguments, check=False)
    return completed.returncode


def import_demo(repo: Path) -> Any:
    """Load the repository's existing demo helpers without copying their logic."""
    demo_path = repo / "demo" / "run_ocr.py"
    spec = importlib.util.spec_from_file_location("unlimited_ocr_demo", demo_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the OCR demo from {demo_path}")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    return demo


def normalize_boxes(value: Any) -> list[list[int | float]] | None:
    """Normalize a parsed coordinate list into one or more four-value boxes."""
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) == 4 and all(
        isinstance(coordinate, (int, float)) and not isinstance(coordinate, bool)
        for coordinate in value
    ):
        return [list(value)]
    if all(
        isinstance(box, (list, tuple))
        and len(box) == 4
        and all(
            isinstance(coordinate, (int, float)) and not isinstance(coordinate, bool)
            for coordinate in box
        )
        for box in value
    ):
        return [list(box) for box in value]
    return None


def pixel_boxes(
    boxes: list[list[int | float]] | None,
    dimensions: dict[str, int] | None,
) -> list[list[int]] | None:
    """Scale model coordinates (0–999) into source-image pixel bounds."""
    if boxes is None or dimensions is None:
        return None
    width = dimensions["width"]
    height = dimensions["height"]
    return [
        [
            int(box[0] / 999 * width),
            int(box[1] / 999 * height),
            int(box[2] / 999 * width),
            int(box[3] / 999 * height),
        ]
        for box in boxes
    ]


def extract_objects(
    raw_text: str,
    image_dimensions: list[dict[str, int]],
) -> dict[str, Any]:
    """Extract ref/det tags into structured detections without discarding raw text."""
    page_parts = raw_text.split("<PAGE>")
    if len(page_parts) > 1:
        page_segments = [(index, text) for index, text in enumerate(page_parts[1:])]
    else:
        page_index = 0 if len(image_dimensions) == 1 else None
        page_segments = [(page_index, raw_text)]

    objects: list[dict[str, Any]] = []
    unmatched_references: list[dict[str, Any]] = []
    unparsed_detections: list[dict[str, Any]] = []

    for page_index, text in page_segments:
        pending_reference: str | None = None
        for event in EVENT_PATTERN.finditer(text):
            reference, detection = event.groups()
            if reference is not None:
                pending_reference = reference
                continue

            if detection is None:
                continue
            open_bracket = detection.find("[")
            if open_bracket < 0:
                unparsed_detections.append(
                    {
                        "page_index": page_index,
                        "reference_text": pending_reference,
                        "raw_detection": detection,
                    }
                )
                pending_reference = None
                continue

            label = detection[:open_bracket].strip()
            try:
                parsed_coordinates = ast.literal_eval(detection[open_bracket:])
            except (SyntaxError, ValueError):  # fmt: skip
                parsed_coordinates = None
            boxes = normalize_boxes(parsed_coordinates)
            if boxes is None:
                unparsed_detections.append(
                    {
                        "page_index": page_index,
                        "reference_text": pending_reference,
                        "raw_detection": detection,
                    }
                )
                pending_reference = None
                continue

            dimensions = (
                image_dimensions[page_index]
                if page_index is not None and page_index < len(image_dimensions)
                else None
            )
            objects.append(
                {
                    "page_index": page_index,
                    "reference_text": pending_reference,
                    "label": label or None,
                    "boxes_normalized_999": boxes,
                    "boxes_pixels_xyxy": pixel_boxes(boxes, dimensions),
                    "image_dimensions": dimensions,
                    "raw_detection": detection,
                }
            )
            pending_reference = None

        if pending_reference is not None:
            unmatched_references.append(
                {"page_index": page_index, "reference_text": pending_reference}
            )

    return {
        "objects": objects,
        "unmatched_references": unmatched_references,
        "unparsed_detections": unparsed_detections,
    }


def source_dimensions(image_paths: list[Path]) -> list[dict[str, int]]:
    """Read the EXIF-corrected dimensions of source images used in this run."""
    from PIL import Image, ImageOps

    dimensions = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            corrected = ImageOps.exif_transpose(image)
            dimensions.append({"width": corrected.width, "height": corrected.height})
    return dimensions


def write_utf8_stdout(text: str) -> None:
    """Write Unicode OCR output without relying on the Windows console encoding."""
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


@contextlib.contextmanager
def quiet_model_output() -> Generator[None]:
    """Mute model output using UTF-8 streams that accept all OCR characters."""
    with (
        open(os.devnull, "w", encoding="utf-8") as devnull,
        contextlib.redirect_stdout(devnull),
        contextlib.redirect_stderr(devnull),
    ):
        yield


def run_worker(args: argparse.Namespace) -> int:
    """Run model inference, retaining demo artifacts and complete raw generations."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    repo = args.repo.resolve()
    demo = import_demo(repo)
    device = demo.pick_device()
    dtype = "bfloat16" if device == "cuda" else "float32"
    print(
        f"Loading Unlimited-OCR on {device!r} ({dtype}); running OCR (this is slow)...",
        file=sys.stderr,
        flush=True,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    if args.kind == "pdf":
        pages_dir = args.output / "input_pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        image_paths = [
            Path(path)
            for path in demo.pdf_to_images(args.inputs[0], pages_dir, args.dpi)
        ]
    else:
        image_paths = args.inputs
    dimensions = source_dimensions(image_paths)

    generations: list[dict[str, Any]] = []
    with quiet_model_output():
        tokenizer, model = demo.load(device)
        original_generate = model.generate

        def capture_generate(*call_args: Any, **kwargs: Any) -> Any:
            """Capture returned prompt/output token IDs while preserving model behavior."""
            input_ids = kwargs.get("input_ids")
            if input_ids is None and call_args:
                input_ids = call_args[0]
            generated = original_generate(*call_args, **kwargs)
            sequences = getattr(generated, "sequences", generated)
            if isinstance(sequences, tuple):
                sequences = sequences[0]
            sequence_rows = sequences.detach().to("cpu").tolist()
            prompt_rows = input_ids.detach().to("cpu").tolist()
            for prompt_ids, sequence_ids in zip(
                prompt_rows, sequence_rows, strict=True
            ):
                continuation_ids = sequence_ids[len(prompt_ids) :]
                raw_text = tokenizer.decode(
                    continuation_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                generations.append(
                    {
                        "prompt_token_ids": prompt_ids,
                        "generated_token_ids": continuation_ids,
                        "full_sequence_token_ids": sequence_ids,
                        "raw_generated_text": raw_text,
                    }
                )
            return generated

        model.generate = capture_generate
        if args.kind == "image":
            image_size, crop_mode = (
                (640, True) if args.mode == "gundam" else (1024, False)
            )
            model.infer(
                tokenizer,
                prompt=args.prompt,
                image_file=str(image_paths[0]),
                output_path=str(args.output),
                base_size=1024,
                image_size=image_size,
                crop_mode=crop_mode,
                max_length=args.max_length,
                no_repeat_ngram_size=args.ngram_size,
                ngram_window=args.ngram_window,
                temperature=args.temperature,
                save_results=True,
            )
        else:
            model.infer_multi(
                tokenizer,
                prompt=args.prompt,
                image_files=[str(path) for path in image_paths],
                output_path=str(args.output),
                image_size=1024,
                max_length=args.max_length,
                no_repeat_ngram_size=args.ngram_size,
                ngram_window=args.ngram_window,
                temperature=args.temperature,
                save_results=True,
            )

    result_path = args.output / "result.md"
    processed_markdown = (
        result_path.read_text(encoding="utf-8") if result_path.is_file() else None
    )
    capture_path = args.output / "raw_generation.json"
    capture_document = {
        "model_repository": str(repo),
        "input_files": [str(path) for path in args.inputs],
        "rendered_input_pages": (
            [str(path) for path in image_paths] if args.kind == "pdf" else []
        ),
        "image_dimensions": dimensions,
        "run_parameters": {
            "kind": args.kind,
            "prompt": args.prompt,
            "device": device,
            "dtype": dtype,
            "mode": args.mode if args.kind == "image" else "base",
            "image_size": 640
            if args.kind == "image" and args.mode == "gundam"
            else 1024,
            "crop_mode": args.kind == "image" and args.mode == "gundam",
            "max_length": args.max_length,
            "no_repeat_ngram_size": args.ngram_size,
            "ngram_window": args.ngram_window,
            "temperature": args.temperature,
            "pdf_dpi": args.dpi if args.kind == "pdf" else None,
        },
        "processed_markdown": processed_markdown,
        "generations": [
            {
                **generation,
                **extract_objects(generation["raw_generated_text"], dimensions),
            }
            for generation in generations
        ],
        "existing_demo_outputs": {
            "markdown": str(result_path) if result_path.is_file() else None,
            "geometry_visualization": (
                str(args.output / "geo.jpg")
                if (args.output / "geo.jpg").is_file()
                else None
            ),
            "annotated_images": [
                str(path) for path in sorted(args.output.glob("result_with_boxes*.jpg"))
            ],
            "image_crops": [
                str(path) for path in sorted((args.output / "images").glob("*.jpg"))
            ],
        },
    }
    capture_path.write_text(
        json.dumps(capture_document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if processed_markdown is not None:
        write_utf8_stdout(processed_markdown)
        print(f"\nSaved to: {result_path}", file=sys.stderr)
    else:
        print(f"No result written to {result_path}", file=sys.stderr)
    print(
        f"Raw generations and structured detections saved to: {capture_path}",
        file=sys.stderr,
    )
    return 0 if processed_markdown is not None else 1


def main() -> int:
    """Dispatch to the outer launcher or inner model-environment worker."""
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        try:
            return run_worker(parse_worker_args(sys.argv[2:]))
        except Exception as error:  # noqa: BLE001 - report any model/runtime failure at the CLI boundary
            print(f"Unlimited-OCR failed: {error}", file=sys.stderr)
            return 1

    args = parse_args()
    try:
        if not args.prompt.startswith("<image>"):
            raise ValueError("The prompt must begin with the literal '<image>' token")
        if args.kind == "image":
            inputs = [args.image]
        elif args.kind == "multi":
            inputs = args.images
        else:
            inputs = [args.pdf]

        output = (
            args.output.expanduser().resolve()
            if args.output
            else PROCESSED_DIR / inputs[0].stem / "unlimited_ocr"
        )
        return launch_worker(
            argparse.Namespace(
                repo=args.repo,
                python=args.python,
                output=output,
                prompt=args.prompt,
                kind=args.kind,
                mode=getattr(args, "mode", "base"),
                dpi=getattr(args, "dpi", 300),
                max_length=args.max_length,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                ngram_window=args.ngram_window,
                temperature=args.temperature,
                inputs=inputs,
            )
        )
    except (OSError, ValueError) as error:
        print(f"Unlimited-OCR failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
