"""Shared filesystem layout for raw inputs and OCR provider outputs."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RASTER_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def provider_dir(input_stem: str, provider: str) -> Path:
    """Return the output directory for one document and provider."""
    return PROCESSED_DIR / input_stem / provider


def provider_result_path(
    input_stem: str, provider: str, filename: str = "result.json"
) -> Path:
    """Return the default result JSON path for one document and provider."""
    return provider_dir(input_stem, provider) / filename


def reconstruction_path(input_stem: str, source: str) -> Path:
    """Return the reconstruction HTML path for one document and source."""
    return provider_dir(input_stem, source) / "reconstruction.html"


def venv_python(venv_root: Path) -> Path:
    """Return the Python executable inside a virtual environment directory."""
    executable = "python.exe" if os.name == "nt" else "python"
    return venv_root / ("Scripts" if os.name == "nt" else "bin") / executable
