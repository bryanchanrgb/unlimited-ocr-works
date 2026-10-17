"""Create layout-preserving, token-efficient OCR JSON for LLM consumption."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT = DEFAULT_INPUT / "llm_optimized"
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for compacting OCR JSON files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Directory containing raw OCR JSON (default: data/processed)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for compact JSON (default: data/processed/llm_optimized)",
    )
    return parser.parse_args()


def basename(value: Any) -> str | None:
    """Return a source path's filename without retaining machine-specific paths."""
    return Path(value).name if isinstance(value, str) and value else None


def number(value: Any) -> int | float | None:
    """Return a JSON numeric value while excluding booleans and invalid data."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value


def box_from_values(value: Any) -> list[int | float] | None:
    """Return a four-coordinate box only when every coordinate is numeric."""
    if not isinstance(value, list) or len(value) != 4:
        return None
    coordinates = [number(coordinate) for coordinate in value]
    return (
        coordinates
        if all(coordinate is not None for coordinate in coordinates)
        else None
    )


def geometry(value: Any) -> dict[str, list[Any]] | None:
    """Compress a Textract geometry object while retaining non-rectangular quads."""
    if not isinstance(value, dict):
        return None
    bounds = value.get("BoundingBox")
    if not isinstance(bounds, dict):
        return None
    box = [number(bounds.get(key)) for key in ("Left", "Top", "Width", "Height")]
    if any(coordinate is None for coordinate in box):
        return None
    compact: dict[str, list[Any]] = {"b": box}
    polygon = value.get("Polygon")
    if isinstance(polygon, list) and len(polygon) == 4:
        quad = [
            [number(point.get("X")), number(point.get("Y"))]
            for point in polygon
            if isinstance(point, dict)
        ]
        if len(quad) == 4 and all(None not in point for point in quad):
            left, top, width, height = box
            rectangular = [
                [left, top],
                [left + width, top],
                [left + width, top + height],
                [left, top + height],
            ]
            if quad != rectangular:
                compact["q"] = quad
    return compact


def sort_key(item: dict[str, Any]) -> tuple[float, float]:
    """Return top-left geometry coordinates for visual reading-order sorting."""
    bounds = item.get("Geometry", {}).get("BoundingBox", {})
    top = number(bounds.get("Top")) if isinstance(bounds, dict) else None
    left = number(bounds.get("Left")) if isinstance(bounds, dict) else None
    return (float(top or 0), float(left or 0))


def child_ids(block: dict[str, Any], relationship_type: str = "CHILD") -> list[str]:
    """Return IDs from relationships of the requested Textract relationship type."""
    return [
        identifier
        for relationship in block.get("Relationships", [])
        if isinstance(relationship, dict)
        and relationship.get("Type") == relationship_type
        and isinstance(relationship.get("Ids"), list)
        for identifier in relationship["Ids"]
        if isinstance(identifier, str)
    ]


def descendant_text(
    identifier: str,
    blocks_by_id: dict[str, dict[str, Any]],
    visited: set[str] | None = None,
) -> str:
    """Collect a block's text descendants, including selected checkboxes."""
    seen = set() if visited is None else visited
    if identifier in seen:
        return ""
    seen.add(identifier)
    block = blocks_by_id.get(identifier)
    if block is None:
        return ""
    text = block.get("Text")
    parts = [text] if isinstance(text, str) and text else []
    if block.get("BlockType") == "SELECTION_ELEMENT":
        status = block.get("SelectionStatus")
        if isinstance(status, str):
            parts.append(f"[{status.lower()}]")
    for child_identifier in child_ids(block):
        child_text = descendant_text(child_identifier, blocks_by_id, seen)
        if child_text:
            parts.append(child_text)
    return " ".join(parts)


def descendants(
    identifier: str,
    blocks_by_id: dict[str, dict[str, Any]],
) -> set[str]:
    """Return all descendants of a Textract block without following cycles."""
    result: set[str] = set()
    pending = [identifier]
    while pending:
        current = pending.pop()
        if current in result:
            continue
        result.add(current)
        block = blocks_by_id.get(current)
        if block:
            pending.extend(child_ids(block))
    return result


def compact_textract(document: dict[str, Any]) -> dict[str, Any]:
    """Convert Textract blocks into ordered lines, tables, forms, and selections."""
    raw_blocks = document.get("blocks")
    if not isinstance(raw_blocks, list):
        raise TypeError("Textract JSON does not contain a blocks list")
    blocks = [
        block
        for block in raw_blocks
        if isinstance(block, dict) and isinstance(block.get("Id"), str)
    ]
    blocks_by_id = {block["Id"]: block for block in blocks}
    pages = [block for block in blocks if block.get("BlockType") == "PAGE"]
    if not pages:
        pages = [{"Id": "page-1"}]

    compact_pages: list[dict[str, Any]] = []
    for page_number, page in enumerate(pages, start=1):
        page_ids = descendants(page["Id"], blocks_by_id)
        if page["Id"] not in blocks_by_id:
            page_ids = {block["Id"] for block in blocks}
        page_blocks = [
            blocks_by_id[identifier]
            for identifier in page_ids
            if identifier in blocks_by_id
        ]
        tables = [block for block in page_blocks if block.get("BlockType") == "TABLE"]
        compact_tables = []
        for table in sorted(tables, key=sort_key):
            cells = []
            for cell_id in child_ids(table):
                cell = blocks_by_id.get(cell_id)
                if not cell or cell.get("BlockType") not in {"CELL", "MERGED_CELL"}:
                    continue
                cell_geometry = geometry(cell.get("Geometry"))
                if cell_geometry is None:
                    continue
                cells.append(
                    [
                        cell.get("RowIndex"),
                        cell.get("ColumnIndex"),
                        cell.get("RowSpan", 1),
                        cell.get("ColumnSpan", 1),
                        descendant_text(cell_id, blocks_by_id),
                        cell_geometry,
                        number(cell.get("Confidence")),
                    ]
                )
            compact_tables.append(
                {
                    "g": geometry(table.get("Geometry")),
                    "c": sorted(cells, key=lambda cell: (cell[0] or 0, cell[1] or 0)),
                }
            )

        compact_forms = []
        for key in page_blocks:
            entity_types = key.get("EntityTypes")
            if (
                key.get("BlockType") != "KEY_VALUE_SET"
                or not isinstance(entity_types, list)
                or "KEY" not in entity_types
            ):
                continue
            for value_id in child_ids(key, "VALUE"):
                value = blocks_by_id.get(value_id)
                if value is None:
                    continue
                compact_forms.append(
                    [
                        descendant_text(key["Id"], blocks_by_id),
                        geometry(key.get("Geometry")),
                        descendant_text(value_id, blocks_by_id),
                        geometry(value.get("Geometry")),
                    ]
                )

        lines = [
            [
                block["Text"],
                geometry(block.get("Geometry")),
                number(block.get("Confidence")),
            ]
            for block in sorted(page_blocks, key=sort_key)
            if block.get("BlockType") == "LINE" and isinstance(block.get("Text"), str)
        ]
        selections = [
            [block.get("SelectionStatus"), geometry(block.get("Geometry"))]
            for block in sorted(page_blocks, key=sort_key)
            if block.get("BlockType") == "SELECTION_ELEMENT"
        ]
        compact_pages.append(
            {
                "n": page_number,
                "lines": lines,
                "tables": compact_tables,
                "forms": compact_forms,
                "selections": selections,
            }
        )

    return {
        "v": SCHEMA_VERSION,
        "kind": "textract",
        "source": basename(document.get("source")),
        "pages": compact_pages,
    }


def compact_paddle(document: dict[str, Any]) -> dict[str, Any]:
    """Convert PaddleOCR regions, lines, and table predictions into compact JSON."""
    raw_pages = document.get("pages")
    if not isinstance(raw_pages, list):
        raise TypeError("PaddleOCR JSON does not contain a pages list")
    compact_pages = []
    for page_number, page in enumerate(raw_pages, start=1):
        result = page.get("res") if isinstance(page, dict) else None
        if not isinstance(result, dict):
            raise TypeError(f"PaddleOCR page {page_number} has no result object")
        dimensions = [number(result.get("width")), number(result.get("height"))]
        if None in dimensions:
            raise ValueError(f"PaddleOCR page {page_number} has no dimensions")
        regions = []
        for region in result.get("parsing_res_list", []):
            if not isinstance(region, dict):
                continue
            bounds = box_from_values(region.get("block_bbox"))
            content = region.get("block_content")
            if bounds is None or not isinstance(content, str):
                continue
            regions.append(
                [
                    region.get("block_label"),
                    bounds,
                    content,
                    region.get("block_order"),
                ]
            )
        regions.sort(
            key=lambda region: (
                region[3] is None,
                region[3] or 0,
                region[1][1],
                region[1][0],
            )
        )

        ocr = result.get("overall_ocr_res")
        if not isinstance(ocr, dict):
            ocr = {}
        lines = compact_paddle_lines(ocr)
        tables = []
        for table in result.get("table_res_list", []):
            if not isinstance(table, dict):
                continue
            table_ocr = table.get("table_ocr_pred")
            tables.append(
                {
                    "html": table.get("pred_html"),
                    "cells": [
                        bounds
                        for cell in table.get("cell_box_list", [])
                        if (bounds := box_from_values(cell)) is not None
                    ],
                    "lines": compact_paddle_lines(
                        table_ocr if isinstance(table_ocr, dict) else {}
                    ),
                }
            )
        compact_pages.append(
            {
                "n": page_number,
                "size": dimensions,
                "regions": regions,
                "lines": lines,
                "tables": tables,
            }
        )
    return {
        "v": SCHEMA_VERSION,
        "kind": "paddle_ocr",
        "source": basename(document.get("source")),
        "pages": compact_pages,
    }


def compact_paddle_lines(ocr: dict[str, Any]) -> list[list[Any]]:
    """Pair PaddleOCR text lines, bounding boxes, polygons, and confidence scores."""
    texts = ocr.get("rec_texts")
    boxes = ocr.get("rec_boxes")
    scores = ocr.get("rec_scores")
    polygons = ocr.get("rec_polys")
    if not all(isinstance(value, list) for value in (texts, boxes, scores, polygons)):
        return []
    compact = []
    for text, bounds, score, polygon in zip(
        texts, boxes, scores, polygons, strict=False
    ):
        if not isinstance(text, str) or (box := box_from_values(bounds)) is None:
            continue
        line: list[Any] = [text, box, number(score)]
        if isinstance(polygon, list):
            line.append(polygon)
        compact.append(line)
    return compact


def compact_unlimited(document: dict[str, Any]) -> dict[str, Any]:
    """Convert Unlimited-OCR Markdown and detected-object geometry into compact JSON."""
    dimensions = document.get("image_dimensions")
    if not isinstance(dimensions, list):
        raise TypeError("Unlimited-OCR JSON does not contain image dimensions")
    generations = document.get("generations")
    if not isinstance(generations, list):
        raise TypeError("Unlimited-OCR JSON does not contain generations")
    objects = []
    unparsed = []
    for generation in generations:
        if not isinstance(generation, dict):
            continue
        for detected in generation.get("objects", []):
            if not isinstance(detected, dict):
                continue
            boxes = [
                box
                for value in detected.get("boxes_pixels_xyxy", [])
                if (box := box_from_values(value)) is not None
            ]
            objects.append(
                [
                    detected.get("page_index"),
                    detected.get("label"),
                    boxes,
                    detected.get("reference_text"),
                ]
            )
        unparsed.extend(
            value
            for value in generation.get("unparsed_detections", [])
            if isinstance(value, str)
        )
    return {
        "v": SCHEMA_VERSION,
        "kind": "unlimited_ocr",
        "source": [basename(value) for value in document.get("input_files", [])],
        "sizes": [
            [number(item.get("width")), number(item.get("height"))]
            for item in dimensions
            if isinstance(item, dict)
        ],
        "markdown": document.get("processed_markdown"),
        "objects": objects,
        "unparsed_detections": unparsed,
    }


def compact_run_metadata(document: dict[str, Any]) -> dict[str, Any]:
    """Retain provider outcomes while dropping commands, timings, and diagnostics."""
    images = document.get("images")
    if not isinstance(images, list):
        raise TypeError("OCR run metadata does not contain an images list")
    compact_images = []
    for image in images:
        if not isinstance(image, dict):
            continue
        providers = []
        for provider in image.get("providers", []):
            if not isinstance(provider, dict):
                continue
            providers.append([provider.get("provider"), provider.get("returncode")])
        compact_images.append([basename(image.get("input")), providers])
    return {
        "v": SCHEMA_VERSION,
        "kind": "ocr_run_metadata",
        "success": document.get("success"),
        "images": compact_images,
    }


def compact_document(document: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a raw OCR document to its provider-specific compact converter."""
    if "blocks" in document:
        return compact_textract(document)
    if "pages" in document and document.get("pipeline") == "PPStructureV3":
        return compact_paddle(document)
    if "processed_markdown" in document and "generations" in document:
        return compact_unlimited(document)
    if "images" in document and "success" in document:
        return compact_run_metadata(document)
    raise ValueError("Unrecognized OCR JSON schema")


def input_files(input_directory: Path, output_directory: Path) -> Iterable[Path]:
    """Yield raw JSON files without reprocessing files already written as output."""
    output_resolved = output_directory.resolve()
    for path in sorted(input_directory.rglob("*.json")):
        if output_resolved not in path.resolve().parents:
            yield path


def write_compact_file(
    input_path: Path,
    input_directory: Path,
    output_directory: Path,
) -> tuple[Path, int, int]:
    """Compact one JSON file and return its output path and byte counts."""
    raw_text = input_path.read_text(encoding="utf-8")
    document = json.loads(raw_text)
    if not isinstance(document, dict):
        raise TypeError(f"Expected a JSON object in {input_path}")
    compact = compact_document(document)
    output_path = output_directory / input_path.relative_to(input_directory)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compact_text = json.dumps(compact, ensure_ascii=False, separators=(",", ":")) + "\n"
    output_path.write_text(compact_text, encoding="utf-8")
    return output_path, len(raw_text.encode("utf-8")), len(compact_text.encode("utf-8"))


def main() -> int:
    """Compact every supported OCR JSON file and print a size comparison."""
    args = parse_args()
    input_directory = args.input.expanduser().resolve()
    output_directory = args.output.expanduser().resolve()
    if not input_directory.is_dir():
        raise ValueError(f"Input directory does not exist: {input_directory}")
    if input_directory == output_directory:
        raise ValueError("Output directory must differ from the input directory")

    total_before = 0
    total_after = 0
    count = 0
    for input_path in input_files(input_directory, output_directory):
        output_path, before, after = write_compact_file(
            input_path, input_directory, output_directory
        )
        total_before += before
        total_after += after
        count += 1
        print(
            f"{input_path.relative_to(input_directory)} -> {output_path.relative_to(output_directory)}: {before:,} -> {after:,} bytes"
        )
    if not count:
        raise ValueError(f"No JSON files found in {input_directory}")
    reduction = (1 - total_after / total_before) * 100 if total_before else 0
    print(
        f"Total: {total_before:,} -> {total_after:,} bytes ({reduction:.1f}% smaller)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
