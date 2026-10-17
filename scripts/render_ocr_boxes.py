"""Render Textract and Unlimited-OCR bounding boxes over a raw image in HTML."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    """Parse paths for the raw image, OCR results, and rendered HTML."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Raw PNG or JPEG image")
    parser.add_argument("--textract", type=Path, help="Textract result JSON path")
    parser.add_argument(
        "--unlimited",
        type=Path,
        help="Unlimited-OCR raw_generation.json path",
    )
    parser.add_argument("--paddle", type=Path, help="PaddleOCR result JSON path")
    parser.add_argument("--output", type=Path, help="Rendered HTML output path")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from a UTF-8 file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def image_data_url(path: Path) -> str:
    """Return an image file as a self-contained data URL."""
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type not in {"image/jpeg", "image/png"}:
        raise ValueError("The raw image must be a PNG or JPEG file")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def child_ids(block: dict[str, Any]) -> list[str]:
    """Return the IDs of every direct child in a Textract block."""
    return [
        child_id
        for relationship in block.get("Relationships", [])
        if relationship.get("Type") == "CHILD"
        for child_id in relationship.get("Ids", [])
    ]


def textract_text(block_id: str, blocks_by_id: dict[str, dict[str, Any]]) -> str | None:
    """Collect text from a block or its descendants, without traversing cycles."""
    texts: list[str] = []
    visited: set[str] = set()

    def visit(current_id: str) -> None:
        """Visit one block and recursively collect its descendant text."""
        if current_id in visited:
            return
        visited.add(current_id)
        block = blocks_by_id.get(current_id)
        if block is None:
            return
        text = block.get("Text")
        if isinstance(text, str) and text:
            texts.append(text)
        for child_id in child_ids(block):
            visit(child_id)

    visit(block_id)
    return " ".join(texts) or None


def textract_depths(blocks_by_id: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Calculate block depth so descendants receive hover precedence over parents."""
    parents: dict[str, list[str]] = {block_id: [] for block_id in blocks_by_id}
    for block_id, block in blocks_by_id.items():
        for child_id in child_ids(block):
            if child_id in parents:
                parents[child_id].append(block_id)

    depths: dict[str, int] = {}

    def depth(block_id: str, visiting: set[str]) -> int:
        """Return the maximum ancestor depth for one block."""
        if block_id in depths:
            return depths[block_id]
        if block_id in visiting:
            return 0
        value = max(
            (
                depth(parent_id, visiting | {block_id}) + 1
                for parent_id in parents[block_id]
            ),
            default=0,
        )
        depths[block_id] = value
        return value

    return {block_id: depth(block_id, set()) for block_id in blocks_by_id}


def textract_boxes(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize every Textract block with a bounding box for browser rendering."""
    raw_blocks = document.get("blocks")
    if not isinstance(raw_blocks, list):
        raise TypeError("Textract JSON does not contain a blocks array")
    blocks = [
        block
        for block in raw_blocks
        if isinstance(block, dict) and isinstance(block.get("Id"), str)
    ]
    blocks_by_id = {block["Id"]: block for block in blocks}
    depths = textract_depths(blocks_by_id)
    rendered: list[dict[str, Any]] = []
    for block in blocks:
        bounding_box = block.get("Geometry", {}).get("BoundingBox")
        if not isinstance(bounding_box, dict):
            continue
        try:
            left = float(bounding_box["Left"])
            top = float(bounding_box["Top"])
            width = float(bounding_box["Width"])
            height = float(bounding_box["Height"])
        except (KeyError, TypeError, ValueError):  # fmt: skip
            continue
        block_id = block["Id"]
        rendered.append(
            {
                "tool": "textract",
                "type": block.get("BlockType", "UNKNOWN"),
                "text": textract_text(block_id, blocks_by_id),
                "id": block_id,
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "depth": depths[block_id],
                "details": {
                    "confidence": block.get("Confidence"),
                    "entity_types": block.get("EntityTypes"),
                    "selection_status": block.get("SelectionStatus"),
                    "row_index": block.get("RowIndex"),
                    "column_index": block.get("ColumnIndex"),
                    "row_span": block.get("RowSpan"),
                    "column_span": block.get("ColumnSpan"),
                },
            }
        )
    return rendered


def unlimited_boxes(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize every parsed Unlimited-OCR detection box for browser rendering."""
    generations = document.get("generations")
    if not isinstance(generations, list):
        raise TypeError("Unlimited-OCR JSON does not contain a generations array")
    rendered: list[dict[str, Any]] = []
    for generation_index, generation in enumerate(generations):
        if not isinstance(generation, dict):
            continue
        objects = generation.get("objects", [])
        if not isinstance(objects, list):
            continue
        detected_contents = [
            content.strip()
            for content in re.findall(
                r"<\|det\|>.*?<\|/det\|>(.*?)(?=<\|det\|>|$)",
                generation.get("raw_generated_text", ""),
                re.DOTALL,
            )
        ]
        for object_index, detected_object in enumerate(objects):
            if not isinstance(detected_object, dict):
                continue
            dimensions = detected_object.get("image_dimensions")
            boxes = detected_object.get("boxes_pixels_xyxy")
            if not isinstance(dimensions, dict) or not isinstance(boxes, list):
                continue
            try:
                image_width = float(dimensions["width"])
                image_height = float(dimensions["height"])
            except (KeyError, TypeError, ValueError):  # fmt: skip
                continue
            for box_index, box in enumerate(boxes):
                if not isinstance(box, list) or len(box) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = (float(value) for value in box)
                except (TypeError, ValueError):  # fmt: skip
                    continue
                rendered.append(
                    {
                        "tool": "unlimited",
                        "type": detected_object.get("label") or "UNLABELED",
                        "text": (
                            detected_object.get("reference_text")
                            or indexed_value(detected_contents, object_index)
                        ),
                        "id": f"generation-{generation_index}-object-{object_index}-box-{box_index}",
                        "left": x1 / image_width,
                        "top": y1 / image_height,
                        "width": (x2 - x1) / image_width,
                        "height": (y2 - y1) / image_height,
                        "depth": 0,
                        "details": {
                            "page_index": detected_object.get("page_index"),
                            "box_pixels_xyxy": box,
                            "box_normalized_999": detected_object.get(
                                "boxes_normalized_999", []
                            )[box_index],
                            "raw_detection": detected_object.get("raw_detection"),
                        },
                    }
                )
    return rendered


def normalized_pixel_box(
    box: Any,
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float] | None:
    """Convert a pixel xyxy box into normalized left/top/width/height values."""
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError):  # fmt: skip
        return None
    return (
        x1 / image_width,
        y1 / image_height,
        (x2 - x1) / image_width,
        (y2 - y1) / image_height,
    )


def indexed_value(values: Any, index: int) -> Any:
    """Return an indexed list value or None when optional metadata is absent."""
    if isinstance(values, list) and index < len(values):
        return values[index]
    return None


def paddle_boxes(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize PaddleOCR layout, text-line, and table-cell boxes."""
    pages = document.get("pages")
    if not isinstance(pages, list):
        raise TypeError("PaddleOCR JSON does not contain a pages array")
    rendered: list[dict[str, Any]] = []

    def append_box(
        box: Any,
        image_width: float,
        image_height: float,
        box_type: str,
        text: Any,
        identifier: str,
        depth: int,
        details: dict[str, Any],
    ) -> None:
        """Append one normalized PaddleOCR box when its geometry is valid."""
        normalized = normalized_pixel_box(box, image_width, image_height)
        if normalized is None:
            return
        left, top, width, height = normalized
        rendered.append(
            {
                "tool": "paddle",
                "type": box_type,
                "text": text if isinstance(text, str) else None,
                "id": identifier,
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "depth": depth,
                "details": details,
            }
        )

    for page_index, page in enumerate(pages):
        if not isinstance(page, dict) or not isinstance(page.get("res"), dict):
            continue
        result = page["res"]
        try:
            image_width = float(result["width"])
            image_height = float(result["height"])
        except (KeyError, TypeError, ValueError):  # fmt: skip
            continue
        for block in result.get("parsing_res_list", []):
            if not isinstance(block, dict):
                continue
            append_box(
                block.get("block_bbox"),
                image_width,
                image_height,
                str(block.get("block_label", "LAYOUT")),
                block.get("block_content"),
                f"page-{page_index}-layout-{block.get('block_id')}",
                0,
                {
                    "page_index": page_index,
                    "block_order": block.get("block_order"),
                    "source": "parsing_res_list",
                },
            )
        ocr_result = result.get("overall_ocr_res", {})
        if isinstance(ocr_result, dict):
            for text_index, box in enumerate(ocr_result.get("rec_boxes", [])):
                append_box(
                    box,
                    image_width,
                    image_height,
                    "TEXT_LINE",
                    indexed_value(ocr_result.get("rec_texts"), text_index),
                    f"page-{page_index}-text-{text_index}",
                    1,
                    {
                        "page_index": page_index,
                        "confidence": indexed_value(
                            ocr_result.get("rec_scores"), text_index
                        ),
                        "source": "overall_ocr_res",
                    },
                )
        for table_index, table in enumerate(result.get("table_res_list", [])):
            if not isinstance(table, dict):
                continue
            table_ocr = table.get("table_ocr_pred", {})
            if not isinstance(table_ocr, dict):
                table_ocr = {}
            for cell_index, box in enumerate(table.get("cell_box_list", [])):
                append_box(
                    box,
                    image_width,
                    image_height,
                    "TABLE_CELL",
                    None,
                    f"page-{page_index}-table-{table_index}-cell-{cell_index}",
                    2,
                    {
                        "page_index": page_index,
                        "table_index": table_index,
                        "source": "table_res_list.cell_box_list",
                    },
                )
            for text_index, box in enumerate(table_ocr.get("rec_boxes", [])):
                append_box(
                    box,
                    image_width,
                    image_height,
                    "TABLE_TEXT_LINE",
                    indexed_value(table_ocr.get("rec_texts"), text_index),
                    f"page-{page_index}-table-{table_index}-text-{text_index}",
                    3,
                    {
                        "page_index": page_index,
                        "table_index": table_index,
                        "confidence": indexed_value(
                            table_ocr.get("rec_scores"), text_index
                        ),
                        "source": "table_res_list.table_ocr_pred",
                    },
                )
    return rendered


def safe_json(value: Any) -> str:
    """Serialize JSON safely for embedding inside a script element."""
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


TABLE_HTML_PATTERN = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)
UNLIMITED_SEGMENT_PATTERN = re.compile(
    r"<\|det\|>.*?<\|/det\|>(.*?)(?=<\|det\|>|$)", re.DOTALL
)


def textract_segments(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Build reading-order text segments linked to Textract line blocks."""
    lines = [
        block
        for block in document.get("blocks", [])
        if isinstance(block, dict)
        and block.get("BlockType") == "LINE"
        and isinstance(block.get("Text"), str)
    ]
    lines.sort(
        key=lambda block: (
            block.get("Geometry", {}).get("BoundingBox", {}).get("Top", 0),
            block.get("Geometry", {}).get("BoundingBox", {}).get("Left", 0),
        )
    )
    return [{"box_id": block["Id"], "content": block["Text"]} for block in lines]


def unlimited_segments(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Build reading-order segments linked to Unlimited-OCR detection boxes."""
    generations = document.get("generations")
    if not isinstance(generations, list):
        raise TypeError("Unlimited-OCR JSON does not contain a generations array")
    segments: list[dict[str, Any]] = []
    for generation_index, generation in enumerate(generations):
        if not isinstance(generation, dict):
            continue
        raw_text = generation.get("raw_generated_text")
        if not isinstance(raw_text, str):
            continue
        for object_index, match in enumerate(
            UNLIMITED_SEGMENT_PATTERN.finditer(raw_text)
        ):
            content = match.group(1).strip()
            if content:
                segments.append(
                    {
                        "box_id": (
                            f"generation-{generation_index}-object-{object_index}-box-0"
                        ),
                        "content": content,
                    }
                )
    return segments


def paddle_segments(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Build reading-order segments linked to PaddleOCR layout blocks."""
    segments: list[dict[str, Any]] = []
    for page_index, page in enumerate(document.get("pages", [])):
        result = page.get("res") if isinstance(page, dict) else None
        if not isinstance(result, dict):
            continue
        blocks = [
            block
            for block in result.get("parsing_res_list", [])
            if isinstance(block, dict) and isinstance(block.get("block_content"), str)
        ]
        blocks.sort(
            key=lambda block: (
                block.get("block_order") is None,
                block.get("block_order", 0),
                block.get("block_bbox", [0, 0])[1],
            )
        )
        for block in blocks:
            content = block["block_content"].strip()
            box_id = f"page-{page_index}-layout-{block.get('block_id')}"
            table_match = TABLE_HTML_PATTERN.search(content)
            if table_match:
                before = re.sub(
                    r"</?(?:html|body)>", "", content[: table_match.start()]
                ).strip()
                if before:
                    segments.append({"box_id": box_id, "content": before})
                content = table_match.group(0)
            if content:
                segments.append({"box_id": box_id, "content": content})
    return segments


def assign_spans(boxes: list[dict[str, Any]], segments: list[dict[str, Any]]) -> None:
    """Attach each box the indexes of the text segments geometrically inside it."""
    box_by_id = {box["id"]: box for box in boxes}
    for box in boxes:
        spans = []
        for index, segment in enumerate(segments):
            segment_box = box_by_id.get(segment["box_id"])
            if segment_box is None:
                continue
            center_x = segment_box["left"] + segment_box["width"] / 2
            center_y = segment_box["top"] + segment_box["height"] / 2
            if (
                box["left"] <= center_x <= box["left"] + box["width"]
                and box["top"] <= center_y <= box["top"] + box["height"]
            ):
                spans.append(index)
        if spans:
            box["spans"] = spans


def html_document(
    image_url: str,
    boxes: list[dict[str, Any]],
    segments_by_tool: dict[str, list[dict[str, Any]]],
) -> str:
    """Build a standalone interactive HTML document for the supplied OCR boxes."""
    return (
        """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCR bounding boxes</title>
<style>
:root { color: #1d1d1f; font: 15px/1.6 -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif; }
* { box-sizing: border-box; }
body { background: #fff; margin: 0; }
header { align-items: center; backdrop-filter: blur(12px); background: #fffffff0; border-bottom: 1px solid #e5e5e5; display: flex; gap: 22px; padding: 13px max(24px, calc((100vw - 1500px) / 2)); position: sticky; top: 0; z-index: 10000; }
header strong { color: #1d1d1f; font-size: 14px; font-weight: 600; letter-spacing: -.01em; }
.provider { display: flex; gap: 2px; margin-left: auto; }
.provider input { appearance: none; position: absolute; }
.provider label { border-bottom: 2px solid transparent; color: #6e6e73; cursor: pointer; font-size: 13px; padding: 7px 10px 5px; }
.provider label:has(input:checked) { border-color: #0071e3; color: #0071e3; font-weight: 600; }
main { display: grid; gap: 26px; grid-template-columns: minmax(0, 1.08fr) minmax(360px, .92fr); margin: 28px auto; max-width: 1500px; padding: 0 26px 40px; }
.panel { background: #fff; border: 1px solid #e5e5e5; min-width: 0; overflow: hidden; }
.panel-heading { align-items: center; border-bottom: 1px solid #ededed; color: #6e6e73; display: flex; font-size: 12px; justify-content: space-between; letter-spacing: .04em; padding: 11px 15px; text-transform: uppercase; }
#image-wrap { overflow: auto; padding: 16px; }
#stage { display: inline-block; line-height: 0; position: relative; }
#source-image { display: block; max-width: 100%; }
.box { background: var(--fill); border: 1px solid var(--stroke); cursor: crosshair; position: absolute; transition: background .12s, border-width .12s; }
.box.hovered { border-width: 2px; filter: saturate(1.35); }
.box.selected { border-width: 2px; box-shadow: 0 0 0 2px #fff9; }
.type-page { --stroke: #94a3b8; --fill: #cbd5e11c; }
.type-table { --stroke: #54a6a0; --fill: #9bd7d22c; }
.type-cell { --stroke: #8b79c9; --fill: #c6b7eb2b; }
.type-word, .type-line, .type-text { --stroke: #c99539; --fill: #f5dca52b; }
.type-heading { --stroke: #cf7777; --fill: #f3c4c42b; }
.type-footer, .type-aside, .type-key-value, .type-selection { --stroke: #7ca89c; --fill: #b9d9cd2b; }
.type-other { --stroke: #a4a4a4; --fill: #d5d5d51f; }
#reading { max-height: calc(100vh - 145px); min-height: 640px; overflow: auto; padding: 22px 28px 34px; }
.seg { border-left: 2px solid transparent; display: block; margin: 0 0 6px; padding: 2px 8px; transition: background .12s, border-color .12s; white-space: pre-wrap; }
.seg.hovered { background: #f7e7c1; border-color: #d9a62e; }
.seg.selected { background: #d9efe8; border-color: #2fa08c; }
.seg p { margin: 0 0 .6em; }
.seg table { border-collapse: collapse; font-size: 12px; margin: .5em 0; white-space: normal; width: 100%; }
.seg td, .seg th { border: 1px solid #d2d2d7; padding: 4px 6px; vertical-align: top; }
#tooltip { background: #1d1d1ff0; box-shadow: 0 5px 18px #0002; color: #fff; display: none; font: 12px/1.45 -apple-system, system-ui, sans-serif; max-width: min(460px, calc(100vw - 32px)); padding: 9px 11px; pointer-events: none; position: fixed; white-space: pre-wrap; z-index: 20000; }
@media (max-width: 900px) { main { grid-template-columns: 1fr; } #reading { max-height: none; min-height: 280px; } .provider { margin-left: 0; } header { flex-wrap: wrap; } }
</style>
</head>
<body>
<header>
  <strong>Document reader</strong>
  <div class="provider">
    <label><input checked type="radio" name="provider" value="textract"> Textract</label>
    <label><input type="radio" name="provider" value="unlimited"> Unlimited OCR</label>
    <label><input type="radio" name="provider" value="paddle"> PaddleOCR</label>
  </div>
</header>
<main>
  <section class="panel"><div class="panel-heading"><span>Source image</span><span id="counts"></span></div><div id="image-wrap"><div id="stage"><img id="source-image" alt="Raw OCR input"></div></div></section>
  <section class="panel"><div class="panel-heading"><span>Extracted text</span><span id="reading-label"></span></div><article id="reading"></article></section>
</main>
<div id="tooltip" role="tooltip"></div>
<script>
const imageUrl = __IMAGE_URL__;
const boxes = __BOXES__;
const segmentsByTool = __SEGMENTS_BY_TOOL__;
const image = document.getElementById("source-image");
const stage = document.getElementById("stage");
const tooltip = document.getElementById("tooltip");
const counts = document.getElementById("counts");
const reading = document.getElementById("reading");
let activeTool = "textract";
let selectedBox = null;
let hoveredBox = null;
let boxById = {};

function semanticType(box) {
  const raw = String(box.type).toUpperCase();
  if (raw === "CELL" || raw === "MERGED_CELL") return "CELL";
  if (raw.includes("CELL")) return "CELL";
  if (raw === "WORD") return "WORD";
  if (raw === "LINE" || raw.includes("TEXT_LINE")) return "LINE";
  if (raw === "TEXT") return "TEXT";
  if (raw.includes("TABLE")) return "TABLE";
  if (raw.includes("HEADER") || raw.includes("TITLE") || raw === "HEADING") return "HEADING";
  if (raw.includes("FOOTER")) return "FOOTER";
  if (raw.includes("ASIDE")) return "ASIDE";
  if (raw.includes("KEY_VALUE")) return "KEY_VALUE";
  if (raw.includes("SELECTION")) return "SELECTION";
  if (raw === "PAGE") return "PAGE";
  return "OTHER";
}

function providerName(tool) {
  return tool === "textract" ? "Textract" : tool === "unlimited" ? "Unlimited OCR" : "PaddleOCR";
}

function formatValue(value) {
  if (value === null || value === undefined || value === "" || (Array.isArray(value) && !value.length)) return null;
  return Array.isArray(value) || typeof value === "object" ? JSON.stringify(value) : String(value);
}

function tooltipText(box) {
  const lines = [
    `${providerName(box.tool)}: ${semanticType(box)}`,
    box.text ? `Text: ${box.text}` : null,
    `Coordinates (normalized): left=${box.left.toFixed(6)}, top=${box.top.toFixed(6)}, width=${box.width.toFixed(6)}, height=${box.height.toFixed(6)}`,
    `Coordinates (pixels): x1=${Math.round(box.left * image.naturalWidth)}, y1=${Math.round(box.top * image.naturalHeight)}, x2=${Math.round((box.left + box.width) * image.naturalWidth)}, y2=${Math.round((box.top + box.height) * image.naturalHeight)}`,
    `ID: ${box.id}`,
    ...Object.entries(box.details).map(([key, value]) => {
      const rendered = formatValue(value);
      return rendered === null ? null : `${key}: ${rendered}`;
    }),
  ];
  return lines.filter(Boolean).join("\\n");
}

function updateCounts() {
  counts.textContent = `${boxes.filter(box => box.tool === activeTool).length} boxes`;
}

function positionTooltip(event) {
  const gap = 14;
  const maxLeft = window.innerWidth - tooltip.offsetWidth - gap;
  const maxTop = window.innerHeight - tooltip.offsetHeight - gap;
  tooltip.style.left = `${Math.max(gap, Math.min(event.clientX + gap, maxLeft))}px`;
  tooltip.style.top = `${Math.max(gap, Math.min(event.clientY + gap, maxTop))}px`;
}

function addBox(box, rank) {
  const element = document.createElement("div");
  element.className = `box type-${semanticType(box).toLowerCase().replace("_", "-")}`;
  element.dataset.tool = box.tool;
  element.dataset.id = box.id;
  element.style.left = `${box.left * 100}%`;
  element.style.top = `${box.top * 100}%`;
  element.style.width = `${box.width * 100}%`;
  element.style.height = `${box.height * 100}%`;
  element.style.zIndex = String(rank);
  element.addEventListener("mouseenter", event => {
    hoveredBox = box;
    refreshHighlights();
    tooltip.textContent = tooltipText(box);
    tooltip.style.display = "block";
    positionTooltip(event);
  });
  element.addEventListener("mousemove", positionTooltip);
  element.addEventListener("mouseleave", () => { hoveredBox = null; refreshHighlights(); tooltip.style.display = "none"; });
  element.addEventListener("click", () => { selectedBox = selectedBox && selectedBox.id === box.id ? null : box; refreshHighlights(); });
  stage.append(element);
}

function renderBoxes() {
  stage.querySelectorAll(".box").forEach(box => box.remove());
  boxById = {};
  const ordered = boxes.filter(box => box.tool === activeTool).sort((left, right) => {
    const depth = left.depth - right.depth;
    if (depth) return depth;
    return right.width * right.height - left.width * left.height;
  });
  ordered.forEach((box, rank) => { boxById[box.id] = box; addBox(box, rank + 1); });
  updateCounts();
}

function renderReading() {
  reading.replaceChildren();
  document.getElementById("reading-label").textContent = providerName(activeTool);
  const segments = segmentsByTool[activeTool] || [];
  if (!segments.length) {
    reading.textContent = "No extracted text was emitted for this provider.";
    return;
  }
  segments.forEach((segment, index) => {
    const element = document.createElement("div");
    element.className = "seg";
    element.dataset.seg = String(index);
    if (/^\\s*<table/i.test(segment.content)) {
      renderSafeTable(element, segment.content);
    } else {
      segment.content.split(/\\n{2,}/).forEach(paragraph => {
        if (!paragraph.trim()) return;
        const heading = paragraph.match(/^(#{1,6})\\s+([\\s\\S]*)$/);
        const child = document.createElement(heading ? `h${heading[1].length}` : "p");
        child.textContent = heading ? heading[2] : paragraph;
        element.append(child);
      });
    }
    element.addEventListener("mouseenter", () => { hoveredBox = boxById[segment.box_id] || null; refreshHighlights(); });
    element.addEventListener("mouseleave", () => { hoveredBox = null; refreshHighlights(); });
    element.addEventListener("click", () => {
      const linkedBox = boxById[segment.box_id] || null;
      selectedBox = selectedBox && linkedBox && selectedBox.id === linkedBox.id ? null : linkedBox;
      refreshHighlights();
    });
    reading.append(element);
  });
  refreshHighlights();
}

function renderSafeTable(container, markup) {
  const source = new DOMParser().parseFromString(markup, "text/html").querySelector("table");
  if (!source) {
    container.textContent = markup;
    return;
  }
  const table = document.createElement("table");
  source.querySelectorAll("tr").forEach(sourceRow => {
    const row = document.createElement("tr");
    sourceRow.querySelectorAll(":scope > th, :scope > td").forEach(sourceCell => {
      const cell = document.createElement(sourceCell.tagName.toLowerCase());
      cell.colSpan = sourceCell.colSpan;
      cell.rowSpan = sourceCell.rowSpan;
      cell.textContent = sourceCell.textContent;
      row.append(cell);
    });
    table.append(row);
  });
  container.append(table);
}

function refreshHighlights() {
  document.querySelectorAll(".box").forEach(element => {
    element.classList.toggle("hovered", !!hoveredBox && element.dataset.id === hoveredBox.id);
    element.classList.toggle("selected", !!selectedBox && element.dataset.id === selectedBox.id);
  });
  document.querySelectorAll(".seg").forEach(element => {
    const index = Number(element.dataset.seg);
    element.classList.toggle("hovered", !!hoveredBox && (hoveredBox.spans || []).includes(index));
    element.classList.toggle("selected", !!selectedBox && (selectedBox.spans || []).includes(index));
  });
}

document.querySelectorAll('input[name="provider"]').forEach(control => {
  control.addEventListener("change", () => {
    activeTool = control.value;
    selectedBox = null;
    hoveredBox = null;
    tooltip.style.display = "none";
    renderReading();
    renderBoxes();
    updateCounts();
  });
});

image.src = imageUrl;
image.addEventListener("load", () => { renderBoxes(); renderReading(); }, { once: true });
</script>
</body>
</html>
""".replace("__IMAGE_URL__", safe_json(image_url))
        .replace("__BOXES__", safe_json(boxes))
        .replace("__SEGMENTS_BY_TOOL__", safe_json(segments_by_tool))
    )


def main() -> int:
    """Render OCR JSON for one raw image as a self-contained HTML file."""
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise ValueError(f"Raw image does not exist: {image_path}")

    stem = image_path.stem
    textract_path = (
        args.textract.expanduser().resolve()
        if args.textract
        else PROCESSED_DIR / stem / "textract" / "result.json"
    )
    unlimited_path = (
        args.unlimited.expanduser().resolve()
        if args.unlimited
        else PROCESSED_DIR / stem / "unlimited_ocr" / "raw_generation.json"
    )
    paddle_path = (
        args.paddle.expanduser().resolve()
        if args.paddle
        else PROCESSED_DIR / stem / "paddle_ocr" / "result.json"
    )
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else PROCESSED_DIR / stem / "ocr_boxes.html"
    )
    for label, path in (
        ("Textract", textract_path),
        ("Unlimited-OCR", unlimited_path),
        ("PaddleOCR", paddle_path),
    ):
        if not path.is_file():
            raise ValueError(f"{label} JSON does not exist: {path}")

    textract_document = load_json(textract_path)
    unlimited_document = load_json(unlimited_path)
    paddle_document = load_json(paddle_path)
    textract = textract_boxes(textract_document)
    unlimited = unlimited_boxes(unlimited_document)
    paddle = paddle_boxes(paddle_document)
    segments_by_tool = {
        "textract": textract_segments(textract_document),
        "unlimited": unlimited_segments(unlimited_document),
        "paddle": paddle_segments(paddle_document),
    }
    assign_spans(textract, segments_by_tool["textract"])
    assign_spans(unlimited, segments_by_tool["unlimited"])
    assign_spans(paddle, segments_by_tool["paddle"])
    boxes = textract + unlimited + paddle
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        html_document(image_data_url(image_path), boxes, segments_by_tool),
        encoding="utf-8",
    )
    print(f"Saved interactive OCR boxes to {output_path} ({len(boxes)} boxes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
