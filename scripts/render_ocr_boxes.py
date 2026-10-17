"""Render OCR bounding boxes beside pre-generated HTML reconstructions."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

from output_layout import (
    PROCESSED_DIR,
    RAW_DIR,
    provider_result_path,
    reconstruction_path,
)
from textract_blocks import child_ids, descendant_text

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}
IMAGE_MIME_TYPES = {"image/bmp", "image/jpeg", "image/png", "image/webp"}


def parse_args() -> argparse.Namespace:
    """Parse optional raw images, OCR results, and rendered HTML paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        help="Raw BMP, JPEG, PNG, or WebP images (default: every image in data/raw)",
    )
    parser.add_argument("--textract", type=Path, help="Textract result JSON path")
    parser.add_argument(
        "--unlimited",
        type=Path,
        help="Unlimited-OCR raw_generation.json path",
    )
    parser.add_argument("--paddle", type=Path, help="PaddleOCR result JSON path")
    parser.add_argument("--output", type=Path, help="Rendered HTML output path")
    return parser.parse_args()


def resolve_image_paths(images: list[Path]) -> list[Path]:
    """Return explicit images or discover every supported image in data/raw."""
    candidates = images or sorted(
        path
        for path in RAW_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not candidates:
        raise ValueError(
            "No raw BMP, JPEG, PNG, or WebP images were provided or discovered"
        )
    resolved = [path.expanduser().resolve() for path in candidates]
    for path in resolved:
        if not path.is_file():
            raise ValueError(f"Raw image does not exist: {path}")
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Raw image must be a BMP, JPEG, PNG, or WebP: {path}")
    return resolved


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from a UTF-8 file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def image_data_url(path: Path) -> str:
    """Return an image file as a self-contained data URL."""
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type not in IMAGE_MIME_TYPES:
        raise ValueError("The raw image must be a BMP, JPEG, PNG, or WebP file")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def textract_text(block_id: str, blocks_by_id: dict[str, dict[str, Any]]) -> str | None:
    """Collect text from a block or its descendants, without traversing cycles."""
    return descendant_text(block_id, blocks_by_id) or None


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
        if block.get("BlockType") == "PAGE":
            continue
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


def html_document(documents: list[dict[str, Any]]) -> str:
    """Build a standalone interactive HTML document for OCR result collections."""
    return """<!doctype html>
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
.document-picker { background: transparent; border: 0; color: #1d1d1f; font: inherit; font-size: 13px; outline: none; padding: 7px 24px 7px 0; }
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
#reconstruction { border: 0; display: block; width: 100%; }
#tooltip { background: #1d1d1ff0; box-shadow: 0 5px 18px #0002; color: #fff; display: none; font: 12px/1.45 -apple-system, system-ui, sans-serif; max-width: min(460px, calc(100vw - 32px)); padding: 9px 11px; pointer-events: none; position: fixed; white-space: pre-wrap; z-index: 20000; }
@media (max-width: 900px) { main { grid-template-columns: 1fr; } .provider { margin-left: 0; } header { flex-wrap: wrap; } }
</style>
</head>
<body>
<header>
  <strong>Document reader</strong>
  <select class="document-picker" id="document-picker" aria-label="Document"></select>
  <div class="provider">
    <label><input checked type="radio" name="provider" value="textract"> Textract</label>
    <label><input type="radio" name="provider" value="unlimited"> Unlimited OCR</label>
    <label><input type="radio" name="provider" value="paddle"> PaddleOCR</label>
    <label><input type="radio" name="provider" value="raw_image"> Raw image</label>
  </div>
</header>
<main>
  <section class="panel"><div class="panel-heading"><span>Source image</span><span id="counts"></span></div><div id="image-wrap"><div id="stage"><img id="source-image" alt="Raw OCR input"></div></div></section>
  <section class="panel"><div class="panel-heading"><span>HTML reconstruction</span><span id="reading-label"></span></div><iframe id="reconstruction" sandbox="allow-same-origin" title="Selected HTML reconstruction"></iframe></section>
</main>
<div id="tooltip" role="tooltip"></div>
<script>
const documents = __DOCUMENTS__;
const image = document.getElementById("source-image");
const stage = document.getElementById("stage");
const tooltip = document.getElementById("tooltip");
const counts = document.getElementById("counts");
const reconstruction = document.getElementById("reconstruction");
const documentPicker = document.getElementById("document-picker");
const providerControls = [...document.querySelectorAll('input[name="provider"]')];
let activeTool = "textract";
let selectedBox = null;
let hoveredBox = null;
let activeDocument = documents[0];
let boxes = [];

function setDocument(index) {
  activeDocument = documents[index];
  boxes = activeDocument.boxes;
  selectedBox = null;
  hoveredBox = null;
  image.alt = activeDocument.name;
  image.src = activeDocument.image_url;
  updateProviderControls();
  renderReconstruction();
  renderBoxes();
}

function updateProviderControls() {
  const available = new Set(activeDocument.providers);
  providerControls.forEach(control => {
    control.disabled = !available.has(control.value);
    control.parentElement.hidden = !available.has(control.value);
  });
  if (!available.has(activeTool)) activeTool = activeDocument.providers[0];
  providerControls.find(control => control.value === activeTool).checked = true;
}

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
  if (tool === "textract") return "Textract";
  if (tool === "unlimited") return "Unlimited OCR";
  if (tool === "paddle") return "PaddleOCR";
  return "Raw image";
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
  const ordered = boxes.filter(box => box.tool === activeTool).sort((left, right) => {
    const depth = left.depth - right.depth;
    if (depth) return depth;
    return right.width * right.height - left.width * left.height;
  });
  ordered.forEach((box, rank) => addBox(box, rank + 1));
  updateCounts();
}

function renderReconstruction() {
  document.getElementById("reading-label").textContent = providerName(activeTool);
  reconstruction.srcdoc = activeDocument.reconstructions[activeTool];
}

function fitReconstruction() {
  if (!image.clientWidth || !image.clientHeight) return;
  reconstruction.style.height = `${image.clientHeight}px`;
  const reconstructionDocument = reconstruction.contentDocument;
  const page = reconstructionDocument?.querySelector(".page");
  if (!page) return;
  page.style.transform = "none";
  const scale = reconstruction.clientWidth / page.offsetWidth;
  page.style.transformOrigin = "top left";
  page.style.transform = `scale(${scale})`;
  reconstructionDocument.documentElement.style.overflow = "hidden";
  reconstructionDocument.body.style.height = `${page.offsetHeight * scale}px`;
  reconstructionDocument.body.style.margin = "0";
  reconstructionDocument.body.style.overflow = "hidden";
  reconstructionDocument.body.style.width = `${page.offsetWidth * scale}px`;
}

function refreshHighlights() {
  document.querySelectorAll(".box").forEach(element => {
    element.classList.toggle("hovered", !!hoveredBox && element.dataset.id === hoveredBox.id);
    element.classList.toggle("selected", !!selectedBox && element.dataset.id === selectedBox.id);
  });
}

document.querySelectorAll('input[name="provider"]').forEach(control => {
  control.addEventListener("change", () => {
    if (control.disabled) return;
    activeTool = control.value;
    selectedBox = null;
    hoveredBox = null;
    tooltip.style.display = "none";
    renderReconstruction();
    renderBoxes();
    updateCounts();
  });
});

documents.forEach((item, index) => {
  const option = window.document.createElement("option");
  option.value = String(index);
  option.textContent = item.name;
  documentPicker.append(option);
});
documentPicker.addEventListener("change", () => setDocument(Number(documentPicker.value)));
image.addEventListener("load", () => { renderBoxes(); fitReconstruction(); });
reconstruction.addEventListener("load", fitReconstruction);
window.addEventListener("resize", fitReconstruction);
setDocument(0);
</script>
</body>
</html>
""".replace("__DOCUMENTS__", safe_json(documents))


def result_paths(
    image_path: Path,
    args: argparse.Namespace,
    multiple_images: bool,
) -> tuple[Path, Path, Path]:
    """Return provider result paths for one image."""
    if multiple_images and (args.textract or args.unlimited or args.paddle):
        raise ValueError(
            "--textract, --unlimited, and --paddle may only be used with one image"
        )
    stem = image_path.stem
    return (
        args.textract.expanduser().resolve()
        if args.textract
        else provider_result_path(stem, "textract"),
        args.unlimited.expanduser().resolve()
        if args.unlimited
        else provider_result_path(stem, "unlimited_ocr", "raw_generation.json"),
        args.paddle.expanduser().resolve()
        if args.paddle
        else provider_result_path(stem, "paddle_ocr"),
    )


def reconstruction_paths(image_path: Path) -> dict[str, Path]:
    """Return standardized reconstruction HTML paths for one source image."""
    directories = {
        "textract": "textract",
        "unlimited": "unlimited_ocr",
        "paddle": "paddle_ocr",
        "raw_image": "raw_image",
    }
    return {
        provider: reconstruction_path(image_path.stem, directory)
        for provider, directory in directories.items()
    }


def load_reconstructions(
    image_path: Path,
    providers: set[str],
) -> dict[str, str]:
    """Load reconstructions for the available OCR providers and raw image."""
    paths = reconstruction_paths(image_path)
    requested_providers = providers | {"raw_image"}
    missing = [
        paths[provider]
        for provider in requested_providers
        if not paths[provider].is_file()
    ]
    if missing:
        rendered_paths = ", ".join(str(path) for path in missing)
        raise ValueError(f"Reconstruction HTML does not exist: {rendered_paths}")
    return {
        provider: paths[provider].read_text(encoding="utf-8")
        for provider in requested_providers
    }


def make_document(
    image_path: Path,
    args: argparse.Namespace,
    multiple_images: bool,
) -> dict[str, Any]:
    """Load one image and its provider results into an HTML viewer payload."""
    textract_path, unlimited_path, paddle_path = result_paths(
        image_path,
        args,
        multiple_images,
    )
    providers: dict[str, list[dict[str, Any]]] = {}
    reconstruction_files = reconstruction_paths(image_path)
    provider_paths = (
        ("textract", "Textract", textract_path, args.textract, textract_boxes),
        (
            "unlimited",
            "Unlimited-OCR",
            unlimited_path,
            args.unlimited,
            unlimited_boxes,
        ),
        ("paddle", "PaddleOCR", paddle_path, args.paddle, paddle_boxes),
    )
    for provider, label, path, override, box_loader in provider_paths:
        reconstruction_path = reconstruction_files[provider]
        if path.is_file() and reconstruction_path.is_file():
            providers[provider] = box_loader(load_json(path))
        elif override:
            if not path.is_file():
                raise ValueError(f"{label} JSON does not exist: {path}")
            raise ValueError(
                f"{label} reconstruction HTML does not exist: {reconstruction_path}"
            )

    if not providers:
        raise ValueError(f"No OCR JSON results exist for {image_path}")

    return {
        "name": image_path.name,
        "image_url": image_data_url(image_path),
        "boxes": [box for boxes in providers.values() for box in boxes],
        "providers": [*providers, "raw_image"],
        "reconstructions": load_reconstructions(image_path, set(providers)),
    }


def main() -> int:
    """Render OCR JSON for one raw image as a self-contained HTML file."""
    args = parse_args()
    image_paths = resolve_image_paths(args.images)
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else PROCESSED_DIR / "ocr_comparison.html"
    )
    documents = [
        make_document(image_path, args, len(image_paths) > 1)
        for image_path in image_paths
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        html_document(documents),
        encoding="utf-8",
    )
    box_count = sum(len(document["boxes"]) for document in documents)
    print(
        f"Saved interactive OCR boxes to {output_path} "
        f"({len(documents)} documents, {box_count} boxes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
