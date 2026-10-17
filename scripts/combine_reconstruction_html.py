"""Combine standardized reconstruction HTML files into one comparison viewer."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "llm_optimized"
DEFAULT_OUTPUT = DEFAULT_INPUT / "reconstruction_comparison.html"
SOURCE_LABELS = {
    "raw_image": "Raw image",
    "paddle_ocr": "PaddleOCR",
    "textract": "Textract",
    "unlimited_ocr": "Unlimited OCR",
}
SOURCE_ORDER = {source: index for index, source in enumerate(SOURCE_LABELS)}


def parse_args() -> argparse.Namespace:
    """Parse input and output paths for the reconstruction comparison viewer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Standardized reconstruction root (default: data/processed/llm_optimized)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Comparison HTML output (default: data/processed/llm_optimized/reconstruction_comparison.html)",
    )
    return parser.parse_args()


def source_label(source: str) -> str:
    """Return a human-readable label for one standardized source directory."""
    return SOURCE_LABELS.get(source, source.replace("_", " ").title())


def collect_reconstructions(input_directory: Path) -> list[dict[str, str]]:
    """Load each standardized reconstruction HTML file below the input directory."""
    reconstructions: list[dict[str, str]] = []
    for path in input_directory.rglob("reconstruction.html"):
        relative_path = path.relative_to(input_directory)
        if len(relative_path.parts) != 3:
            continue
        document, source, _ = relative_path.parts
        reconstructions.append(
            {
                "document": document,
                "source": source_label(source),
                "html": path.read_text(encoding="utf-8"),
                "sort_source": source,
            }
        )
    reconstructions.sort(
        key=lambda item: (
            item["document"],
            SOURCE_ORDER.get(item["sort_source"], len(SOURCE_ORDER)),
            item["source"],
        )
    )
    for reconstruction in reconstructions:
        del reconstruction["sort_source"]
    return reconstructions


def html_document(reconstructions: list[dict[str, str]]) -> str:
    """Build a self-contained viewer with document and source selectors."""
    documents = sorted(
        {reconstruction["document"] for reconstruction in reconstructions}
    )
    options = "".join(
        f'<option value="{html.escape(document, quote=True)}">'
        f"{html.escape(document)}</option>"
        for document in documents
    )
    payload = json.dumps(reconstructions, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCR reconstruction comparison</title>
<style>
:root {{ color-scheme: light; font-family: system-ui, sans-serif; }}
body {{ background: #f5f5f5; color: #111; margin: 0; }}
header {{ align-items: end; background: #fff; border-bottom: 1px solid #ccc; display: flex; gap: 16px; padding: 14px 20px; position: sticky; top: 0; z-index: 1; }}
h1 {{ font-size: 18px; margin: 0 auto 0 0; }}
label {{ display: grid; font-size: 12px; font-weight: 600; gap: 4px; }}
select {{ font: inherit; min-width: 180px; padding: 5px; }}
main {{ margin: 24px auto; max-width: 1100px; }}
iframe {{ background: #fff; border: 1px solid #bbb; display: block; height: min(1000px, calc(100vh - 130px)); width: 100%; }}
#empty {{ background: #fff; border: 1px solid #bbb; padding: 24px; }}
</style>
</head>
<body>
<header>
  <h1>OCR reconstruction comparison</h1>
  <label>Document<select id="document">{options}</select></label>
  <label>Source<select id="source"></select></label>
</header>
<main>
  <iframe id="reconstruction" sandbox title="Selected reconstruction"></iframe>
  <p id="empty" hidden>No reconstruction is available for this selection.</p>
</main>
<script>
const reconstructions = {payload};
const documentSelector = document.getElementById("document");
const sourceSelector = document.getElementById("source");
const frame = document.getElementById("reconstruction");
const empty = document.getElementById("empty");

function recordsForDocument() {{
  return reconstructions.filter(item => item.document === documentSelector.value);
}}

function renderSources() {{
  const records = recordsForDocument();
  sourceSelector.replaceChildren(...records.map(item => {{
    const option = document.createElement("option");
    option.value = item.source;
    option.textContent = item.source;
    return option;
  }}));
  renderReconstruction();
}}

function renderReconstruction() {{
  const selected = recordsForDocument().find(item => item.source === sourceSelector.value);
  frame.hidden = !selected;
  empty.hidden = Boolean(selected);
  frame.srcdoc = selected ? selected.html : "";
}}

documentSelector.addEventListener("change", renderSources);
sourceSelector.addEventListener("change", renderReconstruction);
renderSources();
</script>
</body>
</html>
"""


def main() -> int:
    """Write a comparison viewer for all standardized reconstruction files."""
    args = parse_args()
    input_directory = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_directory.is_dir():
        raise ValueError(f"Input directory does not exist: {input_directory}")
    reconstructions = collect_reconstructions(input_directory)
    if not reconstructions:
        raise ValueError(f"No reconstruction.html files found in {input_directory}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_document(reconstructions), encoding="utf-8")
    print(f"Saved {len(reconstructions)} reconstructions to {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
