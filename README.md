# Unlimited OCR Works

## AWS Textract forms and tables

The `scripts/run_textract.py` utility analyzes local PNG/JPEG images or PDF/TIFF
documents with Amazon Textract's `FORMS` and `TABLES` features, then saves the
complete Textract block response as JSON. `LINE` and `WORD` blocks are also
included by Textract. Relationships between blocks are preserved, so form
key/value pairs and table/cell contents can be reconstructed from their IDs.

Install dependencies with `uv sync`. Configure AWS credentials using the normal
AWS credential chain, or place `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and
`AWS_DEFAULT_REGION` in the project-root `.env` file. The script loads that file
without overriding variables already set in the environment. Use an IAM role
instead of static keys when running on AWS where possible. Never commit `.env`.
Copy `.env.example` to `.env` to start; add `AWS_SESSION_TOKEN` when using
temporary credentials.

Verify that the configured credentials can authenticate before processing a
document:

`uv run python scripts/verify_aws_connection.py`

The check calls AWS STS `GetCallerIdentity` and prints the authenticated IAM
identity. Pass `--region us-east-1` to override the configured region.

Images are sent using Textract's synchronous `AnalyzeDocument` operation. PDF
and TIFF files use asynchronous analysis: the script uploads a temporary copy to
an S3 bucket, polls Textract for completion, paginates all returned blocks, then
deletes the temporary object. Pass a bucket in the same AWS region as Textract.
The caller needs Textract permissions for the corresponding operation and S3
permissions required to upload, read, and delete the temporary input.

Examples:

- `uv run python scripts/run_textract.py data/raw/scan.png`
- `uv run python scripts/run_textract.py data/raw/scan.pdf --bucket my-textract-inputs`
- `uv run python scripts/run_textract.py data/raw/scan.png --region us-east-1 --output data/processed/scan/textract/result.json`

By default, JSON is saved to
`data/processed/<input-name>/textract/result.json`, alongside the
`data/processed/<input-name>/unlimited_ocr` output structure used by
`run_unlimited_ocr.py`. Use `--output` to choose another path. Async jobs are
polled every five seconds by default, with a 30-minute timeout; use
`--poll-interval` and `--timeout` to change these values. PDF/TIFF uploads are
cleaned up after the job reaches a terminal state. If the client times out while
a job is still running, the temporary S3 object is retained and its URI is
printed so the job can finish.

## PaddleOCR document analysis

`scripts/run_paddle_ocr.py` runs PaddleOCR's PP-StructureV3 pipeline with table
recognition and table-cell results enabled. It writes the complete raw pipeline
response to `data/processed/<input-name>/paddle_ocr/result.json`. Document
orientation correction and unwarping are disabled so its coordinates remain in
the raw source-image coordinate system used by the bounding-box viewer.

PaddlePaddle does not currently provide a CPython 3.14 wheel, so PaddleOCR runs
in its own CPython 3.13 environment:

```shell
uv venv --python 3.13 .venv-paddleocr
uv pip install --python .venv-paddleocr/Scripts/python.exe paddlepaddle==3.2.2 paddleocr==3.7.0 "paddlex[ocr]==3.7.2"
```

Run it against an image:

```shell
uv run python scripts/run_paddle_ocr.py data/raw/scan.png
```

Use `--python` to select a different PaddleOCR environment, `--lang` to change
the OCR language, or `--output` to select a different JSON path.

## Running all providers

`scripts/run_all_ocr.py` runs Textract, Unlimited-OCR, and PaddleOCR for each
supplied image. It preserves each provider's normal output and writes batch
metadata containing UTC start/end times, provider commands and parameters,
return codes, diagnostic output, and elapsed runtime in seconds.

```shell
uv run python scripts/run_all_ocr.py data/raw/*.png data/raw/*.jpg
```

The metadata defaults to `data/processed/ocr_run_metadata.json`. Use
`--metadata` to choose another path, `--region` for Textract, `--unlimited-mode`
for Unlimited-OCR, and `--paddle-lang` or `--paddle-python` for PaddleOCR.

## OCR bounding-box viewer

`scripts/render_ocr_boxes.py` creates a self-contained interactive HTML viewer
for every discovered image under `data/raw`. It embeds each source image
alongside the corresponding pre-generated HTML reconstruction from
`data/processed/llm_optimized`. Select a document and provider to view its
boxes on the left and reconstruction on the right. Select **Raw image** to
display the source image without boxes alongside the raw-image reconstruction.
The viewer exposes only providers available for the selected document. The
reconstruction scales to its pane width and its pane height matches the rendered
source image. Hovering and clicking boxes affects only the box overlay; Textract
`PAGE` blocks are omitted, while Textract child blocks and PaddleOCR table
cells/text take hover precedence over their parent regions.

```shell
uv run python scripts/render_ocr_boxes.py
```

The viewer requires standardized reconstruction files: provider reconstructions
at `data/processed/llm_optimized/<input-name>/<provider>/reconstruction.html`
and the raw-image reconstruction at
`data/processed/llm_optimized/<input-name>/raw_image/reconstruction.html`.
The default output is `data/processed/ocr_boxes.html`. Pass one or more image
paths to restrict the viewer, override a single image's JSON locations with
`--textract`, `--unlimited`, and `--paddle`, or use `--output` to choose another
HTML path.

## Unlimited-OCR raw capture

`scripts/run_unlimited_ocr.py` calls the existing `demo/run_ocr.py` helpers from
the sibling `Unlimited-OCR-Universal` checkout, using that checkout's `.venv`.
Create/sync that environment first. The output directory defaults to
`data/processed/<input-name>/unlimited_ocr`; the model repository's normal demo
artifacts are still produced there (`result.md`, annotated page images, and
image crops). PDF pages are retained in `input_pages/` instead of being deleted.

The helper additionally writes `raw_generation.json` with the prompt, full
generated token IDs, generation text decoded without dropping special tokens,
the processed Markdown, detected ref/det objects, normalized and pixel boxes,
input dimensions, run settings, and paths to the demo artifacts. Boxes use the
model's 0–999 coordinates and are also mapped to source-image pixel `xyxy` bounds.
The token output contains no model confidence scores; the model does not emit
per-object confidence values. The raw text and token IDs are kept even when a
detection tag cannot be parsed into a box.

Examples:

```shell
uv run python scripts/run_unlimited_ocr.py image data/raw/scan.png
uv run python scripts/run_unlimited_ocr.py image data/raw/scan.png --mode base
uv run python scripts/run_unlimited_ocr.py multi data/raw/page1.png data/raw/page2.png
uv run python scripts/run_unlimited_ocr.py pdf data/raw/document.pdf --dpi 300
```

Pass `--repo` if the model checkout is not at the sibling default location, or
`--python` if its virtual-environment interpreter is elsewhere. The prompt must
begin with `<image>`. The image mode applies to one image; multi-image and PDF
inference use the model's base mode.

## LLM-optimized OCR JSON

`scripts/optimize_ocr_for_llm.py` converts every OCR JSON under
`data/processed` into a compact counterpart under
`data/processed/llm_optimized`. It never modifies the raw provider responses.

```shell
uv run python scripts/optimize_ocr_for_llm.py
```

The compact files retain the content and layout evidence required to reproduce a
document: source filename, page dimensions, reading-order text, pixel or
normalized geometry, non-rectangular polygons, recognized table HTML/cells,
form key/value pairs, checkbox states, and Unlimited-OCR Markdown/object boxes.
For each provider, these fields are organized per page and ordered top-to-bottom,
then left-to-right where the provider did not supply an explicit order.

To minimize LLM input, the optimizer removes absolute paths, OCR runtime and
model settings, provider IDs and relationship graphs, word-level decomposition
already represented by lines/cells, duplicate detector output, generated token
IDs, raw generation markup, commands, timestamps, diagnostics, and artifact
paths. Raw JSON remains available when any provider-specific metadata is needed.

Compact field names are deliberately short. `v` is the schema version; `g` is
Textract geometry (`b` = `[left, top, width, height]`, with optional `q`
quadrilateral); `n` is page number; and table cells are
`[row, column, row_span, column_span, text, geometry, confidence]`. PaddleOCR
line records are `[text, xyxy_box, confidence, optional_polygon]`, regions are
`[label, xyxy_box, content, reading_order]`, and Unlimited-OCR objects are
`[page, label, pixel_boxes, reference_text]`.

## End-to-end reconstruction workflow

1. Run each configured OCR provider:

   ```shell
   uv run python scripts/run_all_ocr.py data/raw/document.png
   ```

2. Create compact OCR inputs for agents:

   ```shell
   uv run python scripts/optimize_ocr_for_llm.py
   ```

3. Ask a Copilot agent to apply
   `.agents/skills/ocr-html-reconstruction/SKILL.md` once for each optimized
   provider JSON, and
   `.agents/skills/raw-image-html-reconstruction/SKILL.md` once for the raw
   image. These skills write reconstructions to
   `data/processed/llm_optimized/<document>/<source>/reconstruction.html`.
4. Render the unified cross-document, cross-provider comparison viewer:

   ```shell
   uv run python scripts/render_ocr_boxes.py
   ```
