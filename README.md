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

## OCR bounding-box viewer

`scripts/render_ocr_boxes.py` creates a self-contained interactive HTML viewer
from the Textract, Unlimited-OCR, and PaddleOCR JSON for one source image. It
embeds the source image alongside the provider's extracted text in reading
order. Select one provider at a time; boxes use a shared semantic colour system
(for example, tables, cells, lines, and headings) rather than provider colours.
Hovering a box links to its text, while clicking retains that selection as other
objects are inspected. Textract child blocks and PaddleOCR table cells/text take
hover precedence over their parent regions.

```shell
uv run python scripts/render_ocr_boxes.py data/raw/scan.png
```

The default output is `data/processed/<input-name>/ocr_boxes.html`. Override
the input JSON locations with `--textract`, `--unlimited`, and `--paddle`, or
use `--output` to choose another HTML path.

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
