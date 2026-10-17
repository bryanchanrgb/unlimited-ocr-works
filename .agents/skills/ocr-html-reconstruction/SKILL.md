---
name: ocr-html-reconstruction
description: Create a static, text-only visual HTML reconstruction from LLM-optimized OCR JSON without using the original image, scripts, or code generation.
---

# OCR HTML Reconstruction

Create a directly authored, static HTML visual reconstruction from a file under
`data/processed/llm_optimized`.

## Required constraints

- Read the optimized JSON completely and author the final HTML yourself. Do
  not create or run a generator, conversion script, or build step.
- Use only static HTML and CSS: black text (`#000`) on a white background
  (`#fff`). Do not use JavaScript, images, canvas, or SVG.
- Keep the document at the dimensions provided by OCR where available. Use an
  absolutely positioned page container for fixed-layout forms and reports.
- Preserve OCR content exactly. Do not silently correct spelling, punctuation,
  numbers, or capitalization. A correction is allowed only when another
  optimized provider record explicitly corroborates it; otherwise retain the
  text as extracted.
- Do not invent details absent from the optimized OCR data. Use the extracted
  `[Unreadable]` marker when available.
- Use semantic HTML appropriate to the document. The reconstruction itself
  must remain visible rather than hidden behind a fallback.

## Workflow

1. Identify the optimized file's `kind` and use the provider with the richest
   relevant structure:
   - `paddle_ocr` provides pixel page sizes, semantic regions, OCR lines, and
     recognized table HTML.
   - `textract` provides normalized geometry, table cells with row and column
     spans, form key/value pairs, selection states, and lines.
   - `unlimited_ocr` provides Markdown plus object boxes; use its Markdown for
     structure and its object boxes for placement when reference text exists.
2. Build one `.page` container per source page. Use PaddleOCR dimensions
   directly; scale Textract's normalized geometry proportionally.
3. Use the provider's explicit reading order first. When it is absent, order
   regions by top coordinate and then left coordinate.
4. Place text at its detected bounds. Use fine-grained OCR text only where it
   improves placement; never render the same text twice.
5. Reconstruct tables with actual HTML tables, preserving spans and detected
   dimensions. Use text marks (`X`, `_`) for selections and empty cells for
   blank fields.

## Compact schema reference

### PaddleOCR

```text
pages[].size                         [width, height]
pages[].regions[]                    [label, [left, top, right, bottom], content, order]
pages[].lines[]                      [text, [left, top, right, bottom], ...]
pages[].tables[].html                recognized table HTML
pages[].tables[].cells[]             [[left, top, right, bottom], ...]
```

Map `paragraph_title`, `header`, `footer`, `text`, `table`, and similar region
labels to semantic elements. Match a table prediction to a table region using
overlapping OCR text, not list position alone.

### Textract

```text
pages[].lines[]                      [text, {b: [left, top, width, height]}, ...]
pages[].tables[].c[]                 [row, column, row_span, column_span, text, geometry, ...]
pages[].forms[]                      [key_text, key_geometry, value_text, value_geometry]
pages[].selections[]                 [selection_status, geometry]
```

Coordinates are normalized. Treat `SELECTED` as a visible `X` and
`NOT_SELECTED` as an empty square or blank field, according to the surrounding
form layout.

### Unlimited-OCR

```text
sizes[]                              [width, height]
markdown                             document content and table markup
objects[]                            [page, label, pixel_boxes, reference_text]
```

Use Markdown and HTML table markup directly as content, but place the associated
regions from object boxes only when the reference text makes the association
unambiguous.

## Output requirements

Create one self-contained `.html` file. Its visible `<main>` must contain the
readable document reconstruction, not an image replacement or fallback.
Save it as
`data/processed/llm_optimized/<document>/<provider>/reconstruction.html`,
where `<provider>` is the optimized input's parent directory.

## Final review

Confirm every extracted heading, paragraph, table value, selected mark, and
footer appears once in the visual document. Confirm the output meets every
required constraint.
