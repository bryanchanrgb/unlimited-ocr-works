---
name: raw-image-html-reconstruction
description: Create a static, text-only visual HTML reconstruction by directly reading a raw document image with image modality, without using OCR JSON or displaying the source image.
---

# Raw Image HTML Reconstruction

Create a directly authored, static HTML visual reconstruction of a raw document
image using the agent's built-in image modality.

## Required constraints

- Treat the image as the sole source of truth; do not read, compare, or derive
  content from OCR JSON.
- Author the final HTML directly. Do not create or run an OCR process,
  generator, conversion script, or build step.
- Use only static HTML and CSS: black text (`#000`) on a white background
  (`#fff`). Do not use JavaScript, images, canvas, or SVG.
- Preserve every visible word, number, punctuation mark, checkbox state, and
  paragraph break exactly as read from the image. Do not silently correct apparent
  typos or substitute likely text.
- Mark illegible content as `[Unreadable]` only after examining it closely.
  Do not invent details that cannot be read from the image.
- Use semantic HTML appropriate to the document. The reconstruction itself
  must remain visible rather than hidden behind a fallback.

## Workflow

1. Inspect the image at its natural dimensions. Identify page boundaries,
   layout, text, tables, fields, checkbox states, signatures, and identifiers.
2. Transcribe the document in visual reading order. Reinspect small or blurred
   areas before labeling them unreadable.
3. Build one fixed-size `.page` container per image page.
4. Place content with absolute pixel coordinates, matching observed position,
   dimensions, typography, alignment, and wrapping.
5. Recreate tables and forms using semantic HTML tables. Match observed spans,
   dimensions, and blank fields; use `X` for selected checkboxes.

## Output requirements

Create one self-contained `.html` file. Its visible `<main>` must contain the
readable document reconstruction, not an image replacement or fallback.
Save it as
`data/processed/<document>/raw_image/reconstruction.html`.

## Final review

Compare the authored HTML against the image using image modality. Confirm all
visible text appears once, layout and tables match, selected marks are
represented, and the output meets every required constraint.
