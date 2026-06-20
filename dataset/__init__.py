"""Data pipeline: build (image, latex, typst) triples from the raw corpora.

Two sources feed a single JSONL manifest:

- im2latex (printed): index-aligned LaTeX/PNG pairs, images referenced in place.
- MathWriting (handwritten): InkML strokes rasterised to PNG on the fly.

Every record's LaTeX is run through the transpiler and gated by the typst
compiler before admission, so the manifest only contains compilable Typst.
"""
