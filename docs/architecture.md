# ScanParse Architecture

This document describes the internals of ScanParse so contributors can extend it, tune it, or wire it into larger systems (RAG pipelines, digitization workflows, search indexes).

## Module map

| Module | Responsibility |
|---|---|
| `scanparse/security.py` | Input validation: magic-byte sniffing, size/dimension limits, path-traversal prevention, filename sanitization |
| `scanparse/preprocessing.py` | The poor-scan repair pipeline |
| `scanparse/lang.py` | Unicode-range script classification (en / hi / mixed) |
| `scanparse/parsers.py` | Multi-engine OCR orchestration, auto-retry, `Document` assembly |
| `scanparse/document.py` | Structured data model + Markdown/JSON/DOCX exporters |
| `scanparse/cli.py` | Command-line interface |
| `scanparse/app.py` | Gradio web UI |
| `tests/benchmark/` | Synthetic degraded-page generator + CER/WER runner |

## Preprocessing pipeline

Order matters, and the pipeline is deliberately model-free (pure OpenCV) so it stays fast and portable:

1. **Upscale (LANCZOS, up to 3×)** — Devanagari matras and small type are only reliably recognized when strokes are roughly 20 px tall. Upscaling first also makes the filters that follow behave more predictably.
2. **Denoise (fastNlMeansDenoising)** — removes grain and speckle from cheap scanner sensors and low-quality photocopies.
3. **Illumination flattening** — the single most important step for poor scans. A large-kernel morphological background estimate is divided out, which removes shadows, stains, and yellowing while preserving thin strokes.
4. **CLAHE contrast normalization** — recovers faded ink and low-contrast type.
5. **Deskew (projection-profile analysis)** — searches ±15° on a downscaled copy for the rotation that maximizes row-projection variance, then applies the inverse rotation at full resolution. Accurate to roughly ±0.25°.
6. **Binarization (optional, off by default)** — Sauvola adaptive thresholding keeps thin strokes that Otsu destroys. Off by default because the Tesseract LSTM engine reads enhanced grayscale better than hard binary images on degraded scans; enable it for legacy workflows.

## OCR engine strategy

`fast` mode (the default) uses Tesseract 5 LSTM with `--psm 6` (uniform block of text), OEM 1 (LSTM only, which has the best Devanagari support), and an effective DPI that reflects the preprocessing upscale. It requires no model downloads and runs in about 2–3 seconds per page on a CPU.

`accurate` mode uses Surya for joint layout analysis and recognition. It is heavier (first invocation downloads ~8 GB of weights) but produces typed structural blocks and handles severely degraded pages better.

`vlm` mode uses PaddleOCR's Devanagari recognition and is marked experimental.

### Auto-retry quality layer

After each `fast` pass, pages with suspiciously short output (few words) get a second pass with maximum denoise; the run with the higher quality score (word count weighted by mean line confidence) is kept. The retry only fires on genuinely bad pages, so the median cost stays at one pass per page.

## Language detection

`scanparse/lang.classify_text` uses Unicode-range heuristics over the Devanagari block (U+0900–U+097F, plus extensions) and Latin letters. Digits (ASCII and Devanagari ०–९) and punctuation are ignored so pure-number lines default to English. Lines with both scripts are tagged `mixed`, which is common in Indian documents (English headings, Hindi body text, or vice versa).

## Adding a new OCR backend

1. Write a function `def _myengine_ocr(image, langs, dpi) -> tuple[list[Block], str]` following the signature used in `parsers.py`.
2. Return `Block` objects with `block_type`, `text`, `order`, `language` (from `classify_text`), and optionally `confidence`/`bbox`.
3. Add a `mode` branch in `parse()` and update `cli.py`'s choices.
4. Add a benchmark entry and update the accuracy table in the README.

## Benchmark harness

`tests/benchmark/run_benchmark.py` renders clean text pages (English + Hindi), applies realistic degradation (shadows, grain, stains, motion blur, skew) with known ground truth, runs the full pipeline, and reports CER/WER per language. To evaluate on your own data, drop pairs of `name.png` + `name.txt` (ground truth) into `tests/benchmark/data/` and run with `--regenerate` omitted.
