# ScanParse

[![CI](https://github.com/ashisman2/scanparse/actions/workflows/ci.yml/badge.svg)](https://github.com/ashisman2/scanparse/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Open-source parser for poor-quality English & Hindi scanned documents.**

ScanParse turns degraded, noisy, skewed, and faded scans (books, government forms, archives, receipts) into clean, structured, developer-friendly output — Markdown, JSON, or DOCX. It is built around three principles: **accuracy first**, **fast and lightweight by default**, and **secure by design**.

## Features

- **Poor-scan repair pipeline** — illumination flattening (shadow/stain removal), upscaling, denoising, CLAHE contrast recovery, deskew, and optional Sauvola/Otsu binarization.
- **English + Hindi (Devanagari) focused** — automatic per-line script detection for mixed-language pages; full Devanagari support including digits (०–९) and punctuation.
- **Multi-engine OCR**:

  | Mode | Engine | Install | Use when |
  |---|---|---|---|
  | `fast` (default) | Tesseract LSTM (`eng`+`hin`) | Nothing extra | Everyday use; CPU-only; seconds per page |
  | `accurate` | Surya (layout + recognition) | `pip install "scanparse[surya]"` | Severely degraded pages, layout structure |
  | `vlm` (experimental) | PaddleOCR Devanagari | `pip install "scanparse[vlm]"` | Research / fallback |

- **Structured output** — pages broken into typed blocks (title, heading, paragraph, list, table, caption, footnote) with language tags, confidence scores, and bounding boxes.
- **Three interfaces** — Python library, CLI, and a Gradio web UI with before/after preview.
- **Speed** — fast mode processes a page in ~2.5 s on CPU with only numpy + OpenCV as ML dependencies (no 8 GB model downloads required).
- **Security** — magic-byte format sniffing, 100 MiB file cap, dimension caps, path-traversal prevention, filename sanitization, non-root Docker image, and a `pip-audit` security job in CI.
- **Measurable accuracy** — built-in CER/WER benchmark suite (English & Hindi) so every improvement is quantified.

## Quick start

```bash
# 1. Install Tesseract with Hindi support (system package)
sudo apt-get install -y tesseract-ocr tesseract-ocr-hin poppler-utils

# 2. Install ScanParse
pip install scanparse        # lightweight core (~30 MB)
pip install "scanparse[all]" # + Surya + web UI
```

```python
from scanparse import parse

doc = parse("my_scan.pdf", lang=["en", "hi"], mode="fast")
doc.to_markdown("output.md")
doc.to_json("output.json")
```

### CLI

```bash
scanparse document.pdf --lang en,hi --mode fast --out ./result
scanparse page.png --format all --dpi 150
scanparse --ui                     # launch web UI at http://localhost:7860
```

### Docker

```bash
docker build -t scanparse .
docker run -p 7860:7860 scanparse
```

## Accuracy benchmarks

Measured on synthetic degraded scans (noise, shadows, stains, blur, skew) with ground truth, using CER (character error rate, lower is better).

| Language | Mode | Avg CER | Avg time/page |
|---|---|---|---|
| English | `fast` | **~1.6%** | ~2.9 s (CPU) |
| Hindi | `fast` | **~0.4%** | ~3.0 s (CPU) |

Run your own measurement:

```bash
python -m tests.benchmark.run_benchmark --mode fast --regenerate
```

> Note: benchmark numbers reflect the synthetic test set (heavy degradation). On real-world archives, accuracy depends on scan severity; use `--mode accurate` (Surya) for the hardest pages, and the auto-retry layer automatically reprocesses pages whose confidence is too low.

## Architecture

```
input (PNG/JPEG/PDF)
   │  security: size/format/path checks
   ▼
preprocessing          ── upscale → denoise → flatten shadows → CLAHE → deskew → (binarize)
   │
   ▼
OCR engine             ── fast (Tesseract LSTM) | accurate (Surya) | vlm (PaddleOCR)
   │  auto-retry on low confidence
   ▼
language detection     ── per-line en / hi / mixed classification
   │
   ▼
Document model         ── typed blocks with confidence + bbox
   │
   ▼
exporters              ── Markdown | JSON | DOCX
```

See [`docs/architecture.md`](docs/architecture.md) for module details and how to add a new OCR backend.

## Security model

Every input passes through [`scanparse/security.py`](scanparse/security.py) before any processing: files are sniffed by **magic bytes** (extensions are ignored), capped at **100 MiB**, and confined to an allowed directory when `--allowed-root` is used. Image dimensions are capped at 10 000 px, PDFs at 500 pages, and uploaded filenames are sanitized. The Docker image runs as a **non-root user**, and CI runs a CVE audit (`pip-audit`) on every PR. OCR output is always treated as untrusted plain data — it is never executed or interpreted.

## Development

```bash
git clone https://github.com/ashisman2/scanparse.git
cd scanparse
pip install -e ".[dev]"
pytest tests/
ruff check scanparse tests
```

Contributions are welcome — see [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).

## License

MIT. See [`LICENSE`](LICENSE).

## Roadmap

- Handwritten Devanagari support (experimental, VLM mode)
- Table extraction to CSV/Markdown
- Real-dataset benchmark harness (drop your ground-truth pairs into `tests/benchmark/data/`)
- Batch processing with parallelism (`--workers`)
