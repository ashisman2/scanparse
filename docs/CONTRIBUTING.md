# Contributing to ScanParse

Thank you for your interest in ScanParse. The project is MIT-licensed and welcomes contributions of all sizes — bug fixes, new OCR backends, benchmark datasets, documentation, and UI improvements.

## Code of conduct

Be respectful, be constructive, and keep discussions focused on the code. The project's explicit goals are **accuracy on poor scans**, **speed and low resource usage**, and **security by design**; contributions that move against any of these three goals will not be merged even if the code is correct.

## Getting started

```bash
git clone https://github.com/ashisman2/scanparse.git
cd scanparse
sudo apt-get install -y tesseract-ocr tesseract-ocr-hin tesseract-ocr-eng poppler-utils
pip install -e ".[dev]"
pytest tests/
ruff check scanparse tests
```

## Pull request guidelines

- One logical change per PR. Large refactors should be discussed in an issue first.
- All tests must pass: `pytest tests/` and `ruff check scanparse tests`.
- New OCR backends must include a benchmark entry (see `tests/benchmark/`) so accuracy impact is measurable.
- Keep the default install lightweight. Heavy models (Surya, torch) must stay behind optional extras in `pyproject.toml`.
- Security is non-negotiable: inputs must flow through `scanparse/security.py`; no `eval`/`exec`/shell interpretation of user content.
- CI runs `pytest` on Python 3.10–3.12 plus a `pip-audit` vulnerability scan; both must pass.

## Testing your accuracy improvements

```bash
# On the synthetic degraded set
python -m tests.benchmark.run_benchmark --mode fast --regenerate

# On your own data (drop name.png + name.txt into tests/benchmark/data/)
python -m tests.benchmark.run_benchmark --mode fast
```

Report before/after CER numbers in the PR description.

## Roadmap opportunities

Handwritten Devanagari support, table extraction, a real-dataset benchmark corpus, and parallel batch processing (`--workers`) are open areas where help is especially welcome.
