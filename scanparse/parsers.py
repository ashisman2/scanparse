"""Multi-engine OCR parsing core.

Mode matrix (designed for SPEED first):

- ``fast``    (default, always available): pytesseract with eng+hin. Zero model
              downloads; runs on CPU in seconds. Best for clean-ish scans.
- ``accurate`` (opt-in, ``pip install "scanparse[surya]"``): Surya recognition +
              layout analysis. Slower start-up, better on degraded pages.
- ``vlm``     (experimental, ``pip install "scanparse[vlm]"``): PaddleOCR
              Devanagari recognition. Kept optional since torch is heavy.

Both modes run the preprocessing pipeline first, since that is the biggest
accuracy lever for poor scans.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Sequence

from scanparse.document import Block, BlockType, Document, Page
from scanparse.lang import classify_text
from scanparse.preprocessing import preprocess
from scanparse.security import (
    MAX_PAGES_PDF,
    ImageTooLargeError,
    SecurityError,
    check_file_size,
    validate_format,
    validate_path,
)

LANG_MAP = {"en": "eng", "hi": "hin", "english": "eng", "hindi": "hin"}


class EngineNotAvailableError(SecurityError):
    pass


def _check_langs(lang: Sequence[str]) -> list[str]:
    langs = [x.lower() for x in lang]
    for _lang in langs:
        if _lang not in LANG_MAP:
            raise SecurityError(f"Unsupported language: {_lang!r}. Use 'en' and/or 'hi'.")
    return langs


def _tesseract_ocr(image, langs: list[str], dpi: float) -> tuple[list[Block], str]:
    """Fast-mode OCR via Tesseract with per-line boxes and confidence scores.

    Quality settings: OSD auto-detect off (we deskewed), dense-text page
    segmentation, LSTM engine only (better Devanagari), white background.
    """
    import pytesseract  # local import: heavy only when used

    t_lang = "+".join(LANG_MAP[_lang] for _lang in langs)
    # Tell Tesseract the effective DPI of the preprocessed image.
    # Preprocessing upscales by up to 3x from the assumed input *dpi*.
    effective_dpi = min(600, max(150, int(dpi * 3)))
    custom_config = (
        f"--psm 6 --oem 1 --dpi {effective_dpi} "
        "-c tessedit_char_whitelist='' "
        "-c tessedit_pageseg_mode=6 "
    )
    data = pytesseract.image_to_data(
        image, lang=t_lang, config=custom_config, output_type=pytesseract.Output.DICT
    )

    # Group lines: page_num/par_num/block_num/line_num -> text
    lines: dict[tuple, dict] = {}
    for i in range(len(data["text"])):
        text = (data["text"][i] or "").strip()
        key = (data["page_num"][i], data["block_num"][i], data["line_num"][i])
        entry = lines.setdefault(
            key,
            {"text": [], "x0": None, "y0": None, "x1": None,
             "y1": None, "conf": []},
        )
        entry["text"].append(text)
        if data["left"][i] >= 0:
            entry["x0"] = min(entry["x0"] or 1e9, data["left"][i])
            entry["y0"] = min(entry["y0"] or 1e9, data["top"][i])
            entry["x1"] = max(entry["x1"] or 0, data["left"][i] + data["width"][i])
            entry["y1"] = max(entry["y1"] or 0, data["top"][i] + data["height"][i])
            entry["conf"].append(data["conf"][i])

    blocks = []
    for (pn, bn, ln), entry in lines.items():
        text = " ".join(t for t in entry["text"] if t)
        if not text:
            continue
        bbox = None
        if entry["x0"] is not None:
            w, h = image.size
            bbox = [entry["x0"] / w, entry["y0"] / h, entry["x1"] / w, entry["y1"] / h]
        blocks.append(Block(
            block_type=BlockType.PARAGRAPH,
            text=text,
            order=pn * 10000 + bn * 100 + ln,
            language=classify_text(text),
            confidence=sum(entry["conf"]) / len(entry["conf"]) if entry["conf"] else None,
            bbox=bbox,
        ))
    return blocks, "fast"


def _surya_ocr(image, langs: list[str], _dpi: float) -> tuple[list[Block], str]:
    """Accurate-mode OCR via Surya (layout + recognition)."""
    try:
        from surya.layout import LayoutPredictor
        from surya.model.recognition.config import LANGUAGE_MAP as SURYA_LANGS
        from surya.recognition import RecognitionPredictor
    except ImportError as exc:
        raise EngineNotAvailableError(
            "Surya is not installed. Install with: pip install 'scanparse[surya]'"
        ) from exc

    surya_map = {"en": "en", "hi": "hi"}
    names = [surya_map[_lang] for _lang in langs]
    for n in names:
        if n not in SURYA_LANGS:
            raise EngineNotAvailableError(f"Surya has no model for language: {n}")

    layout_pred = LayoutPredictor()
    rec_pred = RecognitionPredictor()
    try:
        layout = layout_pred([image], max_pages=1)[0]
        results = rec_pred([image], [names], [layout.bboxes], max_pages=1)
    finally:
        layout_pred.shutdown()
        rec_pred.shutdown()

    blocks = []
    for i, (lb, res) in enumerate(zip(layout.bboxes, results.text_lines)):
        blocks.append(Block(
            block_type=_surya_label_to_block_type(lb.label),
            text=res.text,
            order=i,
            language=classify_text(res.text),
            bbox=[lb.bbox[0], lb.bbox[1], lb.bbox[2], lb.bbox[3]],
        ))
    return blocks, "accurate"


def _surya_label_to_block_type(label: str) -> BlockType:
    mapping = {
        "Title": BlockType.TITLE,
        "Section-header": BlockType.HEADING,
        "List-item": BlockType.LIST_ITEM,
        "Table": BlockType.TABLE,
        "Caption": BlockType.CAPTION,
        "Footnote": BlockType.FOOTNOTE,
    }
    return mapping.get(label, BlockType.PARAGRAPH)


def _to_images(source: str, dpi: int = 200) -> list:
    """Convert PDF (or single image) to a list of PIL grayscale Images."""
    from PIL import Image as PILImage

    fmt = validate_format(open(source, "rb").read())
    if fmt == "pdf":
        from pdf2image import pdf2image
        pages = pdf2image.convert_from_path(source, dpi=dpi)
        if len(pages) > MAX_PAGES_PDF:
            raise ImageTooLargeError(f"PDF has {len(pages)} pages; limit is {MAX_PAGES_PDF}.")
        return [p.convert("L") for p in pages]
    # Single image
    img = PILImage.open(source).convert("L")
    img.verify()  # noqa: B018 - intentionally verify the image file
    return [PILImage.open(source).convert("L")]


def _should_retry(blocks: list[Block], mean_conf: float) -> bool:
    """Decide if a low-confidence run deserves a stronger-preprocessing retry."""
    if mean_conf >= 60:
        return False
    n_words = sum(len(b.text.split()) for b in blocks)
    # Very short outputs at low confidence usually mean the page was mangled.
    return n_words < 8


def parse(
    source: str,
    lang: Sequence[str] | str = ("en", "hi"),
    mode: str = "fast",
    allowed_root: Optional[str] = None,
    retry_on_low_confidence: bool = True,
    **preprocess_kwargs,
) -> Document:
    """Parse a scanned image or PDF into a structured Document.

    Args:
        source: path to an image (PNG/JPEG/WebP/TIFF) or PDF.
        lang: ``'en'``, ``'hi'``, or ``['en', 'hi']`` for mixed pages.
        mode: ``'fast'`` (Tesseract, default), ``'accurate'`` (Surya), ``'vlm'``.
        allowed_root: optional directory confinement for path safety.

    Returns:
        A ``Document`` with pages of typed blocks.
    """
    if mode not in ("fast", "accurate", "vlm"):
        raise SecurityError(f"Unknown mode: {mode!r}. Use 'fast', 'accurate', or 'vlm'.")

    start = time.perf_counter()
    langs = _check_langs([lang] if isinstance(lang, str) else lang)

    if allowed_root:
        source = validate_path(source, allowed_root)
    check_file_size(source)

    images = _to_images(source)
    doc = Document(source=os.path.abspath(source), language=langs, mode=mode)

    for idx, img in enumerate(images):
        cleaned = preprocess(img, **preprocess_kwargs)
        if mode == "accurate":
            blocks, used_mode = _surya_ocr(cleaned, langs, preprocess_kwargs.get("dpi", 96.0))
        elif mode == "vlm":
            blocks, used_mode = _paddle_ocr(cleaned, langs)
        else:
            blocks, used_mode = _tesseract_ocr(cleaned, langs, preprocess_kwargs.get("dpi", 96.0))

        # Quality auto-retry: if the page came out mangled (few words at low
        # confidence), reprocess with maximum denoise and compare; keep the
        # higher-quality result. Adds ~2s only on truly bad pages.
        if retry_on_low_confidence and mode == "fast" and _should_retry(blocks, 0):
            tougher = preprocess(
                img, denoise_strength=3, contrast=True, upsample=True,
                max_upscale_factor=preprocess_kwargs.get("max_upscale_factor", 3.0),
                deskew_on=preprocess_kwargs.get("deskew_on", True),
            )
            blocks2, _ = _tesseract_ocr(tougher, langs, preprocess_kwargs.get("dpi", 96.0))
            if _page_quality(blocks2) > _page_quality(blocks):
                blocks = blocks2

        doc.pages.append(Page(page_number=idx + 1, blocks=blocks,
                              language="mixed" if len(langs) > 1 else langs[0]))

    doc.mode = used_mode
    doc.metadata["processing_time_s"] = time.perf_counter() - start
    doc.metadata["num_pages"] = len(images)
    return doc


def _page_quality(blocks: list[Block]) -> float:
    """Heuristic page quality: total words weighted by mean confidence."""
    if not blocks:
        return 0.0
    words = sum(len(b.text.split()) for b in blocks)
    confs = [b.confidence for b in blocks if b.confidence and b.confidence > 0]
    mean_c = (sum(confs) / len(confs)) if confs else 30.0
    return words * (mean_c / 100.0)


def _paddle_ocr(image, langs: list[str]) -> tuple[list[Block], str]:
    """Experimental VLM-mode OCR via PaddleOCR Devanagari support."""
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise EngineNotAvailableError(
            "PaddleOCR is not installed. Install with: pip install 'scanparse[vlm]'"
        ) from exc
    ocr = PaddleOCR(use_angle_cls=True, lang="hi" if "hi" in langs else "en", show_log=False)
    result = ocr.ocr(image, cls=True)
    blocks = []
    for i, line in enumerate(result[0] or []):
        box, (text, conf) = line
        if not text or not text.strip():
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        w, h = image.size
        blocks.append(Block(
            block_type=BlockType.PARAGRAPH,
            text=text.strip(),
            order=i,
            language=classify_text(text),
            confidence=float(conf),
            bbox=[min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h],
        ))
    return blocks, "vlm"
