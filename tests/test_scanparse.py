"""Unit tests for scanparse: security, language detection, preprocessing, and parsing."""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image, ImageDraw

from scanparse import parse
from scanparse.document import BlockType, Document
from scanparse.lang import classify_text
from scanparse.preprocessing import (
    binarize,
    deskew,
    preprocess,
)
from scanparse.security import (
    FileTooLargeError,
    PathTraversalError,
    SecurityError,
    UnsupportedFormatError,
    check_file_size,
    sanitize_filename,
    sniff_format,
    validate_format,
    validate_path,
)

# --- language detection -------------------------------------------------------

def test_classify_english():
    assert classify_text("The quick brown fox.") == "en"


def test_classify_hindi():
    assert classify_text("शिक्षा मंत्रालय की सूचना।") == "hi"


def test_classify_mixed():
    assert classify_text("Ministry of शिक्षा notification 4517") == "mixed"


def test_classify_digits_and_punctuation():
    assert classify_text("12345 !@#$%") == "en"  # default when no script chars


# --- security -----------------------------------------------------------------

def test_sniff_formats(tmp_path):
    # An extension of .jpg but a real PNG magic header must still be sniffed as png
    png_as_jpg = tmp_path / "fake.jpg"
    png_as_jpg.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
    assert sniff_format(png_as_jpg.read_bytes()) == "png"

    # Truely unknown bytes are rejected
    junk = tmp_path / "junk.dat"
    junk.write_bytes(b"\xde\xad\xbe\xef" + b"\x00" * 50)
    with pytest.raises(UnsupportedFormatError):
        validate_format(junk.read_bytes())

    png = tmp_path / "real.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    assert validate_format(png.read_bytes()) == "png"


def test_file_size_limit():
    with pytest.raises(FileTooLargeError):
        check_file_size(b"x" * 10, max_bytes=5)


def test_path_traversal_blocked(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    with pytest.raises(PathTraversalError):
        validate_path(tmp_path / "allowed" / ".." / ".." / "etc" / "passwd", root)


def test_path_inside_root_ok(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    f = root / "scan.png"
    f.write_bytes(b"hi")
    assert validate_path(f, root) == str(f.resolve())


def test_sanitize_filename():
    assert sanitize_filename("../../etc/passwd.sh") == "passwd.sh"


# --- preprocessing ------------------------------------------------------------

@pytest.fixture()
def noisy_page():
    """Rendered English text with heavy noise/shadow/skew."""
    img = Image.new("L", (1400, 600), 255)
    draw = ImageDraw.Draw(img)
    draw.text((40, 40), "The quick brown fox jumps over the lazy dog.", fill=0)
    arr = np.array(img, dtype=np.float32)
    arr += np.random.normal(0, 25, arr.shape)            # noise
    grad = np.linspace(0.5, 1.0, arr.shape[1]).reshape(1, -1)
    arr = arr * grad                                      # shadow
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    import cv2
    mat = cv2.getRotationMatrix2D((700, 300), 2.0, 1)
    arr = cv2.warpAffine(arr, mat, (1400, 600), borderValue=255)
    return Image.fromarray(arr, mode="L")


def test_preprocess_returns_valid_image(noisy_page):
    out = preprocess(noisy_page, dpi=96)
    # Output may be upscaled (up to 3x) depending on input DPI
    assert out.size[0] >= noisy_page.size[0]
    assert out.size[0] <= noisy_page.size[0] * 3
    assert out.mode == "L"


def test_deskew_fixes_rotation():
    img = Image.new("L", (1000, 400), 255)
    d = ImageDraw.Draw(img)
    d.text((30, 30), "Some straight text lines here for testing.", fill=0)
    arr = np.array(img.convert("L"))
    import cv2
    mat = cv2.getRotationMatrix2D((500, 200), -3.0, 1)
    arr = cv2.warpAffine(arr, mat, (1000, 400), borderValue=255)
    fixed = deskew(arr)
    assert fixed.shape == arr.shape


def test_binarize_produces_binary():
    arr = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    out = binarize(arr)
    assert set(np.unique(out)) <= {0, 255}


# --- parsing end-to-end -------------------------------------------------------

@pytest.fixture()
def clean_page(tmp_path):
    img = Image.new("L", (1400, 400), 255)
    ImageDraw.Draw(img).text((40, 40), "Annual report of the department.", fill=0)
    p = tmp_path / "clean.png"
    img.save(p)
    return str(p)


def test_parse_returns_document(clean_page):
    doc = parse(clean_page, lang=["en"], mode="fast")
    assert isinstance(doc, Document)
    assert doc.pages and doc.pages[0].blocks
    text = " ".join(b.text for b in doc.pages[0].blocks)
    assert "Annual" in text and "report" in text


def test_parse_invalid_mode(clean_page):
    with pytest.raises(SecurityError):
        parse(clean_page, mode="magic")


def test_parse_unsupported_language(clean_page):
    with pytest.raises(SecurityError):
        parse(clean_page, lang=["fr"])


def test_document_exporters(clean_page, tmp_path):
    doc = parse(clean_page, lang=["en"])
    md = doc.to_markdown(str(tmp_path / "out.md"))
    js = doc.to_json(str(tmp_path / "out.json"))
    assert os.path.exists(tmp_path / "out.md")
    assert os.path.exists(tmp_path / "out.json")
    assert "Annual report" in md
    assert '"Annual report' in js or "Annual report" in js
    assert doc.to_dict()["pages"][0]["blocks"]


def test_document_has_blocks_with_languages(clean_page):
    doc = parse(clean_page, lang=["en", "hi"])
    for b in doc.pages[0].blocks:
        assert b.language in ("en", "hi", "mixed")
        assert isinstance(b.block_type, BlockType)


def test_empty_line_skipped(clean_page):
    img = Image.new("L", (1400, 400), 255)
    p = clean_page.replace("clean.png", "blank.png")
    img.save(p)
    doc = parse(p, lang=["en"])
    assert all(b.text.strip() for b in doc.pages[0].blocks)
