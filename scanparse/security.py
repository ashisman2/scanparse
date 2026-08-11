"""Security utilities: input validation and safe file handling.

ScanParse may run on untrusted uploads (web UI, pipelines). This module
enforces hard limits on every input so that a malicious or accidental
file cannot hang the process, exhaust memory, or escape the sandbox:

- File size cap (default 100 MiB) and per-image dimension cap (10 000 px).
- Strict allow-list of MIME signatures for PNG/JPEG/WebP/PDF.
- Path traversal prevention via ``os.path.realpath`` resolution inside an
  allowed root directory.
- No ``eval``/``exec``/shell interpretation of user content anywhere in
  the parsing path; OCR output is treated as plain data.
"""

from __future__ import annotations

import io
import os

MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MiB
MAX_IMAGE_DIM = 10_000
MAX_PAGES_PDF = 500

ALLOWED_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".pdf", ".tif", ".tiff", ".bmp"}

# Magic-byte signatures (file type, not extension):
# PNG, JPEG (start + end-of-image markers), WebP (RIFF/WEBP), PDF
MAGIC_SIGNATURES = [
    ("png", b"\x89PNG\r\n\x1a\n"),
    ("jpeg", b"\xff\xd8\xff"),
    ("pdf", b"%PDF-"),
]


class ScanParseError(Exception):
    """Base error for all ScanParse failures."""


class SecurityError(ScanParseError):
    """Raised when an input fails a security check."""


class FileTooLargeError(SecurityError):
    pass


class UnsupportedFormatError(SecurityError):
    pass


class PathTraversalError(SecurityError):
    pass


class ImageTooLargeError(SecurityError):
    pass


def check_file_size(data_or_path, max_bytes: int = MAX_FILE_BYTES) -> int:
    """Return the size of a file-like/bytes/path object, raising if too large."""
    if isinstance(data_or_path, (bytes, bytearray, memoryview)):
        size = len(data_or_path)
    elif isinstance(data_or_path, (str, os.PathLike)):
        size = os.path.getsize(data_or_path)
    else:  # file-like
        pos = data_or_path.tell()
        data_or_path.seek(0, io.SEEK_END)
        size = data_or_path.tell()
        data_or_path.seek(pos)
    if size > max_bytes:
        raise FileTooLargeError(
            f"File too large: {size:,} bytes exceeds limit of {max_bytes:,} bytes."
        )
    return size


def sniff_format(data: bytes) -> str | None:
    """Identify the true file type from magic bytes (ignores the extension)."""
    for name, sig in MAGIC_SIGNATURES:
        if data[: len(sig)] == sig:
            return name
    # WebP: RIFF....WEBP
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        return "webp"
    # TIFF: little/big endian II / MM + 42
    if data[:2] in (b"II", b"MM") and len(data) >= 8 and data[2:4] in (b"\x2a\x00", b"\x00\x2a"):
        return "tiff"
    return None


def validate_format(data: bytes) -> str:
    """Return the sniffed format or raise UnsupportedFormatError."""
    fmt = sniff_format(data)
    if fmt is None:
        raise UnsupportedFormatError(
            "Unrecognized or unsupported file format. Allowed: PNG, JPEG, WebP, TIFF, PDF."
        )
    return fmt


def validate_path(path: str | os.PathLike, allowed_root: str | os.PathLike) -> str:
    """Resolve *path* and ensure it stays inside *allowed_root* (no traversal)."""
    root = os.path.realpath(str(allowed_root))
    resolved = os.path.realpath(str(path))
    if not (resolved == root or resolved.startswith(root + os.sep)):
        raise PathTraversalError(
            f"Path escapes allowed directory: {path}"
        )
    return resolved


def validate_image_dimensions(image) -> None:
    """Raise ImageTooLargeError if width/height exceed MAX_IMAGE_DIM."""
    w, h = image.size
    if w > MAX_IMAGE_DIM or h > MAX_IMAGE_DIM:
        raise ImageTooLargeError(
            f"Image too large: {w}x{h}px exceeds {MAX_IMAGE_DIM}x{MAX_IMAGE_DIM}px limit."
        )


def sanitize_filename(name: str) -> str:
    """Strip directory components and dangerous characters from an upload name."""
    base = os.path.basename(name)
    base = "".join(ch for ch in base if ch.isalnum() or ch in ".-_ ")
    return base or "unnamed"
