"""Image preprocessing pipeline for poor-quality scans.

Quality-first design for accuracy on degraded scans:

1. Upscale first (LANCZOS) — noise filters and binarization behave far better
   at 2-3x size, and Tesseract reads small Devanagari matras much better
   when strokes are >= 20 px tall.
2. Strong denoise (NLM) on the enlarged image.
3. Illumination flattening (background estimation subtraction) to kill shadows
   and stains — the single most important fix for poor scans.
4. Contrast stretch / CLAHE to recover faded text.
5. Deskew via projection profile (fast, accurate to ~0.1 deg).
6. Sauvola adaptive binarization — keeps thin strokes that Otsu eats.

Usage::

    from scanparse import preprocess

    img = preprocess("noisy_scan.png")          # returns PIL.Image
    img = preprocess(img, dpi=300, binarize_on=True)
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from scanparse.security import validate_image_dimensions


def _to_cv(img) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img
    gray = (
        Image.open(img).convert("L")
        if not isinstance(img, Image.Image)
        else img.convert("L")
    )
    return np.array(gray)


def upscale(
    img: np.ndarray,
    min_dpi: float = 300.0,
    dpi: float = 96.0,
    max_factor: float = 3.0,
) -> np.ndarray:
    """Lanczos upscaling so OCR gets at least ~min_dpi effective resolution.

    Larger images dramatically improve Devanagari matra recognition.
    Capped by *max_factor* to stay fast.
    """
    h, w = img.shape[:2]
    factor = min(max_factor, min_dpi / max(dpi, 1.0))
    if factor <= 1.0 or (w * factor) > 12_000:
        return img
    return cv2.resize(
        img,
        (max(1, int(w * factor)), max(1, int(h * factor))),
        interpolation=cv2.INTER_LANCZOS4,
    )


def denoise(img: np.ndarray, strength: int = 2) -> np.ndarray:
    """Remove grain/speckle noise. Strength 1 (light), 2 (heavy), 3 (max)."""
    strength = max(1, min(strength, 3))
    return cv2.fastNlMeansDenoising(
        img, None, h=7 * strength, templateWindowSize=7, searchWindowSize=21
    )


def flatten_illumination(img: np.ndarray) -> np.ndarray:
    """Remove shadows/stains by dividing out a large-kernel background estimate.

    This is the key step for poorly scanned pages with uneven lighting.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    bg = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, kernel)
    with np.errstate(divide="ignore", invalid="ignore"):
        flat = np.where(bg > 0, img.astype(np.float32) / bg.astype(np.float32) * 255.0, 0.0)
    return np.clip(flat, 0, 255).astype(np.uint8)


def normalize_contrast(img: np.ndarray, grid: int = 8) -> np.ndarray:
    """CLAHE contrast normalization to recover faded text."""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(grid, grid))
    return clahe.apply(img)


def deskew(img: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Detect global text skew via projection-profile analysis and rotate to fix."""
    h, w = img.shape[:2]
    if h < 64 or w < 64:
        return img

    scale = min(1.0, 512.0 / max(h, w))
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    _, bin_img = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img = cv2.bitwise_not(bin_img)  # text = white

    best_angle, best_var = 0.0, 0.0
    steps = 61
    for i in range(steps):
        angle = -max_angle + (2 * max_angle) * i / (steps - 1)
        cx, cy = small.shape[1] / 2, small.shape[0] / 2
        rot = cv2.warpAffine(
            bin_img,
            cv2.getRotationMatrix2D((cx, cy), angle, 1),
            (small.shape[1], small.shape[0]),
            flags=cv2.INTER_NEAREST,
        )
        variance = float(np.var(rot.sum(axis=1)))
        if variance > best_var:
            best_var, best_angle = variance, angle
    if abs(best_angle) < 0.25:
        return img
    return cv2.warpAffine(
        img,
        cv2.getRotationMatrix2D((w / 2, h / 2), -best_angle, 1),
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderValue=255,
    )


def binarize(img: np.ndarray, method: str = "sauvola") -> np.ndarray:
    """Adaptive binarization. 'sauvola' keeps thin strokes under shadows/stains."""
    if method == "otsu":
        _, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return out
    f = img.astype(np.float32)
    win = (25, 25)
    mean = cv2.boxFilter(f, -1, win)
    sqmean = cv2.boxFilter(f * f, -1, win)
    std = cv2.sqrt(np.maximum(sqmean - mean * mean, 0.0))
    k, r = 0.34, 128.0  # Sauvola parameters tuned for document text
    thresh = mean * (1 + k * (std / r - 1))
    return np.where(img > thresh, 255, 0).astype(np.uint8)


def preprocess(
    image,
    dpi: float = 96.0,
    deskew_on: bool = True,
    denoise_strength: int = 2,
    contrast: bool = True,
    binarize_on: bool = False,
    upsample: bool = True,
    flatten: bool = True,
    max_upscale_factor: float = 3.0,
) -> Image.Image:
    """Full preprocessing pipeline. Accepts a path, numpy array, or PIL.Image.

    Default output is ENHANCED GRAYSCALE: the Tesseract LSTM engine (default
    in fast mode) reads enhanced grayscale better than hard binarization on
    degraded scans — Sauvola can eat thin Devanagari matras and serifs.
    Enable ``binarize_on=True`` for legacy workflows or Otsu/Sauvola outputs.

    Order matters for accuracy: enlarge -> denoise -> flatten shadows ->
    contrast -> deskew -> (binarize).
    """
    img = _to_cv(image)

    if upsample:
        img = upscale(img, dpi=dpi, max_factor=max_upscale_factor)
    img = denoise(img, strength=denoise_strength)
    if flatten:
        img = flatten_illumination(img)
    if contrast:
        img = normalize_contrast(img)
    if deskew_on:
        img = deskew(img)
    if binarize_on:
        img = binarize(img)

    out = Image.fromarray(img)
    validate_image_dimensions(out)
    return out
