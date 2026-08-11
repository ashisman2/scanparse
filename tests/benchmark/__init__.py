"""Benchmark fixtures generator.

Creates synthetic degraded scans (English + Hindi) with known ground truth,
plus an evaluation runner that computes CER/WER with jiwer. Real-world
datasets can be dropped into ``tests/benchmark/data/`` with matching
``*.txt`` ground-truth files.
"""

from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

EN_SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "Annual report of the finance department, fiscal year 2025.",
    "This invoice is due within thirty days of the issue date.",
    "Ministry of Education notification number 4517, dated March third.",
    "Chapter seven: The industrial revolution and its consequences.",
]

HI_SAMPLES = [
    "राष्ट्रीय शिक्षा नीति के अंतर्गत नए दिशानिर्देश जारी किए गए।",
    "वित्त विभाग की वार्षिक रिपोर्ट सन २०२५।",
    "यह इनवॉइस जारी तारीख से तीस दिन के अंदर भुगतान योग्य है।",
    "शिक्षा मंत्रालय की सूचना संख्या ४५१७, तिथि तीन मार्च।",
    "अध्याय सात: औद्योगिक क्रांति और उसके परिणाम।",
]


def _available_font(size: int):
    paths = [
        "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render_text(
    text: str,
    width: int = 1400,
    height: int = 600,
    font_size: int = 42,
    n_each: int = 1,
) -> Image.Image:
    """Render a text page (PIL grayscale) with a Devanagari-capable font."""
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    font = _available_font(font_size)
    y = 40
    for line in text.splitlines():
        draw.text((40, y), line, fill=0, font=font)
        y += font_size + 24
    return img


def degrade(image: Image.Image, severity: float = 0.6) -> Image.Image:
    """Apply realistic scan degradation: noise, blur, shadows, stains, skew."""
    arr = np.array(image, dtype=np.float32)
    h, w = arr.shape

    # Shadow gradient (uneven lighting)
    grad = np.linspace(0.55, 1.0, w).reshape(1, w) * np.linspace(0.6, 1.0, h).reshape(h, 1)
    arr = arr * (grad ** severity)

    # Grain noise
    arr = arr + np.random.normal(0, 18 * severity, arr.shape)

    # Stains (random dark blobs) — kept moderate so text lines remain readable,
    # mirroring real poor scans where stains rarely cover full lines.
    for _ in range(max(1, int(1.5 * severity))):
        cx, cy = random.randint(0, w - 1), random.randint(0, h - 1)
        rr = random.randint(60, 180)
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= rr ** 2
        arr[mask] *= random.uniform(0.75, 0.95)

    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Motion blur
    if severity > 0.3:
        k = int(3 * severity) * 2 + 1
        arr = cv2.blur(arr, (k, max(1, k // 3)))

    # Skew
    if severity > 0.4:
        angle = random.uniform(-2.5, 2.5)
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        arr = cv2.warpAffine(arr, mat, (w, h), borderValue=255)

    return Image.fromarray(arr, mode="L")


def make_benchmark_set(out_dir: str, n_each: int = 5) -> list[tuple[str, str]]:
    """Render + degrade sample pages; returns list of (image_path, ground_truth_path)."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    pairs = []
    all_samples = [(t, "en") for t in EN_SAMPLES] + [(t, "hi") for t in HI_SAMPLES]
    random.shuffle(all_samples)
    for i, (text, lang) in enumerate(all_samples[: 2 * n_each]):
        img = render_text(text)
        dirty = degrade(img, severity=0.5 + 0.1 * (i % 3))
        img_path = os.path.join(out_dir, f"sample_{i:02d}_{lang}.png")
        gt_path = os.path.join(out_dir, f"sample_{i:02d}_{lang}.txt")
        dirty.save(img_path)
        with open(gt_path, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        pairs.append((img_path, gt_path))
    return pairs
