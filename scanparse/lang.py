"""Script/language detection for mixed English-Hindi text.

Uses Unicode-range heuristics to classify text lines as English, Hindi
(Devanagari), or mixed. Devanagari numerals and common punctuation are
counted as part of the Devanagari script.
"""

from __future__ import annotations


def _in_range(ch: str, lo: str, hi: str) -> bool:
    return ord(lo) <= ord(ch) <= ord(hi)


def _is_devanagari(ch: str) -> bool:
    return (
        _in_range(ch, "\u0900", "\u097F")
        or _in_range(ch, "\uA8E0", "\uA8FF")
        or _in_range(ch, "\u1CD0", "\u1CFF")
    )


def _is_latin(ch: str) -> bool:
    return ch.isascii() and ch.isalpha()


def _is_ignored(ch: str) -> bool:
    if ch.isspace():
        return True
    if ch in ".,;:!?\"'()[]/\\-—–%@$#&*<>_=+\\" or ch.isdigit():
        return True
    # Devanagari digits ०-९
    if _in_range(ch, "\u0966", "\u096F"):
        return True
    return False


def classify_text(text: str) -> str:
    """Classify a text fragment as 'en', 'hi', or 'mixed'.

    Characters outside Latin/Devanagari/ignored sets are skipped (e.g., symbols).
    An empty or non-text fragment returns 'en' (default).
    """
    hi_count = 0
    en_count = 0
    for ch in text:
        if _is_ignored(ch):
            continue
        if _is_devanagari(ch):
            hi_count += 1
        elif _is_latin(ch):
            en_count += 1
    total = hi_count + en_count
    if total == 0:
        return "en"
    if hi_count > 0 and en_count > 0:
        return "mixed"
    if hi_count > 0:
        return "hi"
    return "en"


def classify_lines(lines: list[str]) -> list[str]:
    """Classify each line; returns a list parallel to *lines*."""
    return [classify_text(line) for line in lines]
