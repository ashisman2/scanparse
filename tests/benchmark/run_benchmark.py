#!/usr/bin/env python3
"""Run the ScanParse accuracy & speed benchmark.

Generates synthetic degraded English/Hindi pages (or uses real data in
tests/benchmark/data/), runs the full parsing pipeline on each, and reports
CER/WER per language plus overall processing time per page.

Usage::

    python -m tests.benchmark.run_benchmark --mode fast
    python -m tests.benchmark.run_benchmark --mode accurate --regenerate
"""

from __future__ import annotations

import argparse
import os
import time

from jiwer import cer, wer

from scanparse import parse

from . import make_benchmark_set

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _norm(text: str) -> str:
    return " ".join(text.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fast", choices=["fast", "accurate"])
    ap.add_argument("--lang", default="both", choices=["en", "hi", "both"])
    ap.add_argument("--regenerate", action="store_true", help="Recreate synthetic benchmark pages")
    ap.add_argument("--n", type=int, default=5, help="Synthetic pages per language")
    args = ap.parse_args()

    if args.regenerate or not os.path.isdir(DATA_DIR) or not os.listdir(DATA_DIR):
        make_benchmark_set(DATA_DIR, n_each=args.n)

    files = sorted(os.listdir(DATA_DIR))
    image_files = [f for f in files if f.endswith(".png")]
    if not image_files:
        print("No benchmark images found.")
        return 1

    results: dict[str, list[float]] = {"en": [], "hi": []}
    total_start = time.perf_counter()

    print(f"{'file':42s} {'mode':9s} {'CER':>6s} {'WER':>6s} {'time(s)':>8s}")
    print("-" * 80)
    for fname in image_files:
        lang = "hi" if "_hi" in fname else "en"
        if args.lang != "both" and lang != args.lang:
            continue
        base, _, _ = fname.rpartition(".png")
        gt_path = os.path.join(DATA_DIR, base + ".txt")
        if not os.path.exists(gt_path):
            continue
        with open(gt_path, encoding="utf-8") as fh:
            gt = _norm(fh.read().strip())

        t0 = time.perf_counter()
        doc = parse(os.path.join(DATA_DIR, fname), lang=[lang], mode=args.mode)
        elapsed = time.perf_counter() - t0
        pred = _norm(" ".join(b.text for p in doc.pages for b in p.blocks))
        c, w = cer(gt, pred), wer(gt, pred)
        results[lang].append(c)
        print(f"{fname:42s} {args.mode:9s} {c*100:6.1f}% {w*100:6.1f}% {elapsed:8.2f}")

    print("-" * 80)
    elapsed = time.perf_counter() - total_start
    for langtag in ("en", "hi"):
        vals = results[langtag]
        if vals:
            print(f"Avg {langtag.upper()} CER: "
                  f"{sum(vals) / len(vals) * 100:.1f}% (total {elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
