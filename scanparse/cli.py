"""ScanParse command-line interface.

Examples::

    scanparse scan.pdf --lang en,hi --mode fast --out ./result
    scanparse page.png --mode accurate --binarize --dpi 200 --out out/
    scanparse --ui                        # launch the web interface
"""

from __future__ import annotations

import argparse
import os
import sys

from scanparse import __version__, parse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scanparse",
        description="Parse poor-quality English & Hindi scanned documents into Markdown/JSON/DOCX.",
    )
    p.add_argument("input", nargs="?", help="Path to a scanned image or PDF (omit with --ui)")
    p.add_argument("--version", action="version", version=f"scanparse {__version__}")
    p.add_argument("--lang", default="en,hi",
                   help="Comma-separated languages (en, hi). Default: en,hi")
    p.add_argument("--mode", choices=["fast", "accurate", "vlm"], default="fast",
                   help="fast=Tesseract (default, fastest), accurate=Surya, vlm=PaddleOCR")
    p.add_argument("--out", default=".", help="Output directory. Default: current dir")
    p.add_argument("--format", default="md", choices=["md", "json", "docx", "all"],
                   help="Output format(s). Default: md")
    p.add_argument("--no-binarize", action="store_true", help="Skip binarization")
    p.add_argument("--no-deskew", action="store_true", help="Skip deskew")
    p.add_argument("--dpi", type=float, default=96.0,
                   help="Assumed input DPI for upscaling. Default: 96")
    p.add_argument("--allowed-root", default=None,
                   help="Confine input reads to this directory (security)")
    p.add_argument("--ui", action="store_true", help="Launch the Gradio web UI instead of parsing")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.ui or not args.input:
        return _launch_ui()

    if not os.path.exists(args.input):
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        return 1

    doc = parse(
        args.input,
        lang=[x.strip() for x in args.lang.split(",")],
        mode=args.mode,
        allowed_root=args.allowed_root,
        dpi=args.dpi,
        deskew_on=not args.no_deskew,
        binarize_on=not args.no_binarize,
    )

    os.makedirs(args.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    fmts = ["md", "json", "docx"] if args.format == "all" else [args.format]
    for fmt in fmts:
        if fmt == "md":
            doc.to_markdown(os.path.join(args.out, f"{base}.md"))
        elif fmt == "json":
            doc.to_json(os.path.join(args.out, f"{base}.json"))
        elif fmt == "docx":
            try:
                doc.to_docx(os.path.join(args.out, f"{base}.docx"))
            except RuntimeError as exc:
                print(f"Warning: {exc}", file=sys.stderr)

            print(f"Parsed {len(doc.pages)} page(s) in "
                  f"{doc.metadata.get('processing_time_s', 0):.2f}s -> {args.out}/")
    return 0


def _launch_ui() -> int:
    try:
        from scanparse.app import create_ui
    except ImportError as exc:
        print("Error: Gradio is not installed. Run: pip install 'scanparse[ui]'", file=sys.stderr)
        raise SystemExit(1) from exc
    create_ui().launch(server_name="0.0.0.0", server_port=7860, share=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
