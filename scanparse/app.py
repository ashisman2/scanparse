"""Lightweight Gradio web UI for ScanParse.

Designed to be secure by default: uploads are validated through the
``security`` module, processing runs only on validated content, and the
UI exposes no shell access.
"""

from __future__ import annotations

import os

import gradio as gr

from scanparse import parse
from scanparse.security import (
    SecurityError,
    check_file_size,
    validate_format,
)

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".pdf", ".tif", ".tiff", ".bmp"}


def _handle(file_obj, langs, mode, deskew, denoise, contrast, binarize, upsample):
    if file_obj is None:
        return "Please upload a scanned image or PDF.", None, None
    path = file_obj.name
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_EXT:
        return f"Unsupported file type: {ext}", None, None
    try:
        check_file_size(path)
        validate_format(open(path, "rb").read())
    except SecurityError as exc:
        return f"Rejected: {exc}", None, None

    doc = parse(
        path,
        lang=[x.strip() for x in langs.split(",")] if langs else ["en", "hi"],
        mode=mode,
        dpi=96.0,
        deskew_on=deskew,
        denoise_strength=2 if denoise else 1,
        contrast=contrast,
        binarize_on=binarize,
        upsample=upsample,
    )
    md = doc.to_markdown()
    preview = md if len(md) < 4000 else md[:4000] + "\n\n... (full output downloaded as .md)"
    return md, preview, doc.metadata.get("processing_time_s")


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="ScanParse") as ui:
        gr.Markdown("# ScanParse — Poor-Quality English & Hindi Scan Parser")
        gr.Markdown("Upload a scan (PNG, JPEG, WebP, TIFF, PDF). Output: Markdown + JSON.")

        with gr.Row():
            with gr.Column():
                file_in = gr.File(label="Upload scan", file_types=list(ALLOWED_EXT))
                with gr.Row():
                    langs = gr.Textbox(label="Languages", value="en,hi", scale=1)
                    mode = gr.Dropdown(
                        label="Mode", choices=["fast", "accurate", "vlm"], value="fast", scale=1
                    )
                with gr.Row():
                    deskew = gr.Checkbox(label="Deskew", value=True)
                    denoise = gr.Checkbox(label="Strong denoise", value=False)
                    contrast = gr.Checkbox(label="Contrast fix", value=True)
                    binarize = gr.Checkbox(label="Binarize", value=True)
                    upsample = gr.Checkbox(label="Upsample", value=True)
                run_btn = gr.Button("Parse", variant="primary")
            with gr.Column():
                preview = gr.Textbox(label="Preview (first 4k chars)", lines=12)
                out_md = gr.File(label="Markdown output")
                time_out = gr.Textbox(label="Processing time (s)", interactive=False)

        run_btn.click(
            _handle,
            inputs=[file_in, langs, mode, deskew, denoise, contrast, binarize, upsample],
            outputs=[out_md, preview, time_out],
        )
    return ui


if __name__ == "__main__":
    create_ui().launch(server_name="0.0.0.0", server_port=7860)
