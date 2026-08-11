"""ScanParse: Open-source parser for poor-quality English & Hindi scanned documents.

ScanParse is designed to turn degraded, low-quality scans (books, forms, archives)
into clean, structured, developer-friendly output (Markdown, JSON, DOCX).

Typical usage::

    from scanparse import parse

    doc = parse("scan.png", lang=["en", "hi"], mode="accurate")
    print(doc.to_markdown())
    doc.to_docx("output.docx")
"""

from scanparse.document import Block, BlockType, Document
from scanparse.parsers import parse
from scanparse.preprocessing import preprocess

__version__ = "0.1.0"

__all__ = ["parse", "preprocess", "Document", "Block", "BlockType"]
