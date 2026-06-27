"""文件类型检测与解析器路由。"""
import os
from typing import Literal

from llama_index.core.schema import Document

FileKind = Literal["pdf", "image", "markdown", "text"]


class UnknownFileTypeError(Exception):
    pass


def detect_type(path: str) -> FileKind:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return "pdf"
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        return "image"
    if ext in (".md", ".markdown"):
        return "markdown"
    if ext in (".txt", ".text"):
        return "text"
    raise UnknownFileTypeError(f"unsupported extension: {ext}")


def parse(path: str) -> list[Document]:
    """根据扩展名分发到对应解析器。"""
    kind = detect_type(path)
    if kind == "pdf":
        from parsers.docling_pdf import parse_pdf
        return parse_pdf(path)
    if kind == "image":
        from parsers.image import parse_image
        return parse_image(path)
    if kind == "markdown":
        from parsers.text import parse_markdown
        return parse_markdown(path)
    if kind == "text":
        from parsers.text import parse_text
        return parse_text(path)
    raise UnknownFileTypeError(kind)
