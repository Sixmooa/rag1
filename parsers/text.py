"""TXT/MD 文本解析。"""
import os
from llama_index.core.schema import Document


def _read(path: str) -> str:
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return ""


def parse_text(path: str) -> list[Document]:
    return [Document(
        text=_read(path),
        metadata={
            "source_type": "text",
            "file_path": os.path.abspath(path),
            "doc_id": os.path.basename(path),
        },
    )]


def parse_markdown(path: str) -> list[Document]:
    return [Document(
        text=_read(path),
        metadata={
            "source_type": "markdown",
            "file_path": os.path.abspath(path),
            "doc_id": os.path.basename(path),
        },
    )]
