import re
import os
from PIL import Image
import fitz  # PyMuPDF

from config.settings import settings


def detect_file_type(file_path: str) -> str:
    """根据文件路径检测类型：返回 'pdf' / 'image' / 'md' / 'txt'。"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return 'pdf'
    if ext in ('.png', '.jpg', '.jpeg', '.bmp', '.webp'):
        return 'image'
    return _detect_text_type(file_path)


def detect_bytes_type(content: bytes, filename: str) -> str:
    """根据文件头魔数 + 扩展名识别文件类型（用于上传场景）。"""
    ext = os.path.splitext(filename)[1].lower()
    if content[:5] == b"%PDF-":
        return "pdf"
    if content[:4] == b"\x89PNG":
        return "image"
    if content[:3] == b"\xff\xd8\xff":
        return "image"
    if content[:2] == b"BM":
        return "image"
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image"
    if ext == ".pdf":
        return "pdf"
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        return "image"
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            text = content.decode(enc)
            if text.strip():
                return "text"
            break
        except (UnicodeDecodeError, ValueError):
            continue
    return "unknown"


def _detect_text_type(file_path: str) -> str:
    """区分 markdown 与纯文本。"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ('.md', '.markdown'):
        return 'md'
    if ext in ('.txt', '.text'):
        return 'txt'

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(4096)
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
            content = f.read(4096)

    md_score = 0
    if re.search(r'^#{1,6}\s+', content, re.MULTILINE):
        md_score += 3
    if re.search(r'\*\*[^*]+\*\*', content):
        md_score += 1
    if re.search(r'\[[^\]]+\]\([^)]+\)', content):
        md_score += 2
    if re.search(r'^```', content, re.MULTILINE):
        md_score += 2
    if re.search(r'^[-*+]\s+', content, re.MULTILINE):
        md_score += 1
    if re.search(r'^>\s+', content, re.MULTILINE):
        md_score += 1

    return 'md' if md_score >= 2 else 'txt'


def read_file_content(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
            return f.read()


def parse_pdf_pages(pdf_path: str, dpi: int = None):
    """逐页返回 (page_image: PIL.Image, page_text: str, page_num: int)。"""
    if dpi is None:
        dpi = settings.retrieval.get("pdf_dpi", 200)
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = page.get_text().strip()
        pages.append((img, text, page_num + 1))
    doc.close()
    return pages


def chunk_text(content: str, file_type: str,
               chunk_size: int = None, chunk_overlap: int = None) -> list[str]:
    """将文本内容分块，返回字符串列表。"""
    if chunk_size is None:
        chunk_size = settings.retrieval["chunk_size"]
    if chunk_overlap is None:
        chunk_overlap = settings.retrieval["chunk_overlap"]

    if file_type == 'md':
        return _chunk_markdown(content, chunk_size)
    return _chunk_plain(content, chunk_size, chunk_overlap)


def _chunk_markdown(content: str, chunk_size: int) -> list[str]:
    sections = re.split(r'^(#{1,6}\s+.+)$', content, flags=re.MULTILINE)
    chunks = []
    current_title = ""
    buffer = ""
    for part in sections:
        if re.match(r'^#{1,6}\s+', part):
            if buffer.strip():
                chunks.append((current_title, buffer.strip()))
            current_title = part.strip()
            buffer = ""
        else:
            buffer += part
    if buffer.strip():
        chunks.append((current_title, buffer.strip()))

    final = []
    for title, text in chunks:
        combined = (title + "\n" + text).strip() if title else text.strip()
        if len(text) > chunk_size * 2:
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            sub_buf = title + "\n" if title else ""
            for para in paragraphs:
                if len(sub_buf) + len(para) > chunk_size and len(sub_buf) > len(title or ""):
                    final.append(sub_buf.strip())
                    sub_buf = (title + "\n" + para) if title else para
                else:
                    sub_buf += "\n\n" + para
            if sub_buf.strip():
                final.append(sub_buf.strip())
        else:
            final.append(combined)
    return final if final else [content.strip()]


def _chunk_plain(content: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    chunks = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) > chunk_size and buf:
            chunks.append(buf.strip())
            buf = buf[-chunk_overlap:] + "\n\n" + para if chunk_overlap > 0 else para
        else:
            buf += "\n\n" + para if buf else para
    if buf.strip():
        chunks.append(buf.strip())
    return chunks if chunks else [content.strip()]
