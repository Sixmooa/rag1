"""用 docling 解析 PDF：保留结构、抽图片。"""
import logging
import os

from llama_index.core.schema import Document

logger = logging.getLogger(__name__)

_converter = None


def _get_converter():
    global _converter
    if _converter is None:
        from docling.document_converter import DocumentConverter
        _converter = DocumentConverter()
    return _converter


def _picture_to_pil(pic):
    """Extract a PIL image from a docling PictureItem.

    docling 2.x stores `pic.image` as an `ImageRef`. Use its `pil_image`
    property when available; fall back to the legacy PIL attribute for
    older versions. Returns None if no image payload is present.
    """
    img_ref = getattr(pic, "image", None)
    if img_ref is None:
        return None
    # docling >= 2.x: ImageRef with a `.pil_image` property
    pil = getattr(img_ref, "pil_image", None)
    if callable(pil):
        try:
            pil = pil()
        except Exception:
            pil = None
    if pil is None:
        # Older API where pic.image was directly a PIL.Image
        pil = img_ref if hasattr(img_ref, "save") else None
    return pil


def parse_pdf(path: str) -> list[Document]:
    """返回 list[Document]：
      - 文本节点：metadata.source_type='pdf'
      - 图片节点：metadata.source_type='image'，metadata.image_path
    """
    abs_path = os.path.abspath(path)
    logger.info("docling 解析 PDF: %s", abs_path)

    conv = _get_converter()
    result = conv.convert(abs_path)
    doc = result.document

    docs: list[Document] = []

    # 文本：导出为 markdown
    md_text = doc.export_to_markdown()
    if md_text and md_text.strip():
        docs.append(Document(
            text=md_text,
            metadata={
                "source_type": "pdf",
                "file_path": abs_path,
                "page_number": 1,
                "doc_id": os.path.basename(abs_path),
            },
        ))

    # 图片：保存到 source/.images/<pdfname>/img_N.png
    img_dir = os.path.join(os.path.dirname(abs_path), ".images", os.path.basename(abs_path))
    os.makedirs(img_dir, exist_ok=True)

    try:
        pictures = list(doc.pictures)
    except Exception:
        pictures = []
    for i, pic in enumerate(pictures, 1):
        try:
            pil_img = _picture_to_pil(pic)
            if pil_img is None:
                logger.debug("跳过无图片负载的 picture %s#%d", abs_path, i)
                continue
            img_path = os.path.join(img_dir, f"img_{i}.png")
            pil_img.save(img_path)
            docs.append(Document(
                text="",
                metadata={
                    "source_type": "image",
                    "file_path": abs_path,
                    "image_path": img_path,
                    "page_number": 1,
                    "doc_id": os.path.basename(abs_path),
                },
            ))
        except Exception as e:
            logger.warning("图片抽取失败 %s#%d: %s", abs_path, i, e)

    logger.info("docling 完成 %s: %d 文本 + %d 图片",
                abs_path,
                sum(1 for d in docs if d.metadata["source_type"] == "pdf"),
                sum(1 for d in docs if d.metadata["source_type"] == "image"))
    return docs
