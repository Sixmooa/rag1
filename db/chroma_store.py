import os

import chromadb
from PIL import Image

from config.settings import settings
from tools.clip_tool import get_clip_tool
from tools.bge_tool import get_bge_tool
from tools.file_parser import (
    detect_file_type, read_file_content, parse_pdf_pages, chunk_text,
)


class ChromaStore:
    """ChromaDB 存储封装，提供统一的入库接口。"""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma["db_path"])
        metric = settings.chroma["metric"]

        self.clip_collection = self.client.get_or_create_collection(
            name=settings.chroma["clip_collection"],
            metadata={"hnsw:space": metric},
        )
        self.text_collection = self.client.get_or_create_collection(
            name=settings.chroma["text_collection"],
            metadata={"hnsw:space": metric},
        )

    # ---- 入库接口 ----

    def add_image(self, image_path: str):
        print(f"正在处理图片: {image_path}")
        clip = get_clip_tool()
        image = Image.open(image_path).convert("RGB")
        embedding = clip.get_image_embedding(image)
        doc_id = f"img_{os.path.basename(image_path)}"

        self.clip_collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[{"source_type": "image", "file_path": image_path}],
            documents=[""],
        )
        print(f"图片 {image_path} 已成功入库！")

    def add_pdf(self, pdf_path: str):
        print(f"正在处理 PDF: {pdf_path}")
        clip = get_clip_tool()
        bge = get_bge_tool()
        pages = parse_pdf_pages(pdf_path)

        clip_embeddings, clip_metas, clip_ids = [], [], []
        text_embeddings, text_metas, text_ids, text_docs = [], [], [], []

        for img, text, page_num in pages:
            # CLIP 向量
            clip_emb = clip.get_image_embedding(img)
            clip_embeddings.append(clip_emb)
            clip_metas.append({"source_type": "pdf", "file_path": pdf_path, "page_number": page_num})
            clip_ids.append(f"pdf_{os.path.basename(pdf_path)}_page_{page_num}")

            # BGE 向量
            if text:
                bge_emb = bge.get_embedding(text)
                text_embeddings.append(bge_emb)
                text_metas.append({"source_type": "pdf", "file_path": pdf_path, "page_number": page_num})
                text_ids.append(f"txt_{os.path.basename(pdf_path)}_page_{page_num}")
                text_docs.append(text)

        if clip_ids:
            self.clip_collection.add(
                ids=clip_ids, embeddings=clip_embeddings,
                metadatas=clip_metas, documents=[""] * len(clip_ids),
            )
        if text_ids:
            self.text_collection.add(
                ids=text_ids, embeddings=text_embeddings,
                metadatas=text_metas, documents=text_docs,
            )

        print(f"PDF {pdf_path} (共 {len(clip_ids)} 页) 已成功入库！"
              f" [CLIP: {len(clip_ids)} 页, BGE: {len(text_ids)} 页有文本]")

    def add_text_file(self, file_path: str):
        file_type = detect_file_type(file_path)
        print(f"检测到文件类型: {file_type} | 正在处理: {file_path}")

        content = read_file_content(file_path)
        if not content.strip():
            print(f"文件 {file_path} 内容为空，跳过。")
            return

        chunks = chunk_text(content, file_type)
        bge = get_bge_tool()

        text_ids, text_embeddings, text_metas, text_docs = [], [], [], []
        for i, chunk in enumerate(chunks):
            emb = bge.get_embedding(chunk)
            text_ids.append(f"txt_{os.path.basename(file_path)}_chunk_{i + 1}")
            text_embeddings.append(emb)
            text_metas.append({
                "source_type": file_type, "file_path": file_path,
                "chunk_index": i + 1, "total_chunks": len(chunks),
            })
            text_docs.append(chunk)

        if text_ids:
            self.text_collection.add(
                ids=text_ids, embeddings=text_embeddings,
                metadatas=text_metas, documents=text_docs,
            )

        print(f"文件 {file_path} (类型: {file_type}, 共 {len(chunks)} 块) 已成功入库！")

    # ---- 统计 ----

    @property
    def clip_count(self) -> int:
        return self.clip_collection.count()

    @property
    def text_count(self) -> int:
        return self.text_collection.count()

    def list_source_files(self) -> list[dict]:
        """返回所有已入库的去重文件摘要。"""
        files: dict[str, dict] = {}
        for col in (self.clip_collection, self.text_collection):
            if col.count() == 0:
                continue
            batch = col.get(include=["metadatas"])
            for meta in batch["metadatas"] or []:
                fp = meta.get("file_path", "")
                if not fp or fp in files:
                    continue
                ext = os.path.splitext(fp)[1].lower()
                files[fp] = {
                    "file_name": os.path.basename(fp),
                    "file_type": meta.get("source_type", ext.lstrip(".")),
                }
        return list(files.values())


_store: ChromaStore | None = None


def get_store() -> ChromaStore:
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store
