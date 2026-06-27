"""CLIP 图像向量索引。"""
import logging

import chromadb
from llama_index.core.schema import ImageDocument
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

from config.settings import settings
from llm.embed import get_image_embed_model

logger = logging.getLogger(__name__)


class ClipIndex:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=settings.chroma.db_path)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma.clip_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._embed = get_image_embed_model()
        self._vs = ChromaVectorStore(chroma_collection=self._collection)
        self._sc = StorageContext.from_defaults(vector_store=self._vs)
        # NOTE: llama_index 0.14.x VectorStoreIndex requires storage_context
        # as keyword; passing nodes=[] explicitly to avoid triggering
        # default transformations / embed_model auto-detection.
        self._index = VectorStoreIndex(
            nodes=[],
            storage_context=self._sc,
            embed_model=self._embed,
        )

    def add_documents(self, docs) -> int:
        """把图片 Document 转 ImageDocument 入库。"""
        count = 0
        for d in docs:
            img_path = d.metadata.get("image_path") or d.metadata.get("file_path")
            self._index.insert(ImageDocument(
                text="",
                image_path=img_path,
                metadata=dict(d.metadata),
            ))
            count += 1
        return count

    def as_retriever(self, top_k: int):
        return self._index.as_retriever(similarity_top_k=top_k)

    @property
    def count(self) -> int:
        return self._collection.count()
