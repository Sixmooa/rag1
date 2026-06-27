"""BGE-M3 文本向量索引。"""
import logging

import chromadb
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

from config.settings import settings
from llm.embed import get_text_embed_model

logger = logging.getLogger(__name__)


class BgeIndex:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=settings.chroma.db_path)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma.text_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._embed = get_text_embed_model()
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
        self._sent_splitter = SentenceSplitter(
            chunk_size=settings.retrieval.chunk_size,
            chunk_overlap=settings.retrieval.chunk_overlap,
        )
        self._md_splitter = MarkdownNodeParser()

    def add_documents(self, docs) -> int:
        total = 0
        for d in docs:
            kind = d.metadata.get("source_type")
            if kind == "markdown":
                nodes = self._md_splitter.get_nodes_from_documents([d])
                # 二次切超长段
                refined = []
                for n in nodes:
                    if len(n.text) > settings.retrieval.markdown_max_tokens:
                        refined.extend(self._sent_splitter.get_nodes_from_documents([n]))
                    else:
                        refined.append(n)
                nodes = refined
            else:
                # pdf / text 都用 sentence splitter
                nodes = self._sent_splitter.get_nodes_from_documents([d])

            for n in nodes:
                n.metadata = {**d.metadata, **(n.metadata or {})}
                self._index.insert(n)
                total += 1
        return total

    def as_retriever(self, top_k: int):
        return self._index.as_retriever(similarity_top_k=top_k)

    @property
    def count(self) -> int:
        return self._collection.count()
