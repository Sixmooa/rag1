"""检索流水线：fusion retriever + reranker + metadata collector。"""
import logging

from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    def __init__(self, fusion_retriever, reranker, meta_collector):
        self._fusion = fusion_retriever
        self._reranker = reranker
        self._meta = meta_collector

    async def retrieve(self, query: str, top_k: int = None) -> list[NodeWithScore]:
        logger.info("融合检索 query=%r", query)
        nodes = await self._fusion.aretrieve(query)
        logger.info("融合返回 %d 节点", len(nodes))

        nodes = self._reranker.postprocess_nodes(nodes, query_str=query)
        logger.info("重排后 %d 节点", len(nodes))

        # 触发 metadata 收集
        self._meta.postprocess_nodes(nodes, query_str=query)
        return nodes

    @property
    def meta_text(self) -> str:
        return self._meta.last_meta or ""
