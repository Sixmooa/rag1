"""双索引融合检索：QueryFusionRetriever 内部 RRF，可启用 LLM 子查询改写。"""
import logging

from llama_index.core.retrievers import QueryFusionRetriever

from config.settings import settings

logger = logging.getLogger(__name__)


def build_fusion_retriever(clip_retriever, bge_retriever, llm=None,
                           top_k: int = None, use_query_gen: bool = True):
    """构造 QueryFusionRetriever。

    Args:
        clip_retriever / bge_retriever: 已 as_retriever() 的实例
        llm: 用于子查询生成；None 表示不做 LLM 改写
        top_k: 每个查询返回数
        use_query_gen: 是否启用 LLM 生成子查询
    """
    if top_k is None:
        top_k = settings.retrieval.top_k

    return QueryFusionRetriever(
        retrievers=[clip_retriever, bge_retriever],
        llm=llm if use_query_gen else None,
        similarity_top_k=top_k,
        num_queries=3 if use_query_gen else 1,
        mode="reciprocal_rerank",
        use_async=True,
    )
