"""BGE-reranker-v2-m3 cross-encoder 精排。

FlagEmbeddingReranker 在当前版本只接受 top_n / model / use_fp16；
device 由底层 FlagEmbedding 按 CUDA 可见性自动选择。
"""
from config.settings import settings
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker


def get_bge_reranker(top_n: int = None) -> FlagEmbeddingReranker:
    if top_n is None:
        top_n = settings.retrieval.rerank_top_n
    return FlagEmbeddingReranker(
        model="BAAI/bge-reranker-v2-m3",
        top_n=top_n,
    )
