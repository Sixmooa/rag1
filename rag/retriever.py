import os

from config.settings import settings
from tools.clip_tool import get_clip_tool
from tools.bge_tool import get_bge_tool
from db.chroma_store import get_store


def search_by_clip(query_text: str, top_k: int = None) -> dict:
    if top_k is None:
        top_k = settings.retrieval["default_top_k"]
    clip = get_clip_tool()
    query_emb = clip.get_text_embedding(query_text)
    store = get_store()
    return store.clip_collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )


def search_by_bge(query_text: str, top_k: int = None) -> dict:
    if top_k is None:
        top_k = settings.retrieval["default_top_k"]
    bge = get_bge_tool()
    query_emb = bge.get_embedding(query_text)
    store = get_store()
    return store.text_collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )


def hybrid_search(query_text: str, top_k: int = None,
                  alpha: float = None, rrf_k: int = None) -> list[dict]:
    """
    混合检索：融合 CLIP 和 BGE 检索结果（RRF 融合）。

    Args:
        query_text: 查询文本
        top_k: 最终返回结果数
        alpha: CLIP 权重 (0~1)，1-alpha 为 BGE 权重
        rrf_k: RRF 参数，经典值 60
    """
    if top_k is None:
        top_k = settings.retrieval["default_top_k"]
    if alpha is None:
        alpha = settings.retrieval["hybrid_alpha"]
    if rrf_k is None:
        rrf_k = settings.retrieval["rrf_k"]

    expand_k = min(top_k * 3, 10)

    clip_results = search_by_clip(query_text, top_k=expand_k)
    bge_results = search_by_bge(query_text, top_k=expand_k)

    scores: dict[tuple, float] = {}
    metadata_map: dict[tuple, dict] = {}
    text_map: dict[tuple, str] = {}

    for rank, (doc_id, meta) in enumerate(
        zip(clip_results['ids'][0], clip_results['metadatas'][0])
    ):
        key = (meta.get("file_path", ""), meta.get("page_number", 0))
        scores[key] = scores.get(key, 0) + alpha / (rrf_k + rank + 1)
        metadata_map[key] = meta
        text_map.setdefault(key, "")

    for rank, (doc_id, meta, doc_text) in enumerate(zip(
        bge_results['ids'][0], bge_results['metadatas'][0], bge_results['documents'][0]
    )):
        key = (meta.get("file_path", ""), meta.get("page_number", 0))
        scores[key] = scores.get(key, 0) + (1 - alpha) / (rrf_k + rank + 1)
        metadata_map[key] = meta
        if doc_text:
            text_map[key] = doc_text

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for (file_path, page_num), score in sorted_results:
        results.append({
            "file_path": file_path,
            "page_number": page_num,
            "rrf_score": score,
            "text": text_map.get((file_path, page_num), ""),
            "metadata": metadata_map.get((file_path, page_num), {}),
        })

    print(f"\n混合检索结果 (alpha={alpha}):")
    for i, r in enumerate(results):
        print(f"  {i + 1}. {os.path.basename(r['file_path'])} 第{r['page_number']}页"
              f" | RRF分数: {r['rrf_score']:.4f}"
              f" | 文本长度: {len(r['text'])}")

    return results
