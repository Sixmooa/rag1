"""单例容器：懒加载所有 pipeline，给 api/cli 用。"""
import logging

from config.settings import settings
from llm.client import get_llm
from indices.clip_index import ClipIndex
from indices.bge_index import BgeIndex
from indices.fusion import build_fusion_retriever
from postprocessors.reranker import get_bge_reranker
from postprocessors.metadata_filter import KnowledgeMetaCollector, list_source_files
from pipeline.indexing import IndexingPipeline
from pipeline.retrieval import RetrievalPipeline
from pipeline.generation import GenerationPipeline

logger = logging.getLogger(__name__)

_clip_idx = None
_bge_idx = None
_indexing = None
_retrieval = None
_generation = None


def get_clip_index() -> ClipIndex:
    global _clip_idx
    if _clip_idx is None:
        _clip_idx = ClipIndex()
    return _clip_idx


def get_bge_index() -> BgeIndex:
    global _bge_idx
    if _bge_idx is None:
        _bge_idx = BgeIndex()
    return _bge_idx


def get_indexing_pipeline() -> IndexingPipeline:
    global _indexing
    if _indexing is None:
        from parsers.registry import parse as parser_parse
        class _ParserAdapter:
            parse = staticmethod(parser_parse)
        _indexing = IndexingPipeline(
            _ParserAdapter(), get_clip_index(), get_bge_index(),
        )
    return _indexing


def get_retrieval_pipeline() -> RetrievalPipeline:
    global _retrieval
    if _retrieval is None:
        clip_r = get_clip_index().as_retriever(settings.retrieval.top_k * 3)
        bge_r = get_bge_index().as_retriever(settings.retrieval.top_k * 3)
        fusion = build_fusion_retriever(clip_r, bge_r, llm=get_llm(),
                                        top_k=settings.retrieval.top_k,
                                        use_query_gen=True)
        reranker = get_bge_reranker()
        meta_collector = KnowledgeMetaCollector(
            list_source_files_fn=lambda: list_source_files(get_clip_index(), get_bge_index())
        )
        _retrieval = RetrievalPipeline(fusion, reranker, meta_collector)
    return _retrieval


def get_generation_pipeline() -> GenerationPipeline:
    global _generation
    if _generation is None:
        from memory import session as session_store
        _generation = GenerationPipeline(
            get_llm(), get_retrieval_pipeline(), session_store,
        )
    return _generation
