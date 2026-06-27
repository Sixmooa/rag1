"""把知识库元信息（文件清单/计数）作为字符串挂到 query_bundle 上，
供 GenerationPipeline 拼到 system prompt。"""
import os
from typing import Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle


class KnowledgeMetaCollector(BaseNodePostprocessor):
    """不修改 nodes；只在 postprocess 时把命中节点的 file_path 去重收集，
    并把全库文件清单写到 _last_meta（供 GenerationPipeline 取用）。"""

    def __init__(self, list_source_files_fn):
        super().__init__()
        self._list_source_files = list_source_files_fn
        self._last_meta: Optional[str] = None

    def _postprocess_nodes(self, nodes: list[NodeWithScore],
                           query_bundle: Optional[QueryBundle] = None):
        files = self._list_source_files()
        pdf_count = sum(1 for f in files if f["file_type"] == "pdf")
        img_count = sum(1 for f in files if f["file_type"] == "image")
        txt_count = len(files) - pdf_count - img_count
        file_list = "\n".join(
            f"  - {f['file_name']} ({f['file_type']})" for f in files
        ) or "  (空)"
        self._last_meta = (
            f"当前知识库统计：共 {len(files)} 个文件"
            f"（PDF {pdf_count} 篇, 图片 {img_count} 张, 文本 {txt_count} 个）。\n"
            f"文件清单：\n{file_list}"
        )
        return nodes  # 不改 nodes，只副作用收集

    @property
    def last_meta(self) -> Optional[str]:
        return self._last_meta


def list_source_files(clip_index, bge_index) -> list[dict]:
    """从两个 collection 的 metadata 里去重收集已入库文件。"""
    files: dict[str, dict] = {}
    for col in (clip_index._collection, bge_index._collection):
        if col.count() == 0:
            continue
        batch = col.get(include=["metadatas"])
        for meta in batch.get("metadatas") or []:
            fp = (meta or {}).get("file_path", "")
            if not fp or fp in files:
                continue
            ext = os.path.splitext(fp)[1].lower()
            files[fp] = {
                "file_name": os.path.basename(fp),
                "file_type": (meta or {}).get("source_type", ext.lstrip(".")),
            }
    return list(files.values())
