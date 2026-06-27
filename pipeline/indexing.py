"""入库流水线：parse → 按 source_type 路由到 ClipIndex/BgeIndex。文件级失败隔离。"""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    file: str
    status: str   # ok / error
    detail: str


class IndexingPipeline:
    def __init__(self, parser, clip_index, bge_index):
        self._parser = parser
        self._clip = clip_index
        self._bge = bge_index

    def ingest(self, file_path: str) -> IngestResult:
        try:
            docs = self._parser.parse(file_path)
        except Exception as e:
            logger.warning("解析失败 %s: %s", file_path, e)
            return IngestResult(file=file_path, status="error",
                                detail=f"解析失败: {type(e).__name__}: {e}")

        image_docs = [d for d in docs if d.metadata.get("source_type") == "image"]
        text_docs = [d for d in docs if d.metadata.get("source_type") != "image"]

        n_img = n_txt = 0
        if image_docs:
            try:
                n_img = self._clip.add_documents(image_docs)
            except Exception as e:
                logger.warning("图像入库失败 %s: %s", file_path, e)
                return IngestResult(file=file_path, status="error",
                                    detail=f"图像入库失败: {type(e).__name__}: {e}")
        if text_docs:
            try:
                n_txt = self._bge.add_documents(text_docs)
            except Exception as e:
                logger.warning("文本入库失败 %s: %s", file_path, e)
                return IngestResult(file=file_path, status="error",
                                    detail=f"文本入库失败: {type(e).__name__}: {e}")

        return IngestResult(file=file_path, status="ok",
                            detail=f"{n_txt} 文本块 / {n_img} 图")
