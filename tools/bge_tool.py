from sentence_transformers import SentenceTransformer

from config.settings import settings


class BgeTool:
    """BGE-M3 多语言文本向量提取工具。"""

    def __init__(self):
        self.device = settings.device
        model_name = settings.models["bge"]
        self.model = SentenceTransformer(model_name, device=self.device)

    def get_embedding(self, text: str) -> list:
        emb = self.model.encode(text, normalize_embeddings=True)
        return emb.tolist()


_bge_tool: BgeTool | None = None


def get_bge_tool() -> BgeTool:
    global _bge_tool
    if _bge_tool is None:
        _bge_tool = BgeTool()
    return _bge_tool
