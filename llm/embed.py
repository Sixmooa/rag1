"""文本/图像 embedding 模型工厂（单例、懒加载）。"""
from config.settings import settings

_text_embed = None
_image_embed = None


def get_text_embed_model():
    """BGE-M3 文本 embedding。"""
    global _text_embed
    if _text_embed is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        _text_embed = HuggingFaceEmbedding(
            model_name=settings.models.bge,
            device=settings.models.device,
        )
    return _text_embed


def get_image_embed_model():
    """多语言 CLIP，文本与图像共享向量空间。"""
    global _image_embed
    if _image_embed is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        # clip-ViT-B-32-multilingual-v1 是 SentenceTransformer，能同时跑文本和图像
        _image_embed = HuggingFaceEmbedding(
            model_name=settings.models.clip,
            device=settings.models.device,
        )
    return _image_embed
