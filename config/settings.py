"""强类型配置：从 yaml 加载，支持 ${ENV_VAR} 替换。"""
import os
import re
from pathlib import Path
import yaml
from pydantic import BaseModel

_ENV_PATTERN = re.compile(r"^\$\{([A-Z_][A-Z0-9_]*)\}$")


class ConfigError(Exception):
    """配置错误。"""


def _resolve_env(value):
    if not isinstance(value, str):
        return value
    m = _ENV_PATTERN.match(value)
    return os.environ.get(m.group(1), "") if m else value


class ModelsConfig(BaseModel):
    clip: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    bge: str = "BAAI/bge-m3"
    clip_image: str = "openai/clip-vit-base-patch32"
    device: str = "auto"


class LLMConfig(BaseModel):
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"


class ChromaConfig(BaseModel):
    db_path: str = "./multimodal_db"
    clip_collection: str = "image_index"
    text_collection: str = "text_index"


class RetrievalConfig(BaseModel):
    top_k: int = 3
    fusion_alpha: float = 0.3
    rrf_k: int = 60
    chunk_size: int = 800
    chunk_overlap: int = 200
    rerank_top_n: int = 3
    markdown_max_tokens: int = 1500
    pdf_dpi: int = 200


class Settings(BaseModel):
    models: ModelsConfig
    llm: LLMConfig
    chroma: ChromaConfig
    retrieval: RetrievalConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Settings":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        llm_raw = {k: _resolve_env(v) for k, v in raw.get("llm", {}).items()}
        if not llm_raw.get("api_key"):
            raise ConfigError("llm.api_key is required (config.yaml or ${ENV})")

        models = raw.get("models", {})
        device = models.get("device", "auto")
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        return cls(
            models=ModelsConfig(**{**models, "device": device}),
            llm=LLMConfig(**llm_raw),
            chroma=ChromaConfig(**raw.get("chroma", {})),
            retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        )


def _load() -> Settings:
    cfg = Path(__file__).parent / "config.yaml"
    return Settings.from_yaml(str(cfg))


settings = _load()
