import os
import yaml
from pathlib import Path


def _resolve_env(value: str) -> str:
    """支持 ${ENV_VAR} 语法从环境变量读取值。"""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_key = value[2:-1]
        return os.environ.get(env_key, "")
    return value


class _Settings:
    """全局配置，从 config.yaml 加载。"""

    def __init__(self):
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        self.models = raw["models"]
        self.llm = {k: _resolve_env(v) for k, v in raw["llm"].items()}
        self.chroma = raw["chroma"]
        self.retrieval = raw["retrieval"]
        self.paths = raw["paths"]

        device = self.models["device"]
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device


settings = _Settings()
