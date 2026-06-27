import pytest
from config.settings import Settings


def test_settings_from_yaml_loads_all_sections():
    s = Settings.from_yaml("config/config.yaml")
    assert s.llm.api_key, "api_key must be set"
    assert s.llm.base_url.startswith("http")
    assert s.chroma.db_path
    assert s.retrieval.top_k > 0
    assert s.models.bge.startswith("BAAI/")


def test_env_var_substitution(monkeypatch, tmp_path):
    yaml_text = """
models: {clip: x, bge: y, clip_image: z, device: cpu}
llm:
  api_key: ${TEST_KEY}
  base_url: http://x
  model: m
chroma: {db_path: ./d, clip_collection: c, text_collection: t}
retrieval: {top_k: 3, fusion_alpha: 0.3, rrf_k: 60, chunk_size: 800, chunk_overlap: 200, rerank_top_n: 3, markdown_max_tokens: 1500, pdf_dpi: 200}
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    monkeypatch.setenv("TEST_KEY", "real-key-123")
    s = Settings.from_yaml(str(p))
    assert s.llm.api_key == "real-key-123"


def test_missing_api_key_raises(tmp_path):
    from config.settings import ConfigError
    yaml_text = """
models: {clip: x, bge: y, clip_image: z, device: cpu}
llm: {api_key: "", base_url: http://x, model: m}
chroma: {db_path: ./d, clip_collection: c, text_collection: t}
retrieval: {top_k: 3, fusion_alpha: 0.3, rrf_k: 60, chunk_size: 800, chunk_overlap: 200, rerank_top_n: 3, markdown_max_tokens: 1500, pdf_dpi: 200}
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ConfigError):
        Settings.from_yaml(str(p))
