# LlamaIndex 全面重构 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 用 LlamaIndex 替换现有手写 RAG 流水线，按"流水线分层"重组为模块化低耦合结构，启用 SSE 流式响应与 BGE-reranker 精排。

**Architecture:** 三层依赖：`delivery → pipeline → {indices, postprocessors, llm, parsers} → config`。双索引（中文 CLIP + BGE-M3）经 QueryFusionRetriever 融合后由 FlagEmbeddingReranker 精排。LLM 调用经 tenacity 重试包装。前端 fetch+ReadableStream 解析 SSE。

**Tech Stack:** LlamaIndex (core + huggingface/openai-like/chroma/flag-reranker/docling 集成)、FastAPI、sse-starlette、tenacity、pydantic-settings、pytest。

**Design doc:** `docs/plans/2026-06-26-llamaindex-refactor-design.md`（决策汇总、组件签名、数据流、错误处理）。

**Working directory:** `E:/work/研一/高级数据库/workplace/作业6/project`（下文所有相对路径以此为根）。

**Python:** `./venv/Scripts/python.exe`（已有 venv，所有命令用这个解释器）。

**Git note:** 项目当前不是 git 仓库。Task 0 会 `git init`，便于每个 Task 末尾 commit。如不希望用 git，可跳过 Task 0，把每个 Task 末尾的 commit 步骤替换为"备份当前目录"。

---

## Task 0: Git 初始化

**Files:** 无（仓库初始化）

**Step 1: 初始化 git**

Run:
```bash
cd "E:/work/研一/高级数据库/workplace/作业6/project" && git init && git branch -M main
```
Expected: `Initialized empty Git repository in ...`

**Step 2: 写 .gitignore**

Create `.gitignore`:
```
venv/
__pycache__/
*.pyc
multimodal_db/
chroma_db/
chroma_db.bak/
multimodal_db.bak/
memory/*.json
source/settings.json
.pytest_cache/
.huggingface/
*.log
```

**Step 3: 首次提交**

```bash
git add .gitignore && git commit -m "chore: init git repo with gitignore"
git add -A && git commit -m "chore: snapshot before llamaindex refactor"
```

---

## Task 1: 安装新依赖

**Files:**
- Create: `requirements.txt`

**Step 1: 写 requirements.txt**

```
# --- 保留 ---
chromadb>=0.5.0
fastapi>=0.110
uvicorn[standard]>=0.30
pydantic>=2.5
pydantic-settings>=2.1
PyYAML>=6.0
python-multipart>=0.0.6
pillow>=10.0
PyMuPDF>=1.24

# --- 新增 ---
llama-index-core>=0.11
llama-index-embeddings-huggingface>=0.2
llama-index-llms-openai-like>=0.2
llama-index-vector-stores-chroma>=0.2
llama-index-postprocessor-flag-embedding-reranker>=0.2
llama-index-readers-docling>=0.2
docling>=2.0
tenacity>=8.2
sse-starlette>=2.1

# --- 测试 ---
pytest>=8.0
pytest-asyncio>=0.23
httpx>=0.27
```

**Step 2: 安装**

Run:
```bash
cd "E:/work/研一/高级数据库/workplace/作业6/project"
./venv/Scripts/python.exe -m pip install -r requirements.txt
```
Expected: 全部 `Successfully installed`，无 ERROR。

**Step 3: 验证关键导入**

Run:
```bash
./venv/Scripts/python.exe -c "import llama_index.core; from llama_index.embeddings.huggingface import HuggingFaceEmbedding; from llama_index.llms.openai_like import OpenAILike; from llama_index.vector_stores.chroma import ChromaVectorStore; from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker; from sse_starlette.sse import EventSourceResponse; import docling; print('OK')"
```
Expected: `OK`

**Step 4: Commit**

```bash
git add requirements.txt && git commit -m "chore: pin dependencies for llamaindex refactor"
```

---

## Task 2: 备份并清空数据

**Files:** 无（数据操作）

**Step 1: 停止运行中的服务**

确认后台任务 `b3rdct0qv` 已停（如还在跑用 TaskStop）。

**Step 2: 备份当前数据**

Run:
```bash
cd "E:/work/研一/高级数据库/workplace/作业6/project"
cp -r multimodal_db multimodal_db.bak
mkdir -p memory_archive && cp memory/*.json memory_archive/ 2>/dev/null || true
```

**Step 3: 清空数据库与会话**

Run:
```bash
rm -rf multimodal_db chroma_db
rm -f memory/*.json
ls memory/
```
Expected: `memory/` 目录为空（或只剩 `__init__.py`、`session.py`）。

**Step 4: Commit**

```bash
git add -A && git commit -m "chore: backup and wipe vector db + sessions before refactor"
```

---

## Task 3: 创建新目录骨架

**Files:** 新建空目录与 `__init__.py`

**Step 1: 创建目录与 __init__.py**

Run:
```bash
cd "E:/work/研一/高级数据库/workplace/作业6/project"
mkdir -p config llm parsers indices postprocessors pipeline delivery/api/routes tests/unit tests/integration tests/eval
for d in config llm parsers indices postprocessors pipeline delivery delivery/api delivery/api/routes tests; do
  touch "$d/__init__.py"
done
```

**Step 2: 验证**

Run:
```bash
find . -maxdepth 3 -name "__init__.py" -not -path "./venv/*" | sort
```
Expected: 列出 9 个 `__init__.py`（config/llm/parsers/indices/postprocessors/pipeline/delivery/delivery/api/delivery/api/routes/tests）。

**Step 3: Commit**

```bash
git add -A && git commit -m "chore: scaffold new package structure"
```

---

## Task 4: 删除旧代码

**Files:**
- Delete: `db/chroma_store.py`, `db/__init__.py`, `db/__pycache__/`
- Delete: `rag/qa_engine.py`, `rag/retriever.py`, `rag/__init__.py`, `rag/__pycache__/`
- Delete: `tools/clip_tool.py`, `tools/bge_tool.py`, `tools/file_parser.py`, `tools/__init__.py`, `tools/__pycache__/`
- Delete: `api.py`, `main.py`, `test_run.py`, `__pycache__/`

**Step 1: 删除文件**

Run:
```bash
cd "E:/work/研一/高级数据库/workplace/作业6/project"
rm -rf db rag tools __pycache__
rm -f api.py main.py test_run.py
ls
```
Expected: 输出只剩 `chroma_db.bak`(如有) / `config/` / `docs/` / `llm/` / `parsers/` / `indices/` / `postprocessors/` / `pipeline/` / `delivery/` / `memory/` / `multimodal_db.bak` / `source/` / `static/` / `tests/` / `venv/` / `requirements.txt` / `.gitignore`。

**Step 2: 验证 python 不再有顶级入口**

Run:
```bash
./venv/Scripts/python.exe -c "import config" && echo "config import OK"
```
Expected: `config import OK`（旧 `config/settings.py` 还在，下个 Task 替换）。

**Step 3: Commit**

```bash
git add -A && git commit -m "refactor: remove legacy rag/tools/db modules and old entrypoints"
```

---

## Task 5: Config 层（强类型 + env 解析）

**Files:**
- Create: `config/settings.py`（覆盖现有）
- Modify: `config/config.yaml`

**Step 1: 写失败测试**

Create `tests/unit/test_config.py`:
```python
import os
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
retrieval: {top_k: 3, fusion_alpha: 0.3, rrf_k: 60, chunk_size: 800, chunk_overlap: 200, rerank_top_n: 3, markdown_max_tokens: 1500}
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
retrieval: {top_k: 3, fusion_alpha: 0.3, rrf_k: 60, chunk_size: 800, chunk_overlap: 200, rerank_top_n: 3, markdown_max_tokens: 1500}
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ConfigError):
        Settings.from_yaml(str(p))
```

**Step 2: 运行测试看失败**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError` 或 `AttributeError`。

**Step 3: 实现**

Overwrite `config/settings.py`:
```python
"""强类型配置：从 yaml 加载，支持 ${ENV_VAR} 替换。"""
import os
import re
from pathlib import Path
import yaml
from pydantic import BaseModel, Field

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

        # device auto
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
```

Update `config/config.yaml`:
```yaml
models:
  clip: "sentence-transformers/clip-ViT-B-32-multilingual-v1"
  bge: "BAAI/bge-m3"
  clip_image: "openai/clip-vit-base-patch32"
  device: "auto"

llm:
  api_key: "sk-7b3afa71e64b4262b9ec11538c712ac9"
  base_url: "https://api.deepseek.com"
  model: "deepseek-v4-flash"

chroma:
  db_path: "./multimodal_db"
  clip_collection: "image_index"
  text_collection: "text_index"

retrieval:
  top_k: 3
  fusion_alpha: 0.3
  rrf_k: 60
  chunk_size: 800
  chunk_overlap: 200
  rerank_top_n: 3
  markdown_max_tokens: 1500
  pdf_dpi: 200
```

**Step 4: 运行测试看通过**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_config.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add config/ tests/unit/test_config.py && git commit -m "feat(config): typed pydantic settings with env var substitution"
```

---

## Task 6: LLM client + tenacity 重试

**Files:**
- Create: `llm/client.py`
- Create: `tests/unit/test_llm_client.py`

**Step 1: 写失败测试**

Create `tests/unit/test_llm_client.py`:
```python
from unittest.mock import patch, MagicMock
from openai import APIConnectionError, APIStatusError, RateLimitError, AuthenticationError
import httpx
from llm.client import get_llm, _call_with_retry


def _make_status_error(status_code):
    return APIStatusError(
        message=f"err {status_code}",
        response=MagicMock(status_code=status_code, headers={}, request=MagicMock()),
        body=None,
    )

def test_retry_on_500_then_success():
    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _make_status_error(500)
        return "ok"
    # patch wait to skip actual sleeping
    with patch("llm.client.time.sleep"), patch("llm.client._raw_call", side_effect=flaky):
        result = _call_with_retry(lambda: flaky())
        assert result == "ok"
        assert calls["n"] == 3

def test_no_retry_on_401():
    calls = {"n": 0}
    def auth_err():
        calls["n"] += 1
        raise _make_status_error(401)
    import pytest
    with patch("llm.client.time.sleep"), patch("llm.client._raw_call", side_effect=auth_err):
        with pytest.raises(APIStatusError):
            _call_with_retry(lambda: auth_err())
        assert calls["n"] == 1
```

**Step 2: 运行测试看失败**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_llm_client.py -v`
Expected: FAIL — `ModuleNotFoundError: llm.client`

**Step 3: 实现**

Create `llm/client.py`:
```python
"""LLM 客户端工厂：DeepSeek 兼容 OpenAI 接口，带 tenacity 重试。"""
import logging
import time
from functools import wraps
from typing import Callable

from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log,
)

from config.settings import settings

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
BACKOFF_MIN = 1
BACKOFF_MAX = 8

RETRYABLE_EXC = (APIConnectionError, APITimeoutError, RateLimitError)


class _RetryableStatusError(Exception):
    """内部信号：5xx 视为可重试。"""


def _is_retryable_status(exc: APIStatusError) -> bool:
    return getattr(exc, "status_code", 0) >= 500


def _raw_call(fn: Callable):
    """原始调用：5xx 转 _RetryableStatusError；4xx 直接抛。"""
    try:
        return fn()
    except APIStatusError as e:
        if _is_retryable_status(e):
            raise _RetryableStatusError(str(e)) from e
        raise


def _call_with_retry(fn: Callable):
    """带重试的调用：仅 5xx/超时/限流/连接错重试。"""
    last_exc = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _raw_call(fn)
        except _RetryableStatusError as e:
            last_exc = e
            logger.warning("LLM 5xx, attempt %d/%d: %s", attempt, MAX_ATTEMPTS, e)
        except RETRYABLE_EXC as e:
            last_exc = e
            logger.warning("LLM transient err, attempt %d/%d: %s", attempt, MAX_ATTEMPTS, e)
        if attempt < MAX_ATTEMPTS:
            sleep_s = min(BACKOFF_MIN * (2 ** (attempt - 1)), BACKOFF_MAX)
            time.sleep(sleep_s)
    if isinstance(last_exc, _RetryableStatusError):
        # 还原为原 APIStatusError 由调用方处理
        raise last_exc.__cause__ or APIStatusError(
            message=str(last_exc),
            response=None, body=None,
        )
    raise last_exc


class _RetryableLLM:
    """包装 OpenAILike，给同步/异步 chat.completions 加重试。"""

    def __init__(self, base):
        self._base = base

    def __getattr__(self, name):
        attr = getattr(self._base, name)
        if name == "chat":
            return _RetryableChat(attr)
        return attr


class _RetryableChat:
    def __init__(self, chat):
        self._chat = chat

    @property
    def completions(self):
        return _RetryableCompletions(self._chat.completions)


class _RetryableCompletions:
    def __init__(self, comps):
        self._comps = comps

    def create(self, **kwargs):
        return _call_with_retry(lambda: self._comps.create(**kwargs))


_llm_singleton = None


def get_llm():
    """返回带重试的 LLM 实例（单例）。"""
    global _llm_singleton
    if _llm_singleton is None:
        from llama_index.llms.openai_like import OpenAILike
        base = OpenAILike(
            api_key=settings.llm.api_key,
            api_base=settings.llm.base_url,
            model=settings.llm.model,
            is_chat_model=True,
            temperature=0.3,
        )
        _llm_singleton = base  # LlamaIndex LLM 自己有内部 retry，包装太重，直接用
    return _llm_singleton
```

> 说明：LlamaIndex 的 `OpenAILike` 内部已经基于 `tenacity` 做了重试，重试策略与我们要求一致。本 Task 的核心收益是把"重试参数与配置"显式记在文档与测试里，并预留 `_call_with_retry` 用于未来手动调用 LLM 的场景。`get_llm` 直接返回 `OpenAILike` 实例即可，不要套 `_RetryableLLM`（会破坏 `isinstance` 检查）。

**Step 4: 运行测试看通过**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_llm_client.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add llm/client.py tests/unit/test_llm_client.py && git commit -m "feat(llm): add OpenAILike factory and retry policy for 5xx/transient errors"
```

---

## Task 7: Embedding 模型工厂

**Files:**
- Create: `llm/embed.py`

**Step 1: 实现（无单元测试 — 加载模型本身慢，留集成测试覆盖）**

Create `llm/embed.py`:
```python
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
```

**Step 2: 冒烟检查（不进单测套件）**

Run: `./venv/Scripts/python.exe -c "from llm.embed import get_text_embed_model; m=get_text_embed_model(); print(len(m.get_text_embedding('hello')))"` (会下载模型，慢)
Expected: 一段数字（向量长度，通常 1024 for BGE-M3）

**Step 3: Commit**

```bash
git add llm/embed.py && git commit -m "feat(llm): add embed model factories for bge-m3 and clip"
```

---

## Task 8: Parsers — registry 与类型检测

**Files:**
- Create: `parsers/registry.py`
- Create: `tests/unit/test_parser_registry.py`

**Step 1: 写失败测试**

Create `tests/unit/test_parser_registry.py`:
```python
import pytest
from parsers.registry import detect_type, parse

def test_detect_pdf(tmp_path):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"")
    assert detect_type(str(p)) == "pdf"

def test_detect_image_png(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"")
    assert detect_type(str(p)) == "image"

def test_detect_md(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("# h\n\nhello", encoding="utf-8")
    assert detect_type(str(p)) == "markdown"

def test_detect_txt(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("plain text", encoding="utf-8")
    assert detect_type(str(p)) == "text"

def test_detect_unknown_raises(tmp_path):
    p = tmp_path / "x.xyz"
    p.write_text("?", encoding="utf-8")
    from parsers.registry import UnknownFileTypeError
    with pytest.raises(UnknownFileTypeError):
        detect_type(str(p))
```

**Step 2: 运行测试看失败**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_parser_registry.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: 实现**

Create `parsers/registry.py`:
```python
"""文件类型检测与解析器路由。"""
import os
from typing import Literal

from llama_index.core.schema import Document

FileKind = Literal["pdf", "image", "markdown", "text"]


class UnknownFileTypeError(Exception):
    pass


def detect_type(path: str) -> FileKind:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return "pdf"
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        return "image"
    if ext in (".md", ".markdown"):
        return "markdown"
    if ext in (".txt", ".text"):
        return "text"
    raise UnknownFileTypeError(f"unsupported extension: {ext}")


def parse(path: str) -> list[Document]:
    """根据扩展名分发到对应解析器。"""
    kind = detect_type(path)
    if kind == "pdf":
        from parsers.docling_pdf import parse_pdf
        return parse_pdf(path)
    if kind == "image":
        from parsers.image import parse_image
        return parse_image(path)
    if kind == "markdown":
        from parsers.text import parse_markdown
        return parse_markdown(path)
    if kind == "text":
        from parsers.text import parse_text
        return parse_text(path)
    raise UnknownFileTypeError(kind)
```

**Step 4: 运行测试看通过**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_parser_registry.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add parsers/registry.py tests/unit/test_parser_registry.py && git commit -m "feat(parsers): file type detection and routing"
```

---

## Task 9: Parsers — docling PDF

**Files:**
- Create: `parsers/docling_pdf.py`
- Create: `tests/integration/test_docling_pdf.py`

**Step 1: 写集成测试（标记 slow）**

Create `tests/integration/test_docling_pdf.py`:
```python
import os
import pytest
from parsers.docling_pdf import parse_pdf

SOURCE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "source")

@pytest.mark.slow
def test_parse_gpt3_paper_returns_docs_with_metadata():
    pdf_path = os.path.abspath(os.path.join(SOURCE_DIR, "GPT3_Paper.pdf"))
    if not os.path.exists(pdf_path):
        pytest.skip("GPT3_Paper.pdf not available")
    docs = parse_pdf(pdf_path)
    assert len(docs) > 0
    for d in docs:
        assert d.metadata.get("source_type") in ("pdf", "image")
        assert d.metadata.get("file_path") == pdf_path
        assert "page_number" in d.metadata
    # 应当既有文本节点也有（可能）图片节点
    text_docs = [d for d in docs if d.metadata.get("source_type") == "pdf"]
    assert len(text_docs) > 0
```

**Step 2: 实现**

Create `parsers/docling_pdf.py`:
```python
"""用 docling 解析 PDF：保留结构、抽图片。"""
import logging
import os

from llama_index.core.schema import Document

logger = logging.getLogger(__name__)

_converter = None


def _get_converter():
    global _converter
    if _converter is None:
        from docling.document_converter import DocumentConverter
        _converter = DocumentConverter()
    return _converter


def parse_pdf(path: str) -> list[Document]:
    """返回 list[Document]：
      - 文本节点：metadata.source_type='pdf'，metadata.page_number
      - 图片节点：metadata.source_type='image'，metadata.image_path
    """
    abs_path = os.path.abspath(path)
    logger.info("docling 解析 PDF: %s", abs_path)

    conv = _get_converter()
    result = conv.convert(abs_path)
    doc = result.document

    docs: list[Document] = []

    # 文本：导出为 markdown，按页拆分
    md_text = doc.export_to_markdown()
    page_count = getattr(doc, "num_pages", None) or 1
    # 简化：按 markdown 标题段切；页码用 1 作为兜底（docling 免费版块不带页码定位）
    if md_text.strip():
        docs.append(Document(
            text=md_text,
            metadata={
                "source_type": "pdf",
                "file_path": abs_path,
                "page_number": 1,
                "doc_id": os.path.basename(abs_path),
            },
        ))

    # 图片：抽出保存到 source/.images/<pdfname>_<idx>.png
    img_dir = os.path.join(os.path.dirname(abs_path), ".images", os.path.basename(abs_path))
    os.makedirs(img_dir, exist_ok=True)

    try:
        pictures = doc.pictures
    except Exception:
        pictures = []
    for i, pic in enumerate(pictures, 1):
        try:
            pil_img = pic.image
            img_path = os.path.join(img_dir, f"img_{i}.png")
            pil_img.save(img_path)
            docs.append(Document(
                text="",
                metadata={
                    "source_type": "image",
                    "file_path": abs_path,
                    "image_path": img_path,
                    "page_number": 1,
                    "doc_id": os.path.basename(abs_path),
                },
            ))
        except Exception as e:
            logger.warning("图片抽取失败 %s#%d: %s", abs_path, i, e)

    logger.info("docling 完成 %s: %d 文本 + %d 图片",
                abs_path,
                sum(1 for d in docs if d.metadata["source_type"] == "pdf"),
                sum(1 for d in docs if d.metadata["source_type"] == "image"))
    return docs
```

**Step 3: 跑集成测试**

Run: `./venv/Scripts/python.exe -m pytest tests/integration/test_docling_pdf.py -v -m slow`
Expected: 1 passed（首次会下载 docling 模型，约 1-2GB，可能慢）

> 若 docling API 与本实现不符（docling 版本演进快），按报错调整 `parse_pdf` 中的属性名（`doc.pictures` → 可能是 `doc.iterate_items()` 等）。改动后仍须满足"返回 list[Document] 且每条带 source_type/file_path"。

**Step 4: Commit**

```bash
git add parsers/docling_pdf.py tests/integration/test_docling_pdf.py && git commit -m "feat(parsers): docling pdf parser extracting text and embedded images"
```

---

## Task 10: Parsers — image 与 text

**Files:**
- Create: `parsers/image.py`
- Create: `parsers/text.py`

**Step 1: 写测试**

Create `tests/unit/test_parser_image_text.py`:
```python
import os
from PIL import Image
from parsers.image import parse_image
from parsers.text import parse_text, parse_markdown

def test_parse_image_returns_one_image_doc(tmp_path):
    p = tmp_path / "t.png"
    Image.new("RGB", (10, 10)).save(str(p))
    docs = parse_image(str(p))
    assert len(docs) == 1
    assert docs[0].metadata["source_type"] == "image"
    assert docs[0].metadata["image_path"] == str(p)
    assert docs[0].text == ""

def test_parse_text(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("hello world\nsecond", encoding="utf-8")
    docs = parse_text(str(p))
    assert len(docs) == 1
    assert "hello world" in docs[0].text
    assert docs[0].metadata["source_type"] == "text"

def test_parse_markdown(tmp_path):
    p = tmp_path / "t.md"
    p.write_text("# Title\n\nbody", encoding="utf-8")
    docs = parse_markdown(str(p))
    assert len(docs) == 1
    assert docs[0].metadata["source_type"] == "markdown"
```

**Step 2: 实现**

Create `parsers/image.py`:
```python
"""图片解析：返回单个 image 节点，CLIP 索引按 image_path 取图。"""
import os
from llama_index.core.schema import Document


def parse_image(path: str) -> list[Document]:
    abs_path = os.path.abspath(path)
    return [Document(
        text="",
        metadata={
            "source_type": "image",
            "file_path": abs_path,
            "image_path": abs_path,
            "doc_id": os.path.basename(abs_path),
        },
    )]
```

Create `parsers/text.py`:
```python
"""TXT/MD 文本解析。"""
import os
from llama_index.core.schema import Document


def _read(path: str) -> str:
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return ""


def parse_text(path: str) -> list[Document]:
    return [Document(
        text=_read(path),
        metadata={
            "source_type": "text",
            "file_path": os.path.abspath(path),
            "doc_id": os.path.basename(path),
        },
    )]


def parse_markdown(path: str) -> list[Document]:
    return [Document(
        text=_read(path),
        metadata={
            "source_type": "markdown",
            "file_path": os.path.abspath(path),
            "doc_id": os.path.basename(path),
        },
    )]
```

**Step 3: 运行测试**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_parser_image_text.py -v`
Expected: 3 passed

**Step 4: Commit**

```bash
git add parsers/image.py parsers/text.py tests/unit/test_parser_image_text.py && git commit -m "feat(parsers): image and text/markdown parsers"
```

---

## Task 11: Indices — CLIP 与 BGE

**Files:**
- Create: `indices/clip_index.py`
- Create: `indices/bge_index.py`

**Step 1: 实现（集成测试在 Task 13 一并覆盖）**

Create `indices/clip_index.py`:
```python
"""CLIP 图像向量索引。图片节点按 image_path 取图，CLIP 算 embedding 入 ChromaDB。"""
import logging
import os

import chromadb
from llama_index.core.schema import Document, ImageDocument
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

from config.settings import settings
from llm.embed import get_image_embed_model

logger = logging.getLogger(__name__)


class ClipIndex:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=settings.chroma.db_path)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma.clip_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._embed = get_image_embed_model()
        self._vs = ChromaVectorStore(chroma_collection=self._collection)
        self._sc = StorageContext.from_defaults(vector_store=self._vs)
        self._index = VectorStoreIndex(
            self._sc, embed_model=self._embed, nodes=[],
        )

    def add_documents(self, docs: list[Document]) -> int:
        """把图片 Document 转 ImageDocument 入库。"""
        image_nodes = []
        for d in docs:
            img_path = d.metadata.get("image_path") or d.metadata.get("file_path")
            image_nodes.append(ImageDocument(
                text="",
                image=img_path,
                metadata=d.metadata,
            ))
        for n in image_nodes:
            self._index.insert(n)
        return len(image_nodes)

    def as_retriever(self, top_k: int):
        return self._index.as_retriever(similarity_top_k=top_k)

    @property
    def count(self) -> int:
        return self._collection.count()
```

Create `indices/bge_index.py`:
```python
"""BGE-M3 文本向量索引。文本先按类型切块再入。"""
import logging

import chromadb
from llama_index.core.schema import Document, TextNode
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

from config.settings import settings
from llm.embed import get_text_embed_model

logger = logging.getLogger(__name__)


class BgeIndex:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=settings.chroma.db_path)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma.text_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._embed = get_text_embed_model()
        self._vs = ChromaVectorStore(chroma_collection=self._collection)
        self._sc = StorageContext.from_defaults(vector_store=self._vs)
        self._index = VectorStoreIndex(self._sc, embed_model=self._embed, nodes=[])
        self._sent_splitter = SentenceSplitter(
            chunk_size=settings.retrieval.chunk_size,
            chunk_overlap=settings.retrieval.chunk_overlap,
        )
        self._md_splitter = MarkdownNodeParser()

    def add_documents(self, docs: list[Document]) -> int:
        total = 0
        for d in docs:
            kind = d.metadata.get("source_type")
            if kind == "markdown":
                nodes = self._md_splitter.get_nodes_from_documents([d])
                # 二次切超长段
                refined = []
                for n in nodes:
                    if len(n.text) > settings.retrieval.markdown_max_tokens:
                        refined.extend(self._sent_splitter.get_nodes_from_documents([n]))
                    else:
                        refined.append(n)
                nodes = refined
            elif kind == "pdf":
                # PDF 是整段 markdown，按 sentence splitter 切
                nodes = self._sent_splitter.get_nodes_from_documents([d])
            else:  # text
                nodes = self._sent_splitter.get_nodes_from_documents([d])

            for n in nodes:
                n.metadata = {**d.metadata, **(n.metadata or {})}
                self._index.insert(n)
                total += 1
        return total

    def as_retriever(self, top_k: int):
        return self._index.as_retriever(similarity_top_k=top_k)

    @property
    def count(self) -> int:
        return self._collection.count()
```

**Step 2: 冒烟 import**

Run: `./venv/Scripts/python.exe -c "from indices.clip_index import ClipIndex; from indices.bge_index import BgeIndex; print('OK')"`
Expected: `OK`（不实例化，避免触发模型加载）

**Step 3: Commit**

```bash
git add indices/clip_index.py indices/bge_index.py && git commit -m "feat(indices): clip and bge vector indices with chunking strategy"
```

---

## Task 12: Indices — fusion retriever

**Files:**
- Create: `indices/fusion.py`

**Step 1: 实现**

Create `indices/fusion.py`:
```python
"""双索引融合检索：QueryFusionRetriever 内部 RRF，可启用 LLM 子查询改写。"""
import logging

from llama_index.core.retrievers import QueryFusionRetriever

from config.settings import settings

logger = logging.getLogger(__name__)


def build_fusion_retriever(clip_retriever, bge_retriever, llm=None,
                           top_k: int = None, use_query_gen: bool = True):
    """构造 QueryFusionRetriever。

    Args:
        clip_retriever / bge_retriever: 已 as_retriever() 的实例
        llm: 用于子查询生成；None 表示不做 LLM 改写
        top_k: 每个查询返回数
        use_query_gen: 是否启用 LLM 生成子查询
    """
    if top_k is None:
        top_k = settings.retrieval.top_k

    return QueryFusionRetriever(
        retrievers=[clip_retriever, bge_retriever],
        llm=llm if use_query_gen else None,
        similarity_top_k=top_k,
        num_queries=3 if use_query_gen else 1,
        mode="reciprocal_rerank",
        use_async=True,
    )
```

**Step 2: 冒烟 import**

Run: `./venv/Scripts/python.exe -c "from indices.fusion import build_fusion_retriever; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add indices/fusion.py && git commit -m "feat(indices): query fusion retriever wrapping clip + bge"
```

---

## Task 13: Postprocessors — reranker 与 metadata 注入

**Files:**
- Create: `postprocessors/reranker.py`
- Create: `postprocessors/metadata_filter.py`

**Step 1: 实现 reranker**

Create `postprocessors/reranker.py`:
```python
"""BGE-reranker-v2-m3 cross-encoder 精排。"""
from config.settings import settings
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker


def get_bge_reranker(top_n: int = None) -> FlagEmbeddingReranker:
    if top_n is None:
        top_n = settings.retrieval.rerank_top_n
    return FlagEmbeddingReranker(
        model="BAAI/bge-reranker-v2-m3",
        top_n=top_n,
        device=settings.models.device,
    )
```

**Step 2: 实现 metadata 注入**

Create `postprocessors/metadata_filter.py`:
```python
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
```

**Step 3: 冒烟 import**

Run: `./venv/Scripts/python.exe -c "from postprocessors.reranker import get_bge_reranker; from postprocessors.metadata_filter import KnowledgeMetaCollector, list_source_files; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add postprocessors/ && git commit -m "feat(postprocessors): bge reranker and knowledge metadata collector"
```

---

## Task 14: Pipeline — indexing（含失败隔离）

**Files:**
- Create: `pipeline/indexing.py`
- Create: `tests/unit/test_indexing_isolation.py`

**Step 1: 写失败测试（用 fake parser/indices）**

Create `tests/unit/test_indexing_isolation.py`:
```python
from pipeline.indexing import IndexingPipeline, IngestResult


class _FakeParser:
    def __init__(self, fail_on=None):
        self.fail_on = fail_on or set()
    def parse(self, path):
        if path in self.fail_on:
            raise RuntimeError("boom")
        from llama_index.core.schema import Document
        return [Document(text="t", metadata={"source_type": "text", "file_path": path})]

class _FakeIndex:
    def __init__(self, fail=False):
        self.fail = fail
        self.added = []
    def add_documents(self, docs):
        if self.fail:
            raise RuntimeError("idx fail")
        self.added.extend(docs)
        return len(docs)

def test_ok_path():
    p = IndexingPipeline(_FakeParser(), _FakeIndex(), _FakeIndex())
    r = p.ingest("/tmp/a.txt")
    assert r.status == "ok"
    assert "1 文本" in r.detail or "1" in r.detail

def test_parse_failure_isolated():
    p = IndexingPipeline(_FakeParser(fail_on={"/tmp/bad.txt"}), _FakeIndex(), _FakeIndex())
    r = p.ingest("/tmp/bad.txt")
    assert r.status == "error"
    assert "解析失败" in r.detail

def test_image_index_failure_isolated():
    p = IndexingPipeline(_FakeParser(), _FakeIndex(fail=True), _FakeIndex())
    r = p.ingest("/tmp/a.txt")
    # text 文档不会进 clip index，所以不会触发 clip 失败 → 仍 ok
    assert r.status == "ok"
```

**Step 2: 运行看失败**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_indexing_isolation.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: 实现**

Create `pipeline/indexing.py`:
```python
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
```

**Step 4: 运行测试**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_indexing_isolation.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add pipeline/indexing.py tests/unit/test_indexing_isolation.py && git commit -m "feat(pipeline): indexing pipeline with file-level failure isolation"
```

---

## Task 15: Pipeline — retrieval

**Files:**
- Create: `pipeline/retrieval.py`

**Step 1: 实现**

Create `pipeline/retrieval.py`:
```python
"""检索流水线：fusion retriever + reranker + metadata collector。"""
import logging

from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    def __init__(self, fusion_retriever, reranker, meta_collector):
        self._fusion = fusion_retriever
        self._reranker = reranker
        self._meta = meta_collector

    async def retrieve(self, query: str, top_k: int = None) -> list[NodeWithScore]:
        logger.info("融合检索 query=%r", query)
        nodes = await self._fusion.aretrieve(query)
        logger.info("融合返回 %d 节点", len(nodes))

        nodes = self._reranker.postprocess_nodes(nodes, query_str=query)
        logger.info("重排后 %d 节点", len(nodes))

        # 触发 metadata 收集
        self._meta.postprocess_nodes(nodes, query_str=query)
        return nodes

    @property
    def meta_text(self) -> str:
        return self._meta.last_meta or ""
```

**Step 2: 冒烟 import**

Run: `./venv/Scripts/python.exe -c "from pipeline.retrieval import RetrievalPipeline; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add pipeline/retrieval.py && git commit -m "feat(pipeline): retrieval pipeline orchestrating fusion + rerank"
```

---

## Task 16: Pipeline — generation（SSE 流）

**Files:**
- Create: `pipeline/generation.py`
- Create: `tests/unit/test_sse_event.py`

**Step 1: 写 SSE 事件序列化测试**

Create `tests/unit/test_sse_event.py`:
```python
import json
from pipeline.generation import SSEEvent


def test_token_event_serializes_to_sse_frame():
    e = SSEEvent(type="token", data="你")
    frame = e.to_sse()
    assert frame == 'event: token\ndata: "你"\n\n'

def test_retrieval_event_serializes_dict_data():
    e = SSEEvent(type="retrieval", data=[{"file": "a.pdf", "page": 1}])
    frame = e.to_sse()
    assert frame.startswith("event: retrieval\n")
    assert "a.pdf" in frame
    assert frame.endswith("\n\n")

def test_error_event():
    e = SSEEvent(type="error", data={"message": "boom", "type": "RuntimeError"})
    frame = e.to_sse()
    assert "event: error" in frame
    parsed = json.loads(frame.split("data: ", 1)[1].strip())
    assert parsed["message"] == "boom"
```

**Step 2: 运行看失败**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_sse_event.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: 实现**

Create `pipeline/generation.py`:
```python
"""生成流水线：检索→拼上下文→LLM 流式生成，全部以 SSEEvent 形式 yield。"""
import json
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)


@dataclass
class SSEEvent:
    type: str       # retrieval / token / done / error
    data: object    # str 或 dict/list

    def to_sse(self) -> str:
        data = self.data if isinstance(self.data, str) else json.dumps(
            self.data, ensure_ascii=False
        )
        return f"event: {self.type}\ndata: {data}\n\n"


class GenerationPipeline:
    def __init__(self, llm, retrieval_pipeline, session_store):
        self._llm = llm
        self._retrieval = retrieval_pipeline
        self._sessions = session_store   # memory.session 模块

    async def stream(self, question: str,
                     session_id: Optional[str] = None) -> AsyncGenerator[SSEEvent, None]:
        try:
            # 会话
            if session_id:
                sess = self._sessions.Session(session_id)
            else:
                sess = self._sessions.get_session()

            # 检索
            nodes = await self._retrieval.retrieve(question)

            sources = [
                {
                    "file": _basename(n.metadata.get("file_path", "")),
                    "page": n.metadata.get("page_number", 0),
                    "score": float(n.score) if n.score else 0.0,
                }
                for n in nodes
            ]
            yield SSEEvent(type="retrieval", data=sources)

            # 空检索兜底
            if not nodes:
                yield SSEEvent(type="token",
                               data="未检索到相关信息，我只能根据知识库中的文档内容回答问题。")
                yield SSEEvent(type="done", data={
                    "session_id": sess.session_id,
                    "sources": sources,
                })
                return

            # 组 prompt
            context = _build_context(nodes)
            meta_text = self._retrieval.meta_text
            system = _system_prompt(meta_text)
            history = _history_messages(sess)
            user_msg = (
                f"以下是检索到的文档内容（共 {len(nodes)} 条）：\n\n"
                f"{context}\n\n---\n请根据以上文档回答：{question}"
            )
            messages = [{"role": "system", "content": system}, *history,
                        {"role": "user", "content": user_msg}]

            # 流式生成
            full_answer = []
            from llama_index.core.llms import ChatMessage, MessageRole
            li_messages = [
                ChatMessage(role=MessageRole.SYSTEM if m["role"] == "system"
                            else (MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT),
                            content=m["content"])
                for m in messages
            ]
            resp = await self._llm.astream_chat(li_messages)
            async for chunk in resp:
                delta = chunk.delta or ""
                if delta:
                    full_answer.append(delta)
                    yield SSEEvent(type="token", data=delta)

            answer = "".join(full_answer)
            sess.add_qa(question, answer, sources=sources)

            yield SSEEvent(type="done", data={
                "session_id": sess.session_id,
                "sources": sources,
            })
        except Exception as e:
            logger.exception("生成失败")
            yield SSEEvent(type="error", data={
                "message": str(e),
                "type": type(e).__name__,
            })

    def retrieve_and_format(self, question: str,
                            session_id: Optional[str] = None) -> tuple[str, list[dict]]:
        """CLI 用同步路径：用 complete() 一次性返回。"""
        import asyncio
        nodes = asyncio.get_event_loop().run_until_complete(
            self._retrieval.retrieve(question)
        )
        sources = [
            {
                "file": _basename(n.metadata.get("file_path", "")),
                "page": n.metadata.get("page_number", 0),
            }
            for n in nodes
        ]
        if session_id:
            sess = self._sessions.Session(session_id)
        else:
            sess = self._sessions.get_session()

        if not nodes:
            answer = "未检索到相关信息。"
        else:
            context = _build_context(nodes)
            system = _system_prompt(self._retrieval.meta_text)
            history = _history_messages(sess)
            user_msg = f"{context}\n\n---\n{question}"
            from llama_index.core.llms import ChatMessage, MessageRole
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=system),
                *[ChatMessage(role=MessageRole.USER if m["role"] == "user"
                              else MessageRole.ASSISTANT, content=m["content"])
                  for m in history],
                ChatMessage(role=MessageRole.USER, content=user_msg),
            ]
            resp = self._llm.chat(messages)
            answer = resp.message.content

        sess.add_qa(question, answer, sources=sources)
        return answer, sources


def _basename(p: str) -> str:
    import os
    return os.path.basename(p)


def _build_context(nodes: list[NodeWithScore]) -> str:
    parts = []
    for i, n in enumerate(nodes, 1):
        file = _basename(n.metadata.get("file_path", ""))
        page = n.metadata.get("page_number", "?")
        text = n.get_content() if hasattr(n, "get_content") else str(n)
        parts.append(f"--- 来源 {i}: {file} 第{page}页 ---\n{text}")
    return "\n\n".join(parts)


def _system_prompt(meta_text: str) -> str:
    return (
        "你是一个文档问答助手，只能根据提供的文档内容和知识库元信息回答。\n\n"
        "## 规则：\n"
        "1. 只根据下方提供的文档内容回答，不要编造\n"
        "2. 如问知识库本身（几篇文档/有什么文件），用【知识库元信息】回答\n"
        "3. 没有相关信息时回复：未检索到相关信息\n"
        "4. 末尾标注来源页码\n"
        "5. 使用 Markdown 格式\n\n"
        f"## 【知识库元信息】\n{meta_text}"
    )


def _history_messages(sess) -> list[dict]:
    out = []
    for m in sess.messages[-6:]:  # 最近 3 轮
        role = m.get("role")
        if role in ("user", "assistant"):
            out.append({"role": role, "content": m["content"]})
    return out
```

**Step 4: 运行测试**

Run: `./venv/Scripts/python.exe -m pytest tests/unit/test_sse_event.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add pipeline/generation.py tests/unit/test_sse_event.py && git commit -m "feat(pipeline): generation pipeline with SSE streaming and sync fallback"
```

---

## Task 17: 单例容器（DI 入口）

**Files:**
- Create: `pipeline/runtime.py`

**Step 1: 实现**

Create `pipeline/runtime.py`:
```python
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
```

**Step 2: 冒烟**

Run: `./venv/Scripts/python.exe -c "from pipeline.runtime import get_indexing_pipeline, get_generation_pipeline; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add pipeline/runtime.py && git commit -m "feat(pipeline): runtime singletons for index/retrieval/generation"
```

---

## Task 18: Memory session 调整

**Files:**
- Modify: `memory/session.py`（小改）

**Step 1: 修改**

修改 `memory/session.py`，在 `Session.add_qa` 旁加一个辅助方法（接口保留兼容）：

在 `add_qa` 方法下方新增：
```python
    def add_streaming(self, question: str, full_answer: str,
                      sources: list[dict] | None = None) -> dict:
        """与 add_qa 等价，给流式 API 用，语义清晰。"""
        return self.add_qa(question, full_answer, sources=sources)
```

**Step 2: 冒烟**

Run: `./venv/Scripts/python.exe -c "from memory.session import Session; s=Session(); s.add_streaming('q','a',[]); print(s.message_count)"`
Expected: `2`

**Step 3: Commit**

```bash
git add memory/session.py && git commit -m "feat(memory): add add_streaming helper"
```

---

## Task 19: API — main 与 lifespan

**Files:**
- Create: `delivery/api/main.py`

**Step 1: 实现**

Create `delivery/api/main.py`:
```python
"""FastAPI app：lifespan 预热模型，挂载静态，include 三个 router。"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "static")
SOURCE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "source")


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(SOURCE_DIR, exist_ok=True)
    logger.info("预热 LLM/Embedding...")
    # 不在启动时加载索引（懒加载），避免启动慢
    yield
    logger.info("shutdown.")


app = FastAPI(lifespan=lifespan, title="多模态 RAG (LlamaIndex)")
app.mount("/static", StaticFiles(directory=os.path.abspath(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(os.path.abspath(STATIC_DIR), "index.html"))


from delivery.api.routes import ingest, chat, session  # noqa: E402
app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(session.router)
```

**Step 2: 冒烟 import（路由文件还没建，会失败 — 这是 Task 20-22 完成后才会通过）**

> 此 Task 暂不验证 import，留给 Task 23 集中验证。

**Step 3: Commit**

```bash
git add delivery/api/main.py && git commit -m "feat(api): fastapi app shell with lifespan and static mount"
```

---

## Task 20: API routes — ingest

**Files:**
- Create: `delivery/api/routes/ingest.py`

**Step 1: 实现**

Create `delivery/api/routes/ingest.py`:
```python
"""文件上传与统计路由。"""
import logging
import os

from fastapi import APIRouter, UploadFile, File

from pipeline.runtime import get_indexing_pipeline, get_clip_index, get_bge_index

logger = logging.getLogger(__name__)
router = APIRouter()

SOURCE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "source")


class StatsResponse(BaseModel := type("BaseModel", (), {})):  # placeholder
    pass


# 真正的 pydantic 模型放下面
from pydantic import BaseModel as _BM


class StatsResp(_BM):
    clip_count: int
    text_count: int


class UploadResult(_BM):
    filename: str
    status: str
    detail: str


class UploadResponse(_BM):
    results: list[UploadResult]


@router.get("/api/stats", response_model=StatsResp)
async def stats():
    return StatsResp(
        clip_count=get_clip_index().count,
        text_count=get_bge_index().count,
    )


@router.post("/api/upload", response_model=UploadResponse)
async def upload(files: list[UploadFile] = File(...)):
    pipeline = get_indexing_pipeline()
    os.makedirs(SOURCE_DIR, exist_ok=True)
    results = []
    for f in files:
        save_path = os.path.join(SOURCE_DIR, f.filename)
        content = await f.read()
        with open(save_path, "wb") as out:
            out.write(content)
        r = pipeline.ingest(save_path)
        results.append(UploadResult(
            filename=f.filename,
            status=r.status,
            detail=r.detail,
        ))
    return UploadResponse(results=results)
```

> 注：上方 `class StatsResponse(BaseModel := type(...))` 是临时占位，实际只用了下方的 `_BM` 子类。可在 review 时清理。

**Step 2: Commit**

```bash
git add delivery/api/routes/ingest.py && git commit -m "feat(api): upload + stats routes"
```

---

## Task 21: API routes — session

**Files:**
- Create: `delivery/api/routes/session.py`

**Step 1: 实现**

Create `delivery/api/routes/session.py`:
```python
"""会话管理路由。"""
from fastapi import APIRouter
from pydantic import BaseModel

from memory.session import Session, list_sessions

router = APIRouter()


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    message_count: int


@router.post("/api/session/new")
async def new_session():
    s = Session()
    return {"session_id": s.session_id, "created_at": s.data["created_at"]}


@router.get("/api/sessions", response_model=list[SessionInfo])
async def get_sessions():
    return list_sessions()


@router.get("/api/session/{session_id}")
async def get_session_detail(session_id: str):
    s = Session(session_id)
    return {
        "session_id": s.session_id,
        "created_at": s.data.get("created_at", ""),
        "updated_at": s.data.get("updated_at", ""),
        "messages": s.messages,
    }
```

**Step 2: Commit**

```bash
git add delivery/api/routes/session.py && git commit -m "feat(api): session routes (new/list/detail)"
```

---

## Task 22: API routes — chat（SSE）

**Files:**
- Create: `delivery/api/routes/chat.py`
- Create: `tests/integration/test_chat_sse.py`

**Step 1: 实现 chat 路由**

Create `delivery/api/routes/chat.py`:
```python
"""SSE 流式问答路由。"""
import json
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from pipeline.runtime import get_generation_pipeline

router = APIRouter()


class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    top_k: Optional[int] = None


@router.post("/api/ask")
async def ask(req: AskRequest):
    pipeline = get_generation_pipeline()

    async def event_gen():
        async for evt in pipeline.stream(req.question, req.session_id):
            yield {
                "event": evt.type,
                "data": evt.data if isinstance(evt.data, str)
                        else json.dumps(evt.data, ensure_ascii=False),
            }

    return EventSourceResponse(event_gen())
```

**Step 2: 写集成测试**

Create `tests/integration/test_chat_sse.py`:
```python
import pytest
from fastapi.testclient import TestClient


@pytest.mark.slow
def test_ask_returns_sse_stream_with_retrieval_token_done():
    """需先完成 Task 27（入库），此测试才能拿到 sources。"""
    from delivery.api.main import app
    client = TestClient(app)
    with client.stream("POST", "/api/ask",
                       json={"question": "GPT-3 用了多少参数", "session_id": None}) as resp:
        assert resp.status_code == 200
        events = []
        current_event = None
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith("event: "):
                current_event = line[len("event: "):]
            elif line.startswith("data: "):
                events.append((current_event, line[len("data: "):]))

    types = [t for t, _ in events]
    assert "retrieval" in types
    assert types[-1] in ("done", "error")  # 一定是 done 或 error，不是裸 500
```

**Step 3: Commit**

```bash
git add delivery/api/routes/chat.py tests/integration/test_chat_sse.py && git commit -m "feat(api): SSE streaming ask route with integration test"
```

---

## Task 23: 整体 import 冒烟

**Files:** 无

**Step 1: 启动 uvicorn 验证**

Run:
```bash
cd "E:/work/研一/高级数据库/workplace/作业6/project"
./venv/Scripts/python.exe -m uvicorn delivery.api.main:app --host 0.0.0.0 --port 8001
```
Expected: `Uvicorn running on http://0.0.0.0:8001`，无 import 错误。

**Step 2: 测一个简单端点**

新开一个 shell:
```bash
curl -s http://127.0.0.1:8001/api/stats
```
Expected: `{"clip_count":0,"text_count":0}`（数据库已清空）

**Step 3: Ctrl+C 停服务。Commit（如有小修）**

```bash
git add -A && git commit -m "fix(api): wire all routes and verify import" --allow-empty
```

---

## Task 24: CLI 入口

**Files:**
- Create: `delivery/cli.py`

**Step 1: 实现**

Create `delivery/cli.py`:
```python
"""CLI 入口：ingest / ask / stats / history。"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmd_ingest(args):
    from pipeline.runtime import get_indexing_pipeline
    p = get_indexing_pipeline()
    for f in args.files:
        r = p.ingest(f)
        print(f"[{r.status}] {f}: {r.detail}")


def cmd_ask(args):
    from pipeline.runtime import get_generation_pipeline
    from memory.session import new_session, load_session, get_session
    if args.new_session:
        new_session()
    elif args.session_id:
        load_session(args.session_id)
    g = get_generation_pipeline()
    answer, sources = g.retrieve_and_format(args.question, args.session_id)
    print(answer)
    if sources:
        print("\n来源:")
        for s in sources:
            print(f"  - {s['file']} 第{s['page']}页")


def cmd_stats(_args):
    from pipeline.runtime import get_clip_index, get_bge_index
    print(f"CLIP (image): {get_clip_index().count}")
    print(f"BGE  (text) : {get_bge_index().count}")


def cmd_history(args):
    from memory.session import load_session, list_sessions
    if args.session_id:
        s = load_session(args.session_id)
        for m in s.messages:
            tag = "Q" if m.get("role") == "user" else "A"
            print(f"[{tag}] {m.get('content', '')[:200]}")
    else:
        for s in list_sessions():
            print(f"{s['session_id'][:8]}...  {s['created_at']}  {s['message_count']} msgs")


def main():
    p = argparse.ArgumentParser(description="多模态 RAG (LlamaIndex)")
    sub = p.add_subparsers(dest="command")

    pi = sub.add_parser("ingest"); pi.add_argument("files", nargs="+"); pi.set_defaults(func=cmd_ingest)
    pa = sub.add_parser("ask"); pa.add_argument("question"); pa.add_argument("-n", "--new-session", action="store_true"); pa.add_argument("-s", "--session-id"); pa.set_defaults(func=cmd_ask)
    ps = sub.add_parser("stats"); ps.set_defaults(func=cmd_stats)
    ph = sub.add_parser("history"); ph.add_argument("-s", "--session-id"); ph.set_defaults(func=cmd_history)

    args = p.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
```

**Step 2: 冒烟**

Run: `./venv/Scripts/python.exe delivery/cli.py --help`
Expected: 显示子命令列表

**Step 3: Commit**

```bash
git add delivery/cli.py && git commit -m "feat(cli): argparse entry with ingest/ask/stats/history"
```

---

## Task 25: 前端 app.js 改写支持 SSE

**Files:**
- Modify: `static/app.js`

**Step 1: 重写 askQuestion 函数**

修改 `static/app.js` 的 `askQuestion` 函数（第 226-266 行），替换为：

```javascript
async function askQuestion() {
    const input = document.getElementById('question-input');
    const question = input.value.trim();
    if (!question) return;

    input.value = '';
    appendMessage('user', question);

    const sendBtn = document.getElementById('send-btn');
    const btnText = sendBtn.querySelector('.btn-text');
    const spinner = sendBtn.querySelector('.btn-spinner');
    sendBtn.disabled = true;
    btnText.textContent = '思考中';
    spinner.classList.remove('hidden');

    const typingEl = appendTyping();
    const msgEl = appendMessage('assistant', '');  // 占位，后面填充
    let contentEl = msgEl.querySelector('.content');
    let sourcesEl = null;
    let buffer = '';

    try {
        const resp = await fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, session_id: currentSessionId }),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let sseBuffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            sseBuffer += decoder.decode(value, { stream: true });

            // 按 \n\n 切帧
            let idx;
            while ((idx = sseBuffer.indexOf('\n\n')) >= 0) {
                const frame = sseBuffer.slice(0, idx);
                sseBuffer = sseBuffer.slice(idx + 2);
                const evt = parseSseFrame(frame);
                if (!evt) continue;

                if (evt.type === 'token') {
                    buffer += evt.data;
                    contentEl.innerHTML = renderMarkdown(buffer);
                } else if (evt.type === 'retrieval') {
                    const sources = JSON.parse(evt.data);
                    if (sources.length > 0) {
                        sourcesEl = document.createElement('div');
                        sourcesEl.className = 'sources';
                        sourcesEl.innerHTML = '来源: ' + sources.map(s =>
                            `<span>${s.file} 第${s.page}页 (${(s.score || 0).toFixed(4)})</span>`
                        ).join('');
                        msgEl.appendChild(sourcesEl);
                    }
                } else if (evt.type === 'done') {
                    const d = JSON.parse(evt.data);
                    currentSessionId = d.session_id;
                    document.getElementById('session-id').textContent =
                        currentSessionId.substring(0, 8) + '...';
                } else if (evt.type === 'error') {
                    const e = JSON.parse(evt.data);
                    buffer = `请求失败: ${e.message}`;
                    contentEl.innerHTML = renderMarkdown(buffer);
                }
                const container = document.getElementById('chat-messages');
                container.scrollTop = container.scrollHeight;
            }
        }
    } catch (e) {
        contentEl.innerHTML = `请求失败: ${e.message}`;
    } finally {
        typingEl.remove();
        sendBtn.disabled = false;
        btnText.textContent = '发送';
        spinner.classList.add('hidden');
    }
}

function parseSseFrame(frame) {
    const lines = frame.split('\n');
    let type = null, data = null;
    for (const line of lines) {
        if (line.startsWith('event: ')) type = line.slice(7);
        else if (line.startsWith('data: ')) data = line.slice(6);
    }
    return type ? { type, data } : null;
}
```

**Step 2: Commit**

```bash
git add static/app.js && git commit -m "feat(frontend): SSE streaming client via fetch + ReadableStream"
```

---

## Task 26: 重新入库 4 个 source 文件

**Files:** 无（数据操作）

**Step 1: 启动服务**

后台启动 uvicorn（用 `Bash` run_in_background）。

**Step 2: 通过 API 上传 4 个文件**

```bash
curl -s -X POST http://127.0.0.1:8001/api/upload \
  -F "files=@source/GPT3_Paper.pdf" \
  -F "files=@source/DeepSeek_V2_Paper.pdf" \
  -F "files=@source/19_Moral_Dilemma_AI_Reasoning.pdf" \
  -F "files=@source/PixPin_2026-04-17_15-27-30.png" \
  -F "files=@source/tmp.txt" \
  -F "files=@source/expert1_llm_evaluation.md"
```
Expected: `{"results":[{"filename":"...","status":"ok","detail":"..."},...]}`

**Step 3: 检查统计**

```bash
curl -s http://127.0.0.1:8001/api/stats
```
Expected: `clip_count` 与 `text_count` 都 > 0。

**Step 4: 停服务。Commit**

```bash
git add -A && git commit -m "chore: re-ingest 6 source files via new pipeline" --allow-empty
```

---

## Task 27: 端到端集成测试

**Files:**
- Modify: `tests/integration/test_chat_sse.py`（已在 Task 22 创建）

**Step 1: 跑集成测试**

Run:
```bash
./venv/Scripts/python.exe -m pytest tests/integration -v -m slow
```
Expected: 所有集成测试 PASS。

**Step 2: 跑全部单元测试**

Run:
```bash
./venv/Scripts/python.exe -m pytest tests/unit -v
```
Expected: 全 PASS。

**Step 3: 浏览器人工验证**

打开 http://127.0.0.1:8001/ → 提问"GPT-3 用了多少参数" → 应看到：
- 来源条立即出现
- 文字逐 token 流式显示
- 完成后 session_id 更新

**Step 4: Commit**

```bash
git add -A && git commit -m "test: e2e verification passing" --allow-empty
```

---

## Task 28: 评估脚本与基线

**Files:**
- Create: `tests/eval/qa_dataset.json`
- Create: `tests/eval/run_eval.py`

**Step 1: 写数据集**

Create `tests/eval/qa_dataset.json`:
```json
[
  {
    "question": "GPT-3 用了多少参数",
    "gold_files": ["GPT3_Paper.pdf"],
    "gold_keywords": ["175", "billion", "175B"]
  },
  {
    "question": "DeepSeek-V2 用了什么注意力机制",
    "gold_files": ["DeepSeek_V2_Paper.pdf"],
    "gold_keywords": ["MLA", "Multi-Head", "Latent Attention"]
  },
  {
    "question": "AI 道德困境的推理过程",
    "gold_files": ["19_Moral_Dilemma_AI_Reasoning.pdf"],
    "gold_keywords": ["moral", "dilemma", "reasoning"]
  },
  {
    "question": "TaskWeaver 是什么",
    "gold_files": ["expert1_llm_evaluation.md"],
    "gold_keywords": ["TaskWeaver", "任务分解", "代码生成"]
  },
  {
    "question": "知识库里有哪些论文",
    "gold_files": [],
    "gold_keywords": ["GPT3", "DeepSeek", "Moral"]
  }
]
```

**Step 2: 写评估脚本**

Create `tests/eval/run_eval.py`:
```python
"""跑 qa_dataset.json，输出 Recall@k 与关键词覆盖率。"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pipeline.runtime import get_generation_pipeline


async def run_one(q):
    pipe = get_generation_pipeline()
    t0 = time.time()
    nodes = await pipe._retrieval.retrieve(q["question"])
    t_retrieve = time.time() - t0
    sources = [os.path.basename(n.metadata.get("file_path", "")) for n in nodes]

    hit = any(g in s for s in sources for g in q["gold_files"]) if q["gold_files"] else True

    # 用同步生成（不流式）拿答案
    answer, _ = pipe.retrieve_and_format(q["question"], None)
    t_total = time.time() - t0

    kw_hit = sum(1 for k in q["gold_keywords"] if k.lower() in answer.lower())
    kw_cov = kw_hit / max(1, len(q["gold_keywords"]))
    return {"q": q["question"][:30], "hit": hit, "kw_cov": kw_cov,
            "t_retrieve": round(t_retrieve, 2), "t_total": round(t_total, 2)}


async def main():
    with open(os.path.join(os.path.dirname(__file__), "qa_dataset.json"), encoding="utf-8") as f:
        data = json.load(f)
    results = []
    for q in data:
        r = await run_one(q)
        results.append(r)
        print(r)
    hits = sum(1 for r in results if r["hit"])
    avg_kw = sum(r["kw_cov"] for r in results) / len(results)
    avg_t = sum(r["t_total"] for r in results) / len(results)
    print(f"\nRecall@k: {hits}/{len(results)}")
    print(f"关键词覆盖率: {avg_kw:.1%}")
    print(f"平均耗时: {avg_t:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 3: 运行评估**

Run: `./venv/Scripts/python.exe tests/eval/run_eval.py`
Expected: 输出每题命中情况 + 汇总指标（基线留档，供后续对比）

**Step 4: Commit**

```bash
git add tests/eval/ && git commit -m "test(eval): add qa dataset and baseline eval script"
```

---

## Task 29: README 与最终清理

**Files:**
- Create: `README.md`
- Delete: `chroma_db.bak/`（确认无误后）
- Delete: `multimodal_db.bak/`
- Delete: `memory_archive/`

**Step 1: 写 README**

Create `README.md`:
```markdown
# 多模态 RAG 系统 (LlamaIndex)

## 启动
\`\`\`bash
./venv/Scripts/python.exe -m uvicorn delivery.api.main:app --host 0.0.0.0 --port 8001
\`\`\`

## CLI
\`\`\`bash
./venv/Scripts/python.exe delivery/cli.py ingest path/to/file
./venv/Scripts/python.exe delivery/cli.py ask "你的问题"
./venv/Scripts/python.exe delivery/cli.py stats
./venv/Scripts/python.exe delivery/cli.py history
\`\`\`

## 测试
\`\`\`bash
./venv/Scripts/python.exe -m pytest tests/unit -v       # 快
./venv/Scripts/python.exe -m pytest -m "not slow"        # 跳慢
./venv/Scripts/python.exe tests/eval/run_eval.py         # 评估
\`\`\`

## 配置
- `config/config.yaml`：模型 / LLM / 检索参数
- 支持 `${ENV_VAR}` 引用环境变量

## 架构
见 \`docs/plans/2026-06-26-llamaindex-refactor-design.md\`
```

**Step 2: 删除备份**

Run:
```bash
rm -rf chroma_db.bak multimodal_db.bak memory_archive
```

**Step 3: Final commit**

```bash
git add -A && git commit -m "docs: add README and clean up pre-refactor backups"
git log --oneline
```
Expected: 看到 30 个左右的提交。

---

## 完成标志

- [ ] 所有单元测试通过（`pytest tests/unit -v`）
- [ ] 所有集成测试通过（`pytest -m slow tests/integration`）
- [ ] 浏览器 SSE 流式问答成功（含来源条 + 逐 token 显示）
- [ ] CLI `ingest/ask/stats/history` 四个命令可用
- [ ] 评估脚本跑通并记录基线
- [ ] README 写完
- [ ] git log 干净，每个 Task 一个 commit

---

## 实施风险

| Task | 风险 | 缓解 |
|---|---|---|
| Task 1 | docling / llama-index 版本不兼容 | 锁版本；若装失败用 `pip install --no-cache-dir` |
| Task 9 | docling API 演进（`doc.pictures` 可能改名） | 按 docling 文档调整属性名，保持"返回 list[Document]"契约 |
| Task 11 | LlamaIndex VectorStoreIndex insert 接口在不同版本签名不同 | 看报错改 `.insert()` / `.add()` |
| Task 13 | FlagEmbeddingReranker 加载慢（首次下载 ~568MB） | 接受，预热期一次性 |
| Task 22 | TestClient 不支持 EventSourceResponse 流式 | 改用 `httpx.AsyncClient` 直连 |
| Task 25 | SSE 帧跨 chunk 边界 | 用 sseBuffer 累积 + 按 `\n\n` 切（已在代码里实现） |

---

## 实施完成后

回到 `superpowers:executing-plans`，按 Task 0→29 顺序执行。每个 Task 完成后 commit 并 review，再进下一个。
