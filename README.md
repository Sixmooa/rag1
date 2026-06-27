# 多模态 RAG 系统 (LlamaIndex)

基于 LlamaIndex 的模块化多模态 RAG：双索引（中文 CLIP + BGE-M3）经 QueryFusionRetriever 融合，由 BGE-reranker-v2-m3 精排，DeepSeek 提供 LLM，前端通过 SSE 流式接收。

## 启动

```bash
"C:/Users/admin/venvs/rag-project/Scripts/python.exe" -m uvicorn delivery.api.main:app --host 0.0.0.0 --port 8001
```

打开 http://127.0.0.1:8001/

## CLI

```bash
"C:/Users/admin/venvs/rag-project/Scripts/python.exe" delivery/cli.py ingest path/to/file
"C:/Users/admin/venvs/rag-project/Scripts/python.exe" delivery/cli.py ask "你的问题"
"C:/Users/admin/venvs/rag-project/Scripts/python.exe" delivery/cli.py stats
"C:/Users/admin/venvs/rag-project/Scripts/python.exe" delivery/cli.py history
```

## 测试

```bash
# 单元测试（快）
"C:/Users/admin/venvs/rag-project/Scripts/python.exe" -m pytest tests/unit -v

# 集成测试（含 SSE 端到端，慢）
"C:/Users/admin/venvs/rag-project/Scripts/python.exe" -m pytest -m slow tests/integration -v

# 跳过 slow
"C:/Users/admin/venvs/rag-project/Scripts/python.exe" -m pytest -m "not slow"

# 评估脚本（命中 DeepSeek API，慢）
"C:/Users/admin/venvs/rag-project/Scripts/python.exe" tests/eval/run_eval.py
```

## 配置

- `config/config.yaml`：模型 / LLM / 检索参数
- ```${ENV_VAR}``` 引用环境变量（如 `api_key: ${DEEPSEEK_API_KEY}`）

## 架构

```
delivery (FastAPI/CLI/SSE)
    ↓
pipeline (indexing / retrieval / generation)
    ↓
indices (CLIP / BGE-M3 / QueryFusion) · postprocessors (reranker / metadata)
    ↓
llm (OpenAILike + tenacity 重试) · parsers (docling / image / text)
    ↓
config (pydantic-settings + YAML)
```

详细设计文档见 `docs/plans/2026-06-26-llamaindex-refactor-design.md`。

## 依赖

- LlamaIndex (core / huggingface / openai-like / chroma / flag-reranker / docling)
- ChromaDB 持久化（双 collection：image_index、text_index）
- BGE-M3 文本 embedding、多语言 CLIP 图像 embedding、BGE-reranker-v2-m3 精排
- DeepSeek (deepseek-v4-flash, OpenAI 兼容)
- sse-starlette 流式响应、fetch + ReadableStream 前端解析

详见 `requirements.txt`。安装：

```bash
"C:/Users/admin/venvs/rag-project/Scripts/python.exe" -m pip install -r requirements.txt
```
