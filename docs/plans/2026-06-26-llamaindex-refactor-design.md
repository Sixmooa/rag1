# LlamaIndex 全面重构设计稿

- 日期：2026-06-26
- 范围：`db/`、`rag/`、`tools/` 全部替换；`memory/` 接口保留；`api.py`、`main.py` 拆分重组；`static/` 前端改写
- 目标：用 LlamaIndex 替换所有手写 RAG 原语；模块化低耦合；不使用任何付费方法

---

## 1. 目标与非目标

### 目标
- 用 LlamaIndex 的 `VectorStoreIndex` / `QueryFusionRetriever` / `NodePostprocessor` / `BaseLLM.astream()` 替换现有手写的检索、融合、重排、生成逻辑
- 按"流水线"切分模块：`pipeline/` 编排，`indices/` 实现，`postprocessors/` 后处理，`llm/`/`parsers/` 适配
- 启用 SSE 流式响应，改善对话体验
- 引入 BGE-reranker-v2-m3 精排，提升 Top-K 召回质量
- 引入 docling 解析 PDF，保留表格/图片结构
- 失败可观测：所有错误以结构化 SSE 事件返回，前端不再崩在 JSON.parse

### 非目标（YAGNI）
- 不做 Ragas/DeepEval/TruLens 等重型评估框架
- 不做跨平台 CI / 容器化 / k8s
- 不做并发负载测试
- 不重写 `static/style.css`（视觉风格保留）
- 不引入 LangChain / Haystack 等其他框架

---

## 2. 决策汇总

| 维度 | 选择 |
|---|---|
| 迁移策略 | 原地大重写，旧代码删除 |
| 现有向量数据 | 清空重入库（`multimodal_db/` 清空） |
| 现有会话历史 | 全部删除（`memory/*.json` 清空） |
| QA 架构 | 纯 LlamaIndex（QueryFusionRetriever + 重排 + BaseLLM.astream） |
| PDF 解析 | docling（图片节点单独抽出，文本节点送 BGE） |
| 文本分块 | `MarkdownNodeParser`（按标题切，超过 max_tokens 时用 SentenceSplitter 二次切） |
| 多模态 | 双索引融合（中文 CLIP 库 + BGE-M3 库），QueryFusionRetriever 做 RRF |
| 重排序 | BGE-reranker-v2-m3（FlagEmbeddingReranker） |
| 响应模式 | SSE 流式（事件类型: retrieval / token / done / error） |
| CLI 输出 | 同步（用 `llm.complete()`，不走流式） |
| 错误重试 | tenacity 3 次，指数退避 1–8s，仅 5xx/超时/限流/连接错 |
| 入库失败隔离 | 文件级 |
| 前端 SSE 客户端 | 原生 `fetch + ReadableStream`，零依赖 |

---

## 3. 架构

### 3.1 模块依赖图

```
                       ┌──────────────────┐
                       │   delivery/      │
                       │   api/  cli.py   │
                       └────────┬─────────┘
                                │
                       ┌────────▼─────────┐
                       │   pipeline/      │  编排层，无业务状态
                       │   indexing.py     │
                       │   retrieval.py    │
                       │   generation.py   │
                       └────────┬─────────┘
                                │
       ┌────────────────────────┼──────────────────────────┐
       │                        │                          │
┌──────▼──────┐          ┌──────▼──────┐          ┌────────▼────────┐
│  indices/   │          │postprocessors│         │     llm/         │
│  clip_index │          │  reranker    │         │  client.py       │
│  bge_index  │          │  meta_filter │         │  embed.py        │
│  fusion.py  │          └─────────────┘          └──────────────────┘
└──────┬──────┘
       │
       │                ┌──────────────┐
       └───────────────►│  parsers/    │
                        │  registry    │
                        │  docling_pdf │
                        │  image       │
                        │  text        │
                        └──────────────┘

横向贯穿：config/settings.py、memory/session.py
```

### 3.2 依赖原则
- **单向依赖**：上层 import 下层，下层绝不反向 import 上层
- **`pipeline/` 不存状态**：只组合 indices + postprocessors + llm
- **`indices/` 暴露统一接口**：`add_documents(docs)` / `as_retriever(top_k)`
- **`llm/` 是适配器**：把 DeepSeek、BGE-M3、CLIP 三个模型藏到工厂函数后
- **`config/` 强类型**：从 `_Settings` 字典式 → `pydantic-settings` 属性式

### 3.3 删除的旧文件
- `db/chroma_store.py`、`db/__init__.py`
- `rag/qa_engine.py`、`rag/retriever.py`、`rag/__init__.py`
- `tools/clip_tool.py`、`tools/bge_tool.py`、`tools/file_parser.py`、`tools/__init__.py`
- `api.py`（拆到 `delivery/api/main.py` + `routes/`）
- `main.py`（重命名为 `delivery/cli.py`）
- `test_run.py`（功能并入 `tests/`）
- `__pycache__/`（重新生成）

### 3.4 新增依赖
```
llama-index-core
llama-index-embeddings-huggingface
llama-index-llms-openai-like
llama-index-vector-stores-chroma
llama-index-postprocessor-flag-embedding-reranker
llama-index-readers-docling
docling
tenacity
sse-starlette
pytest
pytest-asyncio
httpx
```

保留：`chromadb`、`fastapi`、`uvicorn`、`pydantic`、`PyYAML`、`python-multipart`、`pillow`、`PyMuPDF`（docling 间接需要）

---

## 4. 组件规格

### 4.1 `config/settings.py`
```python
class ModelsConfig(BaseSettings):
    clip: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    bge: str = "BAAI/bge-m3"
    clip_image: str = "openai/clip-vit-base-patch32"
    device: str = "auto"

class LLMConfig(BaseSettings):
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"

class ChromaConfig(BaseSettings):
    db_path: str = "./multimodal_db"
    clip_collection: str = "image_index"
    text_collection: str = "text_index"

class RetrievalConfig(BaseSettings):
    top_k: int = 3
    fusion_alpha: float = 0.3
    rrf_k: int = 60
    chunk_size: int = 800
    chunk_overlap: int = 200
    rerank_top_n: int = 3
    markdown_max_tokens: int = 1500

class Settings(BaseSettings):
    models: ModelsConfig
    llm: LLMConfig
    chroma: ChromaConfig
    retrieval: RetrievalConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Settings": ...

settings = Settings.from_yaml("config/config.yaml")
```

### 4.2 `llm/`
```python
# llm/client.py
def get_llm() -> OpenAILike: ...   # 单例，带 tenacity 重试装饰

# llm/embed.py
def get_text_embed_model() -> HuggingFaceEmbedding: ...   # BGE-M3
def get_image_embed_model() -> ClipEmbedding: ...         # 多语言 CLIP
```

### 4.3 `parsers/`
```python
# parsers/registry.py
def parse(file_path: str) -> list[Document]: ...   # 按扩展名分发

# parsers/docling_pdf.py
def parse_pdf(path: str) -> list[Document]: ...
    # 返回文本节点 + 图片节点
    # 文本: source_type='pdf', page_number, doc_id
    # 图片: source_type='image', image_path, page_number, doc_id

# parsers/image.py
def parse_image(path: str) -> list[Document]: ...   # 单个 image 节点

# parsers/text.py
def parse_text(path: str) -> list[Document]: ...   # TXT/MD 内容
```

### 4.4 `indices/`
```python
# indices/clip_index.py
class ClipIndex:
    def __init__(self, chroma_path: str, collection: str, embed_model): ...
    def add_documents(self, docs: list[Document]) -> None: ...
    def as_retriever(self, top_k: int) -> VectorIndexRetriever: ...
    @property
    def count(self) -> int: ...

# indices/bge_index.py
class BgeIndex: ...   # 接口同上，BGE-M3 + SentenceSplitter/MarkdownNodeParser

# indices/fusion.py
def build_fusion_retriever(
    clip_retriever, bge_retriever, llm,
    top_k: int, alpha: float,
) -> QueryFusionRetriever: ...
```

### 4.5 `postprocessors/`
```python
# postprocessors/reranker.py
def get_bge_reranker(top_n: int) -> FlagEmbeddingReranker: ...

# postprocessors/metadata_filter.py
class KnowledgeMetadataPostprocessor(BaseNodePostprocessor):
    """把知识库元信息（文件清单/计数）注入 system prompt。"""
    def _postprocess_nodes(self, nodes, query_bundle): ...
```

### 4.6 `pipeline/`
```python
# pipeline/indexing.py
@dataclass
class IngestResult:
    file: str
    status: str          # ok / error
    detail: str

class IndexingPipeline:
    def __init__(self, parser, clip_idx, bge_idx): ...
    def ingest(self, file_path: str) -> IngestResult: ...
        # parse → 按 source_type 路由到 ClipIndex/BgeIndex
        # 文件级失败隔离

# pipeline/retrieval.py
class RetrievalPipeline:
    def __init__(self, fusion_retriever, reranker, meta_postprocessor): ...
    async def retrieve(self, query: str, top_k: int = None) -> list[NodeWithScore]: ...

# pipeline/generation.py
@dataclass
class SSEEvent:
    type: str            # retrieval / token / done / error
    data: dict | str

class GenerationPipeline:
    def __init__(self, llm, retrieval_pipeline, session_store): ...
    async def stream(self, query: str, session_id: str | None) -> AsyncGenerator[SSEEvent, None]: ...
    def retrieve_and_format(self, query: str, session_id: str | None) -> tuple[str, list[dict]]: ...
        # 给 CLI 用，同步路径
```

### 4.7 `delivery/api/`
```python
# delivery/api/main.py
app = FastAPI(lifespan=lifespan)
app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(session.router)
app.mount("/static", StaticFiles(directory=static_dir))

# delivery/api/routes/ingest.py
@router.post("/api/upload")
@router.get("/api/stats")

# delivery/api/routes/chat.py
@router.post("/api/ask")
async def ask(req: AskRequest) -> EventSourceResponse:
    """EventSourceResponse 包 async generator。
    内部捕获所有异常 → 转 SSE error 事件。"""

# delivery/api/routes/session.py
@router.post("/api/session/new")
@router.get("/api/sessions")
@router.get("/api/session/{session_id}")
```

### 4.8 `delivery/cli.py`
```python
# argparse: ingest / ask / stats / history
# ask 子命令调用 GenerationPipeline.retrieve_and_format()，同步输出
```

### 4.9 `memory/session.py`
接口不变，内部按需补充 `add_streaming(question, full_answer, sources)` 辅助方法。

---

## 5. 数据流

### 5.1 入库（PDF，多文件批量）

```
POST /api/upload
  → routes/ingest.py 保存到 source/
  → IndexingPipeline.ingest(path)
      → parsers/registry.parse(path)
          → docling_pdf.parse_pdf(path) 返回 [Document(text), Document(image), ...]
      → 按 metadata.source_type 路由：
          image 节点  → ClipIndex.add_documents([img_docs])
          pdf/text/md → BgeIndex.add_documents([text_docs])
              (BGE 内部先用 MarkdownNodeParser / SentenceSplitter 切块)
      → return IngestResult{file, status, detail}
  → UploadResponse{results:[IngestResult, ...]}
```

**关键约定**：
- PDF 走 docling 时只抽出内嵌图片节点（不再生成整页位图）入 CLIP
- MD 入库先 `MarkdownNodeParser` 切标题，若单段超 `markdown_max_tokens` 再用 `SentenceSplitter` 二次切
- TXT 走 `SentenceSplitter(chunk_size=800, chunk_overlap=200)`
- 文件级失败隔离：一个文件挂掉不影响同批次其他文件

### 5.2 问答（SSE 流式）

```
POST /api/ask {question, session_id}
  → routes/chat.py 包 EventSourceResponse
  → GenerationPipeline.stream(question, session_id)
      1. 获取/加载 session
      2. RetrievalPipeline.retrieve(question)
           → QueryFusionRetriever.aretrieve(query)
               内部：LLM 生成子查询、CLIP/BGE 双索引并发检索、RRF 融合
           → FlagEmbeddingReranker.postprocess_nodes() 精排
           → KnowledgeMetadataPostprocessor 注入元信息
           → return top_n NodeWithScore
      3. yield SSEEvent(type='retrieval', data=sources)
      4. 空 nodes 时：yield token='未检索到...' → done → return
      5. 组装 messages = [system + 知识库元信息 + session 历史 + user_context]
      6. async for token in llm.astream(messages):
             yield SSEEvent(type='token', data=token)
      7. session.add_qa(question, full_answer, sources)
      8. yield SSEEvent(type='done', data={session_id, message_count})
  → 任一步抛错：内部 try/except → yield SSEEvent(type='error', data={message})
```

### 5.3 CLI 问答（同步）
```
cli.py ask "问题"
  → GenerationPipeline.retrieve_and_format(question, session_id)
      检索同上 → llm.complete(messages) 一次性 → 返回 (answer, sources)
  → print(answer + sources + session_id)
```

---

## 6. 错误处理与重试

### 6.1 错误分级

| 等级 | 例 | 策略 |
|---|---|---|
| 瞬态可重试 | LLM 5xx / 超时 / 429 / Cloudflare 1xxx / 连接错 | tenacity 重试 |
| 客户端错误 | 401 / 403 / 配置缺 key / 不支持的文件 | 不重试，直接抛 + 友好提示 |
| 数据级失败 | 单文件解析失败 / 单 chunk embedding 失败 | 文件级隔离，其他继续 |
| 基础设施失败 | ChromaDB 损坏 / 磁盘满 | SSE error 事件冒泡 |

### 6.2 LLM 重试（`llm/client.py`）
```python
@retry(
    retry=retry_if_exception_type((APIConnectionError, APITimeoutError, RateLimitError)),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_llm(...): ...

# APIStatusError 额外按 status_code >= 500 判断重试
```

### 6.3 入库隔离（`pipeline/indexing.py`）
- `parse_pdf` 异常 → 返回 `IngestResult(status='error', detail='解析失败: ...')`
- `clip_idx.add_documents` 异常 → 同上，detail='图像入库失败'
- `bge_idx.add_documents` 异常 → 同上，detail='文本入库失败'
- 单个文件失败不影响同批次

### 6.4 SSE 流式错误（`delivery/api/routes/chat.py`）
- 外层 `EventSourceResponse` 包裹的 async generator 全部 try/except
- 任何异常 → `yield SSEEvent(type='error', data={message, type})` 后终止流
- 前端永远收到终止事件（done 或 error），不会收到非 JSON 文本

### 6.5 前端降级（`static/app.js`）
```javascript
const resp = await fetch('/api/ask', { method:'POST', body:... });
const reader = resp.body.getReader();
// 手动解析 SSE 帧 (event: xxx\ndata: yyy\n\n)
// 按 event 分发：retrieval → 渲染来源；token → 追加；done → 关流；error → 报错
```
不再有 `Unexpected token 'I'` 这种崩在 JSON.parse 的症状。

### 6.6 启动时检查（`config/settings.py`）
- LLM api_key 为空 → 抛 ConfigError
- chroma.db_path 不可写 → 警告
- device='auto' 且 CUDA 不可用 → 警告并退到 CPU

### 6.7 日志
- 用标准库 `logging`，格式 `%(asctime)s [%(levelname)s] %(name)s: %(message)s`
- 替换所有 `print()`（约 30 处）
- INFO：每跳查询/子查询/命中数；WARNING：降级；ERROR：traceback

---

## 7. 测试方案

### 7.1 测试目录
```
tests/
  conftest.py
  unit/
    test_config.py
    test_parser_registry.py
    test_chunking.py
    test_sse_event.py
    test_retrieval_filter.py
  integration/
    test_docling_pdf.py
    test_indexing.py
    test_retrieval.py
    test_chat_sse.py
  eval/
    qa_dataset.json
    run_eval.py
```

### 7.2 关键 fixture（`conftest.py`）
- `mock_llm`：MockLLM，按关键词返回预设响应
- `mock_embed_model`：确定性 mock embedding（按文本 hash）
- `ephemeral_chroma`：EphemeralClient 内存 ChromaDB
- `real_embed_model`（session 级，仅集成测试用）
- `sample_pdf_path`：source/GPT3_Paper.pdf 的前 5 页
- `fastapi_client`：TestClient，依赖注入覆盖 LLM/embed/chroma

### 7.3 覆盖要点
| 模块 | 测试 | 断言 |
|---|---|---|
| `config/settings.py` | `test_config.py` | `${ENV}` 替换、device auto、缺 key 抛错 |
| `parsers/registry.py` | `test_parser_registry.py` | 扩展名路由正确，未知扩展抛 |
| `pipeline/indexing.py` | `test_chunking.py` | MD 大段被切到 ≤ max_tokens；纯文本走 SentenceSplitter |
| SSE 事件 | `test_sse_event.py` | 帧格式严格匹配 `event: ...\ndata: ...\n\n` |
| 空检索兜底 | `test_retrieval_filter.py` | 空 nodes 时仍发 retrieval/token/done 完整序列 |

### 7.4 集成测试样例
```python
@pytest.mark.slow
def test_ask_returns_sse_stream_with_sources(fastapi_client_real):
    resp = fastapi_client_real.post("/api/ask",
        json={"question":"GPT-3 用了多少参数","session_id":None}, stream=True)
    events = parse_sse_stream(resp)
    types = [e.type for e in events]
    assert "retrieval" in types
    assert types.count("token") > 5
    assert types[-1] == "done"
    assert any("GPT3_Paper" in s["file"] for s in events[0].data)
```

### 7.5 评估脚本（`eval/run_eval.py`）
- 数据集：基于 `source/` 4 个文件编 5-10 条 Q&A
- 指标：Recall@3 / 答案关键词覆盖率 / 首 token 延迟 / 总耗时
- 不引入 Ragas/DeepEval/TruLens

### 7.6 运行命令
```bash
./venv/Scripts/python.exe -m pytest                          # 全部
./venv/Scripts/python.exe -m pytest tests/unit -v            # 仅单元
./venv/Scripts/python.exe -m pytest -m "not slow"            # 跳慢
./venv/Scripts/python.exe tests/eval/run_eval.py             # 评估
```

---

## 8. 迁移操作

执行顺序（写实施计划时细化）：
1. 备份当前 `multimodal_db/`、`memory/*.json` 到 `.bak`
2. 清空 `multimodal_db/`、删除 `memory/*.json`
3. 删除旧代码文件（见 3.3）
4. 装新依赖
5. 创建新目录骨架
6. 按 `config → llm → parsers → indices → postprocessors → pipeline → delivery` 顺序实现
7. 改写前端 `static/app.js` 支持 SSE
8. 跑单元测试 → 集成测试 → 手动验证
9. 重新入库 4 个 source 文件
10. 跑 eval 脚本对比基线

---

## 9. 风险与缓解

| 风险 | 缓解 |
|---|---|
| docling 首次启动下载模型大 | 在 README 标注，提供预热脚本 |
| LlamaIndex 版本碎片化（核心+多个集成包） | `requirements.txt` 锁定版本 |
| 双索引 + 重排内存占用大 | device 默认 CPU 跑得通；提供 device=cuda 配置 |
| SSE 在某些代理后行为异常 | 同时保留 `/api/ask` 的非流式版本作降级（后续可加） |
| 中文 markdown 切块不准 | `MarkdownNodeParser` 后接 `SentenceSplitter` 二次切兜底 |
| 前端 EventSource 不支持 POST | 用 fetch + ReadableStream 手动解析 SSE 帧 |

---

## 10. 后续可选演进（不在本次范围）

- HyDE（hypothetical document embeddings）
- Small-to-big / parent-child retriever
- 多 LLM 路由（cheap vs pro）
- 向量库从 ChromaDB 切到 Qdrant/Milvus
- 多用户隔离与会话持久化到 SQLite
