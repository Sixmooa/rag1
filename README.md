# 多模态 RAG 系统

这是一个基于 LlamaIndex 的多模态检索增强生成（RAG）项目。系统支持上传 PDF、图片、Markdown 和纯文本文件，分别构建图像索引与文本索引；提问时通过融合检索、重排和 DeepSeek 生成答案，并用 SSE 将上传进度、检索来源和回答内容实时推送到前端。

项目同时提供 FastAPI Web 服务和 CLI 两种使用方式，适合作为课程作业、RAG 原型或多模态知识库实验项目。

## 核心逻辑

整体链路可以概括为：

```text
文件上传
  -> parsers 按文件类型解析
  -> pipeline.indexing 分流
  -> CLIP 图像索引 / BGE-M3 文本索引写入 ChromaDB

用户提问
  -> QueryFusionRetriever 融合 CLIP 与 BGE 检索结果
  -> BGE reranker 精排
  -> 拼接知识库元信息、会话历史和检索上下文
  -> DeepSeek 流式生成答案
  -> FastAPI SSE 推送到前端
```

主要模块职责：

| 模块 | 说明 |
|---|---|
| `delivery/` | 对外入口，包含 FastAPI 路由、SSE 接口和 CLI |
| `pipeline/` | 业务编排层，负责入库、检索、生成和运行时单例管理 |
| `parsers/` | 文件解析层，按扩展名路由到 PDF、图片、Markdown、文本解析器 |
| `indices/` | 向量索引层，使用 ChromaDB 持久化 CLIP 图像索引和 BGE 文本索引 |
| `postprocessors/` | 检索后处理，包括 BGE reranker 和知识库元信息收集 |
| `llm/` | Embedding 与 DeepSeek/OpenAI-compatible LLM 客户端封装 |
| `memory/` | JSON 文件形式的会话记忆 |
| `static/` | 原生 HTML/CSS/JS 前端，无需构建步骤 |
| `tests/` | 单元测试、集成测试和简单评估脚本 |

## 功能特点

- 支持 PDF、图片、Markdown、TXT 多种输入。
- 使用 CLIP 处理图片语义，使用 BGE-M3 处理文本语义。
- 使用 LlamaIndex `QueryFusionRetriever` 做双路检索融合。
- 使用 `BAAI/bge-reranker-v2-m3` 对融合结果进行精排。
- 使用 ChromaDB 持久化向量数据。
- Web 端上传和问答均支持 SSE 流式事件。
- 支持多轮会话，历史记录保存为本地 JSON 文件。
- 通过 `config/config.yaml` 集中管理模型、LLM、检索和 ChromaDB 参数。

## 项目结构

```text
project/
├── config/                 # YAML 配置与 pydantic 校验
├── delivery/
│   ├── api/                # FastAPI app 与 HTTP 路由
│   └── cli.py              # 命令行入口
├── indices/                # CLIP / BGE / 融合检索
├── llm/                    # LLM 和 embedding 工厂
├── memory/                 # 会话 JSON 存储
├── parsers/                # PDF、图片、文本解析
├── pipeline/               # indexing / retrieval / generation 编排
├── postprocessors/         # reranker 与元信息收集
├── static/                 # 前端页面
├── tests/                  # 测试与评估
├── source/                 # 上传文件保存目录
├── requirements.txt
└── README.md
```

## 环境准备

建议使用 Python 3.10+。

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

首次运行会下载 BGE-M3、CLIP、reranker 等模型，体积较大。中国大陆网络环境可以先设置 HuggingFace 镜像：

```bash
# Windows CMD
set HF_ENDPOINT=https://hf-mirror.com

# PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

## 配置

主要配置位于 `config/config.yaml`：

```yaml
models:
  clip: "sentence-transformers/clip-ViT-B-32-multilingual-v1"
  bge: "BAAI/bge-m3"
  device: "auto"

llm:
  api_key: "${DEEPSEEK_API_KEY}"
  base_url: "https://api.deepseek.com"
  model: "deepseek-v4-flash"

chroma:
  db_path: "./multimodal_db"
  clip_collection: "image_index"
  text_collection: "text_index"
```

推荐把 API Key 放在环境变量中：

```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY = "sk-xxxxxxxx"

# Linux / macOS
export DEEPSEEK_API_KEY=sk-xxxxxxxx
```

`config/settings.py` 会读取 YAML，并将 `${ENV_VAR}` 形式的值替换为对应环境变量。

## 启动 Web 服务

```bash
python -m uvicorn delivery.api.main:app --host 0.0.0.0 --port 8001
```

打开浏览器访问：

```text
http://127.0.0.1:8001/
```

常用接口：

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/` | 前端页面 |
| `POST` | `/api/upload` | 上传文件并通过 SSE 返回入库进度 |
| `GET` | `/api/stats` | 查看图像索引和文本索引数量 |
| `POST` | `/api/ask` | 流式问答，返回检索来源、token 和完成事件 |
| `POST` | `/api/session/new` | 创建新会话 |
| `GET` | `/api/sessions` | 查看会话列表 |
| `GET` | `/api/session/{id}` | 查看指定会话详情 |

## CLI 用法

```bash
# 文件入库
python delivery/cli.py ingest path/to/file.pdf path/to/file.md

# 提问
python delivery/cli.py ask "请总结文档的主要内容"

# 指定会话继续提问
python delivery/cli.py ask "继续解释第二点" -s <session_id>

# 查看索引统计
python delivery/cli.py stats

# 查看会话历史
python delivery/cli.py history
python delivery/cli.py history -s <session_id>
```

## 测试

```bash
# 单元测试
python -m pytest tests/unit -v

# 跳过 slow 标记的测试
python -m pytest -m "not slow"

# 集成测试，可能需要模型和较长时间
python -m pytest -m slow tests/integration -v

# 简单评估脚本，会调用 DeepSeek API
python tests/eval/run_eval.py
```

## 注意事项

- `transformers` 需要保持 `<5.0.0`，否则部分 tokenizer 兼容性可能出问题，已在 `requirements.txt` 中限制。
- 大 PDF 会优先根据配置阈值切换到 PyMuPDF 兜底解析，避免 docling 在大文件上内存占用过高。
- 当前 ChromaDB 使用本地 `PersistentClient`，适合单进程写入；多 worker 部署时建议切换为 ChromaDB server。
- `multimodal_db/`、`source/` 和会话 JSON 属于运行时数据，是否提交应按实际作业要求决定。

## 设计文档

更详细的设计过程可以查看：

- `docs/plans/2026-06-26-llamaindex-refactor-design.md`
- `docs/plans/2026-06-26-llamaindex-refactor.md`
- `docs/plans/2026-06-29-upload-progress-design.md`
