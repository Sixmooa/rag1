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
        if isinstance(self.data, str):
            data = json.dumps(self.data, ensure_ascii=False)
        else:
            data = json.dumps(self.data, ensure_ascii=False)
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
        """CLI 用同步路径：用 complete() 一次性返回。

        Loop-safe: 若调用方已在事件循环中（如 Jupyter / async 测试），
        退到独立线程跑 retrieve，避免 "This event loop is already running" 错误。
        """
        import asyncio
        nodes = _run_coro_sync(self._retrieval.retrieve(question))
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


def _run_coro_sync(coro):
    """同步执行 coroutine：无活动循环时直接 run_until_complete；
    有活动循环时退到独立线程跑，避免阻塞主循环。"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    if loop is None or not loop.is_running():
        if loop is None:
            loop = asyncio.new_event_loop()
        return loop.run_until_complete(coro)
    # 已在运行中 —— 用线程隔离的新循环
    import threading
    result_box: dict = {}

    def _worker():
        new_loop = asyncio.new_event_loop()
        try:
            result_box["v"] = new_loop.run_until_complete(coro)
        except Exception as e:
            result_box["e"] = e
        finally:
            new_loop.close()

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    if "e" in result_box:
        raise result_box["e"]
    return result_box.get("v")
