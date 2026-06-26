import json
import os

import fitz
from openai import OpenAI

from config.settings import settings
from db.chroma_store import get_store
from rag.retriever import hybrid_search
from memory.session import get_session, Session

def _parse_llm_json(raw: str) -> dict | None:
    """从 LLM 回复中提取 JSON（处理 ```json 代码块包裹）。"""
    try:
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        return json.loads(raw)
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


class QAEngine:
    """多跳 RAG 问答引擎，支持动态跳数与自省早停。"""

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.llm["api_key"],
            base_url=settings.llm["base_url"],
        )
        self.model = settings.llm["model"]

    # ---- 数据库元信息 ----

    @staticmethod
    def _db_meta_text() -> str:
        """生成当前知识库的元信息描述，注入 system prompt。"""
        store = get_store()
        files = store.list_source_files()
        pdf_count = sum(1 for f in files if f["file_type"] == "pdf")
        img_count = sum(1 for f in files if f["file_type"] == "image")
        txt_count = len(files) - pdf_count - img_count
        file_list = "\n".join(
            f"  - {f['file_name']} ({f['file_type']})"
            for f in files
        ) if files else "  (空)"
        return (
            f"当前知识库统计：共 {len(files)} 个文件"
            f"（PDF {pdf_count} 篇, 图片 {img_count} 张, 文本 {txt_count} 个），"
            f"文本向量 {store.text_count} 条, 图像向量 {store.clip_count} 条。\n"
            f"文件清单：\n{file_list}"
        )

    # ---- 自省：判断信息是否充分 ----

    def _reflect(self, question: str, context: str, hop: int) -> dict:
        """让 LLM 判断当前上下文是否足以回答问题，若不足则给出子查询。"""
        reflect_prompt = (
            "你是一个信息充分性判断助手。你的任务是判断当前检索到的文档内容"
            "是否已经足够回答用户的问题。\n\n"
            "请严格按照以下 JSON 格式回复（不要输出任何其他内容）：\n"
            '{"sufficient": true/false, '
            '"reasoning": "你的判断理由", '
            '"sub_queries": ["补充查询1", "补充查询2"]}\n\n'
            "规则：\n"
            "- 如果文档内容已经包含回答问题所需的关键信息，设置 sufficient=true\n"
            "- 如果文档内容明显不足或缺失关键方面，设置 sufficient=false，"
            "并在 sub_queries 中给出 1-3 个更精确的补充查询词/短语\n"
            "- sub_queries 应针对缺失的信息维度，不要重复原始问题"
        )
        user_msg = (
            f"用户问题：{question}\n\n"
            f"当前检索到的文档内容（第 {hop} 跳）：\n"
            f"{'---' * 20}\n{context}\n{'---' * 20}\n\n"
            "请判断这些信息是否足够回答用户的问题。"
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": reflect_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        result = _parse_llm_json(raw)
        if result is not None:
            return {
                "sufficient": bool(result.get("sufficient", False)),
                "reasoning": result.get("reasoning", ""),
                "sub_queries": result.get("sub_queries", []),
            }
        return {"sufficient": True, "reasoning": "解析失败，默认充分", "sub_queries": []}

    # ---- 相关性校验 ----

    def _check_relevance(self, question: str, answer: str, context: str) -> dict:
        """让 LLM 判断生成的回答是否真正回答了用户的问题。"""
        check_prompt = (
            "你是一个回答质量审核员。请判断AI的回答是否**准确且直接地**回答了用户的问题。\n\n"
            "常见错误模式（应判定为 irrelevant）：\n"
            "- 用户问的是关于系统/知识库本身的问题（如'数据库里有几篇论文'、'你有什么文档'），"
            "但回答却在讲文档里的内容\n"
            "- 用户的问题与检索到的文档内容完全不相关，但AI仍然生成了看似合理实则偏题的回答\n"
            "- AI曲解了问题的含义来凑答案\n\n"
            "请严格按照以下 JSON 格式回复（不要输出任何其他内容）：\n"
            '{"relevant": true/false, '
            '"reasoning": "判断理由"}'
        )
        user_msg = (
            f"用户问题：{question}\n\n"
            f"AI的回答：\n{answer}\n\n"
            f"检索到的文档内容：\n{'---' * 10}\n{context[:3000]}\n{'---' * 10}\n\n"
            "请判断AI的回答是否准确、直接地回答了用户的问题。"
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": check_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        result = _parse_llm_json(raw)
        if result is not None:
            return {
                "relevant": bool(result.get("relevant", True)),
                "reasoning": result.get("reasoning", ""),
            }
        return {"relevant": True, "reasoning": "解析失败，默认相关"}

    # ---- 去重合并检索结果 ----

    @staticmethod
    def _merge_results(existing: list[dict], new: list[dict]) -> list[dict]:
        """按 (file_path, page_number) 去重，保留更高 RRF 分数的记录。"""
        seen: dict[tuple, dict] = {}
        for r in existing:
            key = (r["file_path"], r.get("page_number", 0))
            seen[key] = r
        for r in new:
            key = (r["file_path"], r.get("page_number", 0))
            if key not in seen or r["rrf_score"] > seen[key]["rrf_score"]:
                seen[key] = r
        return sorted(seen.values(), key=lambda x: x["rrf_score"], reverse=True)

    # ---- 提取页面文本（惰性加载） ----

    @staticmethod
    def _get_text(r: dict) -> str:
        text = r.get("text", "")
        if text:
            return text
        page = r.get("page_number", 0)
        if not page:
            return ""
        doc = fitz.open(r["file_path"])
        if 0 < page <= len(doc):
            text = doc[page - 1].get_text().strip()
        doc.close()
        return text

    # ---- 构建上下文字符串 ----

    def _build_context(self, results: list[dict]) -> str:
        parts = []
        for r in results:
            text = self._get_text(r)
            if text:
                parts.append(
                    f"--- 来源: {os.path.basename(r['file_path'])} "
                    f"第{r.get('page_number', '?')}页"
                    f" (RRF分数: {r['rrf_score']:.4f}) ---\n{text}"
                )
        return "\n\n".join(parts)

    # ---- 主入口：多跳 RAG 问答 ----

    def ask(self, question: str, top_k: int = None,
            session_id: str = None) -> tuple[str, list[dict]]:
        if top_k is None:
            top_k = settings.retrieval["default_top_k"]

        print(f"\n{'=' * 60}")
        print(f"问题: {question}")
        print(f"{'=' * 60}")

        # === 多跳检索循环 ===
        all_results: list[dict] = []
        queries_this_turn = [question]

        max_hops = settings.retrieval.get("max_hops", 5)
        for hop in range(1, max_hops + 1):
            hop_results: list[dict] = []
            for q in queries_this_turn:
                print(f"\n[跳 {hop}] 检索查询: {q}")
                hop_results.extend(hybrid_search(q, top_k=top_k))

            all_results = self._merge_results(all_results, hop_results)
            context = self._build_context(all_results)

            # 自省：判断是否充分
            print(f"\n[跳 {hop}] 自省中...")
            reflection = self._reflect(question, context, hop)
            print(f"[跳 {hop}] 判断: {'充分' if reflection['sufficient'] else '不充分'}")
            print(f"[跳 {hop}] 理由: {reflection['reasoning']}")

            if reflection["sufficient"]:
                print(f"\n>>> 在第 {hop} 跳后满足充分条件，提前停止检索。")
                break

            if hop < max_hops:
                queries_this_turn = reflection.get("sub_queries", [])
                if not queries_this_turn:
                    print("\n>>> 未生成补充查询，提前停止检索。")
                    break
                print(f"[跳 {hop}] 将执行补充查询: {queries_this_turn}")
            else:
                print(f"\n>>> 已达最大跳数上限 ({max_hops})，停止检索。")

        # === 最终答案生成 ===
        context = self._build_context(all_results)
        db_meta = self._db_meta_text()

        system_prompt = (
            "你是一个文档问答助手，只能根据提供的文档内容和知识库元信息来回答。\n\n"
            "## 核心规则（必须严格遵守）：\n"
            "1. 只根据下方提供的文档内容回答问题，绝不允许编造、推测或曲解\n"
            "2. 如果用户问的是关于知识库/数据库本身的问题（如'有几篇论文'、'你有什么文件'、"
            "'数据库里存了什么'），请直接根据【知识库元信息】如实回答\n"
            "3. 如果文档内容和知识库元信息中都没有相关信息，必须明确回复："
            "'未检索到相关信息，我只能根据知识库中的文档内容回答问题。'\n"
            "4. 严禁将文档中出现的词汇曲解为用户问题的答案（例如用户问'数据库里有几篇论文'，"
            "不能把文档中提到的 CrimeBB/NVD 等外部数据库当作答案）\n"
            "5. 请在回答末尾标注信息来源的页码\n"
            "6. 请使用Markdown格式回答\n\n"
            "## 【知识库元信息】\n" + db_meta
        )
        user_prompt = (
            f"以下是检索到的文档内容（共 {len(all_results)} 条，经过多跳检索）：\n\n"
            f"{context}\n\n---\n请根据以上文档内容和知识库元信息回答：{question}"
        )

        # 获取或创建会话
        if session_id:
            sess = Session(session_id)
        else:
            sess = get_session()

        llm_messages = [{"role": "system", "content": system_prompt}]
        for msg in sess.messages:
            role = msg.get("role")
            if role in ("user", "assistant"):
                llm_messages.append({"role": role, "content": msg["content"]})
        llm_messages.append({"role": "user", "content": user_prompt})

        print("\n正在调用 LLM 生成最终回答...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=llm_messages,
            temperature=0.3,
        )

        answer = response.choices[0].message.content

        # === 相关性校验（仅当检索结果为空或完全无关时触发） ===
        if context and not all_results:
            print("\n>>> 检索结果为空，返回未检索到信息提示。")
            answer = (
                "未检索到相关信息，我只能根据知识库中的文档内容回答问题。\n\n"
                f"**知识库概况：**\n{db_meta}\n\n"
                f"如果您的问题与上述文件内容相关，请换个方式描述后重试。"
            )
        elif context and all_results:
            # 快速检查：回答是否引用了检索到的文件名
            referenced = any(
                os.path.basename(r['file_path']).lower() in answer.lower()
                for r in all_results
            )
            if not referenced:
                # 回答未引用任何源文件，做 LLM 校验
                print("\n相关性校验中...")
                rel = self._check_relevance(question, answer, context)
                print(f"相关性: {'通过' if rel['relevant'] else '不通过'}")
                print(f"理由: {rel['reasoning']}")
                if not rel["relevant"]:
                    answer = (
                        "未检索到相关信息，我只能根据知识库中的文档内容回答问题。\n\n"
                        f"**知识库概况：**\n{db_meta}\n\n"
                        f"如果您的问题与上述文件内容相关，请换个方式描述后重试。"
                    )
                    print(">>> 回答已被相关性校验拦截并替换。")

        print(f"\n回答:\n{answer}")

        sources = [
            {"file": os.path.basename(r['file_path']),
             "page": r.get("page_number", 0),
             "rrf_score": round(r['rrf_score'], 4)}
            for r in all_results
        ]

        sess.add_qa(question, answer, sources=sources)

        return answer, sources


_engine: QAEngine | None = None


def get_qa_engine() -> QAEngine:
    global _engine
    if _engine is None:
        _engine = QAEngine()
    return _engine
