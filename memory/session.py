"""会话记忆系统 —— 每个会话以 UUID 命名存为独立 JSON 文件。"""
import json
import uuid
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path(__file__).parent


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Session:
    """单次会话，记录问答历史。"""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.file_path = MEMORY_DIR / f"{self.session_id}.json"
        self.data: dict = self._load_or_create()

    # ---- 内部方法 ----

    def _load_or_create(self) -> dict:
        if self.file_path.exists():
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "session_id": self.session_id,
            "created_at": _now(),
            "messages": [],
        }

    def _save(self):
        self.data["updated_at"] = _now()
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    # ---- 公开接口 ----

    def add(self, role: str, content: str, **extra) -> dict:
        """追加一条消息并持久化。role: user / assistant / system。"""
        msg = {
            "role": role,
            "content": content,
            "timestamp": _now(),
            **extra,
        }
        self.data["messages"].append(msg)
        self._save()
        return msg

    def add_qa(self, question: str, answer: str,
               sources: list[dict] | None = None) -> dict:
        """快捷方法：记录一次完整问答。"""
        self.add("user", question, type="question")
        return self.add("assistant", answer, type="answer",
                        sources=sources or [])

    def add_streaming(self, question: str, full_answer: str,
                      sources: list[dict] | None = None) -> dict:
        """与 add_qa 等价，给流式 API 用，语义清晰。"""
        return self.add_qa(question, full_answer, sources=sources)

    @property
    def messages(self) -> list[dict]:
        return self.data["messages"]

    @property
    def message_count(self) -> int:
        return len(self.data["messages"])

    def delete(self):
        """删除当前会话文件。"""
        if self.file_path.exists():
            self.file_path.unlink()

    def __repr__(self):
        return (f"Session(id={self.session_id[:8]}..., "
                f"messages={self.message_count})")


# ---- 全局当前会话 ----

_current_session: Session | None = None


def new_session() -> Session:
    """创建新会话。"""
    global _current_session
    _current_session = Session()
    return _current_session


def get_session() -> Session:
    """获取当前会话，不存在则自动创建。"""
    global _current_session
    if _current_session is None:
        _current_session = Session()
    return _current_session


def load_session(session_id: str) -> Session:
    """加载已有会话。"""
    global _current_session
    _current_session = Session(session_id)
    return _current_session


def list_sessions() -> list[dict]:
    """列出所有会话摘要。"""
    sessions = []
    for fp in MEMORY_DIR.glob("*.json"):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({
                "session_id": data.get("session_id", fp.stem),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "message_count": len(data.get("messages", [])),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    sessions.sort(key=lambda s: s["updated_at"] or s["created_at"], reverse=True)
    return sessions
