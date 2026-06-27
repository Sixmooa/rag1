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
