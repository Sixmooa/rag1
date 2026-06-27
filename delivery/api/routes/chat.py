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
                "data": json.dumps(evt.data, ensure_ascii=False),
            }

    return EventSourceResponse(event_gen())
