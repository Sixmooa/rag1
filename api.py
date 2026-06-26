"""多模态 RAG 系统 Web API。"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config.settings import settings
from db.chroma_store import get_store
from rag.qa_engine import get_qa_engine
from memory.session import Session, list_sessions
from tools.file_parser import detect_bytes_type


# ---- Pydantic Schemas ----

class AskRequest(BaseModel):
    question: str
    session_id: str | None = None
    top_k: int | None = None

class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str

class StatsResponse(BaseModel):
    clip_count: int
    text_count: int

class UploadResult(BaseModel):
    filename: str
    status: str
    detail: str

class UploadResponse(BaseModel):
    results: list[UploadResult]

class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    message_count: int


# ---- App ----

static_dir = os.path.join(os.path.dirname(__file__), "static")
source_dir = os.path.join(os.path.dirname(__file__), settings.paths["source_dir"])
os.makedirs(source_dir, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("正在预热模型...")
    get_qa_engine()
    print("模型加载完成，服务已启动。")
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ---- 页面 ----

@app.get("/")
async def index():
    return FileResponse(os.path.join(static_dir, "index.html"))


# ---- API ----

@app.get("/api/stats", response_model=StatsResponse)
async def stats():
    store = get_store()
    return StatsResponse(clip_count=store.clip_count, text_count=store.text_count)


@app.post("/api/upload", response_model=UploadResponse)
async def upload(files: list[UploadFile] = File(...)):
    store = get_store()
    results = []

    for f in files:
        save_path = os.path.join(source_dir, f.filename)
        content = await f.read()
        with open(save_path, "wb") as out:
            out.write(content)

        file_type = detect_bytes_type(content, f.filename)
        try:
            if file_type == "pdf":
                await asyncio.to_thread(store.add_pdf, save_path)
            elif file_type == "image":
                await asyncio.to_thread(store.add_image, save_path)
            elif file_type == "text":
                await asyncio.to_thread(store.add_text_file, save_path)
            else:
                results.append(UploadResult(
                    filename=f.filename, status="error",
                    detail=f"无法识别文件类型",
                ))
                continue
            results.append(UploadResult(
                filename=f.filename, status="ok",
                detail=f"识别为 {file_type}",
            ))
        except Exception as e:
            results.append(UploadResult(
                filename=f.filename, status="error", detail=str(e),
            ))

    return UploadResponse(results=results)


@app.post("/api/session/new")
async def new_session():
    sess = Session()
    return {"session_id": sess.session_id, "created_at": sess.data["created_at"]}


@app.get("/api/sessions", response_model=list[SessionInfo])
async def get_sessions():
    return list_sessions()


@app.get("/api/session/{session_id}")
async def get_session_detail(session_id: str):
    sess = Session(session_id)
    return {
        "session_id": sess.session_id,
        "created_at": sess.data.get("created_at", ""),
        "updated_at": sess.data.get("updated_at", ""),
        "messages": sess.messages,
    }


@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    engine = get_qa_engine()
    answer, sources = await asyncio.to_thread(
        engine.ask, req.question, req.top_k, req.session_id,
    )
    sid = req.session_id or get_current_session_id()
    return AskResponse(answer=answer, sources=sources, session_id=sid)


def get_current_session_id() -> str:
    sessions = list_sessions()
    if sessions:
        return sessions[-1]["session_id"]
    sess = Session()
    return sess.session_id


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
