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
