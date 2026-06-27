"""文件上传与统计路由。"""
import logging
import os

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

from pipeline.runtime import get_indexing_pipeline, get_clip_index, get_bge_index

logger = logging.getLogger(__name__)
router = APIRouter()

SOURCE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "source")


class StatsResp(BaseModel):
    clip_count: int
    text_count: int


class UploadResult(BaseModel):
    filename: str
    status: str
    detail: str


class UploadResponse(BaseModel):
    results: list[UploadResult]


@router.get("/api/stats", response_model=StatsResp)
async def stats():
    return StatsResp(
        clip_count=get_clip_index().count,
        text_count=get_bge_index().count,
    )


@router.post("/api/upload", response_model=UploadResponse)
async def upload(files: list[UploadFile] = File(...)):
    pipeline = get_indexing_pipeline()
    os.makedirs(SOURCE_DIR, exist_ok=True)
    results = []
    for f in files:
        save_path = os.path.join(SOURCE_DIR, f.filename)
        content = await f.read()
        with open(save_path, "wb") as out:
            out.write(content)
        r = pipeline.ingest(save_path)
        results.append(UploadResult(
            filename=f.filename,
            status=r.status,
            detail=r.detail,
        ))
    return UploadResponse(results=results)
