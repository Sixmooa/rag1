"""图片解析：返回单个 image 节点，CLIP 索引按 image_path 取图。"""
import os
from llama_index.core.schema import Document


def parse_image(path: str) -> list[Document]:
    abs_path = os.path.abspath(path)
    return [Document(
        text="",
        metadata={
            "source_type": "image",
            "file_path": abs_path,
            "image_path": abs_path,
            "doc_id": os.path.basename(abs_path),
        },
    )]
