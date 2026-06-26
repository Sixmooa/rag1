"""测试三种文件类型的入库与检索"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config.settings import settings
from db.chroma_store import get_store
from rag.retriever import search_by_clip, search_by_bge, hybrid_search
from rag.qa_engine import get_qa_engine
from tools.file_parser import detect_file_type

SOURCE = os.path.join(os.path.dirname(__file__), "source")
store = get_store()

print("=" * 60)
print("1. 文件类型智能检测测试")
print("=" * 60)

test_files = [
    os.path.join(SOURCE, "tmp.txt"),
    os.path.join(SOURCE, "19_Moral_Dilemma_AI_Reasoning.pdf"),
    os.path.join(SOURCE, "PixPin_2026-04-17_15-27-30.png"),
]
for f in test_files:
    if os.path.exists(f):
        detected = detect_file_type(f)
        print(f"  {os.path.basename(f):45s} -> {detected}")
    else:
        print(f"  {os.path.basename(f):45s} -> 文件不存在!")

print()
print("=" * 60)
print("2. 入库测试")
print("=" * 60)

# 入库 TXT
txt_path = os.path.join(SOURCE, "tmp.txt")
if os.path.exists(txt_path):
    store.add_text_file(txt_path)
    print()

# 入库 PDF
pdf_path = os.path.join(SOURCE, "19_Moral_Dilemma_AI_Reasoning.pdf")
if os.path.exists(pdf_path):
    store.add_pdf(pdf_path)
    print()

# 入库图片
img_path = os.path.join(SOURCE, "PixPin_2026-04-17_15-27-30.png")
if os.path.exists(img_path):
    store.add_image(img_path)
    print()

print("=" * 60)
print("3. 数据库统计")
print("=" * 60)
print(f"  CLIP collection (图像向量): {store.clip_count} 条记录")
print(f"  BGE  collection (文本向量): {store.text_count} 条记录")

print()
print("=" * 60)
print("4. 检索测试")
print("=" * 60)

queries = [
    "道德困境",
    "AI reasoning",
]

for q in queries:
    print(f"\n--- 查询: {q} ---")

    print("\n[BGE 文本检索]")
    bge_res = search_by_bge(q, top_k=3)
    for i, (doc_id, dist, meta) in enumerate(zip(
        bge_res['ids'][0], bge_res['distances'][0], bge_res['metadatas'][0]
    )):
        print(f"  {i + 1}. {doc_id} | 距离: {dist:.4f} | 来源: {os.path.basename(meta.get('file_path', ''))}"
              f" {'第' + str(meta.get('page_number', meta.get('chunk_index', ''))) + '页/块' if meta.get('page_number') or meta.get('chunk_index') else ''}")

    print("\n[CLIP 图像检索]")
    clip_res = search_by_clip(q, top_k=3)
    for i, (doc_id, dist, meta) in enumerate(zip(
        clip_res['ids'][0], clip_res['distances'][0], clip_res['metadatas'][0]
    )):
        print(f"  {i + 1}. {doc_id} | 距离: {dist:.4f} | 类型: {meta.get('source_type', '')}"
              f" | {os.path.basename(meta.get('file_path', ''))}")

print("\n测试完成！")
