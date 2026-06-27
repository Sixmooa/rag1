"""跑 qa_dataset.json，输出 Recall@k 与关键词覆盖率。"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pipeline.runtime import get_generation_pipeline


async def run_one(q):
    pipe = get_generation_pipeline()
    t0 = time.time()
    nodes = await pipe._retrieval.retrieve(q["question"])
    t_retrieve = time.time() - t0
    sources = [os.path.basename(n.metadata.get("file_path", "")) for n in nodes]

    hit = any(g in s for s in sources for g in q["gold_files"]) if q["gold_files"] else True

    # 用同步生成（不流式）拿答案
    answer, _ = pipe.retrieve_and_format(q["question"], None)
    t_total = time.time() - t0

    kw_hit = sum(1 for k in q["gold_keywords"] if k.lower() in answer.lower())
    kw_cov = kw_hit / max(1, len(q["gold_keywords"]))
    return {"q": q["question"][:30], "hit": hit, "kw_cov": kw_cov,
            "t_retrieve": round(t_retrieve, 2), "t_total": round(t_total, 2)}


async def main():
    with open(os.path.join(os.path.dirname(__file__), "qa_dataset.json"), encoding="utf-8") as f:
        data = json.load(f)
    results = []
    for q in data:
        r = await run_one(q)
        results.append(r)
        print(r)
    hits = sum(1 for r in results if r["hit"])
    avg_kw = sum(r["kw_cov"] for r in results) / len(results)
    avg_t = sum(r["t_total"] for r in results) / len(results)
    print(f"\nRecall@k: {hits}/{len(results)}")
    print(f"关键词覆盖率: {avg_kw:.1%}")
    print(f"平均耗时: {avg_t:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
