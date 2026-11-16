"""第 8 课第 5 步：Qdrant 从内存模式搬到 Docker 服务。

跑法（在 ~/Desktop/day1 里，venv 已激活、两个 key 已 export）：
    python3 qdrant_service_test.py            # 写入 4 块，再查一遍
    docker restart qdrant                      # 重启容器
    python3 qdrant_service_test.py query       # 只查不写——数据还在，就是持久化生效了
"""
import os
import sys

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

SILICONFLOW_API_KEY = os.environ["SILICONFLOW_API_KEY"]

client = QdrantClient(url="http://localhost:6333")
embed_client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url="https://api.siliconflow.cn/v1")


def chunk_text(text, size=200, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def embed(texts):
    resp = embed_client.embeddings.create(model="BAAI/bge-m3", input=texts)
    return [d.embedding for d in resp.data]


query_only = len(sys.argv) > 1 and sys.argv[1] == "query"

if not query_only:
    text = open("sample.txt", encoding="utf-8").read()
    chunks = chunk_text(text)
    print("块数：", len(chunks))
    if not client.collection_exists("docs"):
        client.create_collection("docs", vectors_config=VectorParams(size=1024, distance=Distance.COSINE))
    vectors = embed(chunks)
    client.upsert("docs", [PointStruct(id=i, vector=vectors[i], payload={"text": chunks[i]})
                           for i in range(len(chunks))])
    print("已写入", len(chunks), "块")

q = embed(["RAG 是什么"])[0]
hits = client.query_points("docs", query=q, limit=3).points
print("查询 RAG 是什么：", [(h.id, round(h.score, 3)) for h in hits])
