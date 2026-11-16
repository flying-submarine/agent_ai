from qdrant_client import QdrantClient, models
from chunk_search import load_text, chunk_text, search
from app import embed


# ":memory:" 是内存模式：不用装任何服务，先把 API 跑通
client = QdrantClient(":memory:")

# 建集合：向量 1024 维（bge-m3 的维度），距离用余弦（跟你第 4 课一样）
client.create_collection(
    collection_name="docs",
    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
)

# 读 sample.txt，按 200/50 切块，调 SiliconFlow 嵌成向量
chunks = chunk_text(open("sample.txt").read(), 200, 50)
vecs = [embed(c) for c in chunks]

# 入库：向量存进去，原文放 payload 里
# 细扣：向量库只负责"按向量找"，原文存在 payload 里，搜到了连原文一起还你
client.upsert(
    collection_name="docs",
    points=[
        models.PointStruct(id=i, vector=v, payload={"text": c, "chunk_id": i})
        for i, (v, c) in enumerate(zip(vecs, chunks))
    ],
)

# 搜一次：把问题也嵌成向量，找最像的 3 块
hits = client.query_points(
    collection_name="docs",
    query=embed("RAG 是什么"),
    limit=3,
).points

for h in hits:
    print(h.score, h.payload["chunk_id"], h.payload["text"][:40])

