from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient, models
import jieba, requests
from chunk_search import load_text, chunk_text
from app import embed
import os

SILICONFLOW_API_KEY = os.environ["SILICONFLOW_API_KEY"]

# 只切一次：BM25 和 Qdrant 用同一份 chunks，编号才对得上
text = load_text("sample.txt")
chunks = chunk_text(text, 200, 50)

# --- BM25 这一路：先分词，再算分 ---
tokenized = [list(jieba.cut(c)) for c in chunks]
bm25 = BM25Okapi(tokenized)
bm25_scores = bm25.get_scores(list(jieba.cut("pgvector")))

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="docs",
    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
)

vecs = [embed(c) for c in chunks]

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
    query=embed("pgvector"),
    limit=3,
).points

# 向量那路的名次：Qdrant 搜回来的顺序就是名次
vec_order = [h.payload["chunk_id"] for h in hits]

# BM25 那路的名次：分数从高到低排
bm25_order = sorted(range(len(chunks)), key=lambda i: bm25_scores[i], reverse=True)

# RRF 合并：两路名次各算 1/(60+名次)，相加
def rrf(rank, k=60):
    return 1 / (k + rank)

scores = {}
for rank, cid in enumerate(vec_order):
    scores[cid] = scores.get(cid, 0) + rrf(rank)
for rank, cid in enumerate(bm25_order):
    scores[cid] = scores.get(cid, 0) + rrf(rank)

# 总分最高的 3 块，就是送去精排的候选
candidates = sorted(scores, key=scores.get, reverse=True)[:3]

# --- reranker 精排：调 SiliconFlow ---
resp = requests.post(
    "https://api.siliconflow.cn/v1/rerank",
    headers={"Authorization": f"Bearer {SILICONFLOW_API_KEY}"},
    json={
        "model": "BAAI/bge-reranker-v2-m3",
        "query": "pgvector",
        "documents": [chunks[i] for i in candidates],  # 候选块的原文
        "top_n": 3,
    },
).json()

print("向量:", vec_order)
print("BM25:", bm25_order[:3])
print("RRF候选:", candidates)

# index 是候选里的位置，要映射回块编号再打印
for r in resp["results"]:
    print(candidates[r["index"]], r["relevance_score"])

q = list(jieba.cut("pgvector"))
print("query词:", q)
print("分数:", bm25.get_scores(q))
print(bm25.idf.get('pgvector'))  # 0.0


