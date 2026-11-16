
# 第 9 课第 1 步：检索服务（端口 8002）——拥有检索。
# 关键词检索和向量检索都在这；Agent 服务要查东西就来问它。
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_core import vector_search, search, chunks

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class VectorSearchIn(BaseModel):
    q: str = "RAG 是什么"
    top_k: int = 3

@app.post("/vector_search")
def do_vector_search(req: VectorSearchIn):
    results = vector_search(req.q, top_k=req.top_k)
    return {
        "query": req.q,
        "results": [
            {"chunk_id": i, "score": round(s, 4), "text": c.strip()}
            for s, i, c in results
        ],
    }

@app.get("/search")
def do_search(q: str = "RAG", top_k: int = 3):
    all_results = search(chunks, q)
    return {
        "query": q,
        "total": len(all_results),
        "results": [
            {"chunk_id": i, "score": s, "text": c.strip()[:300]}
            for s, i, c in all_results[:top_k]
        ],
    }

@app.get("/health")
def health():
    return {"status": "ok", "service": "retrieval-service"}
