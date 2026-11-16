import os,json,math
from fastapi import FastAPI
from chunk_search import load_text, chunk_text, search
from pydantic import BaseModel
from openai import OpenAI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 学习阶段先放开，上线前改成你的网站域名
    allow_methods=["*"],
    allow_headers=["*"],
)

text = load_text("sample.txt")
chunks = chunk_text(text)

class ToolAskIn(BaseModel):
    question: str

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

embed_client = OpenAI(
    api_key=os.environ.get("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)
def embed(text):
    resp = embed_client.embeddings.create(model="BAAI/bge-m3", input=text)
    return resp.data[0].embedding

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)

def get_doc_stats():
    return {"total_chars": len(text), "num_chunks": len(chunks)}

def get_chunk_lengths():
    return {"lengths": [len(c) for c in chunks]}

def count_keyword(keyword):
    return {"keyword": keyword, "count": text.count(keyword)}


chunk_vecs = [embed(c) for c in chunks]

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_doc_stats",
            "description": "查询当前文档的总字数和切块数量",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chunk_lengths",
            "description": "查询文档每一块各自的字符长度",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_keyword",
            "description": "统计某个关键词在文档中出现的次数",
            "parameters": {
                "type": "object",
                "properties": {"keyword": {"type": "string", "description": "要统计的词"}},
                "required": ["keyword"],
            },
        },
    },
]


SIM_THRESHOLD = 0.4  # 及格线，等火星的分数出来再定

def vector_search(q, top_k=3):
    qv = embed(q)
    scored = sorted(
        ((cosine(qv, v), i) for i, v in enumerate(chunk_vecs)),
        reverse=True,
    )
    return [(s, i, chunks[i]) for s, i in scored[:top_k] if s >= SIM_THRESHOLD]

class AskRequest(BaseModel):
    q: str = "RAG 是什么"

@app.post("/ask")
def do_ask(req: AskRequest):
    try:
        results = vector_search(req.q)
        if not results:
            return {"query": req.q, "answer": "资料里没找到相关内容，换个词试试", "sources": []}
        context = "\n\n".join(
            f"【块 #{i}】\n{c.strip()[:500]}" for _, i, c in results
        )
        resp = client.chat.completions.create(
            model="deepseek-flash",
            messages=[
                {
                    "role": "system",
                    "content": "只根据下面提供的资料回答问题。资料里没有答案就直说不知道，不要编造。",
                },
                {"role": "user", "content": f"资料：\n{context}\n\n问题：{req.q}"},
            ],
        )
        return {
            "query": req.q,
            "answer": resp.choices[0].message.content,
            "sources": [i for _, i, _ in results],
        }
    except Exception as e:
        print("ask 调用失败:", type(e).__name__, e)
        return {"query": req.q, "answer": "外部服务暂时不可用，请稍后重试", "sources": []}

@app.post("/ask_stream")
def do_ask_stream(req: AskRequest):
    try:
        results = vector_search(req.q)
    except Exception as e:
        print("ask_stream 检索失败:", type(e).__name__, e)
        return {"query": req.q, "answer": "外部服务暂时不可用，请稍后重试", "sources": []}
    if not results:
        return {"query": req.q, "answer": "资料里没找到相关内容，换个词试试", "sources": []}

    context = "\n\n".join(
        f"【块 #{i}】\n{c.strip()[:500]}" for _, i, c in results
    )
    sources = [i for _, i, _ in results]

    def gen():
        stream = client.chat.completions.create(
            model="deepseek-flash",
            messages=[
                {
                    "role": "system",
                    "content": "只根据下面提供的资料回答问题。资料里没有答案就直说不知道，不要编造。",
                },
                {"role": "user", "content": f"资料：\n{context}\n\n问题：{req.q}"},
            ],
            stream=True,
        )
        for chunk in stream:
            piece = chunk.choices[0].delta.content
            if piece:
                yield piece
        yield f"\n\n【来源块】{sources}"

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")

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

@app.get("/stats")
def get_stats():
    return {"total_chars": len(text), "num_chunks": len(chunks)}

@app.post("/ask_tool")
def ask_tool(req: ToolAskIn):
    messages = [{"role": "user", "content": req.question}]
    funcs = {"get_doc_stats": get_doc_stats, "get_chunk_lengths": get_chunk_lengths, "count_keyword": count_keyword}
    for round_no in range(1, 6):
        resp = client.chat.completions.create(model="deepseek-flash", messages=messages, tools=tools)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return {"answer": msg.content, "rounds": round_no}
        messages.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            result = funcs[call.function.name](**args)
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, ensure_ascii=False)})
    return {"answer": "工具调用轮数超限，换个问法试试", "rounds": 5}

from agent_graph import run_agent

class GraphAskIn(BaseModel):
    question: str
    session_id: str = "default"

@app.post("/ask_graph")
def ask_graph(req: GraphAskIn):
    try:
        answer = run_agent(req.question, req.session_id)
        return {"answer": answer, "session_id": req.session_id}
    except Exception as e:
        print("ask_graph error:", type(e).__name__, e)
        return {"answer": "外部服务暂时不可用，请稍后重试", "session_id": req.session_id}