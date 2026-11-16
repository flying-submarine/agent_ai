# 第 9 课第 2 步：文档服务（端口 8001）——新增 /ingest + 任务状态查询。
# 原来的 /stats、/chunk_lengths、/count_keyword 一个没动。
# /ingest 只做三件事：生成一个任务号、把文档文本发进 RabbitMQ 队列 ingest_tasks、立刻返回任务号。
# 切块、算向量、写 Qdrant 是 ingest_worker.py 在后台慢慢做的——这就是异步入库。
# 任务状态存在这个服务自己的内存字典里（tasks），worker 做完后调 /task/{task_id}/done 来改状态。
# 注意这个状态的局限：doc 服务重启，状态就丢了——生产上状态要放 Redis 或数据库里（“状态放哪”这个问题，第 3 步会再讲）。
# 每次请求都新开一个 RabbitMQ 连接，用完就关——写法笨，但本课够用，长连接复用是优化项不是现在的事。
import json
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pika
from rag_core import get_doc_stats, get_chunk_lengths, count_keyword

RABBITMQ_HOST = "localhost"
QUEUE = "ingest_tasks"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 任务状态表：task_id -> {"status": "processing" | "done", "chars": N, "chunks": N}
tasks = {}

def publish_ingest(task_id: str, text: str):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    try:
        channel = connection.channel()
        channel.queue_declare(queue=QUEUE, durable=True)
        channel.basic_publish(
            exchange="",
            routing_key=QUEUE,
            body=json.dumps({"task_id": task_id, "text": text}, ensure_ascii=False),
            # delivery_mode=2：消息持久化，RabbitMQ 重启也不丢。
            properties=pika.BasicProperties(delivery_mode=2),
        )
    finally:
        connection.close()

class IngestIn(BaseModel):
    text: str

class TaskDoneIn(BaseModel):
    chunks: int

@app.post("/ingest")
def ingest(req: IngestIn):
    task_id = uuid.uuid4().hex[:12]
    tasks[task_id] = {"status": "processing", "chars": len(req.text), "chunks": 0}
    publish_ingest(task_id, req.text)
    return {"task_id": task_id, "status": "processing", "message": "已收到，正在后台处理", "chars": len(req.text)}

@app.get("/task/{task_id}")
def get_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务号不存在")
    return {"task_id": task_id, **tasks[task_id]}

@app.post("/task/{task_id}/done")
def task_done(task_id: str, req: TaskDoneIn):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务号不存在")
    tasks[task_id]["status"] = "done"
    tasks[task_id]["chunks"] = req.chunks
    return {"task_id": task_id, **tasks[task_id]}

@app.get("/stats")
def stats():
    return get_doc_stats()

@app.get("/chunk_lengths")
def chunk_lengths():
    return get_chunk_lengths()

@app.get("/count_keyword")
def count_keyword_endpoint(keyword: str):
    return count_keyword(keyword)

@app.get("/health")
def health():
    return {"status": "ok", "service": "doc-service"}
