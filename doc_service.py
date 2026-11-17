# 第 9 课第 3 步：文档服务（端口 8001）——三个变化。
# 1. 幂等：/ingest 按文本内容算 doc_id（sha256 前 12 位），同样的文本每次算出来都一样；
#    worker 用 doc_id 生成点的 id，重复入库就是覆盖，不会累加。任务号 task_id 每次还是新的，
#    因为它只用来查状态，不用来标识文档。
# 2. 状态持久化：任务状态不再放内存字典，改放 SQLite 文件 tasks.db——doc 服务重启，状态还在。
# 3. 死信：队列声明加上 x-dead-letter-exchange，处理失败又不重回队列的消息会进死信队列，
#    而不是直接丢掉。注意：队列一旦建好，改参数必须先删掉旧队列再建，否则 RabbitMQ 报 406。

# 引用的包（每个包是干嘛的）：
import hashlib        # 算哈希：给文本内容算一个固定指纹（doc_id），同样的文本算出来一样
import json           # 把 Python 字典转成 JSON 字符串，发给 RabbitMQ 时消息体必须是字符串
import sqlite3        # Python 自带的轻量数据库：任务状态存在 tasks.db 文件里，重启不丢
import uuid           # 生成随机任务号 task_id（uuid4 是随机的，每次都不一样）
from fastapi import FastAPI, HTTPException   # FastAPI：搭 HTTP 服务；HTTPException：返回 404 这类错误
from fastapi.middleware.cors import CORSMiddleware  # 跨域中间件：让浏览器里的前端能直接调这个服务
from pydantic import BaseModel  # 定义请求体的结构（/ingest 收什么字段、什么类型，FastAPI 自动校验）
import pika           # Python 连 RabbitMQ 的客户端库：发消息、声明队列都靠它
from rag_core import get_doc_stats, get_chunk_lengths, count_keyword  # 第 5 课的三个工具函数，复用

RABBITMQ_HOST = "localhost"
QUEUE = "ingest_tasks"
DLX = "dlx"
DLQ = "ingest_tasks_dlq"
DB_PATH = "tasks.db"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def db():
    # 打开 SQLite 连接，并保证 tasks 表存在（不存在就建一张）。
    # 每次操作都新开一个连接，用完就关——不跨请求共享连接，线程安全。
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS tasks "
        "(task_id TEXT PRIMARY KEY, status TEXT, chars INTEGER, chunks INTEGER, error TEXT)"
    )
    conn.commit()
    return conn

def save_task(task_id, status, chars, chunks=0, error=""):
    # 把一条任务状态写进 tasks.db。任务号已存在就覆盖（INSERT OR REPLACE），不存在就新建。
    conn = db()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO tasks VALUES (?, ?, ?, ?, ?)",
            (task_id, status, chars, chunks, error),
        )
        conn.commit()
    finally:
        conn.close()

def load_task(task_id):
    # 按任务号从 tasks.db 查一条任务状态。查不到返回 None（调用方决定报 404 还是新建）。
    conn = db()
    try:
        row = conn.execute(
            "SELECT status, chars, chunks, error FROM tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {"status": row[0], "chars": row[1], "chunks": row[2], "error": row[3]}

def publish_ingest(task_id: str, doc_id: str, text: str):
    # 把入库任务发进 RabbitMQ 队列。发之前先把死信拓扑声明好（交换机、死信队列、绑定、主队列），
    # 声明和 worker 那边必须一模一样，否则 RabbitMQ 报 406。
    # 每调一次就新开一个连接，用完就关——写法笨，本课够用。
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    try:
        channel = connection.channel()
        # 死信交换机 + 死信队列：先声明，队列才能引用它。
        channel.exchange_declare(exchange=DLX, exchange_type="direct", durable=True)
        channel.queue_declare(queue=DLQ, durable=True)
        channel.queue_bind(queue=DLQ, exchange=DLX, routing_key=QUEUE)
        channel.queue_declare(
            queue=QUEUE,
            durable=True,
            arguments={"x-dead-letter-exchange": DLX, "x-dead-letter-routing-key": QUEUE},
        )
        channel.basic_publish(
            exchange="",
            routing_key=QUEUE,
            body=json.dumps({"task_id": task_id, "doc_id": doc_id, "text": text}, ensure_ascii=False),
            # delivery_mode=2：消息持久化，RabbitMQ 重启也不丢。
            properties=pika.BasicProperties(delivery_mode=2),
        )
    finally:
        connection.close()

class IngestIn(BaseModel):
    # /ingest 的请求体：只收一个 text 字段（要入库的文档全文）。
    text: str

class TaskDoneIn(BaseModel):
    # /task/{task_id}/done 的请求体：worker 做完后回告入库了几块。
    chunks: int

class TaskFailedIn(BaseModel):
    # /task/{task_id}/failed 的请求体：worker 失败时回告错误信息，默认空字符串。
    error: str = ""

@app.post("/ingest")
def ingest(req: IngestIn):
    # 接收一段文本，生成任务号和文档号，记状态、发消息、立刻返回。
    # doc_id 按内容算：同样的文本每次都一样，重复入库时 worker 生成的点 id 也一样，upsert 就是覆盖。
    task_id = uuid.uuid4().hex[:12]
    doc_id = hashlib.sha256(req.text.encode("utf-8")).hexdigest()[:12]
    save_task(task_id, "processing", len(req.text))
    publish_ingest(task_id, doc_id, req.text)
    return {"task_id": task_id, "doc_id": doc_id, "status": "processing", "message": "已收到，正在后台处理", "chars": len(req.text)}

@app.get("/task/{task_id}")
def get_task(task_id: str):
    # 按任务号查状态（前端轮询就调这个）。任务号不存在报 404。
    task = load_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务号不存在")
    return {"task_id": task_id, **task}

@app.post("/task/{task_id}/done")
def task_done(task_id: str, req: TaskDoneIn):
    # worker 做完后回告：把任务状态改成 done，并记下入库了几块。任务号不存在报 404。
    task = load_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务号不存在")
    save_task(task_id, "done", task["chars"], req.chunks)
    return {"task_id": task_id, **load_task(task_id)}

@app.post("/task/{task_id}/failed")
def task_failed(task_id: str, req: TaskFailedIn):
    # worker 失败时回告：把任务状态改成 failed，并存下错误信息。
    # 失败回告可能来自没经过 /ingest 的坏消息，任务号在库里没有——这时直接建一条 failed 记录，而不是 404。
    task = load_task(task_id)
    chars = task["chars"] if task else 0
    save_task(task_id, "failed", chars, 0, req.error[:200])
    return {"task_id": task_id, **load_task(task_id)}

@app.get("/stats")
def stats():
    # 返回 sample.txt 的总字数和块数（第 2 课的 get_doc_stats）。
    return get_doc_stats()

@app.get("/chunk_lengths")
def chunk_lengths():
    # 返回每块的长度（第 5 课的 get_chunk_lengths）。
    return get_chunk_lengths()

@app.get("/count_keyword")
def count_keyword_endpoint(keyword: str):
    # 返回某个关键词在文档里出现了几次（第 5 课的 count_keyword）。
    return count_keyword(keyword)

@app.get("/health")
def health():
    # 健康检查：服务活着就返回 ok，给部署和监控用。
    return {"status": "ok", "service": "doc-service"}
