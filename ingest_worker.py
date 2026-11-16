# 第 9 课第 2 步：入库消费者 worker——RabbitMQ 队列的另一头。
# 跑法：单独开一个终端（和 8001/8002/8003 平行），export 两个 key 后 python3 ingest_worker.py。
# 它做的事就是第 1 步里启动时干的那套，只是换成了“收到消息才做”：
#   取任务 -> chunk_text 切块 -> SiliconFlow 算向量 -> 写进 Qdrant -> 回告 doc 服务 done -> ack 确认。
# ack 的意思：只有写进 Qdrant 成功了，才告诉 RabbitMQ“这条处理完了，可以删了”。
# 写失败不 ack（requeue=True），消息回到队列等下一次——但注意：如果这条消息本身就有问题，
# 它会一直失败、一直回队、一直重试，这叫毒消息，第 3 步讲死信队列时再处理。
import json
import httpx
import pika
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from chunk_search import chunk_text
from rag_core import embed_client

QUEUE = "ingest_tasks"
COLLECTION = "docs"
DOC_URL = "http://127.0.0.1:8001"

qclient = QdrantClient(url="http://localhost:6333")
if not qclient.collection_exists(COLLECTION):
    qclient.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

def embed_batch(texts):
    # 和第 8 课 qdrant_service_test.py 里一样的批量 embedding：一次请求算所有块。
    resp = embed_client.embeddings.create(model="BAAI/bge-m3", input=texts)
    return [d.embedding for d in resp.data]

def handle(task_id, text):
    chunks = chunk_text(text)
    vectors = embed_batch(chunks)
    # 新块的 id 从“集合里现有的块数”往后排：第 8 课的 4 块是 id 0-3，新来的第一块就是 id 4。
    # 这只在本课成立——本课只追加不删除，id 不会撞；生产上该用 UUID，第 3 步讲幂等时会回头说。
    base_id = qclient.count(COLLECTION).count
    points = [
        PointStruct(id=base_id + i, vector=vectors[i], payload={"doc_id": task_id, "text": chunks[i]})
        for i in range(len(chunks))
    ]
    qclient.upsert(COLLECTION, points)
    print(f"已入库 {len(chunks)} 块，id {base_id}..{base_id + len(chunks) - 1}")
    return len(chunks)

def report_done(task_id, n_chunks):
    # 告诉 doc 服务任务完成了，前端轮询 /task/{task_id} 就能看到 done。
    # 回告失败也不影响数据——数据已经在 Qdrant 里了，只是前端会一直看到 processing。
    try:
        httpx.post(f"{DOC_URL}/task/{task_id}/done", json={"chunks": n_chunks}, timeout=5.0)
    except Exception as e:
        print("回告 doc 服务失败（数据已入库，状态没同步）:", type(e).__name__, e)

def callback(ch, method, properties, body):
    task = json.loads(body.decode("utf-8"))
    try:
        n = handle(task["task_id"], task["text"])
        report_done(task["task_id"], n)
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        print("入库失败，消息回队列：", type(e).__name__, e)
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))
channel = connection.channel()
channel.queue_declare(queue=QUEUE, durable=True)
channel.basic_consume(queue=QUEUE, on_message_callback=callback, auto_ack=False)
print("worker 已启动，等 ingest 任务……（Ctrl+C 停）")
channel.start_consuming()
