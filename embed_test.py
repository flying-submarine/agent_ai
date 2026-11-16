import config  # noqa: F401 —— 自动加载 .env 里的密钥（见 config.py）
import os
import math
from openai import OpenAI
from chunk_search import load_text, chunk_text, search

client = OpenAI(
    api_key=os.environ["SILICONFLOW_API_KEY"],
    base_url="https://api.siliconflow.cn/v1",
)

def embed(text):
    resp = client.embeddings.create(model="BAAI/bge-m3", input=text)
    return resp.data[0].embedding

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)

text = load_text("sample.txt")
chunks = chunk_text(text)
chunk_vecs = [embed(c) for c in chunks]
print("每块的向量长度:", len(chunk_vecs[0]))

q = "火星上有水吗"
print("关键词版结果:", [(s, i) for s, i, _ in search(chunks, q)])

qv = embed(q)
scored = sorted(
    ((cosine(qv, v), i) for i, v in enumerate(chunk_vecs)),
    reverse=True,
)
print("向量版排名:")
for s, i in scored:
    print(f"{s:.3f}  块 #{i}: {chunks[i].strip()[:40]}")
