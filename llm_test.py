import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

resp = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {"role": "system", "content": "你是一个简洁的助手，回答不超过三句话。"},
        {"role": "user", "content": "用一句话解释什么是 RAG"},
    ],
)
print(resp.choices[0].message.content)
print("用量:", resp.usage)
