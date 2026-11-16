import os, json
from openai import OpenAI
from chunk_search import load_text, chunk_text

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

text = load_text("sample.txt")
chunks = chunk_text(text)

def get_doc_stats():
    return {"total_chars": len(text), "num_chunks": len(chunks)}

def get_chunk_lengths():
    return {"lengths": [len(c) for c in chunks]}

def count_keyword(keyword):
    return {"keyword": keyword, "count": text.count(keyword)}


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

messages = [{"role": "user", "content": "文档一共有多少块？RAG 出现了几次？每块平均多长？"}]

for round_no in range(1, 6):
   
    resp = client.chat.completions.create(
            model="deepseek-flash",
            messages=messages, 
            tools=tools
        )
    msg = resp.choices[0].message
    if not msg.tool_calls:
        print(msg.content)
        break
    # print(f"第 {round_no} 轮，模型决定:", [c.function.name for c in msg.tool_calls])
    funcs = {"get_doc_stats": get_doc_stats, "get_chunk_lengths": get_chunk_lengths, "count_keyword": count_keyword}
    messages.append(msg)
    for call in msg.tool_calls:
        args = json.loads(call.function.arguments or "{}")
        result = funcs[call.function.name](**args)
        messages.append({"role": "tool", "tool_call_id": call.id,
                         "content": json.dumps(result, ensure_ascii=False)})
    



