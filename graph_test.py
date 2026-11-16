import config  # noqa: F401 —— 自动加载 .env 里的密钥（见 config.py）
import os
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(
    model="deepseek-flash",
    base_url="https://api.deepseek.com",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    temperature=0,
)

from langchain_core.tools import tool
from chunk_search import chunk_text
from langgraph.prebuilt import ToolNode, tools_condition

text = open("sample.txt", encoding="utf-8").read()
chunks = chunk_text(text)

@tool
def get_doc_stats() -> dict:
    """返回文档的总字数和切块数量"""
    return {"total_chars": len(text), "num_chunks": len(chunks)}

@tool
def get_chunk_lengths() -> dict:
    """返回每个文本块的长度列表"""
    return {"lengths": [len(c) for c in chunks]}

@tool
def count_keyword(keyword: str) -> dict:
    """统计某个关键词在文档中出现的次数"""
    return {"keyword": keyword, "count": text.count(keyword)}

tools = [get_doc_stats, get_chunk_lengths, count_keyword]
llm_with_tools = llm.bind_tools(tools)


def model_node(state: State):
    reply = llm_with_tools.invoke(state["messages"])
    return {"messages": [reply]}

# graph = StateGraph(State)
# graph.add_node("model", model_node)
# graph.add_edge(START, "model")
# graph.add_edge("model", END)
# app = graph.compile()

from langgraph.prebuilt import ToolNode, tools_condition
# model_node 干的事，就是循环里「调模型 + append」那一步
graph = StateGraph(State)
graph.add_node("model", model_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "model")
graph.add_conditional_edges("model", tools_condition)
# tools_condition 就是你那个 if 的官方版：看最后一条消息，有 tool_calls 就去 "tools"，没有就去 END。
graph.add_edge("tools", "model")
app = graph.compile()


result = app.invoke({"messages": [HumanMessage("这篇文档一共多少字、切了几块？每块多长？RAG 出现几次？")]})
last = result["messages"][-1]
print("content:", repr(last.content))
print("tool_calls:", last.tool_calls)


from langgraph.checkpoint.memory import MemorySaver

app = graph.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "dong-1"}, "recursion_limit": 10}


r1 = app.invoke({"messages": [HumanMessage("这篇文档一共多少字、切了几块？RAG 出现几次？")]}, config)
print("第一轮:", r1["messages"][-1].content)

r2 = app.invoke({"messages": [HumanMessage("我刚才第一个问题问了你什么？")]}, config)
print("第二轮:", r2["messages"][-1].content)


