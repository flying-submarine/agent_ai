import asyncio, os
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(model="deepseek-flash", base_url="https://api.deepseek.com",
                 api_key=os.environ["DEEPSEEK_API_KEY"], temperature=0)

async def main():
    client = MultiServerMCPClient({
        "doc": {"command": "python3", "args": ["mcp_server.py"], "transport": "stdio"}
    })
    tools = await client.get_tools()
    print("从 MCP 拿到的工具:", [t.name for t in tools])

    llm_with_tools = llm.bind_tools(tools)

    def model_node(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph = StateGraph(State)
    graph.add_node("model", model_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")
    app = graph.compile()

    result = await app.ainvoke({"messages": [HumanMessage("这篇文档一共多少字、切了几块？每块多长？RAG 出现几次？")]})
    print("最终回答:", result["messages"][-1].content)

asyncio.run(main())


