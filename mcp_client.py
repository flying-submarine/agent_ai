import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command="python3", args=["mcp_server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("=== server 递出的菜单 ===")
            for t in tools.tools:
                print("-", t.name, "|", t.description, "| 参数表:", t.inputSchema)

            r = await session.call_tool("count_keyword", {"keyword": "RAG"})
            print("=== 调用 count_keyword 的结果 ===")
            print(r.content[0].text)

asyncio.run(main())


