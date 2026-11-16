from mcp.server.fastmcp import FastMCP
from chunk_search import chunk_text

mcp = FastMCP("doc-tools")

text = open("sample.txt", encoding="utf-8").read()
chunks = chunk_text(text)

@mcp.tool()
def get_doc_stats() -> dict:
    """返回文档的总字数和切块数量"""
    return {"total_chars": len(text), "num_chunks": len(chunks)}

@mcp.tool()
def get_chunk_lengths() -> dict:
    """返回每个文本块的长度列表"""
    return {"lengths": [len(c) for c in chunks]}

@mcp.tool()
def count_keyword(keyword: str) -> dict:
    """统计某个关键词在文档中出现的次数"""
    return {"keyword": keyword, "count": text.count(keyword)}

if __name__ == "__main__":
    mcp.run()

