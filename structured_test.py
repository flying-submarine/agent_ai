import config  # noqa: F401 —— 自动加载 .env 里的密钥（见 config.py）
"""第 8 课第 4 步：结构化输出 + 校验。先在小脚本里验，再搬进 app.py。

跑法（在 ~/Desktop/day1 里，venv 已激活、key 已 export）：
    python3 structured_test.py

两道实验题：
  1. trap-01（文档里没答案）：看模型会不会老实 refused
  2. rag-def-01（文档里有答案）：看正常回答能不能通过校验

背景（已查证）：DeepSeek 只支持 response_format json_object，
不支持 json_schema 严格模式——只保证输出是合法 JSON，不保证字段和类型。
所以类型把关只能靠两处：提示词里给例子 + 代码里宽容校验、归一化。
"""
import os
import re

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]


def _to_confidence(v):
    """模型常写 "high"、null、"0.9" 这种人话，统一归一化成 0-1 的数字。

    high/medium/low → 0.9/0.5/0.2 是拍板定的映射，不是模型的意思；
    归一化规则要写进代码、写进文档，不能藏在心里。
    """
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("high", "高"):
            return 0.9
        if s in ("medium", "中"):
            return 0.5
        if s in ("low", "低"):
            return 0.2
        try:
            return float(s)
        except ValueError:
            return 0.0
    return 0.0


def _to_chunk_id(v):
    """块号只认数字："块 0"、"chunk 1"、"0" 都取里面的数字。"""
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        m = re.search(r"\d+", v)
        if m:
            return int(m.group())
    raise ValueError(f"解析不出块号：{v!r}")


class RagAnswer(BaseModel):
    """模型必须按这个结构回答案。

    细扣：answer/sources/confidence 都给了默认值——拒绝回答时它们本来就没意义；
    多余字段直接丢掉；confidence 和 sources 进来先归一化再校验。
    """
    model_config = ConfigDict(extra="ignore")

    answer: str = Field(default="", description="用检索到的资料回答问题")
    sources: list[int] = Field(default=[], description="答案依据的块编号，只许填检索给你的块")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="0 到 1，对答案的把握")
    refused: bool = Field(default=False, description="资料里没有答案时填 true")

    @field_validator("confidence", mode="before")
    @classmethod
    def _norm_confidence(cls, v):
        return _to_confidence(v)

    @field_validator("sources", mode="before")
    @classmethod
    def _norm_sources(cls, v):
        if isinstance(v, list):
            return [_to_chunk_id(x) for x in v]
        return v


client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# 例子只讲格式不讲答案：拿切分那句当例子，两道实验题的答案都没泄露
EXAMPLE = '{"answer": "块太大噪声多，块太小上下文不完整。", "sources": [1], "confidence": 0.8, "refused": false}'


def ask(question, chunks):
    prompt = (
        "只用下面给的资料回答问题。资料里没有答案就 refused 填 true，"
        "其余字段空着就行。\n"
        + "\n".join(f"[块 {i}] {t}" for i, t in chunks.items())
        + f"\n问题：{question}"
    )
    resp = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": "你是一个 RAG 问答助手。只输出 JSON，字段只能是 answer（字符串）、sources（块号组成的数字数组，如 [0]）、confidence（0 到 1 的数字）、refused（true 或 false），不要输出其他字段和文字。格式例子：" + EXAMPLE},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = resp.choices[0].message.content
    print("模型原样返回：", raw)
    try:
        parsed = RagAnswer.model_validate_json(raw)
        print("校验通过：refused =", parsed.refused, "| sources =", parsed.sources,
              "| confidence =", parsed.confidence, "| answer =", parsed.answer)
    except ValidationError as e:
        print("校验失败：", e)
    print()


# 实验 1：trap-01，文档里只有名字，从没比较过性能
ask("Milvus 和 Qdrant 哪个性能更好？", {
    1: "常见的向量数据库有 Milvus、Qdrant 和 pgvector。检索环节除了向量检索，还经常配合关键词检索做混合检索，再用重排序模型把结果精排一遍。",
    2: "drant 和 pgvector。检索环节除了向量检索，还经常配合关键词检索做混合检索，再用重排序模型把结果精排一遍。最后，把排在前面的几个片段连同用户的问题一起组装成提示词，交给大模型生成答案，并且在答案里标出引用来源，这就是引用溯源。",
})

# 实验 2：rag-def-01，文档里有答案
ask("RAG 是什么的缩写？", {
    0: "RAG 是检索增强生成的缩写，它的核心思想是：在让大模型回答问题之前，先从外部知识库里检索出相关的资料，把资料作为上下文一起喂给模型。这样做可以解决大模型的两个问题：一是模型不知道企业内部的私有数据，二是模型容易一本正经地编造，也就是所谓的幻觉。",
})
