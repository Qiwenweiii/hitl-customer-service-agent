import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"
CHROMA_DIR = BASE_DIR / "data" / "chroma"

# RAG 技术参数：换 Embedding 模型必须重建整个知识库（新旧向量不在同一空间）
CHROMA_COLLECTION = "customer_service_knowledge"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
# 先粗排召回 RETRIEVAL_K 条，重排后只把前 CONTEXT_TOP_K 条塞进上下文，控制 token 与噪声
RETRIEVAL_K = 10
CONTEXT_TOP_K = 3

# 退款时效：自物流签收**次日**起 N 个自然日（含周末与法定节假日），口径须与
# data/knowledge/refund_policy.md 第二章第 1 条「7 天无理由退货」一致；改这个数字必须同步改政策文档，
# 否则 Agent 检索到的政策会与工具判定互相矛盾。边界：now.date() <= delivered_at.date() + N 天。
REFUND_WINDOW_DAYS = 7

load_dotenv(BASE_DIR / ".env")

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
# 显式关闭 GLM-4.5 系列的动态思考（该系列默认开启）。必须从源头关掉：这个端点会把思考正文
# 连同正式回答一起放进响应的 content 字段，而不是独立的 reasoning_content，
# 所以 ChatOpenAI 无法分离它，用户侧就会看到 think 闭合标签和重复了两遍的回答。
LLM_EXTRA_BODY = {"thinking": {"type": "disabled"}}
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "customer-service-agent")
