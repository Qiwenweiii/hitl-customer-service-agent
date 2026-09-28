"""RAG 检索服务：Chroma 向量召回 -> CrossEncoder 重排 -> 相关性过滤 -> 拼装上下文。

本模块只负责"取资料"，不生成答案：最终回答由 Agent 的大模型完成，
因此 search() 全程不依赖 LLM_API_KEY，也不导入任何 Chat 模型。

向量库与 Reranker 都做成惰性属性：`import app.tools.rag_tools` 时不加载
sentence-transformers 模型，避免拖慢 Agent 启动；测试也可直接注入假组件。
"""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from app.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CONTEXT_TOP_K,
    RERANKER_MODEL,
    RETRIEVAL_K,
)
from app.rag.knowledge_base import build_vector_store, create_default_embeddings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

RerankedDocument = tuple[float, Document]

# 相关性底线：实测 bge-reranker-base 上，政策类有效命中分数 0.19–0.99，
# 而完全无关的问题（如问天气）全部候选都是 0.00，两者在这条线上分得很开。
MIN_RERANK_SCORE = 0.1


def create_default_reranker() -> "CrossEncoder":
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name_or_path=RERANKER_MODEL)


class RAGService:
    def __init__(
        self,
        *,
        embeddings: Embeddings | None = None,
        vector_store: Chroma | None = None,
        reranker: "CrossEncoder | None" = None,
        persist_dir: Path = CHROMA_DIR,
        collection_name: str = CHROMA_COLLECTION,
        retrieval_k: int = RETRIEVAL_K,
        context_top_k: int = CONTEXT_TOP_K,
        min_rerank_score: float = MIN_RERANK_SCORE,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._reranker = reranker
        self._retriever: BaseRetriever | None = None
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.retrieval_k = retrieval_k
        self.context_top_k = context_top_k
        self.min_rerank_score = min_rerank_score

    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            self._embeddings = create_default_embeddings()
        return self._embeddings

    @property
    def vector_store(self) -> Chroma:
        if self._vector_store is None:
            self._vector_store = build_vector_store(
                self.embeddings,
                persist_dir=self.persist_dir,
                collection_name=self.collection_name,
            )
        return self._vector_store

    @property
    def retriever(self) -> BaseRetriever:
        if self._retriever is None:
            self._retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.retrieval_k}
            )
        return self._retriever

    @property
    def reranker(self) -> "CrossEncoder":
        if self._reranker is None:
            self._reranker = create_default_reranker()
        return self._reranker

    def retrieve(self, query: str) -> list[Document]:
        """向量粗排，召回最多 retrieval_k 个候选 chunk。"""
        return self.retriever.invoke(query)

    def rerank(self, query: str, documents: Sequence[Document]) -> list[RerankedDocument]:
        """交叉编码器按 query-document 相关度重新排序，返回 (分数, 文档) 降序列表（不做阈值过滤）。"""
        if not documents:
            return []
        pairs = [[query, document.page_content] for document in documents]
        scores = self.reranker.predict(pairs)
        ranked = zip((float(score) for score in scores), documents, strict=True)
        return sorted(ranked, key=lambda item: item[0], reverse=True)

    def build_context(self, ranked_documents: Sequence[RerankedDocument], top_k: int | None = None) -> str:
        """取重排结果的前 top_k 条拼成上下文字符串。"""
        limit = self.context_top_k if top_k is None else top_k
        return "\n\n".join(document.page_content for _, document in ranked_documents[:limit])

    def search(self, query: str) -> str:
        """只检索、不调用 LLM 的完整链路：召回 -> 重排 -> 阈值过滤 -> 上下文。

        低于 min_rerank_score 的结果一律不进上下文；全部低于阈值时返回空字符串，
        由 Tool 层转成"未检索到"提示，避免把无关政策当成答案依据。
        """
        normalized = query.strip()
        if not normalized:
            return ""
        relevant = [
            (score, document)
            for score, document in self.rerank(normalized, self.retrieve(normalized))
            if score >= self.min_rerank_score
        ]
        if not relevant:
            return ""
        return self.build_context(relevant)
