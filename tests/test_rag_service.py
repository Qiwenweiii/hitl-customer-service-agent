"""RAGService 与 search_knowledge 测试。

全部注入假向量与假 Reranker：不加载 bge 模型，不调用真实 LLM，不依赖网络。
"""

import ast
import inspect
from collections.abc import Callable

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.config import CONTEXT_TOP_K, RETRIEVAL_K
from app.rag import rag_service as rag_service_module
from app.rag.knowledge_base import (
    build_knowledge_base,
    build_vector_store,
    list_knowledge_files,
    load_documents,
    split_documents,
)
from app.rag.rag_service import MIN_RERANK_SCORE, RAGService
from app.tools import rag_tools
from app.tools.rag_tools import search_knowledge

FAKE_EMBEDDINGS = DeterministicFakeEmbedding(size=64)


class FakeReranker:
    """按内容长度打分的交叉编码器替身，记录每次 predict 的入参。"""

    def __init__(self, score_of: Callable[[list[str]], float] | None = None) -> None:
        self.calls: list[list[list[str]]] = []
        self._score_of = score_of or (lambda pair: float(len(pair[1])))

    def predict(self, pairs: list[list[str]]) -> list[float]:
        self.calls.append(pairs)
        return [self._score_of(pair) for pair in pairs]


class StubService:
    """search_knowledge 的服务替身：记录 query 并返回预设上下文。"""

    def __init__(self, context: str) -> None:
        self.context = context
        self.queries: list[str] = []

    def search(self, query: str) -> str:
        self.queries.append(query)
        return self.context


def make_doc(index: int, content: str) -> Document:
    return Document(
        page_content=content,
        metadata={
            "document_id": "refund_policy.md",
            "source": "/data/knowledge/refund_policy.md",
            "chunk_id": f"refund_policy-{index}",
        },
    )


FILTER_DOCS = [make_doc(1, "片段甲"), make_doc(2, "片段乙"), make_doc(3, "片段丙")]


def build_scoring_service(monkeypatch, scores: dict[str, float], **kwargs) -> RAGService:
    """按内容查表打分的 RAGService，retrieve 固定返回 FILTER_DOCS（不打开 Chroma）。"""
    service = RAGService(reranker=FakeReranker(lambda pair: scores[pair[1]]), **kwargs)
    monkeypatch.setattr(service, "retrieve", lambda query: list(FILTER_DOCS))
    return service


def imported_names(module) -> set[str]:
    """收集模块的导入符号，用于分层守卫（比文本匹配可靠，不会被注释误伤）。"""
    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


@pytest.fixture(scope="session")
def corpus() -> list[Document]:
    return split_documents(load_documents(list_knowledge_files()))


@pytest.fixture(scope="session")
def store(tmp_path_factory):
    """真实知识库 + 假向量入库一次，供各测试共享（入库比断言慢，重复建没必要）。"""
    persist_dir = tmp_path_factory.mktemp("rag_chroma")
    build_knowledge_base(
        persist_dir=persist_dir,
        collection_name="rag_service_test",
        embeddings=FAKE_EMBEDDINGS,
    )
    return build_vector_store(
        FAKE_EMBEDDINGS, persist_dir=persist_dir, collection_name="rag_service_test"
    )


@pytest.fixture()
def service(store) -> RAGService:
    """每个测试一个全新 RAGService：共享向量库，但 Reranker 调用记录互不污染。"""
    return RAGService(embeddings=FAKE_EMBEDDINGS, vector_store=store, reranker=FakeReranker())


def test_retrieve_returns_stored_chunks_within_k(service: RAGService):
    documents = service.retrieve("退款到账要多久")

    assert 0 < len(documents) <= RETRIEVAL_K
    assert all(doc.metadata.get("chunk_id") for doc in documents)
    assert all(doc.page_content.strip() for doc in documents)


def test_retriever_k_comes_from_service_config(service: RAGService):
    strict = RAGService(vector_store=service.vector_store, reranker=FakeReranker(), retrieval_k=2)

    assert len(strict.retrieve("运费规则")) <= 2 < len(service.retrieve("运费规则"))


def test_rerank_sorts_by_score_descending(service: RAGService):
    documents = [make_doc(1, "短"), make_doc(2, "中等长度片段"), make_doc(3, "最长最完整的政策片段内容")]

    ranked = service.rerank("退款", documents)

    scores = [score for score, _ in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(score, float) for score in scores)
    assert [doc.page_content for _, doc in ranked] == [documents[2].page_content,
                                                      documents[1].page_content,
                                                      documents[0].page_content]


def test_rerank_feeds_query_with_each_candidate():
    reranker = FakeReranker()

    RAGService(reranker=reranker).rerank("会员权益", [make_doc(1, "片段一"), make_doc(2, "片段二")])

    assert reranker.calls == [[["会员权益", "片段一"], ["会员权益", "片段二"]]]


def test_rerank_without_candidates_skips_model():
    reranker = FakeReranker()

    assert RAGService(reranker=reranker).rerank("退款", []) == []
    assert reranker.calls == []


def test_build_context_keeps_only_top_ranked_chunks(service: RAGService):
    ranked = [
        (float(CONTEXT_TOP_K + 1 - index), make_doc(index, f"片段{index}"))
        for index in range(1, CONTEXT_TOP_K + 2)
    ]
    expected = "\n\n".join(f"片段{index}" for index in range(1, CONTEXT_TOP_K + 1))

    assert service.build_context(ranked) == expected
    assert service.build_context(ranked, top_k=1) == "片段1"


def test_search_runs_retrieve_then_rerank_without_llm(service: RAGService):
    """search() 只做检索与重排：服务层没有模型对象，也不需要任何大模型凭据。"""
    assert not hasattr(service, "llm")

    context = service.search("退款到账要多久")

    assert context.strip()
    candidates = service.reranker.calls[0]
    assert 0 < len(candidates) <= RETRIEVAL_K
    # 上下文只取重排后的前 CONTEXT_TOP_K 条，必然短于全部候选拼接
    assert len(context) < sum(len(pair[1]) for pair in candidates)


def test_search_context_comes_from_indexed_knowledge(service: RAGService, corpus: list[Document]):
    stored = [chunk.page_content for chunk in corpus]

    context = service.search("会员等级 积分 退换货 配送时效")

    assert any(text in context for text in stored)
    assert all(
        any(piece in text for text in stored) for piece in context.split("\n\n")
    ), "上下文必须全部来自入库 chunk，不能凭空生成"


def test_search_ignores_blank_query(service: RAGService):
    assert service.search("   ") == ""
    assert service.reranker.calls == []


def test_search_returns_context_when_all_scores_meet_threshold(monkeypatch):
    service = build_scoring_service(monkeypatch, {"片段甲": 0.92, "片段乙": 0.31, "片段丙": 0.19})

    assert service.search("退款到账要多久") == "片段甲\n\n片段乙\n\n片段丙"


def test_search_returns_empty_when_all_scores_below_threshold(monkeypatch):
    service = build_scoring_service(monkeypatch, {"片段甲": 0.0, "片段乙": 0.05, "片段丙": 0.09})

    assert service.search("今天天气怎么样") == ""
    assert len(service.reranker.calls) == 1, "应确认走到了重排，而不是检索提前落空"


def test_search_drops_only_subthreshold_documents(monkeypatch):
    service = build_scoring_service(monkeypatch, {"片段甲": 0.92, "片段乙": 0.04, "片段丙": 0.31})

    assert service.search("退款") == "片段甲\n\n片段丙"


def test_rerank_itself_does_not_filter(monkeypatch):
    """阈值只作用于 search()：rerank() 仍返回全部候选与原始分数。"""
    service = build_scoring_service(monkeypatch, {"片段甲": 0.92, "片段乙": 0.04, "片段丙": 0.0})

    ranked = service.rerank("退款", FILTER_DOCS)

    assert [round(score, 2) for score, _ in ranked] == [0.92, 0.04, 0.0]
    assert [doc.page_content for _, doc in ranked] == ["片段甲", "片段乙", "片段丙"]


def test_threshold_defaults_to_constant_and_is_overridable(monkeypatch):
    assert RAGService().min_rerank_score == MIN_RERANK_SCORE

    lenient = build_scoring_service(
        monkeypatch, {"片段甲": 0.92, "片段乙": 0.04, "片段丙": 0.0}, min_rerank_score=0.0
    )
    assert lenient.search("退款") == "片段甲\n\n片段乙\n\n片段丙"


def test_models_load_lazily_and_only_when_needed(tmp_path, monkeypatch):
    """构造 RAGService 不能加载模型，否则 Agent 启动会被 sentence-transformers 拖慢十几秒。"""
    loaded: list[str] = []
    monkeypatch.setattr(
        rag_service_module,
        "create_default_embeddings",
        lambda: loaded.append("embeddings") or FAKE_EMBEDDINGS,
    )
    monkeypatch.setattr(
        rag_service_module,
        "create_default_reranker",
        lambda: loaded.append("reranker") or FakeReranker(),
    )
    docs = [make_doc(1, "片段一"), make_doc(2, "片段二")]
    service = RAGService(persist_dir=tmp_path / "chroma", collection_name="rag_test_lazy")

    assert loaded == []

    assert service.rerank("退款", docs)
    assert loaded == ["reranker"]

    assert service.retrieve("退款") == []
    service.retrieve("退款")
    assert loaded == ["reranker", "embeddings"]


def test_rag_layers_never_import_or_call_an_llm():
    """需求：RAG 只取资料，最终回答由 Agent 的大模型完成，本层不得引入模型。"""
    forbidden = {"langchain_openai", "openai", "ChatOpenAI", "init_chat_model", "build_chat_model"}
    for module in (rag_service_module, rag_tools):
        hits = imported_names(module) & forbidden
        assert not hits, f"{module.__name__} 引入了大模型依赖：{hits}"


def test_tool_layer_only_depends_on_rag_service():
    """Tool 层不直接操作向量库与模型，保持 Tool -> Service -> 组件 的分层。"""
    forbidden = {
        "langchain_chroma",
        "chromadb",
        "sentence_transformers",
        "langchain_huggingface",
        "Chroma",
        "HuggingFaceEmbeddings",
        "CrossEncoder",
        "knowledge_base",
    }
    assert not imported_names(rag_tools) & forbidden
    assert "as_retriever" not in inspect.getsource(rag_tools)


def test_search_knowledge_tool_returns_indexed_content(monkeypatch):
    stub = StubService("退款将于 1-3 个工作日内到账。")
    monkeypatch.setattr(rag_tools, "_rag_service", stub)

    output = search_knowledge.invoke({"query": "退款到账要多久"})

    assert stub.queries == ["退款到账要多久"]
    assert "以下是知识库中与问题相关的内容" in output
    assert "退款将于 1-3 个工作日内到账。" in output


def test_search_knowledge_tool_reports_miss_without_fabricating(monkeypatch):
    stub = StubService("")
    monkeypatch.setattr(rag_tools, "_rag_service", stub)

    output = search_knowledge.invoke({"query": "有没有免费停车"})

    assert stub.queries == ["有没有免费停车"]
    assert "未检索到" in output
    assert "不要" in output


def test_search_knowledge_tool_signature():
    assert search_knowledge.name == "search_knowledge"
    assert "query" in search_knowledge.args
    assert "query_order" in search_knowledge.description
    assert "query_logistics" in search_knowledge.description
