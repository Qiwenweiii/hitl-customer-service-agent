"""知识库构建测试：加载 -> 切分 -> 入库 -> 幂等。全部使用假向量，不加载 bge 模型、不调用 LLM。"""

from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.config import BASE_DIR, CHROMA_DIR, KNOWLEDGE_DIR
from app.rag.knowledge_base import (
    CHUNK_SIZE,
    build_knowledge_base,
    build_vector_store,
    list_knowledge_files,
    load_documents,
    split_documents,
)

FAKE_EMBEDDINGS = DeterministicFakeEmbedding(size=64)
REQUIRED_TOPICS = ("退款", "物流", "售后", "会员")


@pytest.fixture()
def collection_name(request) -> str:
    return f"kb_test_{request.node.name}"


@pytest.fixture()
def knowledge_files() -> list[Path]:
    files = list_knowledge_files()
    assert files, "data/knowledge/ 下没有可入库的文档"
    return files


def test_knowledge_dir_covers_required_topics(knowledge_files: list[Path]):
    assert len(knowledge_files) >= 4
    combined = "".join(path.read_text(encoding="utf-8") for path in knowledge_files)
    for topic in REQUIRED_TOPICS:
        assert topic in combined, f"知识库缺少主题：{topic}"


def test_each_knowledge_doc_is_substantial(knowledge_files: list[Path]):
    """知识库文档要能支撑真实问答，不能只有几条占位数据。"""
    for path in knowledge_files:
        text = path.read_text(encoding="utf-8")
        assert len(text) >= 800, f"{path.name} 内容过短（{len(text)} 字）"
        assert text.count("\n## ") >= 4, f"{path.name} 章节结构过少"


def test_load_documents_sets_identifying_metadata(knowledge_files: list[Path]):
    documents = load_documents(knowledge_files)

    assert len(documents) == len(knowledge_files)
    for document, path in zip(documents, knowledge_files, strict=True):
        assert document.metadata["document_id"] == path.name
        assert document.metadata["source"] == str(path)
        assert document.page_content.startswith("# ")


def test_split_documents_produces_bounded_chunks_with_stable_ids(knowledge_files: list[Path]):
    chunks = split_documents(load_documents(knowledge_files))

    assert len(chunks) > len(knowledge_files)
    assert all(len(chunk.page_content) <= CHUNK_SIZE for chunk in chunks)
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == len(chunks)

    first = chunks[0].metadata
    assert first["chunk_index"] == 1
    assert first["chunk_id"] == f"{Path(first['document_id']).stem}-1"


def test_build_knowledge_base_persists_all_chunks(tmp_path: Path, collection_name: str):
    persist_dir = tmp_path / "chroma"

    stats = build_knowledge_base(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embeddings=FAKE_EMBEDDINGS,
    )

    assert persist_dir.exists()
    assert stats["chunks"] > 0
    assert stats["vectors"] == stats["chunks"]

    store = build_vector_store(
        FAKE_EMBEDDINGS, persist_dir=persist_dir, collection_name=collection_name
    )
    stored_ids = store.get()["ids"]
    assert sorted(stored_ids) == sorted(
        chunk.metadata["chunk_id"] for chunk in split_documents(load_documents(list_knowledge_files()))
    )


def test_build_knowledge_base_rerun_does_not_duplicate(tmp_path: Path, collection_name: str):
    """重复执行只替换旧 chunk，不累积重复数据。"""
    persist_dir = tmp_path / "chroma"
    args = {
        "persist_dir": persist_dir,
        "collection_name": collection_name,
        "embeddings": FAKE_EMBEDDINGS,
    }

    first = build_knowledge_base(**args)
    second = build_knowledge_base(**args)

    assert first["chunks"] == second["chunks"]
    assert second["vectors"] == first["vectors"]
    assert second["replaced"] == first["chunks"]


def test_build_knowledge_base_fails_loudly_on_empty_dir(tmp_path: Path, collection_name: str):
    empty_dir = tmp_path / "knowledge"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        build_knowledge_base(
            knowledge_dir=empty_dir,
            persist_dir=tmp_path / "chroma",
            collection_name=collection_name,
            embeddings=FAKE_EMBEDDINGS,
        )


def test_rag_paths_are_base_dir_relative(tmp_path: Path):
    """持久化目录必须走 BASE_DIR 统一配置，不能依赖进程当前工作目录。"""
    assert CHROMA_DIR == BASE_DIR / "data" / "chroma"
    assert KNOWLEDGE_DIR == BASE_DIR / "data" / "knowledge"
    assert CHROMA_DIR.is_absolute() and KNOWLEDGE_DIR.is_absolute()
