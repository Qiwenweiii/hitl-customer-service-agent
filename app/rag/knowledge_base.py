"""知识库构建：data/knowledge/ -> 加载 -> 切分 -> Embedding -> Chroma 持久化。

这里只负责"入库"，检索与问答由后续 RAGService 承担。
Embedding 模型在函数内部延迟导入：sentence-transformers 首次导入需十几秒，
测试可注入轻量假向量，不必加载真实模型。
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHROMA_COLLECTION, CHROMA_DIR, EMBEDDING_MODEL, KNOWLEDGE_DIR

KNOWLEDGE_SUFFIXES = (".md", ".txt")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 60

# 中文政策文档以段落与句号为自然边界，只按 \\n\\n 切会把整节塞进一个 chunk 并硬截断句子。
SEPARATORS = ["\n\n", "\n", "。", "，", ""]


def list_knowledge_files(knowledge_dir: Path = KNOWLEDGE_DIR) -> list[Path]:
    return sorted(
        path
        for path in knowledge_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in KNOWLEDGE_SUFFIXES
    )


def load_documents(files: list[Path]) -> list[Document]:
    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={
                "document_id": path.name,
                "source": str(path),
                "title": path.stem,
            },
        )
        for path in files
    ]


def split_documents(documents: list[Document]) -> list[Document]:
    """切分为 chunk，并补上文档内序号与稳定 ID（同一文档重复构建时 ID 不变）。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    counters: dict[str, int] = {}
    chunks: list[Document] = []
    for chunk in splitter.split_documents(documents):
        text = chunk.page_content.strip()
        if not text:
            continue
        chunk.page_content = text
        document_id = chunk.metadata["document_id"]
        counters[document_id] = counters.get(document_id, 0) + 1
        index = counters[document_id]
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = f"{Path(document_id).stem}-{index}"
        chunks.append(chunk)
    return chunks


def create_default_embeddings() -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vector_store(
    embeddings: Embeddings,
    persist_dir: Path = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION,
) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def sync_document(vector_store: Chroma, document_id: str, chunks: list[Document]) -> int:
    """按源文档先删后写：重复执行只会得到同一批 chunk，不会累积重复数据。"""
    stale_ids = vector_store.get(where={"document_id": document_id}).get("ids") or []
    if stale_ids:
        vector_store.delete(ids=list(stale_ids))
    vector_store.add_documents(
        documents=chunks,
        ids=[chunk.metadata["chunk_id"] for chunk in chunks],
    )
    return len(stale_ids)


def count_stored_chunks(vector_store: Chroma) -> int:
    """统计 collection 中实际落库的 chunk 数（走 Chroma 公开的 get 接口）。

    不能再额外开一个 chromadb.PersistentClient：同一进程内对同一路径使用不同
    Settings 会被 chromadb 判定为冲突实例。
    """
    return len(vector_store.get().get("ids") or [])


def group_by_document(chunks: list[Document]) -> dict[str, list[Document]]:
    grouped: dict[str, list[Document]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.metadata["document_id"], []).append(chunk)
    return grouped


def build_knowledge_base(
    knowledge_dir: Path = KNOWLEDGE_DIR,
    persist_dir: Path = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION,
    embeddings: Embeddings | None = None,
) -> dict:
    files = list_knowledge_files(knowledge_dir)
    if not files:
        raise FileNotFoundError(f"知识库目录中没有可入库的文档：{knowledge_dir}")

    chunks = split_documents(load_documents(files))
    vector_embeddings = embeddings or create_default_embeddings()
    vector_store = build_vector_store(
        vector_embeddings, persist_dir=persist_dir, collection_name=collection_name
    )

    replaced = 0
    for document_id, document_chunks in group_by_document(chunks).items():
        replaced += sync_document(vector_store, document_id, document_chunks)

    # 重新打开一次向量库，确认数据真的持久化到了磁盘而不是只在内存里
    persisted_store = build_vector_store(
        vector_embeddings, persist_dir=persist_dir, collection_name=collection_name
    )

    return {
        "files": len(files),
        "chunks": len(chunks),
        "replaced": replaced,
        "vectors": count_stored_chunks(persisted_store),
        "persist_dir": str(persist_dir),
        "collection": collection_name,
    }
