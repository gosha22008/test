import os

from qdrant_client import AsyncQdrantClient, QdrantClient

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore

from embedding_documents import get_embedding_model


COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")
DOCSTORE_PATH = os.path.join(STORAGE_DIR, "docstore.json")


def get_hybrid_vector_store() -> QdrantVectorStore:
    """
    Vector store с ГИБРИДНЫМ поиском (замечание ментора №3).

    enable_hybrid=True + fastembed_sparse_model="Qdrant/bm25": Qdrant хранит
    ДВА вектора на чанк — dense (bge-m3, семантика) и sparse (BM25, точные
    термины/номера пунктов) — и объединяет результаты фьюжном на своей
    стороне. Отдельный BM25-ретривер и ансамбль руками не нужны.

    Пропорцию dense/sparse задаёт фьюжн Qdrant (по умолчанию RRF —
    Reciprocal Rank Fusion, примерно равный вклад); на этапе запроса можно
    крутить sparse_top_k / similarity_top_k (см. retriever.py).
    """
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    aclient = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return QdrantVectorStore(
        client=client,
        aclient=aclient,
        collection_name=COLLECTION_NAME,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
    )


def create_index(all_nodes: list[BaseNode], leaf_nodes: list[BaseNode]) -> VectorStoreIndex:
    """
    Индексация для AutoMerging + гибридного поиска.

      1. Листья -> векторы в Qdrant (dense + sparse).
      2. ВСЕ узлы -> docstore (нужны для схлопывания в AutoMergingRetriever).
      3. docstore -> на диск (STORAGE_DIR), чтобы FastAPI-процесс (отдельный
         от индексации) смог его загрузить.

    Коллекцию с нужной схемой (два именованных вектора) создаёт сам
    QdrantVectorStore при первой записи — вручную создавать не нужно.
    """
    vector_store = get_hybrid_vector_store()

    docstore = SimpleDocumentStore()
    docstore.add_documents(all_nodes)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore,
    )

    index = VectorStoreIndex(
        leaf_nodes,                       # в векторный индекс — ТОЛЬКО листья
        storage_context=storage_context,
        embed_model=get_embedding_model(),
        show_progress=True,
    )

    os.makedirs(STORAGE_DIR, exist_ok=True)
    docstore.persist(persist_path=DOCSTORE_PATH)

    print(f"Проиндексировано листьев: {len(leaf_nodes)}; "
          f"узлов в docstore: {len(all_nodes)} -> {DOCSTORE_PATH}")
    return index


if __name__ == "__main__":
    from load_documents import load_documents
    from chunk_documents import split_documents

    documents = load_documents()
    all_nodes, leaf_nodes = split_documents(documents)
    create_index(all_nodes, leaf_nodes)
