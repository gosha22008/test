import os

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore

from create_index import DOCSTORE_PATH, get_hybrid_vector_store
from embedding_documents import get_embedding_model


def get_retriever(similarity_top_k: int = 5, sparse_top_k: int = 10):
    """
    Ретривер: гибридный поиск (dense+sparse) + AutoMerging.

    Поток:
      1. Гибридный поиск по листьям в Qdrant: sparse_top_k кандидатов из
         каждого пространства (dense/sparse) -> фьюжн -> similarity_top_k.
      2. AutoMergingRetriever: если в выдаче много листьев одного родителя,
         заменяет их родительским чанком из docstore (полный контекст вместо
         обрывков). docstore грузится с диска (его пишет create_index).

    similarity_top_k=5 — компромисс: достаточно для синтеза из разных
    документов, но не разбавляет промпт мусором.

    Raises:
        FileNotFoundError: если индексация ещё не выполнялась (нет docstore).
    """
    if not os.path.exists(DOCSTORE_PATH):
        raise FileNotFoundError(
            f"Docstore не найден ({DOCSTORE_PATH}). "
            f"Сначала выполните индексацию: python create_index.py"
        )

    vector_store = get_hybrid_vector_store()
    docstore = SimpleDocumentStore.from_persist_path(DOCSTORE_PATH)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore,
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=get_embedding_model(),
        storage_context=storage_context,
    )

    base_retriever = index.as_retriever(
        similarity_top_k=similarity_top_k,
        sparse_top_k=sparse_top_k,
        vector_store_query_mode="hybrid",
    )

    return AutoMergingRetriever(base_retriever, storage_context, verbose=False)


if __name__ == "__main__":
    retriever = get_retriever()

    query = "что такое баланс?"
    nodes = retriever.retrieve(query)

    print(f"Найдено чанков: {len(nodes)}\n")
    for i, node in enumerate(nodes, start=1):
        print(f"Чанк №{i}")
        print(f"Источник: {node.metadata.get('file_name')}")
        print(f"Страница: {node.metadata.get('page_label')}")
        print(node.text)
        print("-" * 80)
