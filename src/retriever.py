from qdrant_client import QdrantClient

from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

from embedding_documents import get_embedding_model


COLLECTION_NAME = "documents"


def get_retriever(similarity_top_k: int = 3):
    """
    Создает retriever для поиска наиболее релевантных документов.

    Args:
        similarity_top_k: Количество возвращаемых документов.

    Returns:
        Экземпляр Retriever.
    """

    client = QdrantClient(
        host="localhost",
        port=6333,
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=get_embedding_model(),
    )

    return index.as_retriever(
        similarity_top_k=similarity_top_k
    )


if __name__ == "__main__":

    retriever = get_retriever()

    query = "что такое баланс?"

    nodes = retriever.retrieve(query)

    print(f"Найдено документов: {len(nodes)}\n")

    for i, node in enumerate(nodes, start=1):
        print(f"Документ №{i}")
        print(f"Источник: {node.metadata.get('file_name')}")
        print(f"Страница: {node.metadata.get('page_label')}")
        print(node.text)
        print("-" * 80)