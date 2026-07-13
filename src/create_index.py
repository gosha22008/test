from qdrant_client import QdrantClient, models

from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode

from llama_index.vector_stores.qdrant import QdrantVectorStore

from embedding_documents import get_embedding_model


COLLECTION_NAME = "documents"


def create_index(
    nodes: list[BaseNode],
) -> VectorStoreIndex:
    """
    Создает индекс и сохраняет его в Qdrant.

    Args:
        nodes: Список подготовленных чанков.

    Returns:
        Индекс LlamaIndex.
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
        vector_store=vector_store,
    )

    embed_model=get_embedding_model() # получаем embed модель
    vector_size = len(embed_model.get_text_embedding("test")) # получаем vector size

    collections = client.get_collections().collections

    if COLLECTION_NAME not in [c.name for c in collections]:
        client.create_collection(collection_name=COLLECTION_NAME,
                                 vectors_config=models.VectorParams(
                                     size=vector_size,      
                                     distance=models.Distance.COSINE))

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    print(f"Indexed {len(nodes)} chunks.")

    return index


if __name__ == "__main__":
    from load_documents import load_documents
    from chunk_documents import split_documents

    documents = load_documents("./data/docs")

    nodes = split_documents(documents)

    create_index(nodes)