from llama_index.core.node_parser import SentenceSplitter, HierarchicalNodeParser
from llama_index.core.schema import BaseNode, Document


def split_documents(
    documents: list[Document],
    chunk_size: list[int] = [1024, 512, 256],
    chunk_overlap: int = 100,
) -> list[BaseNode]:
    """
    Разбивает документы на чанки.

    Args:
        documents: Список документов LlamaIndex.
        chunk_size: Максимальный размер чанка (в токенах).
        chunk_overlap: Размер перекрытия между соседними чанками.

    Returns:
        Список чанков (Node).
    """

    splitter = HierarchicalNodeParser.from_defaults(
        chunk_sizes=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    nodes = splitter.get_nodes_from_documents(documents)

    print(f"Created {len(nodes)} chunks.")

    return nodes


if __name__ == "__main__":
    from load_documents import load_documents

    documents = load_documents("./data/docs")

    nodes = split_documents(documents)
    print(nodes[0])