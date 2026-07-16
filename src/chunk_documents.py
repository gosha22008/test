from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import BaseNode, Document


def split_documents(
    documents: list[Document],
    chunk_sizes: list[int] = [1024, 512, 256],
) -> tuple[list[BaseNode], list[BaseNode]]:
    """
    Иерархическая разбивка на чанки для AutoMergingRetriever.

    HierarchicalNodeParser строит ДЕРЕВО узлов: родители (1024, 512) и
    листья (256), связанные отношениями parent/child. Возвращаем ОБА набора:

      - all_nodes  — все узлы дерева; кладутся в docstore. Нужны, чтобы при
                     поиске «схлопнуть» листья обратно в родительский чанк и
                     вернуть агенту полный контекст, а не обрывок.
      - leaf_nodes — только листья; ТОЛЬКО их индексируем как векторы в Qdrant.

    Почему только листья в векторах: если положить все три уровня, 
    один и тот же факт попадёт в индекс трижды на разных
    масштабах — и в топ-5 приедут дубли вместо пяти разных фактов.

    Args:
        documents: Документы LlamaIndex.
        chunk_sizes: Размеры уровней в токенах (от родителя к листу).

    Returns:
        (all_nodes, leaf_nodes)
    """
    parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)

    all_nodes = parser.get_nodes_from_documents(documents, show_progress=True)
    # Пустые листья (пустые страницы PDF) убираем только из тех, что пойдут
    # в векторный индекс — дерево отношений в all_nodes не трогаем.
    leaf_nodes = [n for n in get_leaf_nodes(all_nodes) if n.text.strip()]

    print(f"Всего узлов: {len(all_nodes)}, листовых: {len(leaf_nodes)}")
    return all_nodes, leaf_nodes


if __name__ == "__main__":
    from load_documents import load_documents

    documents = load_documents()
    all_nodes, leaf_nodes = split_documents(documents)
    print(leaf_nodes[0])
