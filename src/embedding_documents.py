from llama_index.embeddings.ollama import OllamaEmbedding


def get_embedding_model() -> OllamaEmbedding:
    """
    Создает экземпляр локальной embedding-модели Ollama.

    Returns:
        Настроенная модель эмбеддингов.
    """

    return OllamaEmbedding(
        model_name="bge-m3",
        base_url="http://localhost:11434",
    )


if __name__ == "__main__":
    from load_documents import load_documents
    from chunk_documents import split_documents

    embedding_model = get_embedding_model()

    documents = load_documents("./data/docs")

    nodes = split_documents(documents)
    nodes = [node for node in nodes if node.text.strip()]
    for i, node in enumerate(nodes):
        if not node.text.strip():
            print(f"Node {i} is empty")
            print(repr(node.text))

    texts = [node.text for node in nodes]
    embeddings = embedding_model.get_text_embedding_batch(texts)
    print(len(embeddings), "Всего эмеддингов")
    print(len(embeddings[0]), "Длина одного эмбеддинга")