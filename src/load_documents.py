from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.readers.file import PDFReader


def load_documents(data_dir: str) -> list[Document]:
    """
    Загружает все PDF-документы из указанной директории.

    Args:
        data_dir: Путь к директории с PDF-файлами.

    Returns:
        Список документов LlamaIndex.
    """

    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Directory '{data_dir}' does not exist."
        )

    reader = SimpleDirectoryReader(
        input_dir=str(data_path),
        required_exts=[".pdf"],
        recursive=True,
        filename_as_id=True,
        file_extractor={
        ".pdf": PDFReader(),
        },
    )

    documents = reader.load_data()

    if not documents:
        raise ValueError(
            "No PDF documents were found."
        )

    print(f"Loaded {len(documents)} document(s).")

    return documents


if __name__ == "__main__":
    docs = load_documents("./data/docs")