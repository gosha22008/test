import os
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.readers.file import PDFReader


def load_documents(data_dir: str | None = None) -> list[Document]:
    """
    Загружает все PDF-документы из директории.

    Только PDF: в задании явно указано «10 PDF-файлов». PDFReader извлекает
    текст постранично с метаданными (file_name, page_label) — они нужны для
    ссылки на источник в ответе агента.

    Args:
        data_dir: Путь к директории с PDF (по умолчанию из ENV DOCS_DIR).

    Returns:
        Список документов LlamaIndex.
    """
    data_dir = data_dir or os.getenv("DOCS_DIR", "../data/docs")
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Directory '{data_dir}' does not exist.")

    reader = SimpleDirectoryReader(
        input_dir=str(data_path),
        required_exts=[".pdf"],
        recursive=True,
        filename_as_id=True,
        file_extractor={".pdf": PDFReader()},
    )
    documents = reader.load_data()

    if not documents:
        raise ValueError("No PDF documents were found.")

    print(f"Loaded {len(documents)} document(s).")
    return documents


if __name__ == "__main__":
    docs = load_documents()
