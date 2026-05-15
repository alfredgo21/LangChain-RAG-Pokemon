"""Utilities to load supported documents from the raw dataset folder."""

from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".pdf"}


def _load_single_file(path: Path) -> list[Document]:
    """Load one file into LangChain documents based on extension."""
    ext = path.suffix.lower()

    if ext in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
    elif ext == ".csv":
        loader = CSVLoader(file_path=str(path), encoding="utf-8")
        docs = loader.load()
    elif ext == ".pdf":
        loader = PyPDFLoader(str(path))
        docs = loader.load()
    else:
        return []

    for d in docs:
        d.metadata["source"] = path.name
    return docs


def iter_supported_files(raw_data_dir: Path) -> Iterable[Path]:
    """Yield all supported files from the raw dataset directory recursively."""
    if not raw_data_dir.exists():
        return []

    return (
        path
        for path in sorted(raw_data_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_documents(raw_data_dir: Path) -> list[Document]:
    """Load all supported documents from the dataset folder recursively."""
    files = list(iter_supported_files(raw_data_dir))
    if not files:
        return []

    documents: list[Document] = []
    for file_path in files:
        documents.extend(_load_single_file(file_path))

    return documents


def latest_dataset_mtime(raw_data_dir: Path) -> float:
    """Return latest modification time across supported dataset files."""
    files = list(iter_supported_files(raw_data_dir))
    if not files:
        return 0.0

    return max(path.stat().st_mtime for path in files)
