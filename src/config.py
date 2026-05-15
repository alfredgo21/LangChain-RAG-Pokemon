"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """Centralized runtime settings for the RAG application."""

    project_root: Path = Path(__file__).resolve().parents[1]
    raw_data_dir: Path = project_root / "data" / "raw"
    vectorstore_dir: Path = project_root / "data" / "vectorstore"

    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-002")

    top_k: int = int(os.getenv("TOP_K", "4"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))


settings = Settings()
