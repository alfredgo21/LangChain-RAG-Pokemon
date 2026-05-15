"""CLI script to build the FAISS index from files in data/raw."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_pipeline import RAGAssistant


def main() -> None:
    """Build the index and print a summary for terminal usage."""
    assistant = RAGAssistant()
    chunks = assistant.build_index()

    if chunks == 0:
        print("No se encontraron documentos en data/raw.")
        return

    print(f"Indice generado correctamente con {chunks} chunks.")


if __name__ == "__main__":
    main()
