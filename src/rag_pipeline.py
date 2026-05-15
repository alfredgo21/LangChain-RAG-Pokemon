"""Core RAG pipeline: indexing, retrieval, and LLM response generation."""

from dataclasses import dataclass
from pathlib import Path
import time

from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.data_loader import latest_dataset_mtime, load_documents


RAG_PROMPT = PromptTemplate.from_template(
    """
Eres un asistente que responde usando solo el contexto proporcionado.
Si no hay informacion suficiente, di claramente que no esta en el dataset.
Responde en espanol, de forma clara y breve.

Pregunta:
{question}

Contexto:
{context}

Respuesta:
""".strip()
)


@dataclass
class RetrievalResult:
    """Structured response with model answer and cited source files."""

    answer: str
    sources: list[str]


@dataclass
class ModelValidationResult:
    """Validation outcome for one Gemini model and one API key."""

    model_name: str
    status: str
    message: str


class RAGAssistant:
    """RAG service that can build/load an index and answer questions."""

    def __init__(self) -> None:
        self.embeddings: HuggingFaceEmbeddings | None = None
        self.llm: ChatGoogleGenerativeAI | None = None
        self.vectorstore: FAISS | None = None
        self.session_api_key: str | None = None
        self.session_model_name: str | None = None
        self.active_model_name: str = settings.gemini_model

    def _resolve_model_name(self) -> str:
        """Return active model from session selector or environment config."""
        return self.session_model_name or settings.gemini_model

    def _ensure_embeddings(self) -> HuggingFaceEmbeddings:
        """Create embeddings client lazily to avoid heavy startup work."""
        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        return self.embeddings

    def _resolve_api_key(self) -> str:
        """Return active API key from session or environment."""
        active_api_key = self.session_api_key or settings.google_api_key
        if not active_api_key:
            raise ValueError("Falta GOOGLE_API_KEY. Puedes ingresarla temporalmente en la interfaz.")
        return active_api_key

    def _build_llm(self, model_name: str, api_key: str) -> ChatGoogleGenerativeAI:
        """Create a Gemini chat client for a specific model name."""
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.2,
        )

    def _classify_model_error(self, exc: Exception) -> tuple[str, str]:
        """Map Gemini exceptions to a short status code and user-facing message."""
        error_text = str(exc).lower()

        if "api_key_invalid" in error_text or "api key not valid" in error_text:
            return (
                "invalid_key",
                "API key invalida. Revisa que este completa, activa y sin espacios extra.",
            )

        if "resource_exhausted" in error_text or "quota" in error_text or "429" in error_text:
            return (
                "quota",
                "API key valida pero sin cuota disponible para este modelo.",
            )

        if "not_found" in error_text or "not found" in error_text:
            return (
                "unavailable",
                "El modelo no esta disponible para esta API/version.",
            )

        return ("error", f"No se pudo validar el modelo. Detalle: {exc}")

    def _ensure_llm(self) -> ChatGoogleGenerativeAI:
        """Create Gemini chat client lazily using session key or environment key."""
        active_api_key = self._resolve_api_key()
        model_name = self._resolve_model_name()

        if self.llm is None:
            self.llm = self._build_llm(model_name, active_api_key)
            self.active_model_name = model_name
        return self.llm

    def _invoke_with_model_fallback(self, prompt: str) -> object:
        """Invoke Gemini and fallback to alternative models on recoverable model errors."""
        llm = self._ensure_llm()
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            error_text = str(exc).lower()
            status_code, status_message = self._classify_model_error(exc)
            has_invalid_key = status_code == "invalid_key"
            has_not_found = "not_found" in error_text or "not found" in error_text
            has_quota = "resource_exhausted" in error_text or "quota" in error_text or "429" in error_text

            if has_invalid_key:
                raise ValueError(status_message) from exc

            if not has_not_found and not has_quota:
                raise

            fallback_models = [
                "gemini-1.5-flash-002",
                "gemini-1.5-pro-002",
                "gemini-2.0-flash",
            ]
            fallback_models = [m for m in fallback_models if m != self.active_model_name]

            api_key = self._resolve_api_key()
            last_exc: Exception = exc

            if has_quota:
                # Small delay helps in burst-limit cases before trying alternatives.
                time.sleep(1.2)

            for model_name in fallback_models:
                try:
                    candidate = self._build_llm(model_name, api_key)
                    response = candidate.invoke(prompt)
                    self.llm = candidate
                    self.active_model_name = model_name
                    return response
                except Exception as candidate_exc:
                    last_exc = candidate_exc

            if has_quota:
                raise ValueError(
                    "Cuota de Gemini agotada para esta API key/proyecto. "
                    "Prueba otro modelo en el selector, espera unos minutos, o usa otra API key con cuota disponible."
                ) from last_exc

            raise ValueError(
                "No se pudo usar el modelo Gemini configurado ni los modelos alternativos. "
                "Actualiza GEMINI_MODEL o revisa permisos de tu API key."
            ) from last_exc

    def set_session_api_key(self, api_key: str | None) -> None:
        """Set or clear a temporary API key kept only in process memory."""
        clean_key = api_key.strip() if api_key else None
        if clean_key != self.session_api_key:
            self.session_api_key = clean_key
            # Recreate client when key changes.
            self.llm = None
            self.active_model_name = self._resolve_model_name()

    def set_session_model(self, model_name: str | None) -> None:
        """Set or clear a temporary model name kept only in process memory."""
        clean_model_name = model_name.strip() if model_name else None
        if clean_model_name != self.session_model_name:
            self.session_model_name = clean_model_name
            # Recreate client when model changes.
            self.llm = None
            self.active_model_name = self._resolve_model_name()

    def validate_session_credentials(self, api_key: str | None, model_name: str | None) -> str:
        """Validate API key/model by making a minimal Gemini request."""
        self.set_session_api_key(api_key)
        self.set_session_model(model_name)

        active_key = self._resolve_api_key()
        active_model = self._resolve_model_name()

        try:
            validator_llm = self._build_llm(active_model, active_key)
            _ = validator_llm.invoke("Responde solo con: OK")
            self.llm = validator_llm
            self.active_model_name = active_model
            return f"API key valida. Modelo disponible: {active_model}"
        except Exception as exc:
            _, message = self._classify_model_error(exc)
            return message

    def validate_available_models(self, api_key: str | None, model_names: list[str]) -> list[ModelValidationResult]:
        """Validate a set of Gemini models for the provided API key."""
        self.set_session_api_key(api_key)

        active_key = self._resolve_api_key()
        results: list[ModelValidationResult] = []
        first_available_model: str | None = None

        for model_name in model_names:
            try:
                validator_llm = self._build_llm(model_name, active_key)
                _ = validator_llm.invoke("Responde solo con: OK")
                results.append(
                    ModelValidationResult(
                        model_name=model_name,
                        status="available",
                        message="Disponible",
                    )
                )
                if first_available_model is None:
                    first_available_model = model_name
            except Exception as exc:
                status, message = self._classify_model_error(exc)
                results.append(
                    ModelValidationResult(
                        model_name=model_name,
                        status=status,
                        message=message,
                    )
                )
                if status == "invalid_key":
                    break

        if first_available_model is not None:
            self.set_session_model(first_available_model)
            self.active_model_name = first_available_model
            self.llm = None

        return results

    def _index_marker_path(self) -> Path:
        """Return the path to the marker file storing index metadata."""
        return settings.vectorstore_dir / ".index_meta"

    def _write_index_marker(self) -> None:
        """Persist the latest dataset modification timestamp after indexing."""
        marker_path = self._index_marker_path()
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(str(latest_dataset_mtime(settings.raw_data_dir)), encoding="utf-8")

    def _read_index_marker(self) -> float:
        """Read index metadata timestamp; return 0.0 when unavailable."""
        marker_path = self._index_marker_path()
        if not marker_path.exists():
            return 0.0

        try:
            return float(marker_path.read_text(encoding="utf-8").strip())
        except ValueError:
            return 0.0

    def _index_is_available(self) -> bool:
        """Return True when required FAISS files exist in vectorstore folder."""
        return (settings.vectorstore_dir / "index.faiss").exists() and (
            settings.vectorstore_dir / "index.pkl"
        ).exists()

    def should_rebuild_index(self) -> bool:
        """Decide whether the vector index is missing or outdated."""
        if not self._index_is_available():
            return True

        raw_mtime = latest_dataset_mtime(settings.raw_data_dir)
        if raw_mtime == 0.0:
            return False

        return raw_mtime > self._read_index_marker()

    def build_index(self) -> int:
        """Build and persist FAISS index from current dataset documents."""
        docs = load_documents(settings.raw_data_dir)
        if not docs:
            return 0

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = splitter.split_documents(docs)

        settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore = FAISS.from_documents(chunks, self._ensure_embeddings())
        self.vectorstore.save_local(str(settings.vectorstore_dir))
        self._write_index_marker()

        return len(chunks)

    def load_index(self) -> None:
        """Load existing FAISS index from disk into memory."""
        if not settings.vectorstore_dir.exists():
            raise FileNotFoundError(
                "No existe indice vectorial. Ejecuta primero: python scripts/build_index.py"
            )

        self.vectorstore = FAISS.load_local(
            str(settings.vectorstore_dir),
            self._ensure_embeddings(),
            allow_dangerous_deserialization=True,
        )

    def prepare_index(self) -> str:
        """Build or load the index depending on freshness and availability."""
        if self.should_rebuild_index():
            chunks = self.build_index()
            if chunks == 0:
                return "No se encontraron documentos en data/raw para indexar."
            return f"Indice regenerado automaticamente con {chunks} chunks."

        self.load_index()
        return "Indice existente cargado correctamente."

    def ask(self, question: str) -> RetrievalResult:
        """Answer a question using retrieved context and Gemini generation."""
        if not (self.session_api_key or settings.google_api_key):
            raise ValueError("Falta GOOGLE_API_KEY. Puedes ingresarla temporalmente en la interfaz.")

        if self.vectorstore is None:
            self.load_index()

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": settings.top_k})
        try:
            docs = retriever.invoke(question)
        except AttributeError:
            # Backward compatibility with older retriever API.
            docs = retriever.get_relevant_documents(question)

        if not docs:
            return RetrievalResult(
                answer="No encontre informacion relevante en el dataset.",
                sources=[],
            )

        context = "\n\n".join(d.page_content for d in docs)
        prompt = RAG_PROMPT.format(question=question, context=context)
        response = self._invoke_with_model_fallback(prompt)

        sources = sorted({d.metadata.get("source", "desconocido") for d in docs})

        return RetrievalResult(answer=response.content, sources=sources)
