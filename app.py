"""Gradio UI for the RAG assistant with session-only API key input."""

import os
import socket
import inspect
from typing import Any

import gradio as gr

from src.rag_pipeline import RAGAssistant

ChatMessageDict = dict[str, str]
ChatHistory = list[ChatMessageDict]
MODEL_OPTIONS: list[str] = [
    "gemini-1.5-flash-002",
    "gemini-1.5-pro-002",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
]

assistant = RAGAssistant()
startup_status: str = assistant.prepare_index()


def _supports_parameter(callable_obj: Any, parameter_name: str) -> bool:
    """Return True when a callable exposes a specific named parameter."""
    try:
        return parameter_name in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False


def _resolve_server_port(default_port: int = 7860) -> int:
    """Return an available TCP port for local Gradio runs."""

    def _is_port_free(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
            except OSError:
                return False
            return True

    env_port = os.getenv("GRADIO_SERVER_PORT")
    if env_port and env_port.isdigit():
        candidate = int(env_port)
        if _is_port_free(candidate):
            return candidate

    if _is_port_free(default_port):
        return default_port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("0.0.0.0", 0))
        return int(sock.getsockname()[1])


def _normalize_history(history: Any) -> ChatHistory:
    """Convert incoming history to Gradio messages format consistently."""
    normalized: ChatHistory = []
    for item in history or []:
        if isinstance(item, dict) and "role" in item and "content" in item:
            normalized.append({"role": str(item["role"]), "content": str(item["content"])})
            continue

        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, assistant_msg = item
            normalized.append({"role": "user", "content": str(user_msg)})
            normalized.append({"role": "assistant", "content": str(assistant_msg)})

    return normalized


def answer_question(
    message: str,
    history: ChatHistory,
    api_key: str,
    selected_model: str,
) -> tuple[str, ChatHistory]:
    """Handle one user question and append assistant response to chat history."""
    history = _normalize_history(history)
    question = message.strip()
    if not question:
        return "", history

    # Session-only key: kept in memory, never written to files or env vars.
    if api_key and api_key.strip():
        assistant.set_session_api_key(api_key)

    # Session-only model: kept in memory, never written to files or env vars.
    if selected_model and selected_model.strip():
        assistant.set_session_model(selected_model)

    try:
        result = assistant.ask(question)
        sources_block = "\n".join(f"- {s}" for s in result.sources) if result.sources else "- Sin fuentes"
        answer = f"{result.answer}\n\nFuentes consultadas:\n{sources_block}"
    except Exception as exc:
        answer = (
            "Error al consultar el sistema RAG. "
            "Verifica la API key (campo temporal o .env) y que exista el indice (python scripts/build_index.py).\n\n"
            f"Detalle: {exc}"
        )

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return "", history


def validate_api_key(api_key: str, selected_model: str) -> str:
    """Validate API key against suggested models and report which ones are usable."""
    try:
        candidate_models = list(dict.fromkeys([selected_model, *MODEL_OPTIONS])) if selected_model else MODEL_OPTIONS
        results = assistant.validate_available_models(api_key=api_key, model_names=candidate_models)

        available = [result.model_name for result in results if result.status == "available"]
        unavailable = [result.model_name for result in results if result.status == "unavailable"]
        quota = [result.model_name for result in results if result.status == "quota"]
        invalid_key = next((result for result in results if result.status == "invalid_key"), None)
        other_errors = [
            f"- {result.model_name}: {result.message}"
            for result in results
            if result.status not in {"available", "unavailable", "quota", "invalid_key"}
        ]

        if invalid_key is not None:
            return invalid_key.message

        lines: list[str] = []
        if available:
            lines.append("Modelos disponibles para esta API key:")
            lines.extend(f"- {model_name}" for model_name in available)

        if quota:
            if lines:
                lines.append("")
            lines.append("Modelos con quota agotada o no habilitada:")
            lines.extend(f"- {model_name}" for model_name in quota)

        if unavailable:
            if lines:
                lines.append("")
            lines.append("Modelos no disponibles para esta API/version:")
            lines.extend(f"- {model_name}" for model_name in unavailable)

        if other_errors:
            if lines:
                lines.append("")
            lines.append("Otros resultados:")
            lines.extend(other_errors)

        if not lines:
            return "No se pudo validar ningun modelo con esta API key."

        return "\n".join(lines)
    except Exception as exc:
        return f"Error al validar API key/modelo. Detalle: {exc}"


POKEMON_CSS = """
/* ---- Pokemon background ---- */
body {
    background-image:
        url('https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/906.png'),
        url('https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/935.png'),
        url('https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/963.png'),
        url('https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/988.png');
    background-repeat: no-repeat, no-repeat, no-repeat, no-repeat;
    background-position: left 10px top 10px, right 10px top 10px, left 10px bottom 10px, right 10px bottom 10px;
    background-size: 130px, 130px, 130px, 130px;
    background-color: #1a1a2e;
}
/* main content panel: dark semi-transparent card */
.gradio-container {
    background-color: rgba(20, 20, 40, 0.93) !important;
    color: #f0f0f0 !important;
    border-radius: 12px;
}
/* labels and general text */
label, .label-wrap, span, p, .prose, .output-markdown {
    color: #e8e8e8 !important;
}
/* input/textarea boxes */
input, textarea {
    background-color: #2a2a4a !important;
    color: #f0f0f0 !important;
    border: 1px solid #4a4a7a !important;
}
/* chatbot bubbles */
.message.user {
    background-color: #3d5a80 !important;
    color: #ffffff !important;
}
.message.bot {
    background-color: #293241 !important;
    color: #e0e0e0 !important;
}
/* buttons */
button.primary, button.secondary {
    background-color: #c62828 !important;
    color: #ffffff !important;
    border: none !important;
}
button.primary:hover, button.secondary:hover {
    background-color: #e53935 !important;
}
/* dropdown */
.wrap-inner {
    background-color: #2a2a4a !important;
    color: #f0f0f0 !important;
}
/* title */
#pokemon-title {
    text-align: center;
    font-size: 1.6em;
    font-weight: bold;
    color: #ffcb05;
    text-shadow: 2px 2px 4px #000000, -1px -1px 3px #c62828;
    padding: 12px 0 4px 0;
}
"""

blocks_kwargs: dict[str, Any] = {"title": "Asistente RAG - Gen 10 Pokemon"}
if _supports_parameter(gr.Blocks.__init__, "css"):
    blocks_kwargs["css"] = POKEMON_CSS

with gr.Blocks(**blocks_kwargs) as demo:
    gr.Markdown(
        """
<div id="pokemon-title">⚡ Asistente RAG con Gradio para informacion sobre la Generacion 10 de Pokemon ⚡</div>
<p style="text-align:center;color:#555;">Haz preguntas sobre el dataset de la decima generacion indexado en <code>data/raw/</code>.</p>
""".strip()
    )
    gr.Markdown(f"Estado de indice: {startup_status}")

    with gr.Row():
        api_key_input = gr.Textbox(
            label="Google API Key (temporal para esta sesion)",
            type="password",
            placeholder="Pega tu key aqui. No se guarda para futuras sesiones.",
        )
        model_selector = gr.Dropdown(
            label="Modelo Gemini (temporal para esta sesion)",
            choices=MODEL_OPTIONS,
            value=assistant.active_model_name if assistant.active_model_name in MODEL_OPTIONS else MODEL_OPTIONS[0],
            allow_custom_value=False,
            info="Selecciona un modelo de la lista fija.",
        )

    if _supports_parameter(gr.Chatbot.__init__, "type"):
        chatbot = gr.Chatbot(height=460, type="messages")
    else:
        chatbot = gr.Chatbot(height=460)
    question_input = gr.Textbox(label="Tu pregunta", placeholder="Escribe tu pregunta sobre el dataset...")
    send_btn = gr.Button("Enviar")
    clear_btn = gr.Button("Limpiar chat")

    send_btn.click(
        fn=answer_question,
        inputs=[question_input, chatbot, api_key_input, model_selector],
        outputs=[question_input, chatbot],
    )
    question_input.submit(
        fn=answer_question,
        inputs=[question_input, chatbot, api_key_input, model_selector],
        outputs=[question_input, chatbot],
    )
    clear_btn.click(lambda: [], outputs=[chatbot])

    gr.Markdown("### Validacion")
    validate_btn = gr.Button("Validar API Key y Modelos")
    validation_status = gr.Textbox(
        label="Estado de validacion",
        value="Aun no validado.",
        interactive=False,
    )

    validate_btn.click(
        fn=validate_api_key,
        inputs=[api_key_input, model_selector],
        outputs=[validation_status],
    )


if __name__ == "__main__":
    server_port = _resolve_server_port()
    launch_kwargs: dict[str, Any] = {
        "server_name": "0.0.0.0",
        "server_port": server_port,
        "share": False,
        "prevent_thread_lock": False,
    }
    if "css" not in blocks_kwargs and _supports_parameter(demo.launch, "css"):
        launch_kwargs["css"] = POKEMON_CSS

    demo.launch(**launch_kwargs)
