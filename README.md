---
title: Asistente RAG - Pokemon
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
---

# Proyecto Final - Python para IA (Sesion 4)

Aplicacion web con Gradio + LangChain (RAG) usando un dataset propio.

## Que incluye

- Interfaz web con Gradio
- Pipeline RAG con LangChain
- Indexado offline del dataset
- Respuesta con Gemini (via API)
- Deployment listo para HuggingFace Spaces
- Repositorio estructurado para entrega en GitHub

## Estructura

- `app.py`: interfaz web
- `scripts/build_index.py`: indexacion del dataset
- `src/config.py`: configuracion
- `src/rag_pipeline.py`: logica RAG
- `data/raw/`: coloca aqui tus archivos fuente
- `data/vectorstore/`: indice FAISS persistente (se genera automaticamente)

## 1) Instalacion local

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Configuracion

1. Copia `.env.example` a `.env`
2. Completa `GOOGLE_API_KEY`

## 3) Carga tu dataset

Copia tus archivos en `data/raw/`.
Formatos soportados por defecto:

- `.txt`
- `.md`
- `.csv`
- `.pdf`

## 4) Construir el indice vectorial

```bash
python scripts/build_index.py
```

## 5) Ejecutar la app

```bash
python app.py
```

Gradio abrira una URL local.

Al iniciar, la app revisa automaticamente el indice vectorial:

- Si no existe, lo construye.
- Si los archivos en `data/raw/` cambiaron, lo regenera.
- Si no hubo cambios, solo carga el indice existente.

## Deployment en HuggingFace Spaces

1. Crea un Space de tipo Gradio
2. Sube este repositorio
3. En `Settings > Variables and secrets`, agrega:
   - `GOOGLE_API_KEY`
4. El comando de inicio por defecto de Gradio detectara `app.py`

## Rubrica (mapa rapido)

- Interfaz web: Gradio en `app.py`
- Modelo IA: Gemini + LangChain en `src/rag_pipeline.py`
- Deployment online: HF Spaces
- Repo GitHub: estructura y documentacion lista

## Notas

- El indice se construye offline para evitar recalcular embeddings en cada inicio.
- Para actualizar el dataset, agrega/quita archivos en `data/raw/` y vuelve a ejecutar `scripts/build_index.py`.
