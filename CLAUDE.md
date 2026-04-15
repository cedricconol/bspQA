# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

bspQA is a RAG-powered Q&A system over Bangko Sentral ng Pilipinas (BSP) Monetary Policy and Inflation Reports. Users ask questions about Philippine monetary policy and get cited, grounded answers.

**Prerequisites:** Python 3.12+, Node.js 18+, OpenAI API key, Qdrant Cloud account.

## Commands

### Backend (Python)

```bash
# Install dependencies (uses uv)
uv sync

# Run the FastAPI server (from repo root)
uvicorn backend.app.main:app --reload

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_rag_pipeline.py

# Run a single test by name
uv run pytest tests/test_rag_pipeline.py::test_run_rag_pipeline_returns_fallback_when_no_hits

# Lint and format
uv run ruff check .
uv run ruff format .
```

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev      # dev server
npm run build    # production build
npm run lint     # ESLint
```

### Ingestion Pipeline (run in order)

```bash
# 1. Download PDFs listed in manifest.json → ingestion/data/raw/
uv run python ingestion/fetch_from_manifest.py

# 2. Parse PDFs to text → ingestion/data/parsed/
uv run python ingestion/parse_pdfs.py

# 3. Chunk, embed, and upsert to Qdrant → ingestion/data/embeddings/
uv run python ingestion/chunk_and_embed.py
```

## Required Environment Variables

```
OPENAI_API_KEY
QDRANT_CLUSTER_ENDPOINT
QDRANT_API_KEY
QDRANT_COLLECTION_NAME
```

Copy `.env.example` to `.env` and fill in the values. The backend reads config through a `pydantic-settings` `Settings` class in `backend/app/config.py` — accessed via `get_settings()` (cached singleton). Ingestion scripts call `_load_dotenv_files()` which loads from the repo root and `backend/.env`.

## Architecture

### Data Flow

```
BSP PDFs → ingestion/fetch_from_manifest.py → ingestion/data/raw/
         → ingestion/parse_pdfs.py (MarkItDown) → ingestion/data/parsed/
         → ingestion/chunk_and_embed.py (tiktoken + OpenAI embeddings) → Qdrant
                                                                              ↓
User query → FastAPI POST /query → RAG pipeline → Qdrant search + OpenAI → answer + sources
```

### Backend (`backend/app/`)

- **`config.py`** — `Settings` (pydantic-settings `BaseSettings`) and `get_settings()` cached singleton. All env var access in the backend goes through here.
- **`main.py`** — FastAPI app entry point; loads `.env` and registers the router.
- **`api/routes/router.py`** — Single `POST /query` endpoint accepting `QueryRequest` (query, top_k, score_threshold, optional date/period filters) and returning `QueryResponse` (answer, sources list).
- **`rag/pipeline.py`** — `run_rag_pipeline()` orchestrates retrieval then generation; the sole entry point from the API layer.
- **`rag/retriever.py`** — Embeds query with `text-embedding-3-small`, queries Qdrant, applies optional metadata filters (period, date range), then re-ranks results with recency boosting (`RECENCY_WEIGHT = 0.85`). Client factories (`get_openai_client`, `get_qdrant_client`) read from `get_settings()`.
- **`rag/generator.py`** — `_build_context()` and `_build_sources()` helpers that format Qdrant hits for the LLM prompt and API response respectively. Also holds `SYSTEM_PROMPT` and `DEFAULT_GENERATION_MODEL`.

### Ingestion (`ingestion/`)

- **`manifest.json`** — Source of truth for all BSP report URLs, with `period_label`, `publication_date`, and `ingested` status.
- **`fetch_from_manifest.py`** — Downloads PDFs from manifest URLs; skips already-downloaded files.
- **`parse_pdfs.py`** — Converts PDFs to `.txt` using MarkItDown.
- **`chunk_and_embed.py`** — Chunks text with a 512-token window and 64-token overlap (tiktoken `cl100k_base`), embeds in batches of 128, upserts to Qdrant with stable UUID5 point IDs. Uses `QDRANT_RECREATE_COLLECTION` env var to control whether the collection is wiped on re-ingest (blocked in production environments).

### Qdrant Payload Schema

Each vector point has: `source_file`, `chunk_index`, `text`, `embedding_model`, `period_label`, `period_label_key` (normalized slug), `publication_date` (ISO 8601). Keyword indexes on `source_file` and `period_label_key`; datetime index on `publication_date`.

### Frontend (`frontend/`)

Next.js 16 app (App Router). Currently scaffold-only — `frontend/app/page.tsx` is unimplemented. **Important:** This version of Next.js has breaking API/convention changes; read `node_modules/next/dist/docs/` before writing frontend code.

### Tests (`tests/`)

- `test_rag_pipeline.py` — Unit tests for `run_rag_pipeline()` using `monkeypatch` to stub `retrieve_chunks` and the OpenAI client. No real network calls.
- `test_chunk_and_embed_collection_env.py` — Parametrized tests for `_should_recreate_qdrant_collection()` production-safety logic.

Import path for tests: `backend.app.rag.pipeline` and `ingestion.chunk_and_embed` (both roots are on `pythonpath` per `pyproject.toml`).
