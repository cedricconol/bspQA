# bspQA

A RAG-powered Q&A system over official **Bangko Sentral ng Pilipinas (BSP)** Monetary Policy Reports. Ask questions about Philippine inflation, interest rates, and economic outlook — and get cited, grounded answers backed by the source documents.

> Built as a portfolio project to learn and demonstrate end-to-end Retrieval-Augmented Generation (RAG) engineering, from ingestion pipeline to deployed web app.

**Live demo:** [bspqa.onrender.com](https://bspqa.onrender.com) *(first query may take 30–60 s while the server wakes up on the free tier)*

---

## Table of Contents

- [What is RAG?](#what-is-rag)
  - [The core problem](#the-core-problem)
  - [Key concepts](#key-concepts)
  - [How retrieval works: vector search](#how-retrieval-works-vector-search)
  - [The RAG loop](#the-rag-loop)
- [System Architecture](#system-architecture)
  - [High-level overview](#high-level-overview)
  - [Ingestion pipeline](#ingestion-pipeline)
  - [Query pipeline](#query-pipeline)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment variables](#environment-variables)
  - [Backend](#backend)
  - [Frontend](#frontend)
  - [Running ingestion](#running-ingestion)
- [Design Decisions](#design-decisions)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [License](#license)

---

## What is RAG?

### The core problem

Large language models (LLMs) like GPT-4 are trained on a fixed snapshot of the internet. They have two key limitations for domain-specific Q&A:

1. **Knowledge cutoff** — they don't know about documents published after their training date.
2. **Hallucination** — they may confidently generate plausible-sounding but incorrect facts when asked about topics not well represented in their training data.

For a system answering questions about BSP monetary policy reports — a highly specific, time-sensitive corpus — both of these are disqualifying.

**RAG (Retrieval-Augmented Generation)** solves this by separating the *knowledge store* from the *language model*. Instead of relying on the LLM's memorized knowledge, you first retrieve the relevant passages from your own documents, then pass them as context to the LLM. The LLM's job becomes answering *from the provided text*, not from memory.

---

### Key concepts

| Concept | What it is |
|---|---|
| **Embedding** | A numerical vector (list of floats) that encodes the semantic meaning of text. Similar text produces similar vectors. |
| **Vector store** | A database optimized for storing embeddings and finding the most similar ones quickly (approximate nearest-neighbor search). |
| **Chunk** | A short passage of text (here: 512 tokens) cut from a larger document. LLMs have context limits, so documents must be broken into retrievable pieces. |
| **Retrieval** | Given a query embedding, find the top-K most similar chunk embeddings in the vector store. |
| **Generation** | Passing the retrieved chunks as context to an LLM and asking it to produce a grounded answer. |
| **Grounding** | Constraining the LLM to answer only from the provided sources, not its own parametric memory. Enables verifiable, citeable answers. |

---

### How retrieval works: vector search

Every chunk of text is converted into an embedding — a point in high-dimensional space (1 536 dimensions here). The geometry of this space is meaningful: semantically similar texts land close together regardless of the exact words used.

```
        "inflation rate"   "CPI increase"
               ●                ●
                 \             /
                  ●-----------●   ← similar meaning, close in space
                  "price level growth"

        "interest rate"
               ●                   ← different topic, far away
```

When a user asks a question, the question is also embedded into this same space. Retrieval is then a nearest-neighbor lookup: find the chunks whose vectors are closest to the query vector. This is measured with **cosine similarity** — the angle between two vectors — which captures semantic closeness even when the vocabulary differs.

---

### The RAG loop

```
                        ┌─────────────────────────────────────────────────┐
                        │                   RAG LOOP                      │
                        │                                                 │
  User question ──────► │  1. EMBED question → query vector               │
                        │                                                 │
                        │  2. RETRIEVE top-K similar chunks               │
                        │     from vector store                           │
                        │                                                 │
                        │  3. BUILD CONTEXT: number the chunks,           │
                        │     attach metadata (source, date)              │
                        │                                                 │
                        │  4. GENERATE: send question + context to LLM   │
                        │     with instructions to cite sources           │
                        │                                                 │
                        │  5. RETURN answer + structured source list      │
                        └─────────────────────────────────────────────────┘
                                              │
                                              ▼
                                  Cited answer + source links
```

The LLM never accesses the document store directly. It only sees the numbered passages the retriever selected. This makes the system **auditable**: every claim in the answer maps back to a specific chunk from a specific report.

---

## System Architecture

### High-level overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              bspQA System                                    │
│                                                                              │
│  ┌─────────────────────────────┐        ┌──────────────────────────────────┐ │
│  │      INGESTION PIPELINE     │        │          QUERY PIPELINE          │ │
│  │   (run once / on update)    │        │       (runs on every query)      │ │
│  │                             │        │                                  │ │
│  │  BSP PDFs (17 reports)      │        │  User ──► Next.js frontend       │ │
│  │       │                     │        │                │                  │ │
│  │       ▼                     │        │                ▼                  │ │
│  │  fetch_from_manifest.py     │        │  FastAPI  POST /query             │ │
│  │       │                     │        │                │                  │ │
│  │       ▼                     │        │                ▼                  │ │
│  │  parse_pdfs.py              │        │  run_rag_pipeline()               │ │
│  │  (MarkItDown → .txt)        │        │       │          │                │ │
│  │       │                     │        │   RETRIEVE    GENERATE            │ │
│  │       ▼                     │        │       │          │                │ │
│  │  chunk_and_embed.py    ─────┼──────► │   Qdrant     OpenAI              │ │
│  │  (tiktoken + OpenAI)        │        │   Cloud      GPT-4o-mini          │ │
│  │       │                     │        │                │                  │ │
│  │       ▼                     │        │                ▼                  │ │
│  │   Qdrant Cloud  ◄───────────┘        │  Answer + sources ──► User       │ │
│  │   (vector store)            │        │                                  │ │
│  └─────────────────────────────┘        └──────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

### Ingestion pipeline

Ingestion is a one-time (or on-update) offline process that builds the vector store from raw PDFs.

```
  manifest.json
  (17 BSP report URLs)
        │
        ▼
┌───────────────────┐
│ fetch_from_       │   Downloads PDFs, skips already-downloaded files
│ manifest.py       │   Output: ingestion/data/raw/*.pdf
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ parse_pdfs.py     │   MarkItDown converts PDF → plain text
│                   │   Strips form-feeds, BSP page headers/footers
│                   │   Output: ingestion/data/parsed/*.txt
└───────┬───────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│ chunk_and_embed.py                                                │
│                                                                   │
│  For each .txt file:                                              │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Text → tiktoken tokenizer (cl100k_base)                    │  │
│  │                                                             │  │
│  │  ┌──────────────────────────────────────────────────────┐   │  │
│  │  │ CHUNK WINDOW  (512 tokens, 64-token overlap)         │   │  │
│  │  │                                                      │   │  │
│  │  │  [chunk 0][chunk 1][chunk 2] ...                     │   │  │
│  │  │       ↕overlap↕       ↕overlap↕                     │   │  │
│  │  └──────────────────────────────────────────────────────┘   │  │
│  │                                                             │  │
│  │  Chunks → OpenAI text-embedding-3-small → 1536-dim vectors  │  │
│  │  (batches of 128 for throughput)                            │  │
│  │                                                             │  │
│  │  Upsert to Qdrant with payload:                             │  │
│  │  { source_file, chunk_index, text, period_label,           │  │
│  │    period_label_key, publication_date, embedding_model }    │  │
│  └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
   Qdrant Cloud collection
   (17 reports × ~hundreds of chunks each)
   Indexes: source_file (keyword), period_label_key (keyword),
            publication_date (datetime)
```

**Why overlapping chunks?** A 64-token overlap between adjacent chunks ensures that sentences spanning a chunk boundary aren't split in a way that loses meaning. A fact near the end of chunk N also appears near the start of chunk N+1, making it retrievable from either direction.

**Why stable UUIDs?** Each chunk's ID is a deterministic `uuid5` derived from its source file and chunk index. This means re-running ingestion on the same documents produces the same point IDs, enabling safe upserts (update-in-place) rather than duplicating data.

---

### Query pipeline

```
  User query: "What was the BSP policy rate in Q3 2024?"
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│ FastAPI  POST /query                                              │
│                                                                   │
│  QueryRequest {                                                   │
│    query, top_k=15, score_threshold=0.0,                         │
│    period_label_key?, publication_date?                           │
│  }                                                                │
└───────────────────────┬───────────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────────────┐
│ retrieve_chunks()                                                 │
│                                                                   │
│  1. Embed query → 1536-dim vector  (OpenAI text-embedding-3-small)│
│                                                                   │
│  2. Build optional Qdrant filter                                  │
│     └── period_label_key / publication_date range if specified    │
│                                                                   │
│  3. Query Qdrant (top_k × 3 = 45 candidates)                     │
│     └── cosine similarity search                                  │
│                                                                   │
│  4. Recency re-ranking                                            │
│     └── blended score = 0.15 × relevance + 0.85 × recency        │
│         (newer reports are preferred when equally relevant)       │
│     └── trim to top_k=15                                          │
└───────────────────────┬───────────────────────────────────────────┘
                        │
                 top-15 scored chunks
                        │
                        ▼
┌───────────────────────────────────────────────────────────────────┐
│ run_rag_pipeline() → _build_context()                             │
│                                                                   │
│  Formats chunks as numbered sources for the LLM prompt:           │
│                                                                   │
│  [1]                                                              │
│  source_file: FullReport_2024_3.txt                               │
│  publication_date: 2024-08-15                                     │
│  text: "The Monetary Board raised the BSP's key policy rate..."   │
│                                                                   │
│  [2] ...                                                          │
└───────────────────────┬───────────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────────────┐
│ OpenAI GPT-4o-mini                                                │
│                                                                   │
│  System prompt: "Answer using only the provided sources.          │
│  Cite every claim with [N]. Do not speculate."                    │
│                                                                   │
│  Input: question + numbered source blocks                         │
│                                                                   │
│  Output: "The BSP raised its policy rate to 6.50% in             │
│           Q3 2024 [1], citing elevated inflation expectations..." │
└───────────────────────┬───────────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────────────┐
│ QueryResponse {                                                   │
│   answer: "The BSP raised its policy rate to 6.50%...",          │
│   sources: [                                                      │
│     { source_id: 1, title: "Monetary Policy Report Q3 2024",     │
│       url: "https://bsp.gov.ph/...", score: 0.87 },              │
│     ...                                                           │
│   ]                                                               │
│ }                                                                 │
└───────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Embeddings** | OpenAI `text-embedding-3-small` | 1536-dim semantic vectors for chunks and queries |
| **Vector store** | Qdrant Cloud | ANN search with metadata filtering and payload indexes |
| **Generation** | OpenAI `gpt-4o-mini` | Grounded answer synthesis with inline citations |
| **Backend** | FastAPI + Uvicorn | REST API (`POST /query`) |
| **Frontend** | Next.js (App Router) | Q&A UI with citation rendering |
| **PDF parsing** | MarkItDown | PDF → plain text conversion |
| **Tokenizer** | tiktoken (`cl100k_base`) | Token-accurate chunking |
| **Config** | pydantic-settings | Type-safe env var management |
| **Package manager** | uv | Fast, reproducible Python dependency management |
| **Deployment** | Render | Backend API hosting |
| **Testing** | pytest | Unit tests for RAG pipeline and ingestion logic |

---

## Project Structure

```
bspQA/
├── backend/
│   └── app/
│       ├── config.py          # pydantic-settings Settings + get_settings()
│       ├── main.py            # FastAPI app entry point
│       └── api/routes/
│           └── router.py      # POST /query endpoint
│       └── rag/
│           ├── pipeline.py    # run_rag_pipeline() orchestrator
│           ├── retriever.py   # embed_query(), retrieve_chunks(), recency re-ranking
│           └── generator.py   # _build_context(), _build_sources(), SYSTEM_PROMPT
├── frontend/
│   └── app/
│       ├── page.tsx           # Root page
│       └── components/
│           └── QAWidget.tsx   # Main Q&A UI component
├── ingestion/
│   ├── manifest.json          # Source of truth: 17 BSP report URLs + metadata
│   ├── fetch_from_manifest.py # Download PDFs
│   ├── parse_pdfs.py          # PDF → .txt (MarkItDown)
│   ├── chunk_and_embed.py     # Chunk → embed → upsert to Qdrant
│   └── data/
│       ├── raw/               # Downloaded PDFs
│       └── parsed/            # Parsed .txt files
├── tests/
│   ├── test_rag_pipeline.py               # Unit tests for run_rag_pipeline()
│   └── test_chunk_and_embed_collection_env.py  # Tests for production-safety logic
├── evals/                     # Evaluation notebooks
├── pyproject.toml
├── render.yaml                # Render deployment config
└── .env.example
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- Node.js 18+
- [uv](https://docs.astral.sh/uv/) (`pip install uv`)
- OpenAI API key
- Qdrant Cloud account (free tier works)

### Environment variables

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

```bash
OPENAI_API_KEY=sk-...
QDRANT_CLUSTER_ENDPOINT=https://<your-cluster>.qdrant.io
QDRANT_API_KEY=...
QDRANT_COLLECTION_NAME=bsp_reports
```

### Backend

```bash
# Install dependencies
uv sync

# Run the FastAPI server (auto-reload)
uvicorn backend.app.main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

**Run tests:**

```bash
uv run pytest
```

**Lint and format:**

```bash
uv run ruff check .
uv run ruff format .
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI will be available at `http://localhost:3000`. Set `NEXT_PUBLIC_API_URL=http://localhost:8000` (or in `.env.local`).

### Running ingestion

Run these in order. Each step's output feeds the next.

```bash
# 1. Download PDFs listed in manifest.json → ingestion/data/raw/
uv run python ingestion/fetch_from_manifest.py

# 2. Parse PDFs to text → ingestion/data/parsed/
uv run python ingestion/parse_pdfs.py

# 3. Chunk, embed, and upsert to Qdrant
uv run python ingestion/chunk_and_embed.py
```

To wipe and rebuild the Qdrant collection during development:

```bash
QDRANT_RECREATE_COLLECTION=true uv run python ingestion/chunk_and_embed.py
```

> Production environments (detected via `RAILWAY_ENVIRONMENT=production` or `APP_ENV=production`) always block collection recreation, regardless of the flag.

---

## Design Decisions

### Recency re-ranking

BSP reports are time-series documents — a question about "current inflation" should prefer the most recent report, even if an older report has a slightly higher cosine similarity score (e.g. because it uses more matching vocabulary). The retriever fetches 3× more candidates than needed (`top_k × 3`), then re-ranks by a blended score:

```
combined_score = 0.15 × (cosine_similarity / max_similarity)
               + 0.85 × (days_since_oldest / date_span_days)
```

The `RECENCY_WEIGHT = 0.85` strongly favors recent documents while still letting highly relevant older chunks surface if needed.

### Grounding via system prompt

The generator's system prompt explicitly forbids the model from using outside knowledge and requires a citation (`[N]`) for every factual claim. If the retrieved chunks don't contain enough information, the model is instructed to say so rather than guess. This trades recall for precision — appropriate for a system where credibility depends on citeability.

### Stable chunk IDs

Chunk point IDs in Qdrant are deterministic `uuid5` hashes of `(source_file, chunk_index)`. This means ingestion is idempotent: re-running it on the same source file updates vectors in-place rather than creating duplicates. Each ingestion run first deletes all existing points for the source file, then upserts fresh ones.

### Metadata filtering

The query API supports optional `period_label_key` and `publication_date` / `publication_date_from` / `publication_date_to` filters, passed directly to Qdrant as pre-filter conditions evaluated before the ANN search. This lets the frontend (or API clients) scope queries to a specific report period without post-processing.

---

## Evaluation

The `evals/` directory contains notebooks for offline evaluation of retrieval quality and answer faithfulness. Metrics tracked:

- **Retrieval precision** — are the top-K chunks actually relevant to the query?
- **Answer faithfulness** — does the generated answer stay within the provided sources?
- **Citation accuracy** — do the `[N]` citations in the answer map to the right source chunks?

---

## Deployment

The backend is deployed on [Render](https://render.com) as a Python web service. Configuration is in `render.yaml`:

```yaml
services:
  - type: web
    name: bspqa-api
    runtime: python
    buildCommand: uv sync --frozen && uv cache prune --ci
    startCommand: uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
```

All secrets (`OPENAI_API_KEY`, `QDRANT_*`) are injected as environment variables via the Render dashboard — never committed to the repo.

---

## License

[MIT](LICENSE)
