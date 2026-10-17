# Banking Knowledge Assistant — RAG Chatbot

A retrieval-augmented generation (RAG) chatbot that answers questions about banking policies and FAQs, retrieves the relevant passages, and cites its sources.

> **Status:** document ingestion and vector indexing are complete; answer generation is next.
>
> All documents are **synthetic**: policies of *Fernhollow Bank*, a fictional bank. No real customer data.

## Architecture

```
Streamlit UI  ──HTTP──▶  FastAPI backend  ──▶  Retriever (ChromaDB)  ──▶  Gemini (answer + citations)
                                ▲
              Ingestion: data/raw ──▶ chunk ──▶ Gemini embeddings ──▶ ChromaDB (data/chroma)
```

| Layer | Technology |
|---|---|
| Backend API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| RAG orchestration | LangChain |
| Vector store | ChromaDB (local, persisted) |
| LLM & embeddings | Google Gemini API |
| Tests | pytest |

## Project structure

```
backend/
  main.py         FastAPI app (entry point)
  config.py       Settings loaded from environment / .env
  embeddings.py   gemini-embedding-2 adapter (LangChain Embeddings interface)
  vectorstore.py  ChromaDB collection access and config checks
  ingestion/      Load, chunk and index documents (python -m backend.ingestion)
  retrieval/      Query the vector store and generate cited answers
frontend/
  app.py          Streamlit UI
data/
  raw/            Source documents (public/synthetic; committed)
  chroma/         Generated vector store (git-ignored)
tests/            pytest suite
```

## Setup

Requires Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then set GOOGLE_API_KEY
```

Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey). The key is read from `.env` only and is never committed.

## Index the documents

```bash
python -m backend.ingestion          # incremental: only new/changed files are embedded
python -m backend.ingestion --reset  # rebuild from scratch (needed after changing model or dimension)
```

How ingestion works:

- **Loading:** every `.md` / `.txt` file under `data/raw/`. Empty files are skipped with a warning.
- **Chunking:** split on Markdown headings first, then into chunks of at most `CHUNK_SIZE` characters with `CHUNK_OVERLAP` overlap. Each chunk keeps the heading it came from.
- **Metadata for citations:** `source` (file path), `title` (H1), `section` (e.g. `Fees > Daily cap`), `chunk_index`, `chunk_count`.
- **Embeddings:** `gemini-embedding-2` takes no `task_type` field. Instead, [documented text prefixes](https://ai.google.dev/gemini-api/docs/embeddings) are used: `title: {title} | text: {chunk}` for documents and `task: search result | query: {question}` for queries. See [backend/embeddings.py](backend/embeddings.py).
- **Idempotency:** a content fingerprint per file decides what happens. Unchanged files are skipped (no API calls), changed files have all their chunks replaced, and deleted or emptied files are removed from the index. New embeddings are computed before anything is deleted, so a failed API call leaves the index as it was.
- **Consistency:** the Chroma collection uses cosine distance and records the embedding model and dimension. Opening it with a different configuration fails with a clear error instead of mixing incompatible vectors.

## Run

```bash
# Backend (http://localhost:8000, docs at /docs)
uvicorn backend.main:app --reload

# Frontend (in a second terminal)
streamlit run frontend/app.py
```

## Test

```bash
pytest
```

## Configuration

All settings live in `.env` (see [.env.example](.env.example)):

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | — | Gemini API key |
| `GEMINI_CHAT_MODEL` | `gemini-3.8-flash` | Answer generation model |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-2` | Embedding model |
| `EMBEDDING_DIMENSION` | `3072` | Embedding size (128–3072) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Chunking, in characters |
| `CHROMA_PERSIST_DIR` | `data/chroma` | Vector store location |
| `CHROMA_COLLECTION` | `banking_docs` | Collection name |
| `DOCS_DIR` | `data/raw` | Documents to ingest |
| `BACKEND_URL` | `http://localhost:8000` | Used by the frontend (read from the shell environment, not `.env`) |

## Roadmap

- [x] Project foundation, config, health check, tests
- [x] Synthetic banking policy documents and FAQs
- [x] Ingestion pipeline (chunking + embeddings + ChromaDB)
- [ ] Retrieval and `/ask` endpoint with source citations
- [ ] Streamlit chat UI showing answers and sources
- [ ] Retrieval quality evaluation
