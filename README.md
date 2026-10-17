# Banking Knowledge Assistant — RAG Chatbot

A retrieval-augmented generation (RAG) chatbot that answers questions about banking policies and FAQs, retrieves the relevant passages, and cites its sources.

> **Status:** ingestion, retrieval, the grounded `/ask` API and the Streamlit chat UI are complete.
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
  schemas.py      Pydantic request/response models for the API
  embeddings.py   gemini-embedding-2 adapter (LangChain Embeddings interface)
  vectorstore.py  ChromaDB collection access and config checks
  ingestion/      Load, chunk and index documents (python -m backend.ingestion)
  retrieval/
    retriever.py  Similarity search with a relevance threshold
    qa.py         Prompting, answer generation and citation validation
frontend/
  app.py            Streamlit chat UI
  backend_client.py HTTP client for /ask and /health (maps every result to answer/abstain/error)
data/
  raw/            Source documents (public/synthetic; committed)
  chroma/         Generated vector store (git-ignored)
eval/             Live evaluation set (questions.json) and runner
tests/            pytest suite (offline; Gemini and embeddings are mocked)
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

## Run locally

Index the documents first (see above). Then start the API and the UI in two terminals, both from the project root with the virtual environment activated:

```bash
# Terminal 1: FastAPI backend on http://localhost:8000 (interactive docs at /docs)
source .venv/bin/activate
uvicorn backend.main:app --reload

# Terminal 2: Streamlit chat UI on http://localhost:8501
source .venv/bin/activate
streamlit run frontend/app.py
```

If the API runs elsewhere, start the UI with `BACKEND_URL=http://host:port streamlit run frontend/app.py`.

### The chat UI

- Type a question or click an example. Each reply appears as one of three clearly different states:
  - **Answer**, with a collapsible **Sources** list showing each cited document's title, section, filename and an excerpt.
  - **Not covered by the documents** (blue info box): the backend abstained, with a note on why.
  - **Something went wrong** (amber if retrying may help, red otherwise): the API was unreachable, timed out, was rate limited (`429`), was unavailable (`503`) or failed. Messages are written by the UI; backend error bodies and stack traces are never displayed.
- The sidebar shows whether the API is reachable and has a Gemini key configured.
- The conversation is kept for the browser session (across Streamlit reruns) and can be cleared from the sidebar. **Each question is answered independently:** the backend has no conversational memory, so follow-ups like "what about abroad?" aren't understood in context.
- The UI only displays what the API returns. Retrieval, answers, citations and abstention are all decided by the backend.

## Ask a question

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Can I use my mobile phone bill as proof of address?"}'
```

`top_k` (1–10) is optional and overrides `RETRIEVAL_TOP_K` for one request. Example response (abridged):

```json
{
  "answer": "No, mobile phone bills are not accepted as proof of address [1].",
  "answered": true,
  "abstain_reason": null,
  "citations": [
    {
      "id": 1,
      "source": "kyc-account-opening.md",
      "title": "Account Opening and Identity Verification (KYC)",
      "section": "Proof of Address",
      "chunk_index": 4,
      "relevance": 0.7819,
      "excerpt": "## Proof of Address Applicants must also provide one proof of address dated within the last **3 months**, ..."
    }
  ],
  "retrieved": [
    {"id": 1, "source": "kyc-account-opening.md", "title": "Account Opening and Identity Verification (KYC)", "section": "Proof of Address", "relevance": 0.7819}
  ],
  "model": "gemini-3.8-flash"
}
```

When the documents don't cover a question, the API abstains instead of guessing:

```json
{
  "answer": "I couldn't find this in the available Fernhollow Bank documents, so I can't answer it reliably. Please contact Fernhollow Bank customer service for help with this question.",
  "answered": false,
  "abstain_reason": "no_relevant_documents",
  "citations": [],
  "retrieved": [],
  "model": null
}
```

| `abstain_reason` | Meaning |
|---|---|
| `no_relevant_documents` | No passage met `MIN_RELEVANCE`; the chat model was not called |
| `not_in_documents` | Passages were retrieved, but the model found they don't answer the question |
| `no_valid_citations` | The model answered but cited no retrieved passage, so the answer was discarded |

Errors return `{"detail": "..."}` with a fixed message: `422` invalid request, `429` Gemini rate limit (with `Retry-After`), `503` index missing/empty/incompatible, API key not set, or Gemini temporarily overloaded, `502` other model failures.

### How answers are grounded

- **Retrieval:** the question is embedded with the same model and the documented query prefix as the index. The top `RETRIEVAL_TOP_K` chunks are kept only if their cosine relevance is at least `MIN_RELEVANCE`.
- **Prompt:** the model sees only numbered passages and is told to answer from them alone, cite passage numbers, and set `answerable: false` when they don't contain the answer. It replies in JSON enforced by a response schema.
- **Citation validation:** the model returns passage *numbers*, never filenames. Each number is checked against the retrieved passages; invalid ones are dropped, along with their `[n]` markers in the answer text. Source, title and section always come from the index metadata.

## Evaluate

```bash
python -m eval.run_eval    # makes real Gemini API calls (~20 requests; free tier)
```

[eval/questions.json](eval/questions.json) has 12 supported questions (mostly paraphrased, with the expected source and key facts) and 6 unsupported ones, including near misses such as the bank's CEO or an unlisted credit card APR. Latest run (17 Oct 2026, `gemini-3.8-flash` + `gemini-embedding-2`): **12/12 supported answered correctly, 6/6 unsupported abstained.** The set is small and model output varies between runs, so treat this as a smoke test, not a benchmark.

## Test

```bash
pytest
```

The suite runs offline: Gemini and the embeddings are replaced by fakes, and the UI tests mock HTTP (`httpx.MockTransport`) and drive the app with Streamlit's `AppTest`. It makes no network calls and costs nothing.

## Known limitations

- **Small, synthetic corpus.** Eight short fictional documents; behaviour on large or messy real-world documents is untested.
- **Threshold calibrated on this corpus.** `MIN_RELEVANCE=0.65` separated supported (≥0.72) from unrelated (≤0.63) questions here, but questions mentioning the bank's name can score ~0.7 without being answerable. Those rely on the model abstaining. Re-calibrate if the documents change.
- **Single-turn only.** The UI shows the conversation, but each question is answered independently; follow-up questions don't use earlier messages as context.
- **Chat history is per browser session.** It lives in Streamlit session state and is lost when the page is reloaded.
- **No retry/backoff on Gemini errors.** Rate limits and overloads are returned to the caller as `429`/`503`. Free-tier limits are low.
- **Restart after re-indexing.** The API opens the index on the first request; restart it after `python -m backend.ingestion --reset`.
- **Synchronous calls.** Each request blocks a worker thread while Gemini responds; fine for a demo, not tuned for load.
- **Not financial advice.** Answers describe a fictional bank's fictional policies.

## Configuration

All settings live in `.env` (see [.env.example](.env.example)):

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | — | Gemini API key |
| `GEMINI_CHAT_MODEL` | `gemini-3.8-flash` | Answer generation model |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-2` | Embedding model |
| `EMBEDDING_DIMENSION` | `3072` | Embedding size (128–3072) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Chunking, in characters |
| `RETRIEVAL_TOP_K` | `4` | Passages retrieved per question (1–10) |
| `MIN_RELEVANCE` | `0.65` | Minimum cosine relevance (0–1) for a passage to be used |
| `LLM_TIMEOUT_SECONDS` / `LLM_MAX_RETRIES` | `60` / `1` | Chat model request limits |
| `CHROMA_PERSIST_DIR` | `data/chroma` | Vector store location |
| `CHROMA_COLLECTION` | `banking_docs` | Collection name |
| `DOCS_DIR` | `data/raw` | Documents to ingest |
| `BACKEND_URL` | `http://localhost:8000` | API address used by the UI (read from the shell environment, not `.env`) |
| `BACKEND_TIMEOUT_SECONDS` | `90` | How long the UI waits for `/ask` (shell environment) |

## Roadmap

- [x] Project foundation, config, health check, tests
- [x] Synthetic banking policy documents and FAQs
- [x] Ingestion pipeline (chunking + embeddings + ChromaDB)
- [x] Retrieval and `/ask` endpoint with source citations
- [x] Small live evaluation set
- [x] Streamlit chat UI showing answers and sources
