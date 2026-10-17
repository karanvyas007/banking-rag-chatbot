"""FastAPI entry point. Run with: uvicorn backend.main:app --reload"""

import logging
import threading

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from google.genai import errors as genai_errors
from langchain_core.exceptions import ModelRateLimitError
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from backend.config import get_settings
from backend.retrieval.qa import ModelOutputError, NotConfigured, QAService, build_qa_service
from backend.schemas import AskRequest, AskResponse
from backend.vectorstore import IndexConfigMismatch, IndexNotReady

logger = logging.getLogger(__name__)

app = FastAPI(title="Banking Knowledge Assistant", version="0.1.0")

_service: QAService | None = None
_service_lock = threading.Lock()


def get_qa_service() -> QAService:
    """Build the QA service on first use, so the API starts (and /health works) without an index."""
    global _service
    with _service_lock:
        if _service is None:
            _service = build_qa_service(get_settings())
        return _service


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "gemini_configured": settings.gemini_configured,
        "chat_model": settings.gemini_chat_model,
        "embedding_model": settings.gemini_embedding_model,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, service: QAService = Depends(get_qa_service)) -> AskResponse:
    return service.ask(request.question, request.top_k)


# Error handling: return fixed, safe messages. Upstream error text is never echoed or logged,
# only the exception type and status code.


@app.exception_handler(IndexNotReady)
@app.exception_handler(IndexConfigMismatch)
@app.exception_handler(NotConfigured)
async def _service_unavailable(request: Request, exc: Exception) -> JSONResponse:
    logger.warning("/ask unavailable: %s", type(exc).__name__)
    # These messages are written by this app and contain no secrets.
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(genai_errors.APIError)
@app.exception_handler(ChatGoogleGenerativeAIError)
@app.exception_handler(ModelOutputError)
@app.exception_handler(httpx.HTTPError)
async def _upstream_error(request: Request, exc: Exception) -> JSONResponse:
    code = getattr(exc, "code", None)
    logger.warning("Gemini call failed: %s (code=%s)", type(exc).__name__, code)
    if isinstance(exc, ModelRateLimitError) or code == 429:
        return JSONResponse(
            status_code=429,
            content={"detail": "The Gemini API rate limit was reached. Please retry shortly."},
            headers={"Retry-After": "30"},
        )
    if code == 503:  # Gemini reports temporary overload ("high demand") this way
        return JSONResponse(
            status_code=503,
            content={"detail": "The language model is temporarily overloaded. Please retry shortly."},
            headers={"Retry-After": "30"},
        )
    return JSONResponse(
        status_code=502, content={"detail": "The language model service failed. Please retry later."}
    )
