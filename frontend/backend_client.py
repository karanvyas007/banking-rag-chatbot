"""HTTP client for the backend API. No Streamlit imports, so it can be unit-tested directly.

Every call returns an Outcome; this module never raises for network or HTTP problems. User-facing
error messages are written here, so backend error bodies and stack traces are never shown (the
only backend text displayed is the `detail` of a 503, which the backend writes for this purpose).
"""

import os
from dataclasses import dataclass, field
from typing import Literal

import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
# The backend may wait up to LLM_TIMEOUT_SECONDS (60s) for Gemini, plus retrieval.
TIMEOUT_SECONDS = float(os.getenv("BACKEND_TIMEOUT_SECONDS", "90"))
# Mirrors the backend's AskRequest validation, so obvious problems are caught before a request.
MIN_QUESTION_LENGTH, MAX_QUESTION_LENGTH = 3, 1000

ABSTAIN_EXPLANATIONS = {
    "no_relevant_documents": "None of the policy documents matched this question.",
    "not_in_documents": "Related documents were found, but they don't answer this question.",
    "no_valid_citations": "An answer couldn't be backed by citations from the documents, so it was withheld.",
}


@dataclass(frozen=True)
class Citation:
    id: int
    title: str
    section: str
    source: str
    excerpt: str = ""
    relevance: float | None = None


@dataclass(frozen=True)
class Outcome:
    kind: Literal["answer", "abstain", "error"]
    text: str  # the answer, the backend's abstention message, or a user-facing error message
    citations: list[Citation] = field(default_factory=list)
    reason: str | None = None  # abstain_reason for abstentions, an error code for errors
    detail: str | None = None  # extra explanation shown under the message
    retryable: bool = False


def _error(reason: str, text: str, detail: str | None = None, retryable: bool = False) -> Outcome:
    return Outcome(kind="error", text=text, reason=reason, detail=detail, retryable=retryable)


def _backend_detail(response: httpx.Response) -> str | None:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return None
    return detail[:300] if isinstance(detail, str) else None


def _parse_answer(data: dict) -> Outcome:
    if data["answered"]:
        citations = [
            Citation(
                id=int(c["id"]),
                title=str(c["title"]),
                section=str(c.get("section") or ""),
                source=str(c["source"]),
                excerpt=str(c.get("excerpt") or ""),
                relevance=c.get("relevance"),
            )
            for c in data["citations"]
        ]
        return Outcome(kind="answer", text=str(data["answer"]), citations=citations)
    reason = data.get("abstain_reason")
    return Outcome(kind="abstain", text=str(data["answer"]), reason=reason, detail=ABSTAIN_EXPLANATIONS.get(reason))


def ask(
    question: str,
    base_url: str = BACKEND_URL,
    timeout: float = TIMEOUT_SECONDS,
    transport: httpx.BaseTransport | None = None,
) -> Outcome:
    question = question.strip()
    if not MIN_QUESTION_LENGTH <= len(question) <= MAX_QUESTION_LENGTH:
        return _error(
            "invalid_question",
            f"Please enter a question between {MIN_QUESTION_LENGTH} and {MAX_QUESTION_LENGTH} characters.",
        )

    try:
        with httpx.Client(base_url=base_url, timeout=timeout, transport=transport) as client:
            response = client.post("/ask", json={"question": question})
    except httpx.TimeoutException:
        return _error(
            "timeout", "The assistant took too long to respond. Please try again.", retryable=True
        )
    except httpx.HTTPError:
        return _error(
            "backend_unreachable",
            "The assistant service can't be reached right now.",
            detail=f"No response from {base_url}. Is the API running? (uvicorn backend.main:app)",
            retryable=True,
        )

    status = response.status_code
    if status == 200:
        try:
            return _parse_answer(response.json())
        except (ValueError, KeyError, TypeError, AttributeError):
            return _error("unexpected_response", "The assistant returned a response that couldn't be read.")
    if status == 422:
        return _error(
            "invalid_question",
            "That question couldn't be processed. Please rephrase it and try again.",
        )
    if status == 429:
        wait = response.headers.get("retry-after")
        when = f"in about {wait} seconds" if wait and wait.isdigit() else "shortly"
        return _error(
            "rate_limited", f"The assistant is receiving too many requests. Please try again {when}.", retryable=True
        )
    if status == 503:
        return _error(
            "service_unavailable",
            "The assistant is temporarily unavailable. Please try again shortly.",
            detail=_backend_detail(response),
            retryable=True,
        )
    if status == 502:
        return _error(
            "model_error", "The assistant couldn't generate an answer this time. Please try again.", retryable=True
        )
    return _error("unexpected_status", f"Something went wrong (HTTP {status}). Please try again later.")


def check_health(
    base_url: str = BACKEND_URL, transport: httpx.BaseTransport | None = None
) -> dict | None:
    """Return the backend /health payload, or None if it can't be reached."""
    try:
        with httpx.Client(base_url=base_url, timeout=3, transport=transport) as client:
            response = client.get("/health")
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError):
        return None
