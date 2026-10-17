import logging

import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors
from langchain_google_genai.chat_models import (
    ChatGoogleGenerativeAIError,
    GoogleAPIError,
    GoogleRateLimitError,
)

import backend.main as main
from backend.config import Settings

FAKE_KEY = "FAKE-KEY-do-not-leak-123"
ANSWER = {"answerable": True, "answer": "It costs $2.50 [1].", "citations": [1]}


@pytest.fixture
def client_for(make_service):
    def make(reply=ANSWER, error=None):
        service, llm = make_service(reply, error)
        main.app.dependency_overrides[main.get_qa_service] = lambda: service
        return TestClient(main.app), llm

    yield make
    main.app.dependency_overrides.clear()


def test_ask_returns_structured_answer(client_for):
    client, _ = client_for()

    response = client.post("/ask", json={"question": "  What is the ATM fee?  ", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is True
    assert body["abstain_reason"] is None
    assert body["citations"][0]["source"] == "fees.md"
    assert set(body["citations"][0]) == {"id", "source", "title", "section", "chunk_index", "relevance", "excerpt"}
    assert len(body["retrieved"]) <= 2


def test_unsupported_question_returns_abstention(client_for):
    client, llm = client_for()
    body = client.post("/ask", json={"question": "What's the weather in Paris?"}).json()
    assert body["answered"] is False
    assert body["abstain_reason"] == "no_relevant_documents"
    assert llm.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 1001},
        {"question": 123},
        {"question": "ATM fee?", "top_k": 0},
        {"question": "ATM fee?", "top_k": 11},
        {"question": "ATM fee?", "unexpected": True},
    ],
)
def test_malformed_requests_are_rejected(client_for, payload):
    client, llm = client_for()
    assert client.post("/ask", json=payload).status_code == 422
    assert llm.calls == []


def test_non_json_body_is_rejected(client_for):
    client, _ = client_for()
    response = client.post("/ask", content="question=ATM", headers={"content-type": "text/plain"})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "error",
    [
        GoogleRateLimitError(f"429 RESOURCE_EXHAUSTED key={FAKE_KEY}"),
        genai_errors.ClientError(429, {"error": {"code": 429, "message": f"quota {FAKE_KEY}"}}),
    ],
)
def test_rate_limit_returns_429_without_leaking_details(client_for, caplog, error):
    client, _ = client_for(error=error)
    with caplog.at_level(logging.DEBUG):
        response = client.post("/ask", json={"question": "What is the ATM fee?"})

    assert response.status_code == 429
    assert response.headers["retry-after"] == "30"
    assert FAKE_KEY not in response.text
    assert FAKE_KEY not in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        GoogleAPIError(503, {"error": {"code": 503, "message": f"high demand {FAKE_KEY}"}}),
        genai_errors.ServerError(503, {"error": {"code": 503, "message": f"unavailable {FAKE_KEY}"}}),
    ],
)
def test_gemini_overload_returns_503_with_retry_after(client_for, caplog, error):
    client, _ = client_for(error=error)
    with caplog.at_level(logging.DEBUG):
        response = client.post("/ask", json={"question": "What is the ATM fee?"})

    assert response.status_code == 503
    assert response.headers["retry-after"] == "30"
    assert FAKE_KEY not in response.text
    assert FAKE_KEY not in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        ChatGoogleGenerativeAIError(f"500 INTERNAL {FAKE_KEY}"),
        genai_errors.ServerError(500, {"error": {"code": 500, "message": f"internal {FAKE_KEY}"}}),
        genai_errors.ClientError(403, {"error": {"code": 403, "message": f"permission {FAKE_KEY}"}}),
    ],
)
def test_gemini_errors_return_502_without_leaking_details(client_for, caplog, error):
    client, _ = client_for(error=error)
    with caplog.at_level(logging.DEBUG):
        response = client.post("/ask", json={"question": "What is the ATM fee?"})

    assert response.status_code == 502
    assert response.json() == {"detail": "The language model service failed. Please retry later."}
    assert FAKE_KEY not in response.text
    assert FAKE_KEY not in caplog.text


def test_malformed_model_output_returns_502(client_for):
    client, _ = client_for(reply="this is not json")
    assert client.post("/ask", json={"question": "What is the ATM fee?"}).status_code == 502


@pytest.fixture
def real_service_with(monkeypatch, tmp_path):
    """Exercise the real build_qa_service wiring with the given settings (no network calls)."""

    def use(**overrides):
        settings = Settings(_env_file=None, chroma_persist_dir=tmp_path / "chroma", **overrides)
        monkeypatch.setattr(main, "get_settings", lambda: settings)
        monkeypatch.setattr(main, "_service", None)
        return TestClient(main.app)

    return use


def test_missing_index_returns_503(real_service_with):
    client = real_service_with(google_api_key=FAKE_KEY)

    response = client.post("/ask", json={"question": "What is the ATM fee?"})

    assert response.status_code == 503
    assert "python -m backend.ingestion" in response.json()["detail"]
    assert FAKE_KEY not in response.text


def test_missing_api_key_returns_503(real_service_with):
    client = real_service_with(google_api_key=None)
    response = client.post("/ask", json={"question": "What is the ATM fee?"})
    assert response.status_code == 503
    assert "GOOGLE_API_KEY" in response.json()["detail"]


def test_health_still_works_without_index(real_service_with):
    assert real_service_with(google_api_key=None).get("/health").status_code == 200
