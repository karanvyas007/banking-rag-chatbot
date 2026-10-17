"""Frontend tests. HTTP is mocked (httpx.MockTransport / patched client); no backend or Gemini calls."""

import json
import sys
from pathlib import Path

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
sys.path.insert(0, str(FRONTEND))  # the app imports backend_client the way `streamlit run` does

import backend_client as api  # noqa: E402

SECRET = "FAKE-KEY-do-not-leak-123"
ANSWER = {
    "answer": "Other banks' ATMs cost $2.50 per withdrawal; abroad it's $3.00 [1].",
    "answered": True,
    "abstain_reason": None,
    "citations": [
        {
            "id": 1,
            "source": "account-fees-schedule.md",
            "title": "Personal Account Fees Schedule",
            "section": "ATM and Cash Withdrawal Fees",
            "chunk_index": 3,
            "relevance": 0.77,
            "excerpt": "## ATM and Cash Withdrawal Fees - Withdrawals at other domestic ATMs cost **$2.50**.",
        }
    ],
    "retrieved": [],
    "model": "gemini-3.8-flash",
}
ABSTAIN = {
    "answer": "I couldn't find this in the available Fernhollow Bank documents.",
    "answered": False,
    "abstain_reason": "not_in_documents",
    "citations": [],
    "retrieved": [],
    "model": "gemini-3.8-flash",
}


def transport(status=200, body=None, headers=None, raises=None, seen=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if raises:
            raise raises
        content = body if isinstance(body, (str, bytes)) else json.dumps(body)
        return httpx.Response(status, content=content, headers=headers)

    return httpx.MockTransport(handler)


def ask(**kwargs):
    question = kwargs.pop("question", "What is the ATM fee?")
    return api.ask(question, base_url="http://backend.test", transport=transport(**kwargs))


# --- Request handling ------------------------------------------------------------------------


def test_posts_stripped_question_to_ask_endpoint():
    seen = []
    api.ask("  What is the ATM fee?  ", base_url="http://backend.test", transport=transport(body=ANSWER, seen=seen))

    [request] = seen
    assert request.method == "POST"
    assert str(request.url) == "http://backend.test/ask"
    assert json.loads(request.content) == {"question": "What is the ATM fee?"}


@pytest.mark.parametrize("question", ["", "  ", "hi", "x" * 1001])
def test_invalid_questions_are_rejected_without_a_request(question):
    seen = []
    outcome = api.ask(question, base_url="http://backend.test", transport=transport(body=ANSWER, seen=seen))
    assert (outcome.kind, outcome.reason) == ("error", "invalid_question")
    assert seen == []


def test_answer_is_parsed_with_citations():
    outcome = ask(body=ANSWER)

    assert outcome.kind == "answer"
    assert outcome.text == ANSWER["answer"]
    [c] = outcome.citations
    assert (c.id, c.title, c.section, c.source) == (
        1,
        "Personal Account Fees Schedule",
        "ATM and Cash Withdrawal Fees",
        "account-fees-schedule.md",
    )
    assert "$2.50" in c.excerpt


def test_abstention_is_distinct_from_error():
    outcome = ask(body=ABSTAIN)
    assert outcome.kind == "abstain"
    assert outcome.reason == "not_in_documents"
    assert outcome.text == ABSTAIN["answer"]  # the backend's message is shown as-is
    assert "don't answer this question" in outcome.detail


# --- Error states ----------------------------------------------------------------------------

TRACEBACK = f'Traceback (most recent call last):\n  File "qa.py"\nGOOGLE_API_KEY={SECRET}'


@pytest.mark.parametrize(
    ("kwargs", "reason", "retryable"),
    [
        ({"status": 422, "body": {"detail": [{"msg": TRACEBACK}]}}, "invalid_question", False),
        ({"status": 429, "body": {"detail": TRACEBACK}, "headers": {"Retry-After": "30"}}, "rate_limited", True),
        ({"status": 502, "body": {"detail": TRACEBACK}}, "model_error", True),
        ({"status": 500, "body": TRACEBACK}, "unexpected_status", False),
        ({"raises": httpx.ReadTimeout("timed out")}, "timeout", True),
        ({"raises": httpx.ConnectError(f"refused {SECRET}")}, "backend_unreachable", True),
        ({"status": 200, "body": "<html>not json</html>"}, "unexpected_response", False),
        ({"status": 200, "body": {"unexpected": "shape"}}, "unexpected_response", False),
        ({"status": 200, "body": {**ANSWER, "citations": [{"id": "x"}]}}, "unexpected_response", False),
    ],
)
def test_error_states_use_safe_messages(kwargs, reason, retryable):
    outcome = ask(**kwargs)

    assert (outcome.kind, outcome.reason, outcome.retryable) == ("error", reason, retryable)
    shown = f"{outcome.text} {outcome.detail or ''}"
    assert SECRET not in shown
    assert "Traceback" not in shown


def test_rate_limit_mentions_retry_after():
    outcome = ask(status=429, body={"detail": "x"}, headers={"Retry-After": "30"})
    assert "30 seconds" in outcome.text


def test_503_shows_backend_detail_and_ignores_non_json_bodies():
    detail = "Document index 'banking_docs' does not exist. Build it with: python -m backend.ingestion"
    assert ask(status=503, body={"detail": detail}).detail == detail
    html = ask(status=503, body="<html>Service Unavailable</html>")
    assert (html.reason, html.detail) == ("service_unavailable", None)


def test_health_check_handles_unreachable_backend():
    up = api.check_health("http://backend.test", transport=transport(body={"status": "ok"}))
    down = api.check_health("http://backend.test", transport=transport(raises=httpx.ConnectError("x")))
    assert up == {"status": "ok"}
    assert down is None


# --- Streamlit UI ----------------------------------------------------------------------------

HEALTHY = {"status": "ok", "gemini_configured": True, "chat_model": "gemini-3.8-flash"}


@pytest.fixture
def app(monkeypatch):
    """Run the Streamlit app with the backend client patched. Returns (app, asked questions)."""
    st.cache_data.clear()
    asked, replies = [], []
    monkeypatch.setattr(api, "check_health", lambda base_url=None: HEALTHY)

    def fake_ask(question, **kwargs):
        asked.append(question)
        return replies.pop(0)

    monkeypatch.setattr(api, "ask", fake_ask)

    def start(*outcomes, health=HEALTHY):
        monkeypatch.setattr(api, "check_health", lambda base_url=None: health)
        replies.extend(outcomes)
        return AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=10).run(), asked

    return start


def send(at, text):
    at.chat_input[0].set_value(text).run()
    return at


def all_text(at):
    return " ".join(e.value for e in [*at.markdown, *at.caption, *at.info, *at.warning, *at.error])


def test_initial_page_renders_examples_and_status(app):
    at, _ = app()

    assert not at.exception
    assert at.title[0].value == "Fernhollow Bank Assistant"
    assert len(at.chat_input) == 1
    assert [b.label for b in at.button if b.label.endswith("?")] == [
        "How much does it cost to use another bank's ATM?",
        "How long do I have to dispute a card payment?",
        "Can I use a mobile phone bill as proof of address?",
    ]
    assert at.sidebar.success[0].value == "Connected"
    assert "doesn't use earlier messages" in all_text(at)


def test_answer_renders_with_escaped_amounts_and_sources(app):
    at, asked = app(api._parse_answer(ANSWER))

    send(at, "What is the ATM fee?")

    assert not at.exception
    assert asked == ["What is the ATM fee?"]
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    assert r"\$2.50 per withdrawal; abroad it's \$3.00 [1]." in all_text(at)  # not rendered as LaTeX
    [sources] = at.expander
    assert sources.label == "Sources (1)"
    assert "**[1] Personal Account Fees Schedule** — ATM and Cash Withdrawal Fees" in all_text(at)
    assert "`account-fees-schedule.md`" in all_text(at)
    # Excerpt is shown without the repeated heading or Markdown syntax.
    assert r"> Withdrawals at other domestic ATMs cost \$2.50." in all_text(at)


def test_abstention_is_shown_as_not_covered_not_as_error(app):
    at, _ = app(api._parse_answer(ABSTAIN))

    send(at, "Who is the CEO?")

    assert len(at.info) == 1
    assert at.info[0].value.startswith("**Not covered by the documents**")
    assert not at.error and not at.warning
    assert not at.expander


@pytest.mark.parametrize(
    ("outcome", "element"),
    [
        (api.Outcome(kind="error", text="Temporarily unavailable.", reason="service_unavailable", retryable=True), "warning"),
        (api.Outcome(kind="error", text="Something odd.", reason="unexpected_status"), "error"),
    ],
)
def test_errors_are_shown_as_errors_not_abstentions(app, outcome, element):
    at, _ = app(outcome)

    send(at, "What is the ATM fee?")

    assert not at.exception
    [box] = getattr(at, element)
    assert box.value.startswith("**Something went wrong**")
    assert not at.info


def test_history_persists_across_reruns_and_can_be_cleared(app):
    at, asked = app(api._parse_answer(ANSWER), api._parse_answer(ABSTAIN))

    send(at, "First question?")
    send(at, "Second question?")
    at.run()  # an unrelated rerun keeps the conversation

    assert asked == ["First question?", "Second question?"]
    assert [m.name for m in at.chat_message] == ["user", "assistant", "user", "assistant"]
    assert len(at.session_state["messages"]) == 4

    at.sidebar.button[0].click().run()
    assert at.session_state["messages"] == []
    assert not at.chat_message


def test_example_question_button_sends_question(app):
    at, asked = app(api._parse_answer(ANSWER))

    next(b for b in at.button if b.label.startswith("How much")).click().run()

    assert asked == ["How much does it cost to use another bank's ATM?"]
    assert not [b for b in at.button if b.label.endswith("?")]  # examples hidden once chatting


def test_unreachable_backend_is_shown_in_sidebar(app):
    at, _ = app(health=None)
    assert not at.exception
    assert "API not reachable" in at.sidebar.error[0].value
