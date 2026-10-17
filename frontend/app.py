"""Streamlit chat UI for the Fernhollow Bank Knowledge Assistant.

Run with: streamlit run frontend/app.py (the API must be running; see README).
The backend decides everything about retrieval, answers, citations and abstention; this file
only sends questions to POST /ask and displays the results.
"""

import re

import streamlit as st

import backend_client as api

EXAMPLE_QUESTIONS = [
    "How much does it cost to use another bank's ATM?",
    "How long do I have to dispute a card payment?",
    "Can I use a mobile phone bill as proof of address?",
]


def escape_markdown(text: str) -> str:
    # Streamlit renders $...$ as LaTeX, which would mangle amounts like "$2.50 ... $3.00".
    return text.replace("$", r"\$")


def clean_excerpt(excerpt: str, section: str) -> str:
    """Strip Markdown heading/bold syntax and the repeated section heading from an excerpt."""
    text = re.sub(r"^#+\s*", "", excerpt.strip()).replace("**", "")
    if section and text.startswith(section):
        text = text[len(section) :].lstrip(" -:")
    return text


def render_citations(citations: list[api.Citation]) -> None:
    with st.expander(f"Sources ({len(citations)})"):
        for c in citations:
            heading = f"**[{c.id}] {escape_markdown(c.title)}**"
            if c.section:
                heading += f" — {escape_markdown(c.section)}"
            st.markdown(heading)
            st.caption(f"`{c.source}`")
            if c.excerpt:
                st.markdown(f"> {escape_markdown(clean_excerpt(c.excerpt, c.section))}")


def render_outcome(outcome: api.Outcome) -> None:
    if outcome.kind == "answer":
        st.markdown(escape_markdown(outcome.text))
        if outcome.citations:
            render_citations(outcome.citations)
    elif outcome.kind == "abstain":
        st.info(f"**Not covered by the documents**\n\n{escape_markdown(outcome.text)}", icon=":material/search_off:")
        if outcome.detail:
            st.caption(outcome.detail)
    else:
        message = f"**Something went wrong**\n\n{escape_markdown(outcome.text)}"
        if outcome.retryable:
            st.warning(message, icon=":material/warning:")
        else:
            st.error(message, icon=":material/error:")
        if outcome.detail:
            st.caption(escape_markdown(outcome.detail))


@st.cache_data(ttl=15, show_spinner=False)
def backend_health(base_url: str) -> dict | None:
    return api.check_health(base_url)


def render_sidebar() -> None:
    with st.sidebar:
        st.subheader("Fernhollow Bank Assistant")
        health = backend_health(api.BACKEND_URL)
        if health is None:
            st.error(f"API not reachable at {api.BACKEND_URL}", icon=":material/cloud_off:")
        elif not health.get("gemini_configured"):
            st.warning("API is running, but no Gemini API key is configured.", icon=":material/key_off:")
        else:
            st.success("Connected", icon=":material/check_circle:")
            st.caption(f"Model: {health.get('chat_model')}")

        st.markdown(
            "Answers come only from Fernhollow Bank's policy documents and cite their sources. "
            "If the documents don't cover a question, the assistant says so instead of guessing."
        )
        st.caption(
            "Each question is answered on its own: the assistant doesn't use earlier messages as "
            "context. Fernhollow Bank is fictional; this is a demo, not financial advice."
        )
        if st.button("Clear conversation", icon=":material/delete:", disabled=not st.session_state.messages):
            st.session_state.messages = []
            st.rerun()


def ask_example(question: str) -> None:
    st.session_state.pending_question = question


def main() -> None:
    st.set_page_config(page_title="Fernhollow Bank Assistant", page_icon="🏦", layout="centered")
    st.session_state.setdefault("messages", [])

    render_sidebar()
    st.title("Fernhollow Bank Assistant")
    st.caption("Ask about fees, overdrafts, card disputes, savings, transfers, security or complaints.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(escape_markdown(message["content"]))
            else:
                render_outcome(message["outcome"])

    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        for question in EXAMPLE_QUESTIONS:
            st.button(question, on_click=ask_example, args=(question,))

    prompt = st.chat_input("Ask a question about Fernhollow Bank policies", max_chars=api.MAX_QUESTION_LENGTH)
    question = prompt or st.session_state.pop("pending_question", None)
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(escape_markdown(question))
    with st.chat_message("assistant"):
        with st.spinner("Searching the policy documents…"):
            outcome = api.ask(question)
    st.session_state.messages.append({"role": "assistant", "content": outcome.text, "outcome": outcome})
    # Rerun to draw the reply from history (and hide the example questions).
    st.rerun()


main()
