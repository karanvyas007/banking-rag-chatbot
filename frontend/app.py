"""Streamlit UI. Run with: streamlit run frontend/app.py"""

import os

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Banking Knowledge Assistant", page_icon="🏦")
st.title("Banking Knowledge Assistant")
st.caption("Answers questions from banking policy documents and FAQs, with cited sources.")

try:
    health = httpx.get(f"{BACKEND_URL}/health", timeout=5).json()
    st.success(f"Backend reachable at {BACKEND_URL}")
    if not health.get("gemini_configured"):
        st.warning("GOOGLE_API_KEY is not set on the backend. See .env.example.")
except httpx.HTTPError:
    st.error(f"Backend not reachable at {BACKEND_URL}. Start it with: uvicorn backend.main:app --reload")

st.chat_input("Ask about a banking policy (coming soon)", disabled=True)
