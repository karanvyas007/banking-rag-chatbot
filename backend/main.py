"""FastAPI entry point. Run with: uvicorn backend.main:app --reload"""

from fastapi import FastAPI

from backend.config import get_settings

app = FastAPI(title="Banking Knowledge Assistant", version="0.1.0")


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "gemini_configured": settings.gemini_configured,
        "chat_model": settings.gemini_chat_model,
        "embedding_model": settings.gemini_embedding_model,
    }
