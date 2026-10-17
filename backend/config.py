"""Application settings, loaded from environment variables and an optional .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Optional so the app can start (and report its status) before a key is configured.
    google_api_key: SecretStr | None = None
    gemini_chat_model: str = "gemini-3.8-flash"
    gemini_embedding_model: str = "gemini-embedding-2"
    # Fixed per collection: changing it requires re-indexing (python -m backend.ingestion --reset).
    embedding_dimension: int = 3072

    chroma_persist_dir: Path = PROJECT_ROOT / "data" / "chroma"
    chroma_collection: str = "banking_docs"
    docs_dir: Path = PROJECT_ROOT / "data" / "raw"

    chunk_size: int = 1000
    chunk_overlap: int = 150

    @field_validator("chroma_persist_dir", "docs_dir")
    @classmethod
    def _resolve_relative_to_project(cls, path: Path) -> Path:
        # Relative paths in .env refer to the project root, not the current working directory.
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def gemini_configured(self) -> bool:
        return bool(self.google_api_key and self.google_api_key.get_secret_value())


@lru_cache
def get_settings() -> Settings:
    return Settings()
