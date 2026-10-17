"""ChromaDB access. The collection records which embedding model and dimension built it."""

from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

REINDEX_HINT = "Build it with: python -m backend.ingestion"


class IndexConfigMismatch(RuntimeError):
    pass


class IndexNotReady(RuntimeError):
    pass


def _client(persist_dir: Path) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
    )


def get_collection(
    persist_dir: Path,
    name: str,
    embedding_model: str,
    embedding_dimension: int,
    reset: bool = False,
    create: bool = True,
) -> Collection:
    client = _client(persist_dir)
    exists = name in [c.name for c in client.list_collections()]
    if reset and exists:
        client.delete_collection(name)
    elif not create and not exists:
        raise IndexNotReady(f"Document index '{name}' does not exist. {REINDEX_HINT}")

    expected = {"embedding_model": embedding_model, "embedding_dimension": embedding_dimension}
    collection = client.get_or_create_collection(
        name,
        metadata=expected,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,  # embeddings are always supplied explicitly
    )
    actual = {key: (collection.metadata or {}).get(key) for key in expected}
    if actual != expected:
        raise IndexConfigMismatch(
            f"Collection '{name}' was built with {actual}, but config is {expected}. "
            "Re-index with: python -m backend.ingestion --reset"
        )
    return collection


def open_store(
    persist_dir: Path, name: str, embedding_model: str, embedding_dimension: int, embeddings: Embeddings
) -> Chroma:
    """Open an existing, compatible, non-empty index for querying. Never creates one."""
    collection = get_collection(persist_dir, name, embedding_model, embedding_dimension, create=False)
    if collection.count() == 0:
        raise IndexNotReady(f"Document index '{name}' is empty. {REINDEX_HINT}")
    return Chroma(
        client=_client(persist_dir),
        collection_name=name,
        embedding_function=embeddings,
        create_collection_if_not_exists=False,
    )
