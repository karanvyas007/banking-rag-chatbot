"""ChromaDB access. The collection records which embedding model and dimension built it."""

from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings


class IndexConfigMismatch(RuntimeError):
    pass


def get_collection(
    persist_dir: Path, name: str, embedding_model: str, embedding_dimension: int, reset: bool = False
) -> Collection:
    client = chromadb.PersistentClient(
        path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
    )
    if reset and name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)

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
