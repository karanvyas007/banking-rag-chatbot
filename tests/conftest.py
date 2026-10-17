import hashlib

import pytest

from backend.vectorstore import get_collection

DIM = 8


class FakeEmbeddings:
    """Deterministic, offline stand-in for GeminiEmbeddings that records what it embeds."""

    def __init__(self):
        self.embedded: list[str] = []

    def embed_documents(self, texts, titles=None):
        self.embedded.extend(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)

    @staticmethod
    def _vector(text):
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255 for b in digest[:DIM]]


@pytest.fixture
def embeddings():
    return FakeEmbeddings()


@pytest.fixture
def collection(tmp_path):
    return get_collection(tmp_path / "chroma", "test", "fake-model", DIM)


@pytest.fixture
def docs_dir(tmp_path):
    path = tmp_path / "docs"
    path.mkdir()
    return path
