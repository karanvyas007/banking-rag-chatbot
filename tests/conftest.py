import hashlib
import json

import pytest
from langchain_core.messages import AIMessage

from backend.ingestion.indexer import ingest
from backend.retrieval.qa import QAService
from backend.retrieval.retriever import Retriever
from backend.vectorstore import get_collection, open_store

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


# --- Retrieval / QA fakes -------------------------------------------------------------------

# Synonyms map to the same dimension, so paraphrases ("cash machine" vs "ATM") are similar,
# mimicking semantic embeddings without network calls.
CONCEPTS = [
    ("atm", "cash machine"),
    ("overdraft", "overdrawn"),
    ("dispute", "chargeback", "challenge a card payment"),
    ("savings", "saver"),
]


class ConceptEmbeddings:
    model = "concept-fake"
    dimension = len(CONCEPTS) + 2

    def __init__(self):
        self.queries: list[str] = []

    def embed_documents(self, texts, titles=None):
        return [self._vector(t, is_query=False) for t in texts]

    def embed_query(self, text):
        self.queries.append(text)
        return self._vector(text, is_query=True)

    @staticmethod
    def _vector(text, is_query):
        text = text.lower()
        counts = [float(sum(text.count(term) for term in terms)) for terms in CONCEPTS]
        # Separate small doc/query components keep vectors non-zero, and make concept-free
        # queries unrelated to concept-free documents.
        return counts + ([0.0, 0.1] if is_query else [0.1, 0.0])


QA_DOCS = {
    "fees.md": "# Fees Schedule\n\n## ATM Fees\n\nUsing another bank's ATM costs $2.50 per ATM withdrawal."
    "\n\n## Paper Statements\n\nEach paper statement costs $2.00.",
    "overdraft.md": "# Overdraft Policy\n\n## Unarranged Overdrafts\n\nIf your account is overdrawn without an "
    "arranged overdraft, a fee of $25 applies.",
    "disputes.md": "# Card Disputes\n\n## Time Limits\n\nRaise a dispute within 120 days of the transaction.",
}


class FakeLLM:
    """Stand-in chat model: returns a canned JSON reply (or raises) and records prompts."""

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return AIMessage(content=self.reply if isinstance(self.reply, str) else json.dumps(self.reply))


@pytest.fixture
def concept_embeddings():
    return ConceptEmbeddings()


@pytest.fixture
def qa_store(tmp_path, concept_embeddings):
    docs = tmp_path / "qa_docs"
    docs.mkdir()
    for name, text in QA_DOCS.items():
        (docs / name).write_text(text)
    persist = tmp_path / "qa_chroma"
    e = concept_embeddings
    ingest(docs, get_collection(persist, "qa_test", e.model, e.dimension), e, chunk_size=1000, chunk_overlap=0)
    return open_store(persist, "qa_test", e.model, e.dimension, e)


@pytest.fixture
def retriever(qa_store):
    return Retriever(qa_store, top_k=4, min_relevance=0.5)


@pytest.fixture
def make_service(retriever):
    def make(reply=None, error=None):
        llm = FakeLLM(reply, error)
        return QAService(retriever, llm, "fake-chat-model"), llm

    return make
