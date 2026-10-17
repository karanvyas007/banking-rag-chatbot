from types import SimpleNamespace

import pytest

from backend.embeddings import BATCH_SIZE, GeminiEmbeddings


class FakeModels:
    def __init__(self, dimension):
        self.dimension = dimension
        self.calls = []

    def embed_content(self, model, contents, config):
        self.calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.0] * self.dimension) for _ in contents])


def make(dimension=4, returned_dimension=None):
    models = FakeModels(returned_dimension or dimension)
    client = SimpleNamespace(models=models)
    return GeminiEmbeddings("gemini-embedding-2", "unused", dimension, client=client), models


def texts_sent(call):
    return [c.parts[0].text for c in call.contents]


def test_documents_use_title_prefix_and_no_task_type():
    emb, models = make()
    emb.embed_documents(["Fee is $5.", "No title."], titles=["Fees - ATM", None])

    call = models.calls[0]
    assert texts_sent(call) == ["title: Fees - ATM | text: Fee is $5.", "title: none | text: No title."]
    assert call.config.task_type is None
    assert call.config.output_dimensionality == 4


def test_query_uses_search_prefix():
    emb, models = make()
    emb.embed_query("What is the ATM fee?")
    assert texts_sent(models.calls[0]) == ["task: search result | query: What is the ATM fee?"]
    assert models.calls[0].config.task_type is None


def test_documents_are_batched():
    emb, models = make()
    vectors = emb.embed_documents(["x"] * (BATCH_SIZE + 1))
    assert len(vectors) == BATCH_SIZE + 1
    assert [len(c.contents) for c in models.calls] == [BATCH_SIZE, 1]


def test_dimension_mismatch_is_rejected():
    emb, _ = make(dimension=4, returned_dimension=3)
    with pytest.raises(RuntimeError, match="dimension"):
        emb.embed_query("q")
