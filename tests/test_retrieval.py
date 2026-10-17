import pytest

from backend.retrieval.retriever import Retriever
from backend.vectorstore import IndexConfigMismatch, IndexNotReady, get_collection, open_store


def test_relevant_chunk_is_ranked_first_with_metadata(retriever):
    chunks = retriever.retrieve("What is the ATM fee?")

    top = chunks[0]
    assert (top.source, top.title, top.section) == ("fees.md", "Fees Schedule", "ATM Fees")
    assert top.id == "fees.md::0000"
    assert top.chunk_index == 0
    assert "$2.50" in top.text
    assert 0.5 <= top.relevance <= 1.0
    assert [c.relevance for c in chunks] == sorted((c.relevance for c in chunks), reverse=True)


def test_paraphrased_question_retrieves_same_chunk(retriever):
    direct = retriever.retrieve("ATM fee")[0]
    paraphrased = retriever.retrieve("How much is it to use a cash machine at another bank?")[0]
    assert paraphrased.id == direct.id


def test_unrelated_question_returns_nothing_above_threshold(retriever):
    assert retriever.retrieve("What's the weather in Paris?") == []


def test_threshold_and_top_k_limit_results(qa_store):
    assert len(Retriever(qa_store, top_k=4, min_relevance=0.0).retrieve("overdraft")) == 4
    assert len(Retriever(qa_store, top_k=4, min_relevance=0.0).retrieve("overdraft", top_k=2)) == 2
    strict = Retriever(qa_store, top_k=4, min_relevance=0.5).retrieve("overdraft")
    assert [c.source for c in strict] == ["overdraft.md"]


def test_query_is_embedded_with_query_embedding(retriever, concept_embeddings):
    retriever.retrieve("dispute deadline")
    assert concept_embeddings.queries == ["dispute deadline"]


def test_missing_index_raises_not_ready(tmp_path, concept_embeddings):
    with pytest.raises(IndexNotReady, match="does not exist"):
        open_store(tmp_path / "none", "banking_docs", "concept-fake", 6, concept_embeddings)
    # Opening for retrieval must not create the collection as a side effect.
    with pytest.raises(IndexNotReady):
        open_store(tmp_path / "none", "banking_docs", "concept-fake", 6, concept_embeddings)


def test_empty_index_raises_not_ready(tmp_path, concept_embeddings):
    get_collection(tmp_path / "chroma", "banking_docs", "concept-fake", 6)
    with pytest.raises(IndexNotReady, match="empty"):
        open_store(tmp_path / "chroma", "banking_docs", "concept-fake", 6, concept_embeddings)


def test_index_emptied_after_startup_raises_not_ready(qa_store, retriever):
    qa_store.delete(ids=qa_store.get(include=[])["ids"])
    with pytest.raises(IndexNotReady):
        retriever.retrieve("ATM fee")


def test_incompatible_index_raises_mismatch(tmp_path, qa_store, concept_embeddings):
    persist = tmp_path / "qa_chroma"
    with pytest.raises(IndexConfigMismatch):
        open_store(persist, "qa_test", "gemini-embedding-2", 3072, concept_embeddings)
