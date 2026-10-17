import pytest

from backend.ingestion.indexer import ingest
from backend.vectorstore import IndexConfigMismatch, get_collection

DOC_A = "# Fees\n\n## ATM\n\nOther banks' ATMs cost $2.50."
DOC_B = "# Savings\n\n## Rates\n\nEasy Access Saver pays 3.10% AER."


def run(docs_dir, collection, embeddings):
    return ingest(docs_dir, collection, embeddings, chunk_size=1000, chunk_overlap=100)


def test_ingest_indexes_chunks_with_metadata(docs_dir, collection, embeddings):
    (docs_dir / "a.md").write_text(DOC_A)
    (docs_dir / "b.md").write_text(DOC_B)

    report = run(docs_dir, collection, embeddings)

    assert report.added == ["a.md", "b.md"]
    stored = collection.get(ids=["a.md::0000"], include=["metadatas", "documents"])
    assert stored["metadatas"][0]["title"] == "Fees"
    assert stored["metadatas"][0]["section"] == "ATM"
    assert "$2.50" in stored["documents"][0]


def test_rerun_is_idempotent_and_skips_embedding(docs_dir, collection, embeddings):
    (docs_dir / "a.md").write_text(DOC_A)
    (docs_dir / "b.md").write_text(DOC_B)
    run(docs_dir, collection, embeddings)
    count, embedded = collection.count(), len(embeddings.embedded)

    report = run(docs_dir, collection, embeddings)

    assert report.unchanged == ["a.md", "b.md"]
    assert report.chunks_written == 0
    assert collection.count() == count
    assert len(embeddings.embedded) == embedded  # no new embedding calls


def test_changed_document_replaces_old_chunks(docs_dir, collection, embeddings):
    long_doc = DOC_A + "\n\n" + "\n\n".join(f"## Section {i}\n\nDetails {i}." for i in range(5))
    (docs_dir / "a.md").write_text(long_doc)
    run(docs_dir, collection, embeddings)
    before = collection.count()

    (docs_dir / "a.md").write_text(DOC_A.replace("$2.50", "$3.00"))
    report = run(docs_dir, collection, embeddings)

    assert report.updated == ["a.md"]
    assert collection.count() < before  # stale higher-index chunks are gone
    docs = collection.get(include=["documents"])["documents"]
    assert any("$3.00" in d for d in docs)
    assert not any("$2.50" in d for d in docs)


def test_removed_and_emptied_documents_are_deleted(docs_dir, collection, embeddings):
    (docs_dir / "a.md").write_text(DOC_A)
    (docs_dir / "b.md").write_text(DOC_B)
    run(docs_dir, collection, embeddings)

    (docs_dir / "a.md").unlink()
    (docs_dir / "b.md").write_text("")
    report = run(docs_dir, collection, embeddings)

    assert report.removed == ["a.md", "b.md"]
    assert report.skipped_empty == ["b.md"]
    assert collection.count() == 0


def test_empty_directory_indexes_nothing(docs_dir, collection, embeddings):
    report = run(docs_dir, collection, embeddings)
    assert collection.count() == 0
    assert report.chunks_written == 0


def test_embedding_failure_leaves_index_unchanged(docs_dir, collection, embeddings):
    (docs_dir / "a.md").write_text(DOC_A)
    run(docs_dir, collection, embeddings)
    before = collection.get(include=["documents"])

    class FailingEmbeddings:
        def embed_documents(self, texts, titles=None):
            raise RuntimeError("API unavailable")

    (docs_dir / "a.md").write_text(DOC_A.replace("$2.50", "$3.00"))
    with pytest.raises(RuntimeError):
        run(docs_dir, collection, FailingEmbeddings())

    assert collection.get(include=["documents"]) == before


def test_collection_rejects_different_embedding_config(tmp_path):
    get_collection(tmp_path / "chroma", "col", "model-a", 8)

    with pytest.raises(IndexConfigMismatch):
        get_collection(tmp_path / "chroma", "col", "model-b", 8)
    with pytest.raises(IndexConfigMismatch):
        get_collection(tmp_path / "chroma", "col", "model-a", 16)


def test_reset_rebuilds_collection_with_new_config(tmp_path):
    get_collection(tmp_path / "chroma", "col", "model-a", 8)
    collection = get_collection(tmp_path / "chroma", "col", "model-b", 16, reset=True)
    assert collection.metadata["embedding_model"] == "model-b"
    assert collection.configuration["hnsw"]["space"] == "cosine"
