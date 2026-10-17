from backend.config import PROJECT_ROOT
from backend.ingestion.documents import SourceFile, chunk_document, load_documents

POLICY = """# Overdraft Policy

Intro paragraph.

## Fees

Unarranged overdraft fee is $25.

### Daily cap

Maximum two fees per day.
"""


def test_load_skips_unsupported_and_keeps_empty_files(docs_dir):
    (docs_dir / "a.md").write_text(POLICY)
    (docs_dir / "empty.md").write_text("  \n\n")
    (docs_dir / "image.png").write_bytes(b"\x89PNG")
    (docs_dir / "sub").mkdir()
    (docs_dir / "sub" / "b.txt").write_text("plain text")

    files = {f.source: f.text for f in load_documents(docs_dir)}

    assert set(files) == {"a.md", "empty.md", "sub/b.txt"}
    assert files["empty.md"] == ""


def test_empty_document_produces_no_chunks():
    assert chunk_document(SourceFile("empty.md", ""), 1000, 100) == []


def test_chunk_metadata_preserves_source_title_and_section():
    chunks = chunk_document(SourceFile("overdraft.md", POLICY), 1000, 100)

    by_text = {c.page_content.splitlines()[0].strip(): c.metadata for c in chunks}
    assert by_text["## Fees"]["section"] == "Fees"
    assert by_text["### Daily cap"]["section"] == "Fees > Daily cap"
    for index, chunk in enumerate(chunks):
        assert chunk.metadata["source"] == "overdraft.md"
        assert chunk.metadata["title"] == "Overdraft Policy"
        assert chunk.metadata["chunk_index"] == index
        assert chunk.metadata["chunk_count"] == len(chunks)
        # Chroma only accepts scalar metadata values.
        assert all(isinstance(v, (str, int, float, bool)) for v in chunk.metadata.values())


def test_title_falls_back_to_filename_without_h1():
    chunks = chunk_document(SourceFile("card-disputes.txt", "No heading here."), 1000, 100)
    assert chunks[0].metadata["title"] == "Card Disputes"
    assert chunks[0].metadata["section"] == ""


def test_long_sections_are_split_within_size_with_overlap():
    body = " ".join(f"sentence{i}." for i in range(400))
    chunks = chunk_document(SourceFile("long.md", f"# Long\n\n## Body\n\n{body}"), 300, 50)

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 300 for c in chunks)
    assert all(c.metadata["section"] == "Body" for c in chunks[1:])
    # Consecutive chunks overlap: the next chunk starts with text from the end of the previous one.
    first_word_of_second = chunks[2].page_content.split()[0]
    assert first_word_of_second in chunks[1].page_content


def test_sample_corpus_loads_and_chunks():
    files = load_documents(PROJECT_ROOT / "data" / "raw")
    assert 5 <= len(files) <= 10
    for file in files:
        chunks = chunk_document(file, 1000, 150)
        assert chunks, file.source
        assert all(c.metadata["title"] for c in chunks)
