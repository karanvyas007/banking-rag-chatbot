"""Idempotent indexing of source documents into a Chroma collection.

Each source file's chunks are replaced as a unit: unchanged files are skipped (no embedding calls),
changed files have their old chunks deleted and new ones added, and files that were removed or
are now empty have their chunks deleted.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from chromadb.api.models.Collection import Collection
from langchain_core.documents import Document

from backend.embeddings import GeminiEmbeddings
from backend.ingestion.documents import chunk_document, chunk_id, fingerprint, load_documents

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    skipped_empty: list[str] = field(default_factory=list)
    chunks_written: int = 0


def _indexed_fingerprints(collection: Collection) -> dict[str, set[str]]:
    result = collection.get(include=["metadatas"])
    indexed: dict[str, set[str]] = {}
    for metadata in result["metadatas"] or []:
        indexed.setdefault(metadata["source"], set()).add(metadata["fingerprint"])
    return indexed


def ingest(
    docs_dir: Path,
    collection: Collection,
    embeddings: GeminiEmbeddings,
    chunk_size: int,
    chunk_overlap: int,
) -> IngestReport:
    report = IngestReport()
    indexed = _indexed_fingerprints(collection)
    files = load_documents(docs_dir)

    pending: list[tuple[str, list[Document]]] = []
    for file in files:
        if not file.text:
            report.skipped_empty.append(file.source)
            continue
        if indexed.get(file.source) == {fingerprint(file.text, chunk_size, chunk_overlap)}:
            report.unchanged.append(file.source)
            continue
        pending.append((file.source, chunk_document(file, chunk_size, chunk_overlap)))

    # Embed everything before touching the index, so an API failure leaves it unchanged.
    all_chunks = [chunk for _, chunks in pending for chunk in chunks]
    vectors = embeddings.embed_documents(
        [c.page_content for c in all_chunks],
        titles=[" - ".join(filter(None, [c.metadata["title"], c.metadata["section"]])) for c in all_chunks],
    )

    offset = 0
    for source, chunks in pending:
        collection.delete(where={"source": source})
        collection.add(
            ids=[chunk_id(source, c.metadata["chunk_index"]) for c in chunks],
            embeddings=vectors[offset : offset + len(chunks)],
            documents=[c.page_content for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        offset += len(chunks)
        (report.updated if source in indexed else report.added).append(source)
        report.chunks_written += len(chunks)

    present = {f.source for f in files if f.text}
    for source in sorted(set(indexed) - present):
        collection.delete(where={"source": source})
        report.removed.append(source)

    return report
