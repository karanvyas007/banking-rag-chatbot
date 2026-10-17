"""Load source documents and split them into chunks with citation metadata."""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt"}
# Bump when chunking or metadata logic changes, so unchanged files are re-indexed anyway.
CHUNKING_VERSION = "1"

_HEADERS = [("#", "title"), ("##", "section"), ("###", "subsection")]


@dataclass(frozen=True)
class SourceFile:
    source: str  # path relative to the docs directory, used as the citation key
    text: str


def load_documents(docs_dir: Path) -> list[SourceFile]:
    """Read supported files under docs_dir. Empty files are returned with empty text."""
    files = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            logger.warning("Skipping empty file: %s", path)
        files.append(SourceFile(source=path.relative_to(docs_dir).as_posix(), text=text))
    return files


def fingerprint(text: str, chunk_size: int, chunk_overlap: int) -> str:
    key = f"{CHUNKING_VERSION}:{chunk_size}:{chunk_overlap}:{text}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def chunk_document(file: SourceFile, chunk_size: int, chunk_overlap: int) -> list[Document]:
    if not file.text:
        return []

    sections = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=False).split_text(file.text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(sections)

    doc_title = next((c.metadata["title"] for c in chunks if c.metadata.get("title")), None)
    title = doc_title or Path(file.source).stem.replace("-", " ").replace("_", " ").title()
    doc_fingerprint = fingerprint(file.text, chunk_size, chunk_overlap)

    for index, chunk in enumerate(chunks):
        headings = [chunk.metadata.get(key) for key in ("section", "subsection")]
        # Chroma metadata values must be str/int/float/bool, so missing values become "".
        chunk.metadata = {
            "source": file.source,
            "title": title,
            "section": " > ".join(h for h in headings if h),
            "chunk_index": index,
            "chunk_count": len(chunks),
            "fingerprint": doc_fingerprint,
        }
    return chunks


def chunk_id(source: str, index: int) -> str:
    return f"{source}::{index:04d}"
