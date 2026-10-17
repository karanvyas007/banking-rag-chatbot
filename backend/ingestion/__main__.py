"""Index documents into ChromaDB. Usage: python -m backend.ingestion [--reset]"""

import argparse
import logging
import sys

from backend.config import get_settings
from backend.embeddings import GeminiEmbeddings
from backend.ingestion.indexer import ingest
from backend.vectorstore import IndexConfigMismatch, get_collection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="delete and rebuild the collection")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    settings = get_settings()
    if not settings.gemini_configured:
        print("GOOGLE_API_KEY is not set. See .env.example.", file=sys.stderr)
        return 1

    try:
        collection = get_collection(
            settings.chroma_persist_dir,
            settings.chroma_collection,
            settings.gemini_embedding_model,
            settings.embedding_dimension,
            reset=args.reset,
        )
    except IndexConfigMismatch as e:
        print(e, file=sys.stderr)
        return 1

    embeddings = GeminiEmbeddings(
        model=settings.gemini_embedding_model,
        api_key=settings.google_api_key.get_secret_value(),
        dimension=settings.embedding_dimension,
    )
    report = ingest(
        settings.docs_dir, collection, embeddings, settings.chunk_size, settings.chunk_overlap
    )

    for label in ("added", "updated", "unchanged", "removed", "skipped_empty"):
        items = getattr(report, label)
        print(f"{label:>13}: {len(items)} {items if items else ''}")
    print(f"chunks written: {report.chunks_written} | collection total: {collection.count()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
