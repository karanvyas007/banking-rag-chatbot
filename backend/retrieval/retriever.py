"""Similarity search over the Chroma index, filtered by a relevance threshold."""

from dataclasses import dataclass

from langchain_chroma import Chroma

from backend.vectorstore import REINDEX_HINT, IndexNotReady


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    source: str
    title: str
    section: str
    chunk_index: int
    relevance: float


class Retriever:
    def __init__(self, store: Chroma, top_k: int, min_relevance: float):
        self.store = store
        self.top_k = top_k
        self.min_relevance = min_relevance

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return up to top_k chunks scoring at least min_relevance, most relevant first."""
        # The index may have been emptied since startup (e.g. all documents removed).
        if not self.store.get(limit=1, include=[])["ids"]:
            raise IndexNotReady(f"Document index is empty. {REINDEX_HINT}")

        # Uses GeminiEmbeddings.embed_query, i.e. the same model and query prefix as the index.
        results = self.store.similarity_search_with_score(question, k=top_k or self.top_k)
        chunks = []
        for doc, distance in results:
            relevance = round(1.0 - distance, 4)  # the collection uses cosine distance
            if relevance < self.min_relevance:
                continue
            m = doc.metadata
            chunks.append(
                RetrievedChunk(
                    id=doc.id,
                    text=doc.page_content,
                    source=m["source"],
                    title=m["title"],
                    section=m.get("section", ""),
                    chunk_index=m["chunk_index"],
                    relevance=relevance,
                )
            )
        return chunks
