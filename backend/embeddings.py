"""LangChain-compatible embeddings for gemini-embedding-2.

gemini-embedding-2 does not accept the `task_type` field (which langchain-google-genai always
sends); instead the task is expressed as a text prefix. See
https://ai.google.dev/gemini-api/docs/embeddings
"""

from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings

BATCH_SIZE = 100


def format_document(text: str, title: str | None = None) -> str:
    return f"title: {title or 'none'} | text: {text}"


def format_query(text: str) -> str:
    return f"task: search result | query: {text}"


class GeminiEmbeddings(Embeddings):
    def __init__(self, model: str, api_key: str, dimension: int, client: genai.Client | None = None):
        self.model = model
        self.dimension = dimension
        self._client = client or genai.Client(api_key=api_key)

    def embed_documents(self, texts: list[str], titles: list[str] | None = None) -> list[list[float]]:
        titles = titles or [None] * len(texts)
        formatted = [format_document(text, title) for text, title in zip(texts, titles, strict=True)]
        vectors: list[list[float]] = []
        for start in range(0, len(formatted), BATCH_SIZE):
            vectors.extend(self._embed(formatted[start : start + BATCH_SIZE]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([format_query(text)])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        result = self._client.models.embed_content(
            model=self.model,
            # One Content per text, so each text gets its own embedding.
            contents=[types.Content(parts=[types.Part(text=text)]) for text in texts],
            config=types.EmbedContentConfig(output_dimensionality=self.dimension),
        )
        vectors = [list(e.values) for e in result.embeddings]
        if len(vectors) != len(texts):
            raise RuntimeError(f"Expected {len(texts)} embeddings, got {len(vectors)}")
        if any(len(v) != self.dimension for v in vectors):
            raise RuntimeError(f"Embedding dimension mismatch: expected {self.dimension}")
        return vectors
