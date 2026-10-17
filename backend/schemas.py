"""Request and response models for the HTTP API."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=1000)]
AbstainReason = Literal["no_relevant_documents", "not_in_documents", "no_valid_citations"]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Question
    top_k: int | None = Field(default=None, ge=1, le=10, description="Overrides RETRIEVAL_TOP_K")


class Citation(BaseModel):
    id: int = Field(description="Passage number, matching [n] markers in the answer")
    source: str
    title: str
    section: str
    chunk_index: int
    relevance: float
    excerpt: str


class RetrievedPassage(BaseModel):
    id: int
    source: str
    title: str
    section: str
    relevance: float


class AskResponse(BaseModel):
    answer: str
    answered: bool
    abstain_reason: AbstainReason | None = None
    citations: list[Citation]
    retrieved: list[RetrievedPassage] = Field(description="All passages shown to the model")
    model: str | None = Field(description="Chat model used; null if no model call was made")
