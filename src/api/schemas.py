"""Request/response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class UrlRequest(BaseModel):
    url: str = Field(..., min_length=8, max_length=2000)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=4000)
    session_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]{1,64}$",
                                   description="Continue a conversation (follow-up questions).")
    document_ids: list[str] | None = Field(None, description="Restrict retrieval to these documents.")


class Document(BaseModel):
    id: str
    title: str
    source_type: str
    source: str
    mime: str
    pages: int | None
    chars: int
    chunks: int
    created_at: str


class Quote(BaseModel):
    text: str
    verified: bool
    start: int | None = None
    end: int | None = None


class Source(BaseModel):
    n: int
    chunk_id: str
    document_id: str
    title: str
    source: str
    source_type: str
    page_start: int | None
    page_end: int | None
    heading: str | None
    label: str
    chunk_text: str
    quotes: list[Quote]


class Answer(BaseModel):
    session_id: str
    question: str
    standalone: str = Field(description="The question after follow-up rewriting.")
    markdown: str = Field(description="Answer with [n] citation markers.")
    segments: list[dict]
    sources: list[Source]
    all_quotes_verified: bool
    retrieved: list[dict] = Field(description="Every retrieved chunk with vector/keyword ranks and fused score.")
    usage: dict


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    embeddings: str
    database: str
    documents: int
    chunks: int
