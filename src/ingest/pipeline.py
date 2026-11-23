"""Ingestion: load -> chunk -> embed (Voyage) -> store (Postgres + pgvector) in one transaction."""

from __future__ import annotations

from ..models import Embedder
from ..tools import DocumentStore
from ..utils.config import Settings
from .chunker import chunk_sections
from .loaders import LoadedDocument, LoaderError


def ingest(doc: LoadedDocument, store: DocumentStore, embedder: Embedder, settings: Settings) -> dict:
    chunks = chunk_sections(doc.sections, settings.chunk_chars, settings.chunk_overlap)
    if not chunks:
        raise LoaderError("Nothing to index in this document.")
    # Embed the heading together with the text so short sections keep their context.
    texts = [f"{doc.title}\n{c.heading or ''}\n{c.text}".strip() for c in chunks]
    vectors = embedder.embed_documents(texts)
    return store.add(doc, chunks, vectors)
