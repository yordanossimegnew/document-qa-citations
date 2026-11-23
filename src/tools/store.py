"""PostgreSQL + pgvector document store with hybrid (vector + full-text) search.

Retrieval runs two searches inside Postgres and merges them with Reciprocal Rank Fusion:
  * semantic:  pgvector cosine distance over Voyage embeddings (HNSW index)
  * keyword:   Postgres full-text search (tsvector, OR-ed query terms, ts_rank_cd, GIN index)
Keyword search catches exact names, codes and numbers that embeddings blur; vector search
catches paraphrases that share no words. RRF rewards chunks that rank well in either.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (Column, Computed, DateTime, ForeignKey, Index, Integer, LargeBinary, MetaData, String,
                        Table, Text, create_engine, delete, func, insert, select, text)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.engine import Engine

from ..ingest import Chunk, LoadedDocument


def _engine_url(url: str) -> str:
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def build_tables(dim: int) -> tuple[MetaData, Table, Table]:
    metadata = MetaData()
    documents = Table(
        "documents", metadata,
        Column("id", String(32), primary_key=True),
        Column("title", Text, nullable=False),
        Column("source_type", String(20), nullable=False),
        Column("source", Text, nullable=False),
        Column("mime", String(120), nullable=False),
        Column("pages", Integer),
        Column("chars", Integer, nullable=False),
        Column("chunks", Integer, nullable=False, default=0),
        Column("raw", LargeBinary),  # original file, so the UI can render the cited PDF page
        Column("meta", JSONB, nullable=False, default=dict),
        Column("created_at", DateTime(timezone=True), nullable=False),
    )
    chunks = Table(
        "chunks", metadata,
        Column("id", String(40), primary_key=True),
        Column("document_id", String(32), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True),
        Column("ord", Integer, nullable=False),
        Column("text", Text, nullable=False),
        Column("page_start", Integer),
        Column("page_end", Integer),
        Column("heading", Text),
        Column("embedding", Vector(dim), nullable=False),
        Column("tsv", TSVECTOR, Computed("to_tsvector('english', coalesce(heading, '') || ' ' || text)", persisted=True)),
    )
    Index("ix_chunks_embedding_hnsw", chunks.c.embedding, postgresql_using="hnsw",
          postgresql_ops={"embedding": "vector_cosine_ops"})
    Index("ix_chunks_tsv", chunks.c.tsv, postgresql_using="gin")
    return metadata, documents, chunks


HYBRID_SQL = """
WITH vec AS (
    SELECT c.id, row_number() OVER (ORDER BY c.embedding <=> CAST(:qvec AS vector)) AS rnk
    FROM chunks c
    WHERE (CAST(:doc_ids AS text[]) IS NULL OR c.document_id = ANY(CAST(:doc_ids AS text[])))
    ORDER BY c.embedding <=> CAST(:qvec AS vector)
    LIMIT :candidates
),
kw AS (
    SELECT c.id, row_number() OVER (ORDER BY ts_rank_cd(c.tsv, q) DESC) AS rnk
    FROM chunks c, to_tsquery('english', :qtext) q
    WHERE c.tsv @@ q
      AND (CAST(:doc_ids AS text[]) IS NULL OR c.document_id = ANY(CAST(:doc_ids AS text[])))
    ORDER BY ts_rank_cd(c.tsv, q) DESC
    LIMIT :candidates
)
SELECT c.id, c.document_id, c.ord, c.text, c.page_start, c.page_end, c.heading,
       d.title, d.source, d.source_type,
       vec.rnk AS vector_rank, kw.rnk AS keyword_rank,
       COALESCE(1.0 / (:rrf_k + vec.rnk), 0) + COALESCE(1.0 / (:rrf_k + kw.rnk), 0) AS score
FROM (SELECT id FROM vec UNION SELECT id FROM kw) hits
JOIN chunks c ON c.id = hits.id
JOIN documents d ON d.id = c.document_id
LEFT JOIN vec ON vec.id = c.id
LEFT JOIN kw ON kw.id = c.id
ORDER BY score DESC, c.document_id, c.ord
LIMIT :top_k
"""


def keyword_query(question: str) -> str:
    """OR together the question's terms, so natural questions still match (ts_rank_cd rewards
    chunks matching more terms). Stop words are dropped by Postgres' english configuration."""
    terms = list(dict.fromkeys(re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", question.lower())))
    return " | ".join(terms[:30]) or "zzznomatchzzz"


class DocumentStore:
    def __init__(self, database_url: str, dim: int):
        self.engine: Engine = create_engine(_engine_url(database_url), pool_pre_ping=True)
        self.dim = dim
        self.metadata, self.documents, self.chunks = build_tables(dim)

    def init(self) -> None:
        with self.engine.begin() as c:
            c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        self.metadata.create_all(self.engine)

    # ------------------------------------------------------------------ write
    def add(self, doc: LoadedDocument, chunks: list[Chunk], vectors: list[list[float]]) -> dict:
        if len(chunks) != len(vectors):
            raise ValueError("one embedding per chunk is required")
        doc_id = uuid.uuid4().hex[:12]
        with self.engine.begin() as c:  # all-or-nothing: no half-indexed documents
            c.execute(insert(self.documents).values(
                id=doc_id, title=doc.title[:500], source_type=doc.source_type, source=doc.source, mime=doc.mime,
                pages=doc.pages, chars=doc.chars, chunks=len(chunks), raw=doc.raw, meta={},
                created_at=datetime.now(timezone.utc)))
            if chunks:
                c.execute(insert(self.chunks), [
                    {"id": f"{doc_id}-{ch.ord}", "document_id": doc_id, "ord": ch.ord, "text": ch.text,
                     "page_start": ch.page_start, "page_end": ch.page_end, "heading": ch.heading, "embedding": vec}
                    for ch, vec in zip(chunks, vectors)
                ])
        return self.get(doc_id)

    def delete(self, doc_id: str) -> bool:
        with self.engine.begin() as c:
            return bool(c.execute(delete(self.documents).where(self.documents.c.id == doc_id)).rowcount)

    # ------------------------------------------------------------------ read
    def _doc_columns(self):
        d = self.documents.c
        return [d.id, d.title, d.source_type, d.source, d.mime, d.pages, d.chars, d.chunks, d.created_at]

    def list(self) -> list[dict]:
        with self.engine.connect() as c:
            rows = c.execute(select(*self._doc_columns()).order_by(self.documents.c.created_at.desc())).all()
        return [self._doc(r) for r in rows]

    def get(self, doc_id: str) -> dict | None:
        with self.engine.connect() as c:
            row = c.execute(select(*self._doc_columns()).where(self.documents.c.id == doc_id)).first()
        return self._doc(row) if row else None

    def raw(self, doc_id: str) -> tuple[bytes, str] | None:
        with self.engine.connect() as c:
            row = c.execute(select(self.documents.c.raw, self.documents.c.mime)
                            .where(self.documents.c.id == doc_id)).first()
        return (row.raw, row.mime) if row and row.raw else None

    def chunks_of(self, doc_id: str) -> list[dict]:
        ch = self.chunks.c
        with self.engine.connect() as c:
            rows = c.execute(select(ch.id, ch.ord, ch.text, ch.page_start, ch.page_end, ch.heading)
                             .where(ch.document_id == doc_id).order_by(ch.ord)).all()
        return [dict(r._mapping) for r in rows]

    def counts(self) -> dict:
        with self.engine.connect() as c:
            docs = c.execute(select(func.count()).select_from(self.documents)).scalar()
            chunks = c.execute(select(func.count()).select_from(self.chunks)).scalar()
        return {"documents": docs, "chunks": chunks}

    def hybrid_search(self, query: str, query_vector: list[float], top_k: int = 8, candidates: int = 40,
                      rrf_k: int = 60, document_ids: list[str] | None = None) -> list[dict]:
        params = {"qvec": "[" + ",".join(f"{x:.7f}" for x in query_vector) + "]", "qtext": keyword_query(query),
                  "top_k": top_k, "candidates": candidates, "rrf_k": rrf_k, "doc_ids": document_ids or None}
        with self.engine.connect() as c:
            rows = c.execute(text(HYBRID_SQL), params).all()
        return [{**dict(r._mapping), "score": float(r.score)} for r in rows]

    @staticmethod
    def _doc(row) -> dict:
        data = dict(row._mapping)
        data["created_at"] = data["created_at"].isoformat()
        return data
