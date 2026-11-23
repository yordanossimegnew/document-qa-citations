"""Configuration loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    database_url: str = "postgresql://postgres:postgres@localhost:5432/docqa"
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    max_tokens: int = 8000
    voyage_model: str = "voyage-3.5"
    embedding_dim: int = 1024           # voyage-3.5 default output size
    chunk_chars: int = 2400             # ~600 tokens per chunk
    chunk_overlap: int = 300
    top_k: int = 8                      # chunks sent to Claude per question
    candidates: int = 40                # per retriever, before rank fusion
    rrf_k: int = 60                     # reciprocal rank fusion constant
    max_upload_mb: int = 25
    allow_private_urls: bool = False    # SSRF guard for URL ingestion
    log_level: str = "INFO"
    log_dir: Path = PROJECT_ROOT / "logs"

    def __post_init__(self) -> None:
        if not self.database_url.startswith(("postgres://", "postgresql://", "postgresql+psycopg://")):
            raise ValueError("DATABASE_URL must be a PostgreSQL URL (pgvector is required).")


def load_settings(**overrides) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    settings = Settings(
        database_url=os.getenv("DATABASE_URL", Settings.database_url),
        model=os.getenv("AGENT_MODEL", Settings.model),
        effort=os.getenv("AGENT_EFFORT", Settings.effort),
        voyage_model=os.getenv("VOYAGE_MODEL", Settings.voyage_model),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", Settings.embedding_dim)),
        chunk_chars=int(os.getenv("CHUNK_CHARS", Settings.chunk_chars)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", Settings.chunk_overlap)),
        top_k=int(os.getenv("TOP_K", Settings.top_k)),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", Settings.max_upload_mb)),
        log_level=os.getenv("LOG_LEVEL", Settings.log_level),
    )
    return replace(settings, **{k: v for k, v in overrides.items() if v is not None})
