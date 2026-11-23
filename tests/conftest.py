"""Fixtures: a real PostgreSQL + pgvector test database and generated sample files.

Set TEST_DATABASE_URL (environment or .env) to a database used only for tests.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
from docx import Document as DocxDocument
from dotenv import load_dotenv
from fpdf import FPDF
from sqlalchemy import text

from src.agent import open_checkpointer
from src.tools import DocumentStore
from src.utils.config import Settings
from tests.fakes import DIM, CitingModels, HashEmbedder

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


def pytest_collection_modifyitems(config, items):
    if not TEST_DATABASE_URL:
        skip = pytest.mark.skip(reason="TEST_DATABASE_URL is not set")
        for item in items:
            if "db" in item.fixturenames or "store" in item.fixturenames or "client" in item.fixturenames:
                item.add_marker(skip)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(database_url=TEST_DATABASE_URL or "postgresql://x@localhost/x", embedding_dim=DIM,
                    chunk_chars=600, chunk_overlap=120, top_k=4, candidates=20, log_dir=tmp_path / "logs",
                    allow_private_urls=True)


@pytest.fixture
def store(settings):
    s = DocumentStore(settings.database_url, settings.embedding_dim)
    s.init()
    with s.engine.begin() as c:
        c.execute(text("TRUNCATE chunks, documents"))
    yield s
    s.engine.dispose()


@pytest.fixture
def checkpointer(settings):
    saver = open_checkpointer(settings.database_url)
    with saver.conn.connection() as c:
        c.execute("TRUNCATE checkpoints, checkpoint_blobs, checkpoint_writes")
    yield saver
    saver.conn.close()


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def models():
    return CitingModels()


# ----------------------------------------------------------------------------- sample files
PRICING_PAGES = [
    "Acme Cloud Handbook\n\nAcme Cloud runs web applications in 30 regions worldwide. "
    "Every account starts on the Free plan.",
    "Pricing\n\nThe Free plan includes 100 GB of bandwidth per month. "
    "The Pro plan costs $20 per month and includes 1 TB of bandwidth. "
    "Enterprise pricing is negotiated with the sales team.",
    "Support\n\nPro customers get email support with a 24 hour response time. "
    "Enterprise customers get a dedicated support engineer and a 99.99% uptime SLA.",
]


@pytest.fixture
def pdf_bytes() -> bytes:
    pdf = FPDF()
    pdf.set_title("Acme Cloud Handbook")
    pdf.set_font("Helvetica", size=12)
    for page in PRICING_PAGES:
        pdf.add_page()
        pdf.multi_cell(0, 8, page)
    return bytes(pdf.output())


@pytest.fixture
def docx_bytes() -> bytes:
    doc = DocxDocument()
    doc.core_properties.title = "Security Policy"
    doc.add_heading("Passwords", level=1)
    doc.add_paragraph("Passwords must be at least 14 characters long and are rotated every 180 days.")
    doc.add_heading("Access", level=1)
    doc.add_paragraph("Production access requires hardware security keys for every engineer.")
    table = doc.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Role", "Access level"
    table.rows[1].cells[0].text, table.rows[1].cells[1].text = "Engineer", "Read-only"
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
