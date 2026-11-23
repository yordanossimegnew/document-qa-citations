"""Loaders and chunking (no database needed)."""

from __future__ import annotations

import httpx
import pytest

from src.ingest import LoaderError, chunk_sections, load_file, load_url
from src.ingest.loaders import Section


def test_pdf_keeps_page_numbers(pdf_bytes):
    doc = load_file("handbook.pdf", pdf_bytes)
    assert doc.title == "Acme Cloud Handbook" and doc.pages == 3 and doc.source_type == "pdf"
    assert [s.page for s in doc.sections] == [1, 2, 3]
    assert "$20 per month" in doc.sections[1].text
    assert doc.raw == pdf_bytes  # original kept for the PDF viewer


def test_docx_headings_and_tables(docx_bytes):
    doc = load_file("policy.docx", docx_bytes)
    assert doc.title == "Security Policy"
    headings = [s.heading for s in doc.sections]
    assert "Passwords" in headings and "Access" in headings
    assert any("Engineer | Read-only" in s.text for s in doc.sections)


def test_markdown_and_html():
    md = load_file("notes.md", b"# Release notes\n\nVersion 2 adds SSO.\n\n## Fixes\n\nFaster search.")
    assert md.title == "Release notes" and [s.heading for s in md.sections] == ["Release notes", "Fixes"]

    html = load_file("page.html", b"<html><head><title>FAQ</title><script>evil()</script></head><body>"
                                  b"<nav>menu</nav><main><h2>Billing</h2><p>Invoices are sent monthly.</p></main></body></html>")
    assert html.title == "FAQ"
    text = " ".join(s.text for s in html.sections)
    assert "Invoices are sent monthly." in text and "evil" not in text and "menu" not in text


def test_rejects_unsupported_and_empty():
    with pytest.raises(LoaderError, match="Unsupported"):
        load_file("photo.png", b"\x89PNG")
    with pytest.raises(LoaderError, match="empty"):
        load_file("blank.txt", b"   ")


def test_url_loading_with_redirect_and_ssrf_guard():
    def handler(request: httpx.Request):
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "https://docs.example.com/new"})
        return httpx.Response(200, headers={"content-type": "text/html"},
                              text="<title>New page</title><main><p>Moved content.</p></main>")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    doc = load_url("https://docs.example.com/old", client=client, allow_private=True)
    assert doc.title == "New page" and doc.source == "https://docs.example.com/new" and doc.source_type == "url"

    with pytest.raises(LoaderError, match="private"):
        load_url("http://127.0.0.1:8000/admin")
    with pytest.raises(LoaderError, match="http"):
        load_url("file:///etc/passwd")


def test_redirect_to_private_address_is_blocked():
    def handler(request: httpx.Request):
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(LoaderError, match="private"):
        load_url("https://example.com/redirect", client=client)


def test_chunks_track_pages_and_overlap():
    sections = [Section(text="\n\n".join(f"Page {p} paragraph {i} " + "x" * 80 for i in range(6)), page=p)
                for p in (1, 2, 3)]
    chunks = chunk_sections(sections, max_chars=400, overlap=120)
    assert len(chunks) > 3
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 3
    assert all(c.page_start == c.page_end for c in chunks)  # never crosses a page break
    assert {c.page_start for c in chunks} == {1, 2, 3}
    first_tail = chunks[0].text.split("\n\n")[-1]
    assert chunks[1].page_start == 1 and chunks[1].text.startswith(first_tail)  # overlap within a page
    assert all(len(c.text) <= 400 + 120 for c in chunks)
