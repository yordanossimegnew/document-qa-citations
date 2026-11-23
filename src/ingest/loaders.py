"""Turn PDFs, DOCX, Markdown/text, HTML files and web URLs into page-aware text."""

from __future__ import annotations

import io
import ipaddress
import re
import socket
from dataclasses import dataclass, field
from pathlib import PurePath
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader

SUPPORTED = {".pdf": "pdf", ".docx": "docx", ".md": "markdown", ".markdown": "markdown", ".txt": "text",
             ".html": "html", ".htm": "html"}


class LoaderError(ValueError):
    pass


@dataclass
class Section:
    text: str
    page: int | None = None      # 1-based page for PDFs
    heading: str | None = None   # nearest heading (DOCX / Markdown / HTML)


@dataclass
class LoadedDocument:
    title: str
    source_type: str             # pdf | docx | markdown | text | html | url
    source: str                  # file name or URL
    sections: list[Section] = field(default_factory=list)
    pages: int | None = None
    raw: bytes | None = None     # original bytes (kept so the UI can show the PDF page)
    mime: str = "text/plain"

    @property
    def chars(self) -> int:
        return sum(len(s.text) for s in self.sections)


def load_file(filename: str, data: bytes) -> LoadedDocument:
    ext = PurePath(filename).suffix.lower()
    kind = SUPPORTED.get(ext)
    if kind is None:
        raise LoaderError(f"Unsupported file type '{ext or filename}'. Supported: {', '.join(sorted(SUPPORTED))}")
    stem = PurePath(filename).stem
    if kind == "pdf":
        return _pdf(stem, filename, data)
    if kind == "docx":
        return _docx(stem, filename, data)
    text = _decode(data)
    if kind == "html":
        return _html(text, filename, filename, "html", fallback_title=stem)
    return _markdown(stem, filename, text, kind)


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise LoaderError("Could not decode the file as text.")


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _pdf(stem: str, filename: str, data: bytes) -> LoadedDocument:
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise LoaderError(f"Could not read the PDF: {exc}") from exc
    if reader.is_encrypted:
        raise LoaderError("The PDF is password-protected.")
    sections = []
    for number, page in enumerate(reader.pages, 1):
        text = _clean(page.extract_text() or "")
        if text:
            sections.append(Section(text=text, page=number))
    if not sections:
        raise LoaderError("No text found in the PDF (is it a scanned image? OCR isn't supported).")
    meta_title = (reader.metadata.title if reader.metadata else None) or ""
    return LoadedDocument(title=meta_title.strip() or stem, source_type="pdf", source=filename, sections=sections,
                          pages=len(reader.pages), raw=data, mime="application/pdf")


def _docx(stem: str, filename: str, data: bytes) -> LoadedDocument:
    try:
        doc = DocxDocument(io.BytesIO(data))
    except Exception as exc:
        raise LoaderError(f"Could not read the DOCX file: {exc}") from exc
    sections, heading, buffer = [], None, []

    def flush():
        if buffer:
            sections.append(Section(text=_clean("\n\n".join(buffer)), heading=heading))
            buffer.clear()

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if para.style is not None and para.style.name.lower().startswith(("heading", "title")):
            flush()
            heading = text
        buffer.append(text)
    flush()
    for table in doc.tables:  # tables as pipe-separated rows
        rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
        if rows:
            sections.append(Section(text="\n".join(rows), heading=heading))
    if not sections:
        raise LoaderError("The DOCX file has no text.")
    title = (doc.core_properties.title or "").strip() or stem
    return LoadedDocument(title=title, source_type="docx", source=filename, sections=sections,
                          mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


def _markdown(stem: str, filename: str, text: str, kind: str) -> LoadedDocument:
    text = _clean(text)
    if not text:
        raise LoaderError("The file is empty.")
    sections, heading, buffer = [], None, []
    for line in text.splitlines():
        if kind == "markdown" and re.match(r"^#{1,6}\s", line):
            if buffer:
                sections.append(Section(text="\n".join(buffer).strip(), heading=heading))
                buffer = []
            heading = line.lstrip("#").strip()
        buffer.append(line)
    if buffer:
        sections.append(Section(text="\n".join(buffer).strip(), heading=heading))
    first_heading = next((s.heading for s in sections if s.heading), None)
    return LoadedDocument(title=first_heading or stem, source_type=kind, source=filename,
                          sections=[s for s in sections if s.text], mime="text/markdown" if kind == "markdown" else "text/plain")


def _html(html: str, source: str, name: str, source_type: str, fallback_title: str) -> LoadedDocument:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]):
        tag.decompose()
    title = (soup.title.string.strip() if soup.title and soup.title.string else "") or fallback_title
    root = soup.find("main") or soup.find("article") or soup.body or soup
    sections, heading, buffer = [], None, []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "td", "blockquote"]):
        text = " ".join(el.get_text(" ", strip=True).split())
        if not text:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            if buffer:
                sections.append(Section(text="\n".join(buffer), heading=heading))
                buffer = []
            heading = text
        buffer.append(text)
    if buffer:
        sections.append(Section(text="\n".join(buffer), heading=heading))
    if not sections:
        text = _clean(root.get_text("\n"))
        if not text:
            raise LoaderError("No readable text found on the page.")
        sections = [Section(text=text)]
    return LoadedDocument(title=title, source_type=source_type, source=source, sections=sections, mime="text/html")


# ----------------------------------------------------------------------------- URLs
def check_url(url: str, allow_private: bool = False) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise LoaderError("Only http(s) URLs are supported.")
    if allow_private:
        return
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise LoaderError(f"Could not resolve {parsed.hostname}.") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise LoaderError(f"{parsed.hostname} resolves to a private address; refusing to fetch it.")


def load_url(url: str, client: httpx.Client | None = None, max_bytes: int = 25 * 1024 * 1024,
             allow_private: bool = False) -> LoadedDocument:
    own = client is None
    client = client or httpx.Client(timeout=20, headers={"User-Agent": "document-qa-citations/1.0"})
    try:
        # Follow redirects by hand so every hop is checked BEFORE it is requested (SSRF guard).
        current = url
        for _ in range(6):
            check_url(current, allow_private)
            response = client.get(current, follow_redirects=False)
            if response.is_redirect and response.headers.get("location"):
                current = str(response.url.join(response.headers["location"]))
                continue
            break
        else:
            raise LoaderError("Too many redirects.")
    except httpx.HTTPError as exc:
        raise LoaderError(f"Could not fetch {url}: {exc}") from exc
    finally:
        if own:
            client.close()
    if response.status_code >= 400:
        raise LoaderError(f"{url} returned HTTP {response.status_code}.")
    if len(response.content) > max_bytes:
        raise LoaderError("The page is too large.")
    ctype = response.headers.get("content-type", "").split(";")[0].strip().lower()
    final = str(response.url)
    if ctype == "application/pdf" or final.lower().endswith(".pdf"):
        doc = _pdf(PurePath(urlparse(final).path).stem or final, final, response.content)
    elif ctype in ("text/plain", "text/markdown"):
        doc = _markdown(PurePath(urlparse(final).path).stem or final, final, _decode(response.content),
                        "markdown" if ctype == "text/markdown" else "text")
    else:
        doc = _html(response.text, final, final, "url", fallback_title=urlparse(final).netloc)
    doc.source_type, doc.source = ("url" if doc.source_type != "pdf" else "pdf"), final
    return doc
