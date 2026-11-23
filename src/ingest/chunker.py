"""Split sections into overlapping chunks that remember their page range and heading."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .loaders import Section


@dataclass
class Chunk:
    ord: int
    text: str
    page_start: int | None
    page_end: int | None
    heading: str | None


def _paragraphs(section: Section, max_chars: int) -> list[tuple[str, Section]]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", section.text) if p.strip()]
    out = []
    for part in parts:
        while len(part) > max_chars:  # very long paragraph: cut at a sentence end if possible
            cut = part.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 2 else max_chars
            out.append((part[:cut].strip(), section))
            part = part[cut:].strip()
        if part:
            out.append((part, section))
    return out


def chunk_sections(sections: list[Section], max_chars: int = 2400, overlap: int = 300) -> list[Chunk]:
    paragraphs = [p for s in sections for p in _paragraphs(s, max_chars)]
    chunks: list[Chunk] = []
    current: list[tuple[str, Section]] = []
    size = 0

    def emit():
        pages = [s.page for _, s in current if s.page is not None]
        heading = next((s.heading for _, s in current if s.heading), None)
        chunks.append(Chunk(ord=len(chunks), text="\n\n".join(t for t, _ in current),
                            page_start=min(pages) if pages else None, page_end=max(pages) if pages else None,
                            heading=heading))

    for text, section in paragraphs:
        new_page = bool(current) and section.page is not None and current[-1][1].page != section.page
        if new_page:
            # Never let a chunk cross a page break: every citation then points at exactly one page.
            emit()
            current, size = [], 0
        elif current and size + len(text) > max_chars:
            emit()
            # carry the tail of the previous chunk forward so ideas aren't cut in half
            carried, carried_size = [], 0
            for t, s in reversed(current):
                if carried_size + len(t) > overlap:
                    break
                carried.insert(0, (t, s))
                carried_size += len(t)
            current, size = carried, carried_size
        current.append((text, section))
        size += len(text)
    if current:
        emit()
    return chunks
