"""Turn Claude's cited text blocks into numbered sources, and verify every quote.

Claude returns text blocks like:
    {"type": "text", "text": "Plans start at $20 a month",
     "citations": [{"type": "char_location", "document_index": 2, "cited_text": "...",
                    "start_char_index": 120, "end_char_index": 180}]}
`document_index` is the position of the chunk in the request. We map it back to the
chunk (document, pages), number unique sources [1], [2], ... in order of first use, and
check in code that each quoted span really appears in that chunk.
"""

from __future__ import annotations

from typing import Any


def _norm(s: str) -> str:
    return " ".join(s.split())


def parse_cited_answer(content: Any, documents: list[dict]) -> dict:
    """Return {"text", "segments": [{"text", "refs": [n]}], "sources": [...], "citations": [...]}"""
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
    segments, sources, citations = [], [], []
    by_chunk: dict[str, int] = {}

    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        refs = []
        for cit in block.get("citations") or []:
            idx = cit.get("document_index")
            if not isinstance(idx, int) or not 0 <= idx < len(documents):
                continue
            doc = documents[idx]
            if doc["id"] not in by_chunk:
                by_chunk[doc["id"]] = len(sources) + 1
                sources.append({
                    "n": by_chunk[doc["id"]], "chunk_id": doc["id"], "document_id": doc["document_id"],
                    "title": doc["title"], "source": doc["source"], "source_type": doc["source_type"],
                    "page_start": doc["page_start"], "page_end": doc["page_end"], "heading": doc["heading"],
                    "label": doc["label"], "chunk_text": doc["text"], "quotes": [],
                })
            n = by_chunk[doc["id"]]
            quote = cit.get("cited_text") or ""
            verified = bool(quote) and _norm(quote) in _norm(doc["text"])
            src = sources[n - 1]
            if quote and quote not in [q["text"] for q in src["quotes"]]:
                src["quotes"].append({"text": quote, "verified": verified,
                                      "start": cit.get("start_char_index"), "end": cit.get("end_char_index")})
            citations.append({"n": n, "quote": quote, "verified": verified})
            if n not in refs:
                refs.append(n)
        segments.append({"text": block.get("text", ""), "refs": refs})

    text = "".join(s["text"] for s in segments).strip()
    return {
        "text": text,
        "segments": segments,
        "sources": sources,
        "citations": citations,
        "all_quotes_verified": all(c["verified"] for c in citations),
    }


def render_markdown(parsed: dict) -> str:
    """Answer text with [n] markers after each cited segment."""
    out = []
    for seg in parsed["segments"]:
        out.append(seg["text"])
        if seg["refs"]:
            out.append("".join(f"[{n}]" for n in seg["refs"]))
    return "".join(out).strip()
