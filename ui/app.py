"""Streamlit UI for Document QA with citations.

    uvicorn main:app            # terminal 1
    streamlit run ui/app.py     # terminal 2
"""

from __future__ import annotations

import html
import os
import re

import requests
import streamlit as st

API_URL = os.getenv("AGENT_API_URL", "http://localhost:8000")
PUBLIC_API_URL = os.getenv("PUBLIC_API_URL", API_URL)  # what the browser uses to open source files
ICONS = {"pdf": "📕", "docx": "📘", "markdown": "📝", "text": "📄", "html": "🌐", "url": "🔗"}

st.set_page_config(page_title="Document QA", page_icon="📚", layout="wide")
st.markdown("""<style>
sup.ref {background:#4f46e5;color:#fff;border-radius:8px;padding:0 6px;font-size:0.7rem;margin-left:2px;font-weight:700}
.src {border:1px solid #e5e7eb;border-radius:10px;padding:10px 14px;margin:6px 0;background:#fafafa}
.src .hd {font-weight:600} .src .n {background:#4f46e5;color:#fff;border-radius:8px;padding:0 7px;margin-right:6px;font-size:0.8rem}
.quote {border-left:3px solid #4f46e5;padding:4px 10px;margin:6px 0;background:#eef2ff;font-size:0.9rem;border-radius:0 6px 6px 0}
.ok {color:#15803d;font-size:0.75rem;font-weight:600} .warn {color:#b45309;font-size:0.75rem;font-weight:600}
.doc {font-size:0.85rem;padding:4px 0}
mark {background:#fde68a;padding:0 2px}
</style>""", unsafe_allow_html=True)


def api(method, path, **kw):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=180, **kw)
        if r.status_code == 204:
            return {}
        body = r.json()
        return body if r.ok else {"_error": body.get("detail", r.text)}
    except requests.RequestException as exc:
        return {"_error": str(exc)}


def answer_html(markdown_text: str) -> str:
    """Escape, then turn [n] markers into badges and keep simple line breaks."""
    text = html.escape(markdown_text)
    text = re.sub(r"\[(\d+)\]", r'<sup class="ref">\1</sup>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return text.replace("\n", "<br>")


def highlight(chunk: str, quotes: list[dict]) -> str:
    text = html.escape(chunk)
    for q in quotes:
        pattern = r"\s+".join(re.escape(html.escape(w)) for w in q["text"].split())
        text = re.sub(f"({pattern})", r"<mark>\1</mark>", text, count=1)
    return text.replace("\n", "<br>")


def render_answer(result: dict) -> None:
    st.markdown(f"<div>{answer_html(result['markdown'])}</div>", unsafe_allow_html=True)
    if result.get("standalone") and result["standalone"] != result.get("question"):
        st.caption(f"Searched for: “{result['standalone']}”")
    for src in result.get("sources", []):
        icon = ICONS.get(src["source_type"], "📄")
        link = ""
        if src["source_type"] == "pdf" and src.get("page_start"):
            link = f' · <a href="{PUBLIC_API_URL}/documents/{src["document_id"]}/file#page={src["page_start"]}" target="_blank">open page</a>'
        elif src["source"].startswith("http"):
            link = f' · <a href="{src["source"]}" target="_blank">open</a>'
        quotes = "".join(
            f'<div class="quote">“{html.escape(q["text"].strip())}” '
            f'<span class="{"ok" if q["verified"] else "warn"}">{"✓ verified" if q["verified"] else "⚠ not found in source"}</span></div>'
            for q in src["quotes"])
        st.markdown(f'<div class="src"><span class="n">{src["n"]}</span><span class="hd">{icon} {html.escape(src["label"])}</span>'
                    f'{link}{quotes}</div>', unsafe_allow_html=True)
        with st.expander(f"Show [{src['n']}] in context"):
            st.markdown(f'<div style="font-size:0.85rem">{highlight(src["chunk_text"], src["quotes"])}</div>',
                        unsafe_allow_html=True)
    if result.get("retrieved"):
        with st.expander(f"🔎 Retrieval: {len(result['retrieved'])} chunks (hybrid vector + keyword, RRF)"):
            rows = [{"#": i + 1, "chunk": r["label"], "vector rank": str(r.get("vector_rank") or "–"),
                     "keyword rank": str(r.get("keyword_rank") or "–"), "fused score": round(r["score"], 4)}
                    for i, r in enumerate(result["retrieved"])]
            st.dataframe(rows, hide_index=True, width="stretch")


# ----------------------------------------------------------------------------- state
st.session_state.setdefault("messages", [])
st.session_state.setdefault("session_id", None)
health = api("GET", "/health")
ok = "_error" not in health

# ----------------------------------------------------------------------------- sidebar: library
with st.sidebar:
    st.title("📚 Document QA")
    st.caption("Answers grounded in your documents, every claim cited")
    if not ok:
        st.error(f"API not reachable at {API_URL}. Start it with `uvicorn main:app`.", icon="🚫")
    else:
        st.markdown(f"**Claude:** `{health['model']}`  \n**Embeddings:** `{health['embeddings']}`  \n"
                    f"**Index:** {health['documents']} docs · {health['chunks']} chunks (pgvector)")
    st.markdown("### Add documents")
    files = st.file_uploader("PDF, DOCX, Markdown, text or HTML", accept_multiple_files=True,
                             type=["pdf", "docx", "md", "markdown", "txt", "html", "htm"])
    if st.button("Index files", disabled=not (ok and files), width="stretch"):
        with st.spinner("Chunking and embedding…"):
            res = api("POST", "/documents", files=[("files", (f.name, f.getvalue())) for f in files])
        st.error(res["_error"]) if "_error" in res else st.success(f"Indexed {len(res)} file(s)")
    url = st.text_input("…or a web page / PDF URL", placeholder="https://…")
    if st.button("Add URL", disabled=not (ok and url), width="stretch"):
        with st.spinner("Fetching and indexing…"):
            res = api("POST", "/documents/url", json={"url": url})
        st.error(res["_error"]) if "_error" in res else st.success(f"Indexed “{res['title']}”")

    docs = api("GET", "/documents") if ok else []
    if isinstance(docs, list) and docs:
        st.markdown("### Library")
        for d in docs:
            c1, c2 = st.columns([4, 1], vertical_alignment="center")
            meta = f"{d['pages']} pages · " if d.get("pages") else ""
            c1.markdown(f'<div class="doc">{ICONS.get(d["source_type"], "📄")} <b>{html.escape(d["title"])}</b><br>'
                        f'<span style="color:#64748b">{meta}{d["chunks"]} chunk{"" if d["chunks"] == 1 else "s"}</span></div>',
                        unsafe_allow_html=True)
            if c2.button("✕", key=f"del-{d['id']}", help="Remove from the index"):
                api("DELETE", f"/documents/{d['id']}")
                st.rerun()
    if st.button("🧹 New conversation", width="stretch"):
        st.session_state.messages, st.session_state.session_id = [], None
        st.rerun()

# ----------------------------------------------------------------------------- chat
st.markdown("## Ask your documents")
if not st.session_state.messages:
    st.info("Add documents in the sidebar, then ask a question. Follow-ups (“what about the other plan?”) work too.")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["text"])
        else:
            render_answer(msg["result"])

question = st.chat_input("Ask a question about your documents…", disabled=not ok)
if question:
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching and reading…"):
            res = api("POST", "/ask", json={"question": question, "session_id": st.session_state.session_id})
        if "_error" in res:
            st.error(res["_error"])
            st.session_state.messages.pop()
        else:
            st.session_state.session_id = res["session_id"]
            render_answer(res)
            st.session_state.messages.append({"role": "assistant", "result": res})
