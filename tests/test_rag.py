"""Postgres + pgvector store, hybrid search, citations and the LangGraph QA graph."""

from __future__ import annotations

from src.agent import DocumentQA, parse_cited_answer
from src.ingest import ingest, load_file


def add(store, embedder, settings, name, data):
    return ingest(load_file(name, data), store, embedder, settings)


def test_ingest_and_hybrid_search(store, embedder, settings, pdf_bytes, docx_bytes):
    handbook = add(store, embedder, settings, "handbook.pdf", pdf_bytes)
    policy = add(store, embedder, settings, "policy.docx", docx_bytes)
    assert handbook["chunks"] >= 1 and store.counts()["documents"] == 2

    hits = store.hybrid_search("How much does the Pro plan cost?", embedder.embed_query("Pro plan cost"), top_k=3)
    assert hits[0]["document_id"] == handbook["id"]
    assert "$20 per month" in hits[0]["text"]
    assert hits[0]["vector_rank"] is not None and hits[0]["keyword_rank"] is not None  # found by both retrievers

    # Exact term only the keyword index can match well ("14" + "characters")
    hits = store.hybrid_search("14 characters", embedder.embed_query("password length"), top_k=2)
    assert hits[0]["document_id"] == policy["id"]

    # Filter to one document
    hits = store.hybrid_search("support", embedder.embed_query("support"), top_k=5, document_ids=[policy["id"]])
    assert {h["document_id"] for h in hits} == {policy["id"]}

    assert store.delete(handbook["id"]) and store.counts()["documents"] == 1
    assert store.chunks_of(handbook["id"]) == []  # chunks cascade


def test_citations_are_numbered_and_verified():
    docs = [{"id": "c1", "document_id": "d1", "title": "A", "source": "a.pdf", "source_type": "pdf", "page_start": 2,
             "page_end": 2, "heading": None, "label": "A (p. 2)", "text": "The Pro plan costs $20 per month."},
            {"id": "c2", "document_id": "d2", "title": "B", "source": "b.md", "source_type": "markdown",
             "page_start": None, "page_end": None, "heading": "Intro", "label": "B (Intro)", "text": "Other text."}]
    content = [
        {"type": "text", "text": "Pro costs $20 a month", "citations": [
            {"type": "char_location", "document_index": 0, "cited_text": "The Pro plan costs $20 per month."}]},
        {"type": "text", "text": ". "},
        {"type": "text", "text": "Also something invented", "citations": [
            {"type": "char_location", "document_index": 1, "cited_text": "A quote that is not there"}]},
    ]
    parsed = parse_cited_answer(content, docs)
    assert [s["n"] for s in parsed["sources"]] == [1, 2]
    assert parsed["segments"][0]["refs"] == [1] and parsed["segments"][1]["refs"] == []
    assert parsed["citations"][0]["verified"] is True
    assert parsed["citations"][1]["verified"] is False and parsed["all_quotes_verified"] is False


def test_graph_answers_with_page_citations_and_follow_ups(store, embedder, settings, checkpointer, pdf_bytes):
    from tests.fakes import CitingModels

    add(store, embedder, settings, "handbook.pdf", pdf_bytes)
    models = CitingModels(
        rules=[("much", "The Pro plan costs $20 per month and includes 1 TB of bandwidth.", "Pro costs $20/month."),
               ("support", "Pro customers get email support with a 24 hour response time.", "Email support, 24h.")],
        condense_map={"What support does it include?": "What support does the Pro plan include?"},
    )
    qa = DocumentQA(settings, store, embedder, models, checkpointer)

    first = qa.ask("s1", "How much is the Pro plan?")
    assert first["markdown"] == "Pro costs $20/month.[1]"
    [source] = first["sources"]
    assert source["page_start"] == 2 and source["label"] == "Acme Cloud Handbook (p. 2)"
    assert source["quotes"][0]["verified"] is True
    assert models.condense_calls == []  # first turn: nothing to rewrite

    second = qa.ask("s1", "What support does it include?")
    assert second["standalone"] == "What support does the Pro plan include?"
    assert models.condense_calls[0][0][0] == {"role": "user", "text": "How much is the Pro plan?"}
    assert second["sources"][0]["page_start"] == 3
    assert [m["role"] for m in qa.history("s1")] == ["user", "assistant", "user", "assistant"]
    assert qa.history("other") == []


def test_no_documents_retrieved(store, embedder, settings, checkpointer):
    from tests.fakes import CitingModels

    qa = DocumentQA(settings, store, embedder, CitingModels(), checkpointer)
    result = qa.ask("s2", "anything at all")
    assert "couldn't find" in result["markdown"] and result["sources"] == []
