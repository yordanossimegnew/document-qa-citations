"""HTTP API end to end with the fakes."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from tests.fakes import CitingModels


@pytest.fixture
def client(settings, store, embedder, checkpointer):
    models = CitingModels(rules=[("pro", "The Pro plan costs $20 per month and includes 1 TB of bandwidth.",
                                  "The Pro plan is $20 per month.")])

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/html"},
                              text="<title>Status</title><main><p>All systems operational.</p></main>")

    app = create_app(settings, embedder=embedder, models=models, store=store, checkpointer=checkpointer,
                     http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return TestClient(app)


def test_upload_ask_and_manage(client, pdf_bytes, docx_bytes):
    assert client.post("/ask", json={"question": "anything?"}).status_code == 409  # empty library

    r = client.post("/documents", files=[("files", ("handbook.pdf", pdf_bytes, "application/pdf")),
                                         ("files", ("policy.docx", docx_bytes, "application/octet-stream"))])
    assert r.status_code == 201
    handbook, policy = r.json()
    assert handbook["pages"] == 3 and policy["title"] == "Security Policy"
    assert client.post("/documents/url", json={"url": "https://status.example.com"}).json()["title"] == "Status"
    assert client.get("/health").json()["documents"] == 3

    answer = client.post("/ask", json={"question": "What does the Pro plan cost?"}).json()
    assert answer["markdown"] == "The Pro plan is $20 per month.[1]"
    assert answer["sources"][0]["page_start"] == 2 and answer["all_quotes_verified"] is True
    assert answer["retrieved"] and "keyword_rank" in answer["retrieved"][0]

    follow = client.post("/ask", json={"question": "And the Pro plan bandwidth?", "session_id": answer["session_id"]})
    assert follow.status_code == 200
    history = client.get(f"/sessions/{answer['session_id']}").json()["history"]
    assert len(history) == 4

    assert client.get(f"/documents/{handbook['id']}/file").content == pdf_bytes
    detail = client.get(f"/documents/{handbook['id']}").json()
    assert detail["chunk_list"][0]["page_start"] == 1
    hits = client.get("/search", params={"q": "uptime SLA"}).json()
    assert "99.99%" in hits[0]["text"]

    assert client.delete(f"/documents/{handbook['id']}").status_code == 204
    assert client.get(f"/documents/{handbook['id']}").status_code == 404


def test_validation(client):
    bad = client.post("/documents", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))])
    assert bad.status_code == 422 and "Unsupported" in bad.json()["detail"]
    assert client.post("/ask", json={"question": "x"}).status_code == 422
    assert client.post("/ask", json={"question": "hello", "session_id": "../etc"}).status_code == 422
