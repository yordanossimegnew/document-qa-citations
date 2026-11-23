"""FastAPI app: upload documents or URLs, ask questions, get answers with verified citations."""

from __future__ import annotations

import uuid
from typing import Any

import anthropic
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from ..agent import DocumentQA, open_checkpointer
from ..ingest import LoaderError, ingest, load_file, load_url
from ..models import AnthropicQA, Embedder, QAModels, make_embedder
from ..tools import DocumentStore
from ..utils.config import Settings, load_settings
from ..utils.logger import get_logger
from .schemas import Answer, AskRequest, Document, HealthResponse, UrlRequest

log = get_logger("api")


def create_app(settings: Settings | None = None, embedder: Embedder | None = None, models: QAModels | None = None,
               store: DocumentStore | None = None, checkpointer: Any = None, http_client=None) -> FastAPI:
    settings = settings or load_settings()
    store = store or DocumentStore(settings.database_url, settings.embedding_dim)
    store.init()
    state: dict[str, Any] = {"embedder": embedder, "models": models}

    def get_embedder() -> Embedder:
        if state["embedder"] is None:
            try:
                state["embedder"] = make_embedder(settings)
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"Embeddings unavailable: {exc}. Set VOYAGE_API_KEY.") from exc
        return state["embedder"]

    def get_qa() -> DocumentQA:
        if "qa" not in state:
            models_ = state["models"] or AnthropicQA(settings)
            state["qa"] = DocumentQA(settings, store, get_embedder(), models_, checkpointer or open_checkpointer(settings.database_url))
        return state["qa"]

    app = FastAPI(title="Document QA with Citations", version="1.0.0")

    def _ingest(doc) -> dict:
        try:
            return ingest(doc, store, get_embedder(), settings)
        except LoaderError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        db = settings.database_url.split("@")[-1]
        return HealthResponse(model=settings.model, embeddings=f"voyage/{settings.voyage_model}", database=db,
                              **store.counts())

    @app.post("/documents", response_model=list[Document], status_code=201)
    async def upload(files: list[UploadFile] = File(...)) -> list[dict]:
        """Upload PDF, DOCX, Markdown, text or HTML files."""
        out = []
        for f in files:
            data = await f.read()
            if len(data) > settings.max_upload_mb * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"{f.filename} is larger than {settings.max_upload_mb} MB.")
            try:
                doc = load_file(f.filename or "upload", data)
            except LoaderError as exc:
                raise HTTPException(status_code=422, detail=f"{f.filename}: {exc}") from exc
            out.append(_ingest(doc))
        return out

    @app.post("/documents/url", response_model=Document, status_code=201)
    def add_url(request: UrlRequest) -> dict:
        """Fetch a web page or online PDF and index it."""
        try:
            doc = load_url(request.url, client=http_client, max_bytes=settings.max_upload_mb * 1024 * 1024,
                           allow_private=settings.allow_private_urls)
        except LoaderError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _ingest(doc)

    @app.get("/documents", response_model=list[Document])
    def list_documents() -> list[dict]:
        return store.list()

    @app.get("/documents/{doc_id}")
    def document_detail(doc_id: str) -> dict:
        doc = store.get(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return {**doc, "chunk_list": store.chunks_of(doc_id)}

    @app.get("/documents/{doc_id}/file")
    def document_file(doc_id: str) -> Response:
        raw = store.raw(doc_id)
        if raw is None:
            raise HTTPException(status_code=404, detail="No original file stored for this document")
        return Response(content=raw[0], media_type=raw[1])

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str) -> Response:
        if not store.delete(doc_id):
            raise HTTPException(status_code=404, detail="Document not found")
        return Response(status_code=204)

    @app.get("/search")
    def search(q: str = Query(..., min_length=2), top_k: int = Query(8, ge=1, le=50)) -> list[dict]:
        """Hybrid retrieval only (no answer): vector rank, keyword rank and fused score per chunk."""
        hits = store.hybrid_search(q, get_embedder().embed_query(q), top_k=top_k, candidates=settings.candidates,
                                   rrf_k=settings.rrf_k)
        return [{**h, "text": h["text"][:500]} for h in hits]

    @app.post("/ask", response_model=Answer)
    def ask(request: AskRequest) -> dict:
        if store.counts()["documents"] == 0:
            raise HTTPException(status_code=409, detail="Add some documents first.")
        session_id = request.session_id or uuid.uuid4().hex[:12]
        try:
            result = get_qa().ask(session_id, request.question, request.document_ids)
        except anthropic.AuthenticationError as exc:
            raise HTTPException(status_code=503, detail="Claude authentication failed: set ANTHROPIC_API_KEY.") from exc
        except anthropic.RateLimitError as exc:
            raise HTTPException(status_code=429, detail="Rate limited by the model API; retry shortly.") from exc
        except anthropic.APIError as exc:
            raise HTTPException(status_code=502, detail=f"Model API error: {exc}") from exc
        return {**result, "session_id": session_id}

    @app.get("/sessions/{session_id}")
    def session(session_id: str) -> dict:
        return {"session_id": session_id, "history": get_qa().history(session_id)}

    return app
