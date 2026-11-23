"""Question answering as a LangGraph graph, with chat memory in the Postgres checkpointer.

    START -> condense -> retrieve -> answer -> END

* condense - rewrites a follow-up ("what about the second one?") into a standalone question
             using the conversation so far (skipped on the first turn)
* retrieve - hybrid pgvector + full-text search with reciprocal rank fusion
* answer   - Claude answers from the retrieved chunks with the Citations API; citations are
             mapped to numbered sources and every quote is checked against its chunk
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

import anthropic
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from ..models import Embedder, QAModels
from ..tools import DocumentStore
from ..utils.config import Settings
from .citations import parse_cited_answer, render_markdown

RETRY = RetryPolicy(max_attempts=3, initial_interval=1.0,
                    retry_on=(anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError))
NOT_FOUND = "I couldn't find anything about that in your documents."


class QAState(TypedDict, total=False):
    history: Annotated[list[dict], operator.add]  # [{"role": "user"|"assistant", "text": ...}], kept across turns
    question: str
    document_ids: list[str] | None
    standalone: str
    retrieved: list[dict]
    result: dict


def page_label(title: str, start: int | None, end: int | None, heading: str | None) -> str:
    if start and end and start != end:
        where = f"pp. {start}–{end}"
    elif start:
        where = f"p. {start}"
    elif heading and heading.strip().lower() != title.strip().lower():
        where = heading
    else:
        return title
    return f"{title} ({where})"


class DocumentQA:
    def __init__(self, settings: Settings, store: DocumentStore, embedder: Embedder, models: QAModels,
                 checkpointer: Any):
        self.settings, self.store, self.embedder, self.models = settings, store, embedder, models
        self.graph = self._build(checkpointer)

    def _condense(self, state: QAState) -> dict:
        history = state.get("history") or []
        if not history:
            return {"standalone": state["question"]}
        return {"standalone": self.models.condense(history, state["question"])}

    def _retrieve(self, state: QAState) -> dict:
        query = state["standalone"]
        hits = self.store.hybrid_search(query, self.embedder.embed_query(query), top_k=self.settings.top_k,
                                        candidates=self.settings.candidates, rrf_k=self.settings.rrf_k,
                                        document_ids=state.get("document_ids"))
        for hit in hits:
            hit["label"] = page_label(hit["title"], hit["page_start"], hit["page_end"], hit["heading"])
        return {"retrieved": hits}

    def _answer(self, state: QAState) -> dict:
        docs = state.get("retrieved") or []
        if not docs:
            parsed = {"text": NOT_FOUND, "segments": [{"text": NOT_FOUND, "refs": []}], "sources": [],
                      "citations": [], "all_quotes_verified": True}
            usage = {}
        else:
            message = self.models.answer(state["standalone"], docs)
            parsed = parse_cited_answer(message.content, docs)
            meta = getattr(message, "usage_metadata", None) or {}
            usage = {"input_tokens": meta.get("input_tokens", 0), "output_tokens": meta.get("output_tokens", 0)}
        parsed["markdown"] = render_markdown(parsed)
        result = {**parsed, "question": state["question"], "standalone": state["standalone"],
                  "retrieved": [{k: v for k, v in d.items() if k != "text"} | {"preview": d["text"][:240]}
                                for d in docs],
                  "usage": usage}
        return {"result": result,
                "history": [{"role": "user", "text": state["question"]},
                            {"role": "assistant", "text": parsed["markdown"]}]}

    def _build(self, checkpointer):
        g = StateGraph(QAState)
        g.add_node("condense", self._condense, retry_policy=RETRY)
        g.add_node("retrieve", self._retrieve)
        g.add_node("answer", self._answer, retry_policy=RETRY)
        g.add_edge(START, "condense")
        g.add_edge("condense", "retrieve")
        g.add_edge("retrieve", "answer")
        g.add_edge("answer", END)
        return g.compile(checkpointer=checkpointer)

    def ask(self, session_id: str, question: str, document_ids: list[str] | None = None) -> dict:
        config = {"configurable": {"thread_id": session_id}}
        state = self.graph.invoke({"question": question, "document_ids": document_ids}, config)
        return state["result"]

    def history(self, session_id: str) -> list[dict]:
        values = self.graph.get_state({"configurable": {"thread_id": session_id}}).values
        return values.get("history") or []
