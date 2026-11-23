"""Models: Voyage embeddings (LangChain) and Claude with the Citations API (LangChain ChatAnthropic).

Each retrieved chunk is sent to Claude as its own `document` content block with citations
enabled. Claude's answer comes back as text blocks whose `citations` point at a document
index plus the exact quoted span, so every claim can be traced to a page.
"""

from __future__ import annotations

from typing import Any, Protocol

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_voyageai import VoyageAIEmbeddings
from pydantic import BaseModel, Field

from ..prompts import ANSWER_PROMPT, CONDENSE_PROMPT
from ..utils.config import Settings

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class QAModels(Protocol):
    def condense(self, history: list[dict], question: str) -> str: ...
    def answer(self, question: str, documents: list[dict]) -> AIMessage: ...


def make_embedder(settings: Settings) -> Embedder:
    # embed_documents uses input_type="document", embed_query uses "query" (Voyage's asymmetric retrieval)
    return VoyageAIEmbeddings(model=settings.voyage_model, batch_size=64)


class Standalone(BaseModel):
    question: str = Field(description="The follow-up rewritten as a self-contained question.")


def document_block(doc: dict) -> dict:
    """A retrieved chunk as a citable Claude document."""
    return {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": doc["text"]},
        "title": doc["label"],
        "context": f"Excerpt {doc['ord'] + 1} of '{doc['title']}' ({doc['source_type']}).",
        "citations": {"enabled": True},
    }


class AnthropicQA:
    def __init__(self, settings: Settings):
        self.chat = ChatAnthropic(
            model=settings.model,
            max_tokens=settings.max_tokens,
            output_config={"effort": settings.effort},
            betas=[FALLBACK_BETA],
            model_kwargs={"fallbacks": "default"},
            max_retries=3,
        )

    def condense(self, history: list[dict], question: str) -> str:
        transcript = "\n".join(f"{m['role']}: {m['text'][:1500]}" for m in history[-6:])
        model = self.chat.with_structured_output(Standalone, method="json_schema")
        result = model.invoke([SystemMessage(CONDENSE_PROMPT),
                               HumanMessage(f"<conversation>\n{transcript}\n</conversation>\n<follow_up>{question}</follow_up>")])
        return result.question.strip() or question

    def answer(self, question: str, documents: list[dict]) -> AIMessage:
        # Documents first, then the question (long context before the query works best).
        content: list[Any] = [document_block(d) for d in documents]
        content.append({"type": "text", "text": question})
        return self.chat.invoke([SystemMessage(ANSWER_PROMPT), HumanMessage(content=content)])
