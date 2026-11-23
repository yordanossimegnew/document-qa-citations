"""Offline stand-ins: a hashing embedder (instead of Voyage) and a scripted citing model (instead of Claude)."""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.messages import AIMessage

DIM = 1024


class HashEmbedder:
    """Bag-of-words hashed into a fixed-size unit vector: similar wording -> similar vectors."""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.calls = 0

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if len(word) < 3:
                continue
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        self.calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class CitingModels:
    """Answers by quoting a configured sentence from whichever retrieved document contains it.

    rules: list of (trigger_word_in_question, sentence_to_quote, answer_text)
    """

    def __init__(self, rules=(), condense_map=None):
        self.rules = list(rules)
        self.condense_map = condense_map or {}
        self.answer_calls: list[tuple[str, list[dict]]] = []
        self.condense_calls: list[tuple[list, str]] = []

    def condense(self, history, question):
        self.condense_calls.append((history, question))
        return self.condense_map.get(question, question)

    def answer(self, question, documents):
        self.answer_calls.append((question, documents))
        blocks = []
        for trigger, sentence, text in self.rules:
            if trigger.lower() not in question.lower():
                continue
            # PDF extraction can break lines mid-sentence, so match across any whitespace and
            # quote the span exactly as it appears in the chunk (like Claude does).
            pattern = r"\s+".join(re.escape(w) for w in sentence.split())
            for i, doc in enumerate(documents):
                m = re.search(pattern, doc["text"])
                if m:
                    blocks.append({"type": "text", "text": text, "citations": [{
                        "type": "char_location", "cited_text": m.group(0), "document_index": i,
                        "document_title": doc["label"], "start_char_index": m.start(), "end_char_index": m.end()}]})
                    break
        if not blocks:
            blocks = [{"type": "text", "text": "The documents don't cover that."}]
        return AIMessage(content=blocks, usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120})
