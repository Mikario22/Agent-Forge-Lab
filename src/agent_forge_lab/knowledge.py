"""Strict, local-only synthetic document store; documents are untrusted data."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
WORD = re.compile(r"[a-z0-9]+")
STOP = frozenset("the and for from this that with are was were which what when where who why how into about please can could should summarize explain show list report information related its their then of in on to a an is by as at it or be also have has had".split())


def keywords(value: str) -> frozenset[str]:
    return frozenset(word for word in WORD.findall(value.casefold()) if word not in STOP and len(word) > 2)


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    title: str
    score: int


class KnowledgeBase:
    """No network, disk writes, general search, or arbitrary file reading."""

    def __init__(self, documents: Sequence[Document]):
        if not 1 <= len(documents) <= 200:
            raise ValueError("knowledge base must have 1–200 documents")
        if len({d.doc_id for d in documents}) != len(documents):
            raise ValueError("duplicate document IDs")
        for doc in documents:
            if not ID.fullmatch(doc.doc_id) or not doc.title.strip() or not doc.text.strip():
                raise ValueError("invalid document")
            if len(doc.title) > 200 or len(doc.text) > 8000:
                raise ValueError("document exceeds limits")
        self._docs = {doc.doc_id: doc for doc in documents}

    @classmethod
    def from_json(cls, path: str | Path) -> "KnowledgeBase":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) != {"dataset", "documents"} or not isinstance(raw["dataset"], str):
            raise ValueError("knowledge manifest requires dataset and documents")
        rows = raw["documents"]
        if not isinstance(rows, list) or not 1 <= len(rows) <= 200:
            raise ValueError("documents must be a bounded nonempty list")
        docs = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"id", "title", "text"}:
                raise ValueError("invalid document schema")
            if any(not isinstance(row[key], str) for key in ("id", "title", "text")):
                raise ValueError("document values must be strings")
            docs.append(Document(row["id"], row["title"], row["text"]))
        return cls(docs)

    def search(self, query: str, *, limit: int = 3) -> tuple[SearchResult, ...]:
        if not isinstance(query, str) or not query.strip() or len(query) > 240:
            raise ValueError("search query must contain 1–240 characters")
        if type(limit) is not int or not 1 <= limit <= 5:
            raise ValueError("search limit must be 1–5")
        terms = keywords(query)
        ranked = []
        for doc in self._docs.values():
            title_words, content_words = keywords(doc.title), keywords(doc.text)
            score = 3 * len(terms & title_words) + len(terms & content_words)
            if score:
                ranked.append(SearchResult(doc.doc_id, doc.title, score))
        return tuple(sorted(ranked, key=lambda r: (-r.score, r.doc_id))[:limit])

    def read(self, document_id: str) -> Document:
        try:
            return self._docs[document_id]
        except KeyError as exc:
            raise ValueError("unknown document ID") from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._docs))
