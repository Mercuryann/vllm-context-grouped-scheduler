#!/usr/bin/env python3
"""
Lightweight RAG retriever for IK2221 Task 3.

This module builds a tiny TF-IDF embedding space over the context documents in
frontend/data. It intentionally avoids heavyweight dependencies so the benchmark
can run in the same environment as Task 1 and Task 2.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{2,}")

STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "based",
    "between",
    "can",
    "does",
    "for",
    "from",
    "has",
    "how",
    "into",
    "its",
    "main",
    "more",
    "paper",
    "problem",
    "section",
    "short",
    "summary",
    "that",
    "the",
    "their",
    "this",
    "through",
    "what",
    "when",
    "where",
    "which",
    "with",
    "write",
}


def tokenize(text: str) -> list[str]:
    """Tokenize text for retrieval, not for LLM inference."""
    return [
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if token.lower() not in STOPWORDS
    ]


@dataclass(frozen=True)
class SearchHit:
    context_id: str
    score: float
    rank: int


# [Task3] Sparse TF-IDF document index with cosine-similarity search
class TfidfRagIndex:
    """Sparse TF-IDF document index with cosine-similarity search."""

    def __init__(self, contexts: dict[str, str]):
        if not contexts:
            raise ValueError("TfidfRagIndex requires at least one context")
        self.contexts = dict(contexts)
        self._doc_tokens: dict[str, list[str]] = {
            cid: tokenize(text) for cid, text in self.contexts.items()
        }
        self._idf = self._build_idf(self._doc_tokens.values())  # [Task3] IDF = log((N+1)/(df+1)) + 1
        self._vectors = {  # [Task3] TF-IDF vector per document, L2-normalized
            cid: self._vectorize_tokens(tokens)
            for cid, tokens in self._doc_tokens.items()
        }

    @staticmethod
    def _build_idf(docs: Iterable[list[str]]) -> dict[str, float]:
        docs = list(docs)
        n_docs = len(docs)
        df: Counter[str] = Counter()
        for tokens in docs:
            df.update(set(tokens))
        return {
            term: math.log((n_docs + 1) / (freq + 1)) + 1.0
            for term, freq in df.items()
        }

    def _vectorize_tokens(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        weights: dict[str, float] = {}
        for term, count in counts.items():
            if term not in self._idf:
                continue
            tf = 1.0 + math.log(count)  # [Task3] sub-linear TF
            weights[term] = tf * self._idf[term]  # [Task3] TF * IDF
        norm = math.sqrt(sum(value * value for value in weights.values()))  # [Task3] L2 norm
        if norm == 0:
            return {}
        return {term: value / norm for term, value in weights.items()}

    def vectorize_query(self, query: str) -> dict[str, float]:
        return self._vectorize_tokens(tokenize(query))

    @staticmethod
    def _cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        if len(a) > len(b):
            a, b = b, a
        return sum(value * b.get(term, 0.0) for term, value in a.items())

    # [Task3] Core retrieval: cosine similarity between query and all documents
    def search(self, query: str, *, top_k: int = 3) -> list[SearchHit]:
        q_vec = self.vectorize_query(query)
        scored = [
            SearchHit(cid, self._cosine_sparse(q_vec, d_vec), rank=0)
            for cid, d_vec in self._vectors.items()
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return [
            SearchHit(hit.context_id, hit.score, rank=i + 1)
            for i, hit in enumerate(scored[:top_k])
        ]

    # [Task3] Return top-n TF-IDF keywords for a document (used to generate eval questions)
    def top_terms(self, context_id: str, *, n: int = 5) -> list[str]:
        if context_id not in self._vectors:
            raise KeyError(context_id)
        pairs = sorted(
            self._vectors[context_id].items(),
            key=lambda item: item[1],
            reverse=True,
        )
        return [term for term, _ in pairs[:n]]

# [Task3] Load all .txt files from data dir as {filename_stem: content}
def load_text_contexts(
    data_dir: str | Path,
    *,
    exclude: Iterable[str] = ("sample",),
) -> dict[str, str]:
    data_dir = Path(data_dir)
    exclude_set = set(exclude)
    contexts: dict[str, str] = {}
    for path in sorted(data_dir.glob("*.txt")):
        if path.stem in exclude_set:
            continue
        contexts[path.stem] = path.read_text(encoding="utf-8")
    if not contexts:
        raise FileNotFoundError(f"No .txt contexts found in {data_dir}")
    return contexts