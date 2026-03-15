"""Baseline RAG models: TF-IDF and BM25."""

import re
from typing import Iterable, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, alphanumeric tokens."""
    return re.findall(r"\w+", text.lower())


class TfidfRAG:
    """RAG baseline using TF-IDF vectors and cosine similarity."""

    def __init__(self, max_features: int = 50_000, **kwargs):
        self.vectorizer = TfidfVectorizer(max_features=max_features, **kwargs)
        self.matrix_ = None
        self.texts_: list[str] = []

    def fit(self, texts: Iterable[str]) -> 'TfidfRAG':
        self.texts_ = list(texts)
        self.matrix_ = self.vectorizer.fit_transform(self.texts_)

        return self

    def topk(self, query: str, k: int) -> list[int]:
        if self.matrix_ is None:
            raise RuntimeError("Call fit() before topk()")
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(self.matrix_, q).ravel()
        n = min(k, len(scores))
        return np.argsort(scores)[-n:][::-1].tolist()


class BM25RAG:
    """RAG baseline using Okapi BM25."""

    def __init__(self):
        self.bm25_: Optional[BM25Okapi] = None
        self.tokenized_corpus_: list[list[str]] = []

    def fit(self, texts: Iterable[str]) -> 'BM25RAG':
        self.tokenized_corpus_ = [_tokenize(t) for t in texts]
        self.bm25_ = BM25Okapi(self.tokenized_corpus_)

        return self

    def topk(self, query: str, k: int) -> list[int]:
        if self.bm25_ is None:
            raise RuntimeError("Call fit() before topk()")
        tokenized_query = _tokenize(query)
        scores = self.bm25_.get_scores(tokenized_query)
        n = min(k, len(scores))
        return np.argsort(scores)[-n:][::-1].tolist()
