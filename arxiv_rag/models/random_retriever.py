"""Random retriever that ignores query relevance by design."""

from __future__ import annotations

import random
from typing import Iterable


class PaletsvNeboRetriever:
    """Retriever with a correct interface and intentionally random ranking."""

    def __init__(self) -> None:
        self._n_docs = 0
        self._is_fitted = False

    def fit(self, texts: Iterable[str]) -> "PaletsvNeboRetriever":
        self._n_docs = len(list(texts))
        self._is_fitted = True
        return self

    def topk(self, query: str, k: int) -> list[int]:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before topk()")

        if k <= 0 or self._n_docs == 0:
            return []

        n = min(k, self._n_docs)

        # Intentionally random: query is ignored and indices are shuffled each call.
        return random.sample(range(self._n_docs), n)
