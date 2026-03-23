"""Hybrid retrieval combining sparse and dense retrievers."""

from typing import Iterable, List, Protocol


class Retriever(Protocol):
    def fit(self, texts: Iterable[str]) -> None:
        ...

    def topk(self, query: str, k: int) -> List[int]:
        ...


class HybridRetriever:
    """Combine a sparse and a dense retriever via Reciprocal Rank Fusion or weighted scores.

    Parameters
    ----------
    sparse : object
        Sparse retriever (e.g. BM25RAG) with fit/topk interface.
    dense : object
        Dense retriever (e.g. MiniLMRetriever) with fit/topk interface.
    alpha : float
        Weight of the dense retriever in weighted score fusion (0 = only sparse, 1 = only dense).
        Ignored when fusion="rrf".
    fusion : str
        Fusion strategy: "rrf" (Reciprocal Rank Fusion) or "weighted".
    rrf_k : int
        Constant for RRF formula: score = 1 / (k + rank). Default 60.
    retrieval_depth : int
        How many results to fetch from each sub-retriever before fusion.
    """

    def __init__(
        self,
        sparse: Retriever,
        dense: Retriever,
        alpha: float = 0.5,
        fusion: str = "rrf",
        rrf_k: int = 60,
        retrieval_depth: int = 100,
    ):
        self.sparse = sparse
        self.dense = dense
        self.alpha = alpha
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.retrieval_depth = retrieval_depth
        self._n_docs: int = 0

    def fit(self, texts: Iterable[str]) -> "HybridRetriever":
        texts_list = list(texts)
        self._n_docs = len(texts_list)
        self.sparse.fit(texts_list)
        self.dense.fit(texts_list)
        return self

    def topk(self, query: str, k: int) -> list[int]:
        depth = max(k, self.retrieval_depth)
        sparse_ids = self.sparse.topk(query, depth)
        dense_ids = self.dense.topk(query, depth)

        if self.fusion == "rrf":
            return self._rrf_fusion(sparse_ids, dense_ids, k)
        return self._weighted_fusion(sparse_ids, dense_ids, k)

    def _rrf_fusion(
        self, sparse_ids: list[int], dense_ids: list[int], k: int
    ) -> list[int]:
        scores: dict[int, float] = {}
        for rank, doc_id in enumerate(sparse_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, doc_id in enumerate(dense_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        return ranked[:k]

    def _weighted_fusion(
        self, sparse_ids: list[int], dense_ids: list[int], k: int
    ) -> list[int]:
        scores: dict[int, float] = {}
        n_sparse = len(sparse_ids)
        n_dense = len(dense_ids)

        for rank, doc_id in enumerate(sparse_ids):
            norm_score = 1.0 - rank / n_sparse
            scores[doc_id] = scores.get(doc_id, 0.0) + (1 - self.alpha) * norm_score
        for rank, doc_id in enumerate(dense_ids):
            norm_score = 1.0 - rank / n_dense
            scores[doc_id] = scores.get(doc_id, 0.0) + self.alpha * norm_score

        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        return ranked[:k]
