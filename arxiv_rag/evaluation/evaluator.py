"""Architecture-agnostic evaluation pipeline for RAG retrieval models."""

import math
from typing import Any, Iterable

from tqdm import tqdm


class Evaluator:
    """Evaluator for RAG retrieval models.

    Works with any retriever that exposes:
      - ``fit(texts: Iterable[str])`` – index the corpus
      - ``topk(query: str, k: int) -> list[int]`` – return ranked corpus indices

    Parameters
    ----------
    retriever:
        A retriever object (e.g. ``TfidfRAG`` or ``BM25RAG``) that has
        ``fit`` and ``topk`` methods.
    doc_ids : list[str]
        Document identifiers, one per corpus document.  The position in this
        list must correspond to the integer index that ``topk`` returns.
    texts : list[str]
        Raw text for each document.  Passed directly to ``retriever.fit``.
    """

    def __init__(self, retriever: Any, doc_ids: list[str], texts: list[str]) -> None:
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length.")
        self.retriever = retriever
        self.doc_ids = list(doc_ids)
        self.texts = list(texts)
        self.retriever.fit(self.texts)

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
        """Recall@k: fraction of relevant docs found in the top-k results."""
        if not relevant_ids:
            return 0.0
        hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
        return hits / len(relevant_ids)

    @staticmethod
    def _reciprocal_rank(retrieved_ids: list, relevant_ids: set) -> float:
        """Reciprocal rank of the first relevant document in the ranked list."""
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def _ndcg_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
        """nDCG@k with binary relevance judgements."""
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
            if doc_id in relevant_ids:
                dcg += 1.0 / math.log2(rank + 1)

        # Ideal DCG: place all relevant docs at the top
        ideal_hits = min(len(relevant_ids), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

        return dcg / idcg if idcg > 0 else 0.0

    # ------------------------------------------------------------------
    # Main evaluation method
    # ------------------------------------------------------------------

    def evaluate(
        self,
        benchmark: list,
        k: int = 10,
        show_progress: bool = False,
        progress_desc: str = "Evaluating",
    ) -> dict:
        """Run evaluation over all benchmark queries and return averaged metrics.

        Parameters
        ----------
        benchmark : list[dict]
            Each entry must have the keys:
              - ``"query"`` (str)
              - ``"relevant_ids"`` (list[str]) – ground-truth doc IDs
        k : int
            Cut-off depth for Recall@k and nDCG@k (default 10).
        show_progress : bool
            Whether to show a tqdm progress bar over benchmark queries.
        progress_desc : str
            Description to show next to the progress bar.

        Returns
        -------
        dict
            ``{"recall@k": float, "mrr": float, "ndcg@k": float,
               "per_query": list[dict]}``
        """
        if not benchmark:
            raise ValueError("benchmark must contain at least one query.")

        per_query_results = []
        recall_sum = mrr_sum = ndcg_sum = 0.0

        entries: Iterable[dict] = benchmark
        if show_progress:
            entries = tqdm(benchmark, desc=progress_desc, unit="query")

        for entry in entries:
            query: str = entry["query"]
            relevant_ids: set = set(entry["relevant_ids"])

            # Retrieve top-k indices from the model
            indices = self.retriever.topk(query, k)
            retrieved_ids = [self.doc_ids[i] for i in indices]

            recall = self._recall_at_k(retrieved_ids, relevant_ids, k)
            rr = self._reciprocal_rank(retrieved_ids, relevant_ids)
            ndcg = self._ndcg_at_k(retrieved_ids, relevant_ids, k)

            recall_sum += recall
            mrr_sum += rr
            ndcg_sum += ndcg

            per_query_results.append(
                {
                    "query": query,
                    f"recall@{k}": recall,
                    "rr": rr,
                    f"ndcg@{k}": ndcg,
                    "retrieved_ids": retrieved_ids,
                }
            )

        n = len(benchmark)
        return {
            f"recall@{k}": recall_sum / n,
            "mrr": mrr_sum / n,
            f"ndcg@{k}": ndcg_sum / n,
            "per_query": per_query_results,
        }
