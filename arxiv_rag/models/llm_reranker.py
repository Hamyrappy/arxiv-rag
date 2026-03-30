"""LLM-based re-ranker using the Yandex Foundation Models API.

Uses an async worker pattern to score document-query pairs in parallel,
then re-ranks the top candidates returned by a cheap base retriever.

Environment variables
---------------------
YANDEX_API_KEY : str
    Yandex Cloud API key for authentication.
YANDEX_FOLDER_ID : str
    Yandex Cloud folder ID used to build the model URI when ``model_uri``
    is not passed explicitly.

Usage example
-------------
>>> from arxiv_rag.models import BM25RAG, YandexLLMReranker
>>> reranker = YandexLLMReranker(base_retriever=BM25RAG(), top_n=20)
>>> reranker.fit(texts)
>>> indices = reranker.topk("attention mechanism", k=10)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
from typing import Any

import aiohttp
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

_YANDEX_COMPLETION_URL = (
    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
)

_RELEVANCE_PROMPT_TEMPLATE = (
    "You are a search relevance judge.\n"
    "Rate how relevant the document is to the query.\n"
    "Respond with a single decimal number between 0.0 (not relevant) "
    "and 1.0 (perfectly relevant). No other text.\n\n"
    "Query: {query}\n\n"
    "Document: {document}\n\n"
    "Relevance score:"
)

_MAX_DOC_CHARS = 800


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class _YandexMessage(BaseModel):
    role: str
    text: str


class _YandexAlternative(BaseModel):
    message: _YandexMessage
    status: str = ""


class _YandexResult(BaseModel):
    alternatives: list[_YandexAlternative]
    usage: dict[str, Any] = Field(default_factory=dict)
    model_version: str = Field(default="", alias="modelVersion")

    model_config = {"populate_by_name": True}


class _YandexCompletionResponse(BaseModel):
    result: _YandexResult

    @model_validator(mode="after")
    def _check_alternatives(self) -> "_YandexCompletionResponse":
        if not self.result.alternatives:
            raise ValueError("Yandex API returned empty alternatives list.")
        return self


# model_rebuild() ensures forward references between nested Pydantic models are resolved
_YandexMessage.model_rebuild()
_YandexAlternative.model_rebuild()
_YandexResult.model_rebuild()
_YandexCompletionResponse.model_rebuild()


class _RelevanceScore(BaseModel):
    """Validated relevance score in [0, 1]."""

    value: float = Field(..., ge=0.0, le=1.0)

    @classmethod
    def parse_llm_text(cls, text: str) -> "_RelevanceScore":
        """Parse a raw LLM text response into a validated score."""
        stripped = text.strip()
        # Keep only the first token (the LLM sometimes adds punctuation)
        token = stripped.split()[0].rstrip(".,;:") if stripped else "0"
        try:
            raw = float(token)
        except ValueError:
            logger.debug("Could not parse score from %r; defaulting to 0.0", text)
            raw = 0.0
        return cls(value=max(0.0, min(1.0, raw)))


# ---------------------------------------------------------------------------
# Async scoring helpers
# ---------------------------------------------------------------------------


async def _score_document(
    session: aiohttp.ClientSession,
    query: str,
    document: str,
    model_uri: str,
    api_key: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> float:
    """Call the Yandex completions endpoint and return a relevance score."""
    prompt = _RELEVANCE_PROMPT_TEMPLATE.format(
        query=query,
        document=document[:_MAX_DOC_CHARS],
    )
    payload = {
        "modelUri": model_uri,
        "completionOptions": {
            "stream": False,
            "temperature": temperature,
            "maxTokens": 10,
        },
        "messages": [{"role": "user", "text": prompt}],
    }
    headers = {
        "Authorization": f"Api-Key {api_key}",
        "Content-Type": "application/json",
    }

    async with semaphore:
        async with session.post(
            _YANDEX_COMPLETION_URL,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            response.raise_for_status()
            raw = await response.json()

    parsed = _YandexCompletionResponse.model_validate(raw)
    llm_text = parsed.result.alternatives[0].message.text
    return _RelevanceScore.parse_llm_text(llm_text).value


async def _worker(
    queue: asyncio.Queue,
    results: list[tuple[int, float]],
    lock: asyncio.Lock,
    session: aiohttp.ClientSession,
    query: str,
    model_uri: str,
    api_key: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> None:
    """Async worker: pull (position, doc_idx, text) from queue and score it."""
    while True:
        pos, doc_idx, text = await queue.get()
        try:
            score = await _score_document(
                session=session,
                query=query,
                document=text,
                model_uri=model_uri,
                api_key=api_key,
                temperature=temperature,
                semaphore=semaphore,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Scoring failed for doc at position %d (index %d): %s",
                pos,
                doc_idx,
                exc,
            )
            score = 0.0
        finally:
            async with lock:
                results.append((pos, score))
            queue.task_done()


async def _rerank_async(
    query: str,
    texts: list[str],
    candidate_indices: list[int],
    model_uri: str,
    api_key: str,
    temperature: float,
    max_workers: int,
) -> list[float]:
    """Score all candidates concurrently and return scores in original order."""
    queue: asyncio.Queue[tuple[int, int, str]] = asyncio.Queue()
    for pos, doc_idx in enumerate(candidate_indices):
        await queue.put((pos, doc_idx, texts[doc_idx]))

    results: list[tuple[int, float]] = []
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_workers)

    async with aiohttp.ClientSession() as session:
        n_workers = min(max_workers, len(candidate_indices))
        tasks = [
            asyncio.create_task(
                _worker(
                    queue=queue,
                    results=results,
                    lock=lock,
                    session=session,
                    query=query,
                    model_uri=model_uri,
                    api_key=api_key,
                    temperature=temperature,
                    semaphore=semaphore,
                )
            )
            for _ in range(n_workers)
        ]
        await queue.join()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    results.sort(key=lambda x: x[0])
    return [score for _, score in results]


# ---------------------------------------------------------------------------
# Public retriever class
# ---------------------------------------------------------------------------


class YandexLLMReranker:
    """LLM-based re-ranker that calls the Yandex Foundation Models API.

    Architecture
    ------------
    1. **First stage** – a fast base retriever (e.g. BM25 or Hybrid) fetches
       ``top_n`` candidate documents.
    2. **Second stage** – each candidate is scored in parallel via the Yandex
       LLM API using an async worker pool.
    3. Candidates are re-sorted by LLM score and the best ``k`` are returned.

    Parameters
    ----------
    base_retriever :
        Any retriever with ``fit(texts)`` and ``topk(query, k) -> list[int]``.
    api_key : str, optional
        Yandex Cloud API key. Falls back to the ``YANDEX_API_KEY`` env var.
    folder_id : str, optional
        Yandex Cloud folder ID. Falls back to ``YANDEX_FOLDER_ID`` env var.
        Used only when ``model_uri`` is not provided.
    model_uri : str, optional
        Full model URI, e.g. ``"gpt://<folder_id>/yandexgpt-lite"``.
        When omitted, built automatically from ``folder_id``.
    top_n : int
        Number of candidates to fetch from the base retriever (default 20).
    max_workers : int
        Maximum number of concurrent API requests (default 5).
    temperature : float
        Sampling temperature for the LLM (default 0.0 for deterministic output).
    """

    def __init__(
        self,
        base_retriever: Any,
        api_key: str | None = None,
        folder_id: str | None = None,
        model_uri: str | None = None,
        top_n: int = 20,
        max_workers: int = 5,
        temperature: float = 0.0,
    ) -> None:
        self.base_retriever = base_retriever
        self.api_key: str = api_key or os.environ.get("YANDEX_API_KEY", "")
        self.folder_id: str = folder_id or os.environ.get("YANDEX_FOLDER_ID", "")
        if model_uri:
            self.model_uri = model_uri
        elif self.folder_id:
            self.model_uri = f"gpt://{self.folder_id}/yandexgpt-lite"
        else:
            self.model_uri = ""
        self.top_n = top_n
        self.max_workers = max_workers
        self.temperature = temperature
        self.texts: list[str] = []

    # ------------------------------------------------------------------
    # Retriever interface
    # ------------------------------------------------------------------

    def fit(self, texts: list[str]) -> "YandexLLMReranker":
        """Index the corpus in the base retriever and store clean texts."""
        self.texts = [t.replace(" [SEP] ", " ") for t in texts]
        self.base_retriever.fit(texts)
        return self

    def topk(self, query: str, k: int) -> list[int]:
        """Return top-``k`` document indices after LLM-based re-ranking.

        The method is synchronous to match the standard retriever interface.
        Async calls to the Yandex API are dispatched internally.
        """
        if not self.texts:
            raise RuntimeError("Call fit() before topk().")
        if not self.api_key:
            raise ValueError(
                "Yandex API key is required. Set the YANDEX_API_KEY environment "
                "variable or pass api_key to YandexLLMReranker()."
            )
        if not self.model_uri:
            raise ValueError(
                "model_uri is required. Either pass model_uri or set "
                "YANDEX_FOLDER_ID so that the URI can be built automatically."
            )

        candidate_indices = self.base_retriever.topk(query, self.top_n)
        if not candidate_indices:
            return []

        scores = self._run_async_rerank(query, candidate_indices)

        sorted_pairs = sorted(
            zip(candidate_indices, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        final_k = min(k, len(sorted_pairs))
        return [idx for idx, _ in sorted_pairs[:final_k]]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_async_rerank(
        self,
        query: str,
        candidate_indices: list[int],
    ) -> list[float]:
        """Run ``_rerank_async`` regardless of whether a loop is already running."""
        coro = _rerank_async(
            query=query,
            texts=self.texts,
            candidate_indices=candidate_indices,
            model_uri=self.model_uri,
            api_key=self.api_key,
            temperature=self.temperature,
            max_workers=self.max_workers,
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Running inside an existing event loop (e.g. Jupyter / pytest-asyncio).
            # Submit the coroutine in a separate thread that owns its own loop.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return asyncio.run(coro)
