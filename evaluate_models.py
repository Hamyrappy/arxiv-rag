"""Evaluate BM25 and TF-IDF baselines on a benchmark stored under eval/.

The benchmark format is a TSV file with at least two columns:
  - query
  - relevant_ids

The relevant_ids column must contain a JSON list of arXiv IDs, for example:
    ["1706.03762", "1810.04805"]

Usage
-----
    python evaluate_models.py
    python evaluate_models.py --model bm25 --benchmark eval/benchmark.tsv --k 20
    python evaluate_models.py --model all --benchmark eval/benchmark_fast.tsv --limit 50000 --k 20
    python evaluate_models.py --model tfidf --benchmark eval/benchmark_fast.tsv --limit 50000 --k 20 --show-per-query
    python evaluate_models.py --benchmark eval/benchmark.tsv --validate-only

If some relevant_ids are not present in the loaded corpus slice, they are skipped.
Queries with no remaining relevant ids are excluded from evaluation and reported.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from arxiv_rag.dataset import load_arxiv_data
from arxiv_rag.evaluation import Evaluator
from arxiv_rag.models import (
    BM25RAG,
    BGERetriever,
    HybridRetriever,
    MiniLMRetriever,
    Specter1Retriever,
    Specter2Retriever,
    TfidfRAG,
)

DEFAULT_DATA_FOLDER = Path("data/processed")
DEFAULT_BENCHMARK_PATH = Path("eval/benchmark.tsv")
DEFAULT_LIMIT = None
DEFAULT_K = 10
DEFAULT_MODEL = "all"

REQUIRED_BENCHMARK_COLUMNS = {"query", "relevant_ids"}
ARXIV_ID_PATTERN = re.compile(
    r"^(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z-]+)?/\d{7})(?:v\d+)?$",
    re.IGNORECASE,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate BM25 and TF-IDF on a benchmark stored in eval/*.tsv."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Model or retriever to evaluate: bm25, tfidf, all, or a custom "
            "import path in the form module.submodule:factory_or_class."
        ),
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK_PATH,
        help=f"Path to TSV benchmark file (default: {DEFAULT_BENCHMARK_PATH}).",
    )
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=DEFAULT_DATA_FOLDER,
        help=f"Folder with processed part_*.parquet files (default: {DEFAULT_DATA_FOLDER}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of documents to load from processed data (default: all documents).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Cut-off for Recall@k and nDCG@k (default: {DEFAULT_K}).",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--summary-only",
        dest="show_per_query",
        action="store_false",
        default=False,
        help="Print only aggregate metrics. This is the default behavior.",
    )
    output_group.add_argument(
        "--show-per-query",
        dest="show_per_query",
        action="store_true",
        help="Print full per-query breakdown in addition to aggregate metrics.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate benchmark format and exit without loading the corpus or evaluating models.",
    )
    return parser


def _normalize_arxiv_id(value: object) -> str:
    doc_id = str(value).strip()
    if doc_id.lower().startswith("arxiv:"):
        doc_id = doc_id[6:].strip()
    if "/" in doc_id:
        prefix, suffix = doc_id.split("/", 1)
        doc_id = f"{prefix.lower()}/{suffix}"
    return doc_id


def _parse_relevant_ids(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, list):
        return [_normalize_arxiv_id(item) for item in value if str(item).strip()]
    raw = str(value).strip()
    if not raw:
        return []

    parsed: object
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid relevant_ids value: {raw!r}. Expected JSON list like ['id1', 'id2']."
        ) from exc

    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid relevant_ids value: {raw!r}. Expected a JSON list, not a quoted string."
            ) from exc

    if not isinstance(parsed, list):
        raise ValueError(
            f"Invalid relevant_ids value: {raw!r}. Expected JSON list like ['id1', 'id2']."
        )

    normalized_ids = [_normalize_arxiv_id(item) for item in parsed if str(item).strip()]
    invalid_ids = [doc_id for doc_id in normalized_ids if not ARXIV_ID_PATTERN.match(doc_id)]
    if invalid_ids:
        raise ValueError(
            "Invalid arXiv id format in relevant_ids: "
            + ", ".join(sorted(invalid_ids[:5]))
        )
    return normalized_ids


def _validate_benchmark_row(row_num: int, query: str, relevant_ids: list[str]) -> None:
    if not query and not relevant_ids:
        raise ValueError(f"Row {row_num}: both query and relevant_ids are empty.")
    if not query:
        raise ValueError(f"Row {row_num}: query is empty.")
    if not relevant_ids:
        raise ValueError(f"Row {row_num}: relevant_ids is empty after parsing.")


def load_benchmark(benchmark_path: Path) -> list[dict]:
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file does not exist: {benchmark_path}")

    benchmark = []
    with benchmark_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        if not reader.fieldnames:
            raise ValueError(f"Benchmark file is empty: {benchmark_path}")

        missing_columns = REQUIRED_BENCHMARK_COLUMNS - set(reader.fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Benchmark file is missing required columns: {missing}")

        for row_num, row in enumerate(reader, start=2):
            query = str(row.get("query", "")).strip()
            raw_relevant_ids = row.get("relevant_ids", "")

            if not query and not str(raw_relevant_ids).strip():
                continue

            try:
                relevant_ids = _parse_relevant_ids(raw_relevant_ids)
                _validate_benchmark_row(row_num, query, relevant_ids)
            except ValueError as exc:
                raise ValueError(f"Benchmark validation failed at row {row_num}: {exc}") from exc

            benchmark.append({"query": query, "relevant_ids": relevant_ids})

    if not benchmark:
        raise ValueError(
            f"No valid benchmark rows found in {benchmark_path}. Fill query and relevant_ids."
        )

    return benchmark


def load_corpus(
    data_folder: Path,
    limit: int | None,
    must_include_ids: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    df = load_arxiv_data(
        data_folder=str(data_folder),
        limit=limit,
        columns=["id", "title", "abstract"],
        must_include_ids=must_include_ids,
    )
    if df.empty:
        raise ValueError(f"No documents loaded from processed data folder: {data_folder}")

    doc_ids = df["id"].fillna("").astype(str).tolist()
    titles = df["title"].fillna("").astype(str)
    abstracts = df["abstract"].fillna("").astype(str)
    texts = (titles + " [SEP] " + abstracts).str.strip().tolist()
    return doc_ids, texts


def reconcile_benchmark_with_corpus(
    benchmark: list[dict],
    available_doc_ids: set[str],
) -> tuple[list[dict], dict[str, Any]]:
    total_queries = len(benchmark)
    total_relevant_ids = 0
    missing_relevant_ids = 0
    skipped_queries = 0
    missing_ids_set: set[str] = set()

    filtered_benchmark: list[dict] = []
    for item in benchmark:
        relevant_ids = [str(doc_id) for doc_id in item["relevant_ids"]]
        total_relevant_ids += len(relevant_ids)
        present_ids = [doc_id for doc_id in relevant_ids if doc_id in available_doc_ids]
        absent_ids = [doc_id for doc_id in relevant_ids if doc_id not in available_doc_ids]

        missing_relevant_ids += len(absent_ids)
        missing_ids_set.update(absent_ids)

        if not present_ids:
            skipped_queries += 1
            continue

        filtered_benchmark.append(
            {
                "query": item["query"],
                "relevant_ids": present_ids,
            }
        )

    evaluated_queries = len(filtered_benchmark)
    coverage = 0.0
    if total_relevant_ids > 0:
        coverage = (total_relevant_ids - missing_relevant_ids) / total_relevant_ids

    summary = {
        "total_queries": total_queries,
        "evaluated_queries": evaluated_queries,
        "skipped_queries": skipped_queries,
        "total_relevant_ids": total_relevant_ids,
        "missing_relevant_ids": missing_relevant_ids,
        "relevant_id_coverage": coverage,
        "missing_ids": sorted(missing_ids_set),
    }
    return filtered_benchmark, summary


def _build_retriever(model_name: str) -> tuple[str, Any]:
    registry = {
        "bm25": ("BM25RAG", BM25RAG),
        "tfidf": ("TfidfRAG", TfidfRAG),
        "specter1": ("SPECTER-v1", Specter1Retriever),
        "specter2": ("SPECTER-v2", Specter2Retriever),
        "bge": ("BGE-small", BGERetriever),
        "minilm": ("MiniLM-L6", MiniLMRetriever),
        "hybrid-rrf": (
            "Hybrid-RRF(BM25+BGE)",
            lambda: HybridRetriever(BM25RAG(), BGERetriever(), fusion="rrf"),
        ),
        "hybrid-rrf-specter": (
            "Hybrid-RRF(BM25+SPECTER2)",
            lambda: HybridRetriever(BM25RAG(), Specter2Retriever(), fusion="rrf"),
        ),
        "hybrid-weighted": (
            "Hybrid-Weighted(BM25+BGE)",
            lambda: HybridRetriever(BM25RAG(), BGERetriever(), fusion="weighted", alpha=0.5),
        ),
        "hybrid-weighted-specter": (
            "Hybrid-Weighted(BM25+SPECTER2)",
            lambda: HybridRetriever(BM25RAG(), Specter2Retriever(), fusion="weighted", alpha=0.5),
        ),
    }

    if model_name in registry:
        display_name, factory = registry[model_name]
        return display_name, factory()

    if ":" not in model_name:
        valid = ", ".join(["all", *sorted(registry)])
        raise ValueError(
            f"Unknown model '{model_name}'. Use one of: {valid}, or module.submodule:factory_or_class"
        )

    module_name, attr_name = model_name.split(":", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, attr_name)
    retriever = obj() if callable(obj) else obj

    if not callable(getattr(retriever, "fit", None)) or not callable(getattr(retriever, "topk", None)):
        raise TypeError(
            f"Custom retriever '{model_name}' must provide callable fit(texts) and topk(query, k) methods."
        )

    display_name = getattr(retriever, "__class__", type(retriever)).__name__
    return display_name, retriever


def resolve_retrievers(model_arg: str) -> list[tuple[str, Any]]:
    normalized = model_arg.strip().lower()
    if normalized == "all":
        return [_build_retriever("bm25"), _build_retriever("tfidf")]
    return [_build_retriever(model_arg.strip())]


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def run_evaluation(
    model_name: str,
    retriever,
    doc_ids: list[str],
    texts: list[str],
    benchmark: list[dict],
    benchmark_summary: dict[str, Any],
    k: int = 10,
    show_per_query: bool = False,
) -> None:
    import time

    print(f"\nBuilding index for {model_name}...")
    t0 = time.perf_counter()
    evaluator = Evaluator(retriever=retriever, doc_ids=doc_ids, texts=texts)
    index_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    results = evaluator.evaluate(
        benchmark,
        k=k,
        show_progress=True,
        progress_desc=f"Evaluating {model_name}",
    )
    query_time = time.perf_counter() - t0
    n_queries = len(benchmark)
    avg_latency_ms = (query_time / n_queries * 1000) if n_queries else 0.0

    total_queries = benchmark_summary["total_queries"]
    evaluated_queries = benchmark_summary["evaluated_queries"]
    query_factor = evaluated_queries / total_queries if total_queries else 0.0

    penalized_recall = results[f"recall@{k}"] * query_factor
    penalized_mrr = results["mrr"] * query_factor
    penalized_ndcg = results[f"ndcg@{k}"] * query_factor

    print(f"\n{'=' * 60}")
    print(f"Model : {model_name}")
    print(f"{'=' * 60}")
    print(f"  Recall@{k} (evaluated) : {results[f'recall@{k}']:.4f}")
    print(f"  MRR (evaluated)        : {results['mrr']:.4f}")
    print(f"  nDCG@{k} (evaluated)   : {results[f'ndcg@{k}']:.4f}")
    print(f"  Recall@{k} (penalized) : {penalized_recall:.4f}")
    print(f"  MRR (penalized)        : {penalized_mrr:.4f}")
    print(f"  nDCG@{k} (penalized)   : {penalized_ndcg:.4f}")
    print(f"  --- Speed ---")
    print(f"  Index time             : {index_time:.2f}s")
    print(f"  Total query time       : {query_time:.2f}s ({n_queries} queries)")
    print(f"  Avg query latency      : {avg_latency_ms:.1f}ms")

    if not show_per_query:
        return

    print()
    print("  Per-query breakdown:")
    for item in results["per_query"]:
        rr_val = item["rr"]
        hit = "✓" if rr_val > 0 else "✗"
        print(
            f"    [{hit}] RR={rr_val:.2f}  "
            f"Recall@{k}={item[f'recall@{k}']:.2f}  "
            f"nDCG@{k}={item[f'ndcg@{k}']:.2f}  "
            f"| {item['query'][:65]}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        benchmark = load_benchmark(args.benchmark)
        retrievers = resolve_retrievers(args.model)
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc

    if args.validate_only:
        print("arXiv RAG – Benchmark Validation")
        print(f"Benchmark file  : {args.benchmark}")
        print(f"Queries valid   : {len(benchmark)}")
        print("Status          : OK")
        return

    try:
        benchmark_ids = {
            str(doc_id)
            for item in benchmark
            for doc_id in item["relevant_ids"]
        }
        doc_ids, texts = load_corpus(args.data_folder, args.limit, must_include_ids=benchmark_ids)
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc

    benchmark, benchmark_summary = reconcile_benchmark_with_corpus(
        benchmark=benchmark,
        available_doc_ids=set(doc_ids),
    )

    if benchmark_summary["evaluated_queries"] == 0:
        preview = ", ".join(benchmark_summary["missing_ids"][:10])
        suffix = "..." if len(benchmark_summary["missing_ids"]) > 10 else ""
        raise SystemExit(
            "Error: none of the benchmark queries can be evaluated with the loaded corpus. "
            f"Missing ids preview: {preview}{suffix}"
        )

    print("arXiv RAG – Benchmark Evaluation")
    print(f"Corpus size     : {len(doc_ids)} documents")
    print(f"Benchmark file  : {args.benchmark}")
    print("Models          : " + ", ".join(name for name, _ in retrievers))
    print(f"Queries total   : {benchmark_summary['total_queries']}")
    print(f"Queries used    : {benchmark_summary['evaluated_queries']}")
    print(f"Queries skipped : {benchmark_summary['skipped_queries']}")
    print(f"Relevant id coverage : {benchmark_summary['relevant_id_coverage']:.2%}")
    print(f"Cut-off k       : {args.k}")

    if benchmark_summary["missing_relevant_ids"] > 0:
        preview = ", ".join(benchmark_summary["missing_ids"][:10])
        suffix = "..." if len(benchmark_summary["missing_ids"]) > 10 else ""
        print(
            "Warning: some relevant_ids are not in the loaded corpus and were skipped. "
            f"Missing ids: {preview}{suffix}"
        )

    for model_name, retriever in retrievers:
        run_evaluation(
            model_name,
            retriever,
            doc_ids=doc_ids,
            texts=texts,
            benchmark=benchmark,
            benchmark_summary=benchmark_summary,
            k=args.k,
            show_per_query=args.show_per_query,
        )


if __name__ == "__main__":
    main()
