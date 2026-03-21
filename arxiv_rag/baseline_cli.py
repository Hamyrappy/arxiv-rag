"""CLI for running simple retrieval baselines on prepared arXiv data."""

from __future__ import annotations

import argparse

from arxiv_rag.dataset import load_arxiv_data
from arxiv_rag.models import BM25RAG, TfidfRAG

DEFAULT_DATA_FOLDER = "data/processed"
DEFAULT_LIMIT = 2000
DEFAULT_TOPK = 5
DEFAULT_QUERIES = [
    "transformer attention mechanism",
    "reinforcement learning for robotics",
    "graph neural networks",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run BM25 or TF-IDF baseline retrieval on prepared arXiv data."
    )
    parser.add_argument(
        "--data-folder",
        default=DEFAULT_DATA_FOLDER,
        help=f"Folder with part_*.parquet files (default: {DEFAULT_DATA_FOLDER}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of documents to load (default: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--model",
        choices=["bm25", "tfidf"],
        default="bm25",
        help="Retriever to use.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_TOPK,
        help=f"Top-k documents to return per query (default: {DEFAULT_TOPK}).",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=None,
        help="Query to run. Pass multiple times for multiple queries.",
    )
    return parser


def _build_model(model_name: str):
    if model_name == "bm25":
        return BM25RAG()
    return TfidfRAG()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        df = load_arxiv_data(
            data_folder=args.data_folder,
            limit=args.limit,
            columns=["id", "title", "abstract"],
        )
    except Exception as exc:
        raise SystemExit(
            "Failed to load processed dataset. "
            "Run `arxiv-rag-prepare-data` first or pass --data-folder. "
            f"Original error: {exc}"
        ) from exc

    texts = df["abstract"].fillna("").astype(str).tolist()
    model = _build_model(args.model)
    model.fit(texts)

    print(f"Loaded {len(df)} documents")
    print(f"Model: {args.model}")

    queries = args.query if args.query else DEFAULT_QUERIES

    for query in queries:
        indices = model.topk(query, k=args.k) or []
        print(f"\nQuery: {query}")
        for rank, idx in enumerate(indices, 1):
            row = df.iloc[idx]
            title = str(row["title"] or "")[:100]
            print(f"  {rank}. [{idx}] {title}")


if __name__ == "__main__":
    main()
