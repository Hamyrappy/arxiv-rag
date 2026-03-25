"""Quick CLI for searching arXiv papers with a single query."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from arxiv_rag.dataset import load_arxiv_data
from arxiv_rag.models import (
    BGERetriever,
    BM25RAG,
    CrossEncoderReranker,
    HybridRetriever,
    MiniLMRetriever,
    PaletsvNeboRetriever,
    Specter1Retriever,
    Specter2Retriever,
    TfidfRAG,
)

DEFAULT_DATA_FOLDER = "data/processed"
DEFAULT_LIMIT = 50000
DEFAULT_TOPK = 3
MODEL_CHOICES = [
    "bm25",
    "tfidf",
    "specter1",
    "specter2",
    "bge",
    "minilm",
    "hybrid-rrf",
    "hybrid-rrf-specter",
    "hybrid-weighted",
    "hybrid-weighted-specter",
    "cross-encoder",
    "paletsv-nebo",
    "random",
]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quick search for arXiv papers. Returns top-k papers for a query.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  arxiv-rag-quick "transformer attention"
  arxiv-rag-quick "neural networks" --k 5 --model tfidf
  arxiv-rag-quick "reinforcement learning" --limit 100000
        """,
    )
    parser.add_argument(
        "query",
        help="Search query for arXiv papers",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="bm25",
        help="Retriever to use (default: bm25)",
    )
    parser.add_argument(
        "--k",
        type=_positive_int,
        default=DEFAULT_TOPK,
        help=f"Number of top papers to return (default: {DEFAULT_TOPK})",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=DEFAULT_LIMIT,
        help=f"Max documents to load from corpus (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--data-folder",
        default=DEFAULT_DATA_FOLDER,
        help=f"Folder with part_*.parquet files (default: {DEFAULT_DATA_FOLDER})",
    )
    return parser


def _build_model(model_name: str):
    """Build retriever instance."""
    registry = {
        "bm25": BM25RAG,
        "tfidf": TfidfRAG,
        "specter1": Specter1Retriever,
        "specter2": Specter2Retriever,
        "bge": BGERetriever,
        "minilm": MiniLMRetriever,
        "paletsv-nebo": PaletsvNeboRetriever,
        "random": PaletsvNeboRetriever,
        "hybrid-rrf": lambda: HybridRetriever(BM25RAG(), BGERetriever(), fusion="rrf"),
        "hybrid-rrf-specter": lambda: HybridRetriever(BM25RAG(), Specter2Retriever(), fusion="rrf"),
        "hybrid-weighted": lambda: HybridRetriever(BM25RAG(), BGERetriever(), fusion="weighted", alpha=0.5),
        "hybrid-weighted-specter": lambda: HybridRetriever(
            BM25RAG(), Specter2Retriever(), fusion="weighted", alpha=0.5
        ),
        "cross-encoder": lambda: CrossEncoderReranker(
            HybridRetriever(BM25RAG(), BGERetriever(), fusion="weighted", alpha=0.5),
            top_n=100,
        ),
    }
    return registry[model_name]()


def _format_abstract(abstract: str, max_length: int = 150) -> str:
    """Truncate abstract to max_length with ellipsis."""
    if pd.isna(abstract):
        return "[No abstract]"
    abstract = str(abstract).strip()
    if not abstract:
        return "[No abstract]"
    if len(abstract) > max_length:
        return abstract[:max_length] + "..."
    return abstract


def _format_title(title: str) -> str:
    if pd.isna(title):
        return ""
    return str(title).strip()


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
        print(
            "Error: Failed to load processed dataset.",
            file=sys.stderr,
        )
        print(
            "Please run: arxiv-rag-prepare-data",
            file=sys.stderr,
        )
        print(f"Original error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    texts = df["abstract"].fillna("").astype(str).tolist()
    if not texts:
        print(
            "Error: Loaded dataset is empty."
            " Check --data-folder and --limit values.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    model = _build_model(args.model)
    model.fit(texts)

    indices = model.topk(args.query, k=args.k) or []

    if not indices:
        print(f"No results found for query: {args.query}")
        raise SystemExit(0)

    print(f"\nTop {len(indices)} results for: '{args.query}'\n")
    print("=" * 80)

    for rank, idx in enumerate(indices, 1):
        row = df.iloc[idx]
        arxiv_id = str(row["id"])
        title = _format_title(row["title"])
        abstract = _format_abstract(row["abstract"])

        print(f"\n{rank}. arXiv:{arxiv_id}")
        print(f"   Title: {title}")
        print(f"   Abstract: {abstract}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
