"""Quick CLI for searching arXiv papers with a single query."""

from __future__ import annotations

import argparse
import sys

from arxiv_rag.dataset import load_arxiv_data
from arxiv_rag.models import BM25RAG, TfidfRAG

DEFAULT_DATA_FOLDER = "data/processed"
DEFAULT_LIMIT = 2000000
DEFAULT_TOPK = 3


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
        choices=["bm25", "tfidf"],
        default="bm25",
        help="Retriever to use (default: bm25)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_TOPK,
        help=f"Number of top papers to return (default: {DEFAULT_TOPK})",
    )
    parser.add_argument(
        "--limit",
        type=int,
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
    if model_name == "bm25":
        return BM25RAG()
    return TfidfRAG()


def _format_abstract(abstract: str, max_length: int = 150) -> str:
    """Truncate abstract to max_length with ellipsis."""
    if not abstract:
        return "[No abstract]"
    abstract = str(abstract).strip()
    if len(abstract) > max_length:
        return abstract[:max_length] + "..."
    return abstract


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
        title = str(row["title"] or "")
        abstract = _format_abstract(row["abstract"])

        print(f"\n{rank}. arXiv:{arxiv_id}")
        print(f"   Title: {title}")
        print(f"   Abstract: {abstract}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
