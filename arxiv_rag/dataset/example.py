"""Example script for preparing and loading arXiv data without manual path edits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from arxiv_rag.dataset import load_arxiv_data
from arxiv_rag.dataset.prepare_data import prepare_data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare arXiv dataset and show sample loads."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory with raw Kaggle files.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory with processed parquet files.",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Optional local path to arxiv-metadata-oai-snapshot.json.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not call Kaggle API; use local files only.",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip dataset preparation and only run load examples.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if not args.skip_prepare:
        prepare_data(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            input_json=args.input_json,
            skip_download=args.skip_download,
        )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", 50)

    df = load_arxiv_data(
        data_folder=str(args.processed_dir),
        limit=1000,
    )
    print("\nFirst load (limit=1000):")
    print(df.head(10))
    print(df.shape)

    df = load_arxiv_data(
        data_folder=str(args.processed_dir),
        categories=["cs.CL"],
        limit=1000,
        columns=["id", "title"],
    )
    print("\nCategory-filtered load (cs.CL, limit=1000):")
    print(df.head(10))
    print(df.shape)

    df = load_arxiv_data(
        data_folder=str(args.processed_dir),
        limit=5000,
        shuffle=True,
        random_state=42,
    )
    print("\nShuffled load (limit=5000):")
    print(df.head(10))
    print(df.shape)


if __name__ == "__main__":
    main()
