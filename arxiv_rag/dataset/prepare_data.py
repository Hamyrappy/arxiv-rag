"""CLI for downloading and preparing the arXiv dataset."""

from __future__ import annotations

import argparse
import importlib
import shutil
from pathlib import Path

from arxiv_rag.dataset.dataloader import data_converter

DEFAULT_DATASET = "Cornell-University/arxiv"
DEFAULT_METADATA_FILENAME = "arxiv-metadata-oai-snapshot.json"


def _download_from_kaggle(dataset: str, raw_dir: Path, force_download: bool) -> None:
    """Download dataset files from Kaggle into raw_dir."""
    try:
        kaggle_module = importlib.import_module("kaggle.api.kaggle_api_extended")
        KaggleApi = getattr(kaggle_module, "KaggleApi")
    except (ModuleNotFoundError, AttributeError) as exc:
        raise RuntimeError(
            "Kaggle package is not installed. Run `pip install -e .` first."
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        raise RuntimeError(
            "Kaggle authentication failed. Configure ~/.kaggle/kaggle.json "
            "or set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
        ) from exc

    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Kaggle dataset '{dataset}' into {raw_dir}...")
    api.dataset_download_files(
        dataset=dataset,
        path=str(raw_dir),
        unzip=True,
        force=force_download,
        quiet=False,
    )


def _find_metadata_json(raw_dir: Path, metadata_filename: str) -> Path:
    """Locate the metadata json file under raw_dir."""
    by_name = sorted(raw_dir.rglob(metadata_filename))
    if by_name:
        return by_name[0]

    json_files = sorted(raw_dir.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No JSON files were found under {raw_dir}. "
            "Run with download enabled or provide --input-json."
        )
    if len(json_files) == 1:
        return json_files[0]

    candidates = "\n".join(str(path) for path in json_files[:5])
    raise FileNotFoundError(
        "Could not determine which JSON file to use. "
        "Please pass --input-json explicitly. Candidates:\n"
        f"{candidates}"
    )


def _clear_processed_outputs(processed_dir: Path) -> None:
    """Remove previously generated parquet parts in the output directory."""
    if not processed_dir.exists():
        return

    for part_file in processed_dir.glob("part_*.parquet"):
        part_file.unlink()

    temp_dir = processed_dir / "temp_chunks"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def prepare_data(
    raw_dir: Path,
    processed_dir: Path,
    dataset: str = DEFAULT_DATASET,
    metadata_filename: str = DEFAULT_METADATA_FILENAME,
    input_json: Path | None = None,
    skip_download: bool = False,
    force_download: bool = False,
    force_process: bool = False,
    chunksize: int = 100_000,
) -> tuple[Path, Path]:
    """Download (optional) and convert arXiv metadata into parquet parts."""
    raw_dir = raw_dir.resolve()
    processed_dir = processed_dir.resolve()

    if input_json is not None:
        metadata_path = input_json.resolve()
        if not metadata_path.exists():
            raise FileNotFoundError(f"Input JSON does not exist: {metadata_path}")
    else:
        if not skip_download:
            _download_from_kaggle(dataset=dataset, raw_dir=raw_dir, force_download=force_download)
        elif not raw_dir.exists():
            raise FileNotFoundError(
                f"Raw directory does not exist: {raw_dir}. "
                "Disable --skip-download or provide --input-json."
            )

        metadata_path = _find_metadata_json(raw_dir=raw_dir, metadata_filename=metadata_filename)

    processed_dir.mkdir(parents=True, exist_ok=True)

    existing_parts = sorted(processed_dir.glob("part_*.parquet"))
    if existing_parts and not force_process:
        print(
            f"Found {len(existing_parts)} existing parquet parts in {processed_dir}. "
            "Skipping conversion. Use --force-process to rebuild."
        )
        return metadata_path, processed_dir

    if force_process:
        _clear_processed_outputs(processed_dir)

    print(f"Converting {metadata_path} to parquet parts in {processed_dir}...")
    data_converter(
        input_path=str(metadata_path),
        output_path=str(processed_dir),
        chunksize=chunksize,
    )

    return metadata_path, processed_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download arXiv metadata from Kaggle and prepare parquet files."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Kaggle dataset slug (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for downloaded raw files.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed parquet parts.",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Path to local arxiv metadata JSON file. Skips discovery in raw-dir.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not call Kaggle API; use existing local files only.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download from Kaggle even if local files exist.",
    )
    parser.add_argument(
        "--force-process",
        action="store_true",
        help="Rebuild parquet files even if processed output already exists.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Chunk size used while reading the JSON metadata file.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        metadata_path, processed_dir = prepare_data(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            dataset=args.dataset,
            input_json=args.input_json,
            skip_download=args.skip_download,
            force_download=args.force_download,
            force_process=args.force_process,
            chunksize=args.chunksize,
        )
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(f"Metadata file: {metadata_path}")
    print(f"Processed data dir: {processed_dir}")


if __name__ == "__main__":
    main()
