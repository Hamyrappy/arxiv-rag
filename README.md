# arxiv-rag

Small research project for retrieval baselines on arXiv metadata.

## What is included

- Dataset conversion pipeline for the Kaggle arXiv metadata dump.
- Baseline retrievers: BM25 and TF-IDF.
- Lightweight evaluator with Recall@k, MRR and nDCG@k.

## Project layout

```text
arxiv_rag/
	dataset/
		dataloader.py        # conversion + parquet loading
		prepare_data.py      # automated data preparation CLI
	models/
		baseline.py          # BM25 and TF-IDF retrievers
	evaluation/
		evaluator.py         # retrieval metrics
run_simple_baseline.py   # backward-compatible baseline script
evaluate_models.py       # mini-benchmark evaluation
```

## Prerequisites

- Python 3.8+
- Kaggle API credentials (required only for automatic download)

Kaggle authentication options:

1. Put kaggle.json at ~/.kaggle/kaggle.json (Linux/macOS) or %USERPROFILE%\\.kaggle\\kaggle.json (Windows)
2. Or set environment variables KAGGLE_USERNAME and KAGGLE_KEY

## Quickstart

1. Install the project in editable mode:

```bash
pip install -e .
```

2. Download and prepare data (creates data/raw and data/processed automatically):

```bash
arxiv-rag-prepare-data
```

This step can take a while on first run because conversion is chunk-based.

Manual alternative (without Kaggle API credentials):

- Download the dataset archive from Kaggle in browser: https://www.kaggle.com/datasets/Cornell-University/arxiv/data
- Unpack files into data/raw (or any local folder)
- Run preparation without download:

```bash
arxiv-rag-prepare-data --skip-download
```

If the metadata JSON is outside data/raw, pass it explicitly:

```bash
arxiv-rag-prepare-data --skip-download --input-json "path/to/arxiv-metadata-oai-snapshot.json"
```

3. Run the baseline retriever:

```bash
arxiv-rag-run-baseline
```

## Main commands

Prepare data with defaults:

```bash
arxiv-rag-prepare-data
```

Rebuild data from scratch:

```bash
arxiv-rag-prepare-data --force-download --force-process
```

Use local metadata file without download:

```bash
arxiv-rag-prepare-data --skip-download --input-json "path/to/arxiv-metadata-oai-snapshot.json"
```

Manual download + default folder layout:

```text
data/
	raw/
		arxiv-metadata-oai-snapshot.json
	processed/
		part_0000.parquet
		...
```

Then run:

```bash
arxiv-rag-prepare-data --skip-download
```

Run BM25 baseline (default settings):

```bash
arxiv-rag-run-baseline
```

Run TF-IDF with custom limit and query:

```bash
arxiv-rag-run-baseline --model tfidf --limit 5000 --query "graph transformers"
```

Run mini-benchmark evaluation:

```bash
python evaluate_models.py
```

## Data directories

- data/raw: downloaded Kaggle files
- data/processed: generated part_*.parquet files used by loaders and baselines

These directories are ignored by git.

## Backward-compatible script entrypoints

The old script still works:

```bash
python run_simple_baseline.py
```

It now delegates to the same CLI logic as arxiv-rag-run-baseline.

## Troubleshooting

- If download fails: check Kaggle credentials and dataset access.
- If baseline cannot find data: run arxiv-rag-prepare-data first or pass --data-folder.
- If you already have parquet parts and want to keep them: run prepare command without --force-process.
