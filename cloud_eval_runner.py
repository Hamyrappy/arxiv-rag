from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from arxiv_rag.evaluation import Evaluator
from evaluate_models import (
    DEFAULT_DATA_FOLDER,
    _build_retriever,
    load_benchmark,
    load_corpus,
    reconcile_benchmark_with_corpus,
)


DEFAULT_OUTPUT_DIR = Path("outputs/cloud_eval")
DEFAULT_BENCHMARK_GLOB = "*.tsv"
DEFAULT_K = 10
DEFAULT_SUMMARY_SORT = "mrr"
ALL_MODEL_KEYS = [
    "tfidf",
    "bm25",
    "minilm",
    "specter1",
    "specter2",
    "bge",
    "hybrid-rrf",
    "hybrid-rrf-specter",
    "hybrid-weighted",
    "hybrid-weighted-specter",
    "cross-encoder",
    "paletsv-nebo",
]


@dataclass
class AggregateResult:
    benchmark_name: str
    benchmark_path: str
    model_key: str
    model_label: str
    k: int
    corpus_size: int
    queries_total: int
    queries_used: int
    queries_skipped: int
    total_relevant_ids: int
    missing_relevant_ids: int
    relevant_id_coverage: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    penalized_recall_at_k: float
    penalized_mrr: float
    penalized_ndcg_at_k: float
    index_time_sec: float
    query_time_sec: float
    avg_latency_ms: float
    device: str
    cache_dir: str | None
    timestamp_utc: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full structured evaluation for Colab/Kaggle and export JSON/CSV/LaTeX."
    )
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=DEFAULT_DATA_FOLDER,
        help=f"Folder with processed part_*.parquet files (default: {DEFAULT_DATA_FOLDER}).",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("eval"),
        help="Directory with TSV benchmark files.",
    )
    parser.add_argument(
        "--benchmark-glob",
        default=DEFAULT_BENCHMARK_GLOB,
        help=f"Glob pattern for benchmark discovery (default: {DEFAULT_BENCHMARK_GLOB}).",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="Optional explicit benchmark file names to evaluate, relative to --benchmark-dir.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model keys to evaluate. Defaults to all built-in models.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional corpus size limit. Default: full corpus.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Cut-off for Recall@k and nDCG@k (default: {DEFAULT_K}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated artifacts (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--no-per-query",
        action="store_true",
        help="Skip per-query JSONL/CSV export to save space.",
    )
    parser.add_argument(
        "--summary-sort",
        default=DEFAULT_SUMMARY_SORT,
        choices=["mrr", "recall", "ndcg", "penalized_mrr", "penalized_recall", "penalized_ndcg"],
        help="Metric used for leaderboard sorting.",
    )
    return parser


def _discover_benchmark_paths(benchmark_dir: Path, benchmark_glob: str, names: list[str] | None) -> list[Path]:
    if names:
        paths = [benchmark_dir / name for name in names]
    else:
        paths = sorted(path for path in benchmark_dir.glob(benchmark_glob) if path.is_file())

    if not paths:
        raise FileNotFoundError(f"No benchmark files found in {benchmark_dir} with pattern {benchmark_glob!r}")

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing benchmark files: " + ", ".join(missing))
    return paths


def _resolve_model_keys(model_keys: list[str] | None) -> list[str]:
    if not model_keys:
        return list(ALL_MODEL_KEYS)
    return model_keys


def _metric_column(summary_sort: str, k: int) -> str:
    mapping = {
        "mrr": "mrr",
        "recall": f"recall@{k}",
        "ndcg": f"ndcg@{k}",
        "penalized_mrr": "penalized_mrr",
        "penalized_recall": f"penalized_recall@{k}",
        "penalized_ndcg": f"penalized_ndcg@{k}",
    }
    return mapping[summary_sort]


def _device_of(retriever: Any) -> str:
    if hasattr(retriever, "device"):
        return str(retriever.device)
    if hasattr(retriever, "base_retriever") and hasattr(retriever.base_retriever, "dense"):
        dense = retriever.base_retriever.dense
        if hasattr(dense, "device"):
            return str(dense.device)
    if hasattr(retriever, "dense") and hasattr(retriever.dense, "device"):
        return str(retriever.dense.device)
    return "cpu"


def _cache_dir_of(retriever: Any) -> str | None:
    if hasattr(retriever, "cache_dir"):
        return str(retriever.cache_dir)
    if hasattr(retriever, "base_retriever") and hasattr(retriever.base_retriever, "dense"):
        dense = retriever.base_retriever.dense
        if hasattr(dense, "cache_dir"):
            return str(dense.cache_dir)
    if hasattr(retriever, "dense") and hasattr(retriever.dense, "cache_dir"):
        return str(retriever.dense.cache_dir)
    return None


def _cleanup_after_model() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _evaluate_single_model(
    model_key: str,
    model_label: str,
    retriever: Any,
    doc_ids: list[str],
    texts: list[str],
    benchmark_payloads: list[tuple[Path, list[dict], dict[str, Any]]],
    k: int,
    include_per_query: bool,
) -> tuple[list[AggregateResult], list[dict[str, Any]]]:
    print(f"\n=== Building index for {model_label} ({model_key}) ===")
    t0 = perf_counter()
    evaluator = Evaluator(retriever=retriever, doc_ids=doc_ids, texts=texts)
    index_time_sec = perf_counter() - t0

    aggregates: list[AggregateResult] = []
    per_query_rows: list[dict[str, Any]] = []
    device = _device_of(retriever)
    cache_dir = _cache_dir_of(retriever)
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    for benchmark_path, benchmark, benchmark_summary in benchmark_payloads:
        print(f"Evaluating {model_label} on {benchmark_path.name}...")
        t0 = perf_counter()
        results = evaluator.evaluate(
            benchmark,
            k=k,
            show_progress=True,
            progress_desc=f"{model_key}:{benchmark_path.stem}",
        )
        query_time_sec = perf_counter() - t0

        total_queries = benchmark_summary["total_queries"]
        evaluated_queries = benchmark_summary["evaluated_queries"]
        query_factor = evaluated_queries / total_queries if total_queries else 0.0
        recall_at_k = float(results[f"recall@{k}"])
        mrr = float(results["mrr"])
        ndcg_at_k = float(results[f"ndcg@{k}"])
        penalized_recall_at_k = recall_at_k * query_factor
        penalized_mrr = mrr * query_factor
        penalized_ndcg_at_k = ndcg_at_k * query_factor
        avg_latency_ms = (query_time_sec / evaluated_queries * 1000) if evaluated_queries else 0.0

        aggregates.append(
            AggregateResult(
                benchmark_name=benchmark_path.stem,
                benchmark_path=str(benchmark_path),
                model_key=model_key,
                model_label=model_label,
                k=k,
                corpus_size=len(doc_ids),
                queries_total=total_queries,
                queries_used=evaluated_queries,
                queries_skipped=benchmark_summary["skipped_queries"],
                total_relevant_ids=benchmark_summary["total_relevant_ids"],
                missing_relevant_ids=benchmark_summary["missing_relevant_ids"],
                relevant_id_coverage=float(benchmark_summary["relevant_id_coverage"]),
                recall_at_k=recall_at_k,
                mrr=mrr,
                ndcg_at_k=ndcg_at_k,
                penalized_recall_at_k=penalized_recall_at_k,
                penalized_mrr=penalized_mrr,
                penalized_ndcg_at_k=penalized_ndcg_at_k,
                index_time_sec=index_time_sec,
                query_time_sec=query_time_sec,
                avg_latency_ms=avg_latency_ms,
                device=device,
                cache_dir=cache_dir,
                timestamp_utc=timestamp_utc,
            )
        )

        if include_per_query:
            for item in results["per_query"]:
                per_query_rows.append(
                    {
                        "benchmark_name": benchmark_path.stem,
                        "benchmark_path": str(benchmark_path),
                        "model_key": model_key,
                        "model_label": model_label,
                        "k": k,
                        "query": item["query"],
                        f"recall@{k}": float(item[f"recall@{k}"]),
                        "rr": float(item["rr"]),
                        f"ndcg@{k}": float(item[f"ndcg@{k}"]),
                        "retrieved_ids": item["retrieved_ids"],
                    }
                )

    return aggregates, per_query_rows


def _prepare_benchmarks(
    benchmark_paths: list[Path],
    doc_ids: list[str],
) -> list[tuple[Path, list[dict], dict[str, Any]]]:
    payloads: list[tuple[Path, list[dict], dict[str, Any]]] = []
    available_doc_ids = set(doc_ids)
    for benchmark_path in benchmark_paths:
        benchmark = load_benchmark(benchmark_path)
        filtered_benchmark, benchmark_summary = reconcile_benchmark_with_corpus(
            benchmark=benchmark,
            available_doc_ids=available_doc_ids,
        )
        if benchmark_summary["evaluated_queries"] == 0:
            print(f"Skipping {benchmark_path.name}: no evaluable queries after corpus intersection.")
            continue
        payloads.append((benchmark_path, filtered_benchmark, benchmark_summary))

    if not payloads:
        raise ValueError("No benchmark has evaluable queries with the loaded corpus.")
    return payloads


def _summary_dataframe(aggregate_rows: list[AggregateResult], k: int) -> pd.DataFrame:
    rows = []
    for item in aggregate_rows:
        row = asdict(item)
        row[f"recall@{k}"] = row.pop("recall_at_k")
        row[f"ndcg@{k}"] = row.pop("ndcg_at_k")
        row[f"penalized_recall@{k}"] = row.pop("penalized_recall_at_k")
        row[f"penalized_ndcg@{k}"] = row.pop("penalized_ndcg_at_k")
        rows.append(row)
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _generate_latex_tables(summary_df: pd.DataFrame, metric_column: str, output_path: Path) -> None:
    blocks: list[str] = []
    row_break = r"\\"
    for benchmark_name, group in summary_df.groupby("benchmark_name"):
        sorted_group = group.sort_values(metric_column, ascending=False)
        blocks.append(f"% Benchmark: {benchmark_name}")
        blocks.append(r"\begin{tabular}{lrrr}")
        blocks.append(f"Model & MRR & Recall & nDCG {row_break}")
        blocks.append(r"\hline")
        for _, row in sorted_group.iterrows():
            blocks.append(
                "{} & {:.4f} & {:.4f} & {:.4f} {}".format(
                    _latex_escape(row["model_label"]),
                    row["mrr"],
                    row[f"recall@{int(row['k'])}"],
                    row[f"ndcg@{int(row['k'])}"],
                    row_break,
                )
            )
        blocks.append(r"\end{tabular}")
        blocks.append("")

    pivot = summary_df.pivot(index="model_label", columns="benchmark_name", values=metric_column)
    pivot = pivot.sort_index()
    col_spec = "l" + "r" * len(pivot.columns)
    blocks.append("% Consolidated leaderboard")
    blocks.append(rf"\begin{{tabular}}{{{col_spec}}}")
    header = "Model & " + " & ".join(_latex_escape(col) for col in pivot.columns) + f" {row_break}"
    blocks.append(header)
    blocks.append(r"\hline")
    for model_label, row in pivot.iterrows():
        values = " & ".join("-" if pd.isna(val) else f"{val:.4f}" for val in row.tolist())
        blocks.append(f"{_latex_escape(model_label)} & {values} {row_break}")
    blocks.append(r"\end{tabular}")
    output_path.write_text("\n".join(blocks) + "\n", encoding="utf-8")


def run_cloud_evaluation(
    data_folder: Path = DEFAULT_DATA_FOLDER,
    benchmark_dir: Path = Path("eval"),
    benchmark_glob: str = DEFAULT_BENCHMARK_GLOB,
    benchmarks: list[str] | None = None,
    models: list[str] | None = None,
    limit: int | None = None,
    k: int = DEFAULT_K,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    include_per_query: bool = True,
    summary_sort: str = DEFAULT_SUMMARY_SORT,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_paths = _discover_benchmark_paths(benchmark_dir, benchmark_glob, benchmarks)
    model_keys = _resolve_model_keys(models)

    benchmark_ids: set[str] = set()
    for benchmark_path in benchmark_paths:
        benchmark = load_benchmark(benchmark_path)
        for item in benchmark:
            benchmark_ids.update(str(doc_id) for doc_id in item["relevant_ids"])

    doc_ids, texts = load_corpus(data_folder, limit, must_include_ids=benchmark_ids)
    benchmark_payloads = _prepare_benchmarks(benchmark_paths, doc_ids)

    aggregate_rows: list[AggregateResult] = []
    per_query_rows: list[dict[str, Any]] = []

    for model_key in model_keys:
        model_label, retriever_factory = _build_retriever(model_key)
        retriever = None
        try:
            retriever = retriever_factory()
            model_aggregates, model_per_query = _evaluate_single_model(
                model_key=model_key,
                model_label=model_label,
                retriever=retriever,
                doc_ids=doc_ids,
                texts=texts,
                benchmark_payloads=benchmark_payloads,
                k=k,
                include_per_query=include_per_query,
            )
            aggregate_rows.extend(model_aggregates)
            per_query_rows.extend(model_per_query)
        finally:
            if retriever is not None:
                del retriever
            _cleanup_after_model()

    summary_df = _summary_dataframe(aggregate_rows, k)
    sort_column = _metric_column(summary_sort, k)
    summary_df = summary_df.sort_values(["benchmark_name", sort_column], ascending=[True, False])

    summary_csv_path = output_dir / "summary.csv"
    summary_json_path = output_dir / "summary.json"
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8")
    _write_json(summary_json_path, summary_df.to_dict(orient="records"))

    per_query_csv_path = output_dir / "per_query.csv"
    per_query_jsonl_path = output_dir / "per_query.jsonl"
    if include_per_query:
        per_query_df = pd.DataFrame(per_query_rows)
        per_query_df.to_csv(per_query_csv_path, index=False, encoding="utf-8")
        _write_jsonl(per_query_jsonl_path, per_query_rows)

    latex_path = output_dir / "leaderboards.tex"
    _generate_latex_tables(summary_df, sort_column, latex_path)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_folder": str(data_folder),
        "benchmark_dir": str(benchmark_dir),
        "benchmarks": [str(path) for path in benchmark_paths],
        "models": model_keys,
        "limit": limit,
        "k": k,
        "corpus_size": len(doc_ids),
        "artifacts": {
            "summary_csv": str(summary_csv_path),
            "summary_json": str(summary_json_path),
            "latex": str(latex_path),
            "per_query_csv": str(per_query_csv_path) if include_per_query else None,
            "per_query_jsonl": str(per_query_jsonl_path) if include_per_query else None,
        },
    }
    _write_json(output_dir / "manifest.json", manifest)

    print("\nStructured evaluation finished.")
    print(f"Corpus size: {len(doc_ids)}")
    print(f"Benchmarks: {', '.join(path.name for path in benchmark_paths)}")
    print(f"Models: {', '.join(model_keys)}")
    print(f"Artifacts written to: {output_dir}")

    return {
        "summary": summary_df,
        "per_query": pd.DataFrame(per_query_rows) if include_per_query else None,
        "manifest": manifest,
        "latex_path": latex_path,
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run_cloud_evaluation(
        data_folder=args.data_folder,
        benchmark_dir=args.benchmark_dir,
        benchmark_glob=args.benchmark_glob,
        benchmarks=args.benchmarks,
        models=args.models,
        limit=args.limit,
        k=args.k,
        output_dir=args.output_dir,
        include_per_query=not args.no_per_query,
        summary_sort=args.summary_sort,
    )


if __name__ == "__main__":
    main()