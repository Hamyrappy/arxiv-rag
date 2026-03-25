# arxiv-rag

Небольшой исследовательский проект для сравнения retrieval-моделей на метаданных arXiv.

Проект покрывает три практические задачи:
- подготовка корпуса из `arxiv-metadata-oai-snapshot.json` в `part_*.parquet`;
- быстрый ручной поиск по корпусу через CLI или веб-демо;
- оценка retriever-моделей на TSV benchmark.

## Что внутри

- Подготовка данных: `arxiv-rag-prepare-data`.
- Ручной поиск: `arxiv-rag-quick`.
- Базовый baseline-запуск: `arxiv-rag-run-baseline`.
- Оценка на benchmark: `arxiv-rag-evaluate`.
- Метрики: `Recall@k`, `MRR`, `nDCG@k`.
- Поддержка кастомных retriever-ов через интерфейс `fit(texts)` и `topk(query, k)`.

## Требования

- Python 3.10+
- uv
- Kaggle credentials только если нужен автоматический download

Варианты авторизации Kaggle:
1. `~/.kaggle/kaggle.json` (Linux/macOS)
2. `%USERPROFILE%\\.kaggle\\kaggle.json` (Windows)
3. переменные окружения `KAGGLE_USERNAME` и `KAGGLE_KEY`

## Установка

```bash
uv sync --locked
```

Если менялись зависимости в `pyproject.toml`:

```bash
uv lock
uv sync
```

## Quick Start

```bash
# 1) Подготовить данные
uv run arxiv-rag-prepare-data

# 2) Быстрый поиск
uv run arxiv-rag-quick "neural networks"

# 3) Проверить benchmark перед eval
uv run arxiv-rag-evaluate --benchmark eval/benchmark.tsv --validate-only

# 4) Запустить оценку
uv run arxiv-rag-evaluate --model all --benchmark eval/benchmark.tsv --k 20
```

## Структура проекта

```text
arxiv_rag/
  dataset/              # подготовка/загрузка данных
  models/               # retriever-модели
  evaluation/           # evaluator и метрики
  baseline_cli.py       # arxiv-rag-run-baseline
  quick_query_cli.py    # arxiv-rag-quick
evaluate_models.py      # arxiv-rag-evaluate
app.py                  # Flask demo
eval/                   # benchmark-файлы и формат
data/raw/               # исходный JSON
data/processed/         # part_*.parquet
```

## Основные команды

### Подготовка данных

```bash
uv run arxiv-rag-prepare-data
```

Полезные флаги:
- `--skip-download --input-json PATH` если JSON уже скачан вручную
- `--force-download` принудительный перекач из Kaggle
- `--force-process` пересборка `part_*.parquet`
- `--chunksize 50000` меньшие чанки для слабой машины
- `--raw-dir` и `--processed-dir` для кастомных путей

### Быстрый поиск по одному запросу

```bash
uv run arxiv-rag-quick "graph neural networks" --k 10 --model bm25
```

Дефолты `arxiv-rag-quick`:
- модель: `bm25`
- `k`: `3`
- `limit`: `50000`

### Baseline CLI (только bm25/tfidf)

```bash
uv run arxiv-rag-run-baseline --model tfidf --query "graph transformers"
```

Дефолты `arxiv-rag-run-baseline`:
- модель: `bm25`
- `k`: `5`
- `limit`: `2000`

### Evaluation CLI

```bash
# Только проверка benchmark
uv run arxiv-rag-evaluate --benchmark eval/benchmark.tsv --validate-only

# Полная оценка (все встроенные модели)
uv run arxiv-rag-evaluate --model all --benchmark eval/benchmark.tsv --k 20

# Подробный per-query вывод
uv run arxiv-rag-evaluate --model bm25 --benchmark eval/benchmark.tsv --k 20 --show-per-query
```

Дефолты `arxiv-rag-evaluate`:
- модель: `all`
- `k`: `10`
- `limit`: полный корпус (без ограничения)
- вывод: summary-only

## Модели и где они доступны

| Модель | quick | run-baseline | evaluate |
|---|---:|---:|---:|
| bm25 | yes | yes | yes |
| tfidf | yes | yes | yes |
| specter1 | yes | no | yes |
| specter2 | yes | no | yes |
| bge | yes | no | yes |
| minilm | yes | no | yes |
| hybrid-rrf | yes | no | yes |
| hybrid-rrf-specter | yes | no | yes |
| hybrid-weighted | yes | no | yes |
| hybrid-weighted-specter | yes | no | yes |
| cross-encoder | yes | no | yes |
| paletsv-nebo | yes | no | yes |
| yandex-llm | yes | no | yes |
| random | yes | no | no |
| custom `module:factory_or_class` | no | no | yes |

Примечания:
- `random` в quick CLI является алиасом `paletsv-nebo`.
- В evaluate `--model all` включает 12 встроенных моделей, включая `paletsv-nebo`.
- `yandex-llm` требует переменных окружения `YANDEX_API_KEY` и `YANDEX_FOLDER_ID` (или явной передачи параметров).
- В `--model all` модель `yandex-llm` **не** включена автоматически из-за требования внешнего API.

## Benchmark и формат

Основной файл:
- `eval/benchmark.tsv`

Также в `eval/` лежат дополнительные наборы (например `benchmark_smoke.tsv`, `semantic_bench.tsv`) для отдельных сценариев.

Формат benchmark:
- обязательные колонки: `query`, `relevant_ids`
- `relevant_ids` должен быть JSON-списком, например:

```text
query	relevant_ids
How does retrieval-augmented generation work?	["2005.11401", "2401.12345"]
```

Поддерживаются:
- новые и старые arXiv id;
- префикс `arXiv:` (нормализуется автоматически).

Детали формата: `eval/BENCHMARK_FORMAT.md`.

## Важные особенности интерпретации

- `arxiv-rag-run-baseline` и `arxiv-rag-quick` индексируют только `abstract`.
- `arxiv-rag-evaluate` индексирует `title + [SEP] + abstract`.
- Из-за этого метрики eval и ручной baseline-поиск не обязаны совпадать один-в-один.

Evaluator выводит:
- `evaluated` метрики: только по запросам, где после пересечения с корпусом есть релевантные id;
- `penalized` метрики: те же значения с учетом доли пропущенных запросов.

Это особенно важно при запуске с `--limit`.

## Ограничения

- Eval не потоковый: корпус целиком загружается в память.
- Sparse и dense индексы строятся в RAM.
- Dense/Hybrid/Cross-Encoder могут требовать заметно больше RAM/времени.
- На слабой машине лучше:
  - сначала `--validate-only`;
  - уменьшать `--limit`;
  - запускать по одной модели за раз.

Пример для ограниченной RAM:

```bash
uv run arxiv-rag-evaluate --benchmark eval/benchmark.tsv --validate-only
uv run arxiv-rag-evaluate --model bm25 --benchmark eval/benchmark.tsv --limit 30000 --k 20
uv run arxiv-rag-evaluate --model tfidf --benchmark eval/benchmark.tsv --limit 30000 --k 20
```

## Кастомный retriever для eval

Минимальный контракт:
- `fit(texts)`
- `topk(query, k) -> list[int]`

Важно:
- `topk` возвращает индексы документов (не arXiv id);
- индексы должны соответствовать порядку документов из `fit(texts)`.

Пример запуска:

```bash
uv run arxiv-rag-evaluate --model my_retriever:MyRetriever --benchmark eval/benchmark.tsv --k 20
```

## Yandex LLM Re-ranker

`YandexLLMReranker` — двухэтапный re-ranker на базе YandexGPT:

1. **First stage** — быстрый base retriever (по умолчанию BM25) возвращает `top_n` кандидатов.
2. **Second stage** — каждый кандидат асинхронно оценивается через Yandex Foundation Models API;
   кандидаты переупорядочиваются по полученным оценкам релевантности.

### Конфигурация

Обязательные переменные окружения:

| Переменная | Описание |
|---|---|
| `YANDEX_API_KEY` | API-ключ Yandex Cloud |
| `YANDEX_FOLDER_ID` | ID каталога Yandex Cloud (используется для построения model URI) |

Запуск evaluate:

```bash
export YANDEX_API_KEY=your_api_key
export YANDEX_FOLDER_ID=your_folder_id
uv run arxiv-rag-evaluate --model yandex-llm --benchmark eval/benchmark.tsv --k 10
```

Параметры `YandexLLMReranker` (передаются при кастомном использовании):

| Параметр | По умолчанию | Описание |
|---|---|---|
| `base_retriever` | — | Любой retriever с `fit`/`topk` интерфейсом |
| `api_key` | `$YANDEX_API_KEY` | API-ключ |
| `folder_id` | `$YANDEX_FOLDER_ID` | ID каталога |
| `model_uri` | `gpt://<folder_id>/yandexgpt-lite` | Полный URI модели |
| `top_n` | `20` | Кандидаты от base retriever |
| `max_workers` | `5` | Параллельных запросов к API |
| `temperature` | `0.0` | Температура генерации (0 = детерминированный вывод) |

## Веб-демо (Flask)

Запуск:

```bash
uv run python app.py
```

По умолчанию адрес: `http://127.0.0.1:5000`.

В демо можно:
- выбрать модель поиска;
- задать `k`;
- ограничить размер корпуса или загрузить весь корпус;
- выполнить интерактивный поиск по произвольному запросу.

Если данных нет, сначала запустите `arxiv-rag-prepare-data`.

## Troubleshooting

- Ошибка Kaggle auth: проверьте `kaggle.json` или переменные `KAGGLE_USERNAME`/`KAGGLE_KEY`.
- Данные не найдены: выполните `uv run arxiv-rag-prepare-data` или укажите путь через `--data-folder`/`--processed-dir`.
- Benchmark не валидируется: запустите `--validate-only`, исправьте строку из сообщения ошибки.
- Не хватает памяти: уменьшайте `--limit`, отключайте `all`, гоняйте модели по одной.

## Полезные файлы

- `evaluate_models.py` - entrypoint eval
- `arxiv_rag/baseline_cli.py` - baseline CLI
- `arxiv_rag/quick_query_cli.py` - quick CLI
- `arxiv_rag/evaluation/evaluator.py` - логика метрик
- `eval/BENCHMARK_FORMAT.md` - спецификация benchmark
