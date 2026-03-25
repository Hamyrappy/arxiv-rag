# arxiv-rag

Небольшой исследовательский проект для сравнения retrieval baseline-моделей на метаданных arXiv.

Сейчас с помощью кода можно решить три практические задачи:

- подготовка сырых данных arXiv из Kaggle в удобный Parquet-формат;
- запуск простых baseline retriever'ов для ручной проверки поиска;
- оценка retriever'ов на собственном benchmark-файле в формате TSV.

## Что входит в проект

- Pipeline подготовки датасета из `arxiv-metadata-oai-snapshot.json` в `part_*.parquet`.
- Два baseline retriever'а: `BM25RAG` и `TfidfRAG`.
- Общий evaluator с метриками `Recall@k`, `MRR` и `nDCG@k`.
- CLI для подготовки данных, baseline-запуска и eval.
- Поддержка пользовательских retriever'ов через общий интерфейс `fit(texts)` и `topk(query, k)`.
- Веб-интерфейс для интерактивного поиска (демо‑режим).

## Статус проекта

Для условного рабочего релиза под твои цели eval-сценарий сейчас выглядит достаточно зрелым:

- есть отдельный benchmark в `eval/`;
- есть валидация benchmark без загрузки корпуса;
- поддерживаются `bm25`, `tfidf`, `all` и кастомный retriever через `module:factory_or_class`;
- есть summary-only режим по умолчанию и подробный `--show-per-query`;
- есть корректная обработка частичного корпуса через `--limit`;
- при неполном покрытии benchmark evaluator не падает, а показывает coverage и penalized metrics.

При этом есть и текущие ограничения:

- весьма низкое качество бенчмарка
- eval не потоковый: корпус целиком загружается в память;
- BM25 и TF-IDF тоже строят индекс в RAM;
- встроенных автотестов для eval-пайплайна пока нет;
- проект ориентирован на retrieval baseline, а не на полноценный RAG с генерацией ответов.

Если задача именно исследовательская и локальная, а не production deployment, это состояние можно считать рабочим и достаточно чистым.

## Структура проекта

```text
arxiv_rag/
	dataset/
		dataloader.py        # конвертация JSON и загрузка parquet-частей
		prepare_data.py      # CLI подготовки данных
	models/
		baseline.py          # BM25RAG и TfidfRAG
		__init__.py          # реэкспорт baseline retriever'ов
	evaluation/
		evaluator.py         # общая логика расчета retrieval-метрик
evaluate_models.py       # CLI оценки retriever'ов на benchmark
app.py                   # веб-демо
eval/
	benchmark.tsv          # основной benchmark
	benchmark_fast.tsv     # быстрый smoke benchmark
	BENCHMARK_FORMAT.md    # правила формата benchmark-файла
data/
	raw/                   # сырые файлы из Kaggle
	processed/             # подготовленные part_*.parquet
```

## Требования

- Python 3.8+
- uv
- Kaggle API credentials, только если нужен автоматический download

Варианты авторизации Kaggle:

1. Положить `kaggle.json` в `~/.kaggle/kaggle.json` на Linux/macOS.
2. Положить `kaggle.json` в `%USERPROFILE%\\.kaggle\\kaggle.json` на Windows.
3. Или задать переменные окружения `KAGGLE_USERNAME` и `KAGGLE_KEY`.

## Установка

Синхронизировать окружение и установить проект через `uv`:

```bash
uv sync --locked
```

Если позже изменишь зависимости в `pyproject.toml`, обновляй lock-файл командой:

```bash
uv lock
```

После `uv sync` основные CLI запускаются через `uv run`:

- `uv run arxiv-rag-prepare-data`
- `uv run arxiv-rag-run-baseline`
- `uv run arxiv-rag-evaluate`

## Общий интерфейс проекта

Проект устроен вокруг одного и того же корпуса документов и двух уровней работы:

1. Подготовить локальный корпус `data/processed/part_*.parquet`.
2. Либо запускать retriever вручную на отдельных запросах, либо оценивать retriever на benchmark.

Типовой workflow:

1. Установить зависимости.
2. Подготовить данные.
3. Проверить baseline на отдельных запросах.
4. Создать или отредактировать benchmark в `eval/benchmark.tsv`.
5. Сначала прогнать `--validate-only`.
6. Затем запустить eval на `benchmark_fast.tsv` или на полном `benchmark.tsv`.
7. При необходимости подключить свой retriever и сравнить его с baseline.

## Подготовка данных

### Базовый сценарий

Автоматически скачать датасет из Kaggle и подготовить parquet-части:

```bash
uv run arxiv-rag-prepare-data
```

По умолчанию используются:

- raw directory: `data/raw`
- processed directory: `data/processed`
- dataset slug: `Cornell-University/arxiv`

### Если хочешь пересобрать всё с нуля

```bash
uv run arxiv-rag-prepare-data --force-download --force-process
```

### Если JSON уже скачан вручную

```bash
uv run arxiv-rag-prepare-data --skip-download --input-json "path/to/arxiv-metadata-oai-snapshot.json"
```

### Если хочешь использовать другую структуру каталогов

```bash
uv run arxiv-rag-prepare-data --raw-dir "path/to/raw" --processed-dir "path/to/processed"
```

### Если нужны более мелкие чанки на слабой машине

```bash
uv run arxiv-rag-prepare-data --chunksize 50000
```

### Что делает эта команда

- при необходимости скачивает файлы из Kaggle;
- находит `arxiv-metadata-oai-snapshot.json`;
- конвертирует JSON в набор `part_*.parquet`;
- повторно не пересобирает обработанные части без `--force-process`.

## Запуск baseline retriever'ов

### Быстрый поиск по одному запросу

Если нужно быстро найти топ-3 статьи по запросу, используй команду `arxiv-rag-quick`:

```bash
uv run arxiv-rag-quick "neural networks"
```

По умолчанию:

- retriever: `bm25`
- results: `3` (топ-3 статьи)
- limit: `50000` документов

#### Примеры использования quick query

```bash
# Базовый поиск (топ-3)
uv run arxiv-rag-quick "reinforcement learning"

# Получить больше результатов
uv run arxiv-rag-quick "graph neural networks" --k 10

# Использовать TF-IDF вместо BM25
uv run arxiv-rag-quick "transformers" --model tfidf

# Работать с ограниченной частью корпуса
uv run arxiv-rag-quick "attention mechanism" --limit 30000

# Комбинированные параметры
uv run arxiv-rag-quick "machine learning" --k 5 --limit 100000 --model tfidf
```

#### Доступные параметры quick query

- `query` (обязательный) - текст поискового запроса
- `--model {bm25,tfidf}` - retriever для поиска (по умолчанию: `bm25`)
- `--k K` - количество результатов (по умолчанию: `3`)
- `--limit LIMIT` - макс. документов для загрузки (по умолчанию: `50000`)
- `--data-folder FOLDER` - путь до обработанных данных

### Полный baseline с несколькими запросами

Для более подробной проверки baseline'ов используй `arxiv-rag-run-baseline`:

```bash
uv run arxiv-rag-run-baseline
```

По умолчанию это:

- retriever: `bm25`
- limit: `2000`
- top-k: `5`
- несколько встроенных demo-запросов

### Запуск TF-IDF с собственным запросом

```bash
uv run arxiv-rag-run-baseline --model tfidf --limit 5000 --query "graph transformers"
```

### Несколько запросов подряд

```bash
uv run arxiv-rag-run-baseline --model bm25 --query "retrieval augmented generation" --query "graph neural networks"
```

### Что важно знать

- baseline CLI использует только `abstract`, а не `title + abstract`;
- eval CLI использует `title + abstract`;
- это нормально, но полезно помнить при интерпретации результатов.

## Benchmark и eval

Основной eval выполняется командой:

```bash
uv run arxiv-rag-evaluate
```

Это entrypoint для [evaluate_models.py](evaluate_models.py).

### Где хранится benchmark

Benchmark-файлы лежат в каталоге `eval/`:

- `eval/benchmark.tsv` - основной benchmark;
- `eval/benchmark_fast.tsv` - быстрый smoke benchmark;
- `eval/BENCHMARK_FORMAT.md` - краткая спецификация формата.

### Формат benchmark-файла

Файл должен быть TSV в UTF-8. Обязательные колонки:

```text
query	relevant_ids
```

Дополнительные колонки допустимы. Например:

```text
query	relevant_ids	relevance_tier
```

Пример строки:

```text
How does retrieval-augmented generation work?	["2005.11401", "2401.12345"]	high
```

Требования к `relevant_ids`:

- это JSON-список, а не просто строка;
- использовать нужно реальные значения из колонки `id` подготовленного датасета;
- поддерживаются и новые, и старые arXiv id;
- префикс `arXiv:` допускается и автоматически нормализуется.

### Самый важный безопасный шаг: сначала валидация benchmark

Если ты только редактировал benchmark, сначала запускай:

```bash
uv run arxiv-rag-evaluate --benchmark eval/benchmark.tsv --validate-only
```

Эта команда:

- не загружает корпус;
- почти не потребляет RAM;
- проверяет формат TSV и `relevant_ids`;
- падает с понятной ошибкой по строке, если benchmark испорчен.

Если запустить `--benchmark benchmark.tsv` вместо `eval/benchmark.tsv`, команда завершится ошибкой, потому что путь должен указывать на реальный файл в репозитории.

### Полный eval baseline-моделей

```bash
uv run arxiv-rag-evaluate --model all --benchmark eval/benchmark.tsv --k 20
```

### Eval только BM25

```bash
uv run arxiv-rag-evaluate --model bm25 --benchmark eval/benchmark.tsv --k 20
```

### Eval только TF-IDF

```bash
uv run arxiv-rag-evaluate --model tfidf --benchmark eval/benchmark.tsv --k 20
```

### Подробный per-query output

```bash
uv run arxiv-rag-evaluate --model tfidf --benchmark eval/benchmark.tsv --k 20 --show-per-query
```

### Быстрый smoke test

```bash
uv run arxiv-rag-evaluate --model all --limit 50000 --benchmark eval/benchmark_fast.tsv --k 20
```

## Как запускать eval, если полный датасет не помещается в RAM

Сейчас evaluator не потоковый. Это означает:

- корпус сначала загружается в память;
- затем retriever строит индекс тоже в памяти.

Поэтому на слабой машине рабочая стратегия такая:

1. Проверять benchmark через `--validate-only`.
2. Для smoke test использовать `eval/benchmark_fast.tsv`.
3. Ограничивать корпус через `--limit`.
4. Гонять по одному retriever за запуск, а не `all`.

Примеры:

```bash
uv run arxiv-rag-evaluate --benchmark eval/benchmark.tsv --validate-only
uv run arxiv-rag-evaluate --model bm25 --benchmark eval/benchmark_fast.tsv --limit 30000 --k 20
uv run arxiv-rag-evaluate --model tfidf --benchmark eval/benchmark_fast.tsv --limit 30000 --k 20
```

## Какие метрики считает evaluator

Проект считает:

- `Recall@k`
- `MRR`
- `nDCG@k`

И выводит две группы значений:

- `evaluated` - только по тем запросам, для которых после пересечения с корпусом остались релевантные id;
- `penalized` - те же метрики, но с учётом доли пропущенных запросов.

Это важно, когда eval идёт на частичном корпусе через `--limit`.

Дополнительно evaluator сообщает:

- сколько всего запросов в benchmark;
- сколько реально оценено;
- сколько пропущено;
- покрытие `relevant_ids` по загруженному срезу корпуса.

## Как проверять свои модели на eval

В проекте под "своей моделью" для eval имеется в виду retriever-объект, совместимый с интерфейсом evaluator.

### Минимальный контракт

Твой retriever должен предоставлять два метода:

- `fit(texts)` - индексирует список документов;
- `topk(query, k)` - возвращает список индексов документов в порядке релевантности.

Важно:

- `topk` должен возвращать индексы, а не arXiv id;
- индексы должны ссылаться на позиции в том же порядке документов, который был передан в `fit(texts)`;
- возвращаемое значение должно быть совместимо со списком целых чисел.

### Минимальный пример кастомного retriever'а

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MyRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=20000)
        self.matrix = None

    def fit(self, texts):
        self.matrix = self.vectorizer.fit_transform(list(texts))
        return self

    def topk(self, query, k):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(self.matrix, query_vec).ravel()
        n = min(k, len(scores))
        return scores.argsort()[-n:][::-1].tolist()
```

### Как запустить eval такого retriever'а

Если класс лежит, например, в файле `my_retriever.py`, то команда будет такой:

```bash
uv run arxiv-rag-evaluate --model my_retriever:MyRetriever --benchmark eval/benchmark.tsv --k 20
```

Поддерживаются оба варианта:

- класс;
- фабрика, которая возвращает retriever.

То есть можно передать и `module:factory_function`.

## Как добавить новую модель в проект как встроенную

Есть два сценария.

### Сценарий 1. Без изменения кода eval

Самый быстрый путь - использовать `module:factory_or_class`, как в примере выше.

Это удобно, если ты просто хочешь сравнить свой retriever с baseline и не хочешь менять реестр встроенных моделей.

### Сценарий 2. Добавить модель как встроенную

Если хочешь, чтобы модель работала как `--model bm25` или `--model tfidf`, то нужно:

1. Добавить класс в `arxiv_rag/models/`.
2. Реэкспортировать его из `arxiv_rag/models/__init__.py`.
3. Зарегистрировать в словаре `registry` внутри [evaluate_models.py](evaluate_models.py).
4. При желании добавить поддержку этой же модели в [arxiv_rag/baseline_cli.py](arxiv_rag/baseline_cli.py).

Практически это означает, что eval CLI и baseline CLI могут развиваться независимо:

- если нужен только eval, достаточно регистрации в `evaluate_models.py`;
- если нужен ещё и ручной запуск через baseline CLI, надо расширить и `baseline_cli.py`.

## Что считается корректным retriever'ом для этого проекта

Корректный retriever для текущего evaluator:

- принимает список текстов корпуса;
- умеет индексировать их локально;
- возвращает ранжированный список позиций документов;
- не зависит от внешнего stateful API при каждом вызове `topk`, если хочешь воспроизводимый eval.

Подходят, например:

- BM25;
- TF-IDF;
- dense embedding retriever;
- hybrid retriever, если он всё равно возвращает индексы документов.

Не подойдёт без адаптера объект, который:

- возвращает сразу тексты вместо индексов;
- возвращает id, не синхронизированные с порядком корпуса;
- требует другого интерфейса вместо `fit/topk`.

## Практические сценарии использования

### Сценарий 1. Быстро проверить, что benchmark вообще валиден

```bash
uv run arxiv-rag-evaluate --benchmark eval/benchmark.tsv --validate-only
```

### Сценарий 2. Быстро сравнить baseline'ы на ограниченной машине

```bash
uv run arxiv-rag-evaluate --model all --benchmark eval/benchmark_fast.tsv --limit 30000 --k 20
```

### Сценарий 3. Сравнить свой retriever с BM25

```bash
uv run arxiv-rag-evaluate --model bm25 --benchmark eval/benchmark.tsv --k 20
uv run arxiv-rag-evaluate --model my_retriever:MyRetriever --benchmark eval/benchmark.tsv --k 20
```

### Сценарий 4. Оценить benchmark после ручного редактирования TSV

```bash
uv run arxiv-rag-evaluate --benchmark eval/benchmark.tsv --validate-only
uv run arxiv-rag-evaluate --model tfidf --benchmark eval/benchmark.tsv --k 20
```

## Веб-интерфейс для поиска (Demo)

Проект включает простое веб-приложение на Flask, позволяющее интерактивно искать документы в загруженном корпусе. Интерфейс доступен в браузере и позволяет:

* выбирать модель поиска (BM25, TF‑IDF, а также Dense‑ретривер, если он реализован);
* задавать количество возвращаемых результатов `k`;
* управлять размером корпуса (указать лимит вручную или загрузить все документы);
* вводить произвольные поисковые запросы.

### Запуск демо

1. Убедитесь, что данные подготовлены:
   ```bash
   uv run arxiv-rag-prepare-data
   ```
2. Запустите приложение:
   ```bash
   uv run python app.py
   ```
Если вы изменили структуру каталогов, можно указать путь к обработанным данным внутри ```app.py``` (переменная ```DATA_FOLDER```).
3. Откройте браузер по адресу ```http://127.0.0.1:5000```.
При запуске с ```host="0.0.0.0"``` сервер также доступен по локальному IP (например, ```http://192.168.1.8:5000```) для других устройств в сети.

### Возможности интерфейса
* **Поле ввода запроса** – поисковая фраза.
* **Выпадающий список модели** – переключение между доступными ретриверами.
* **Лимит документов** – можно ввести произвольное число (например, 100000) или поставить флажок «All», чтобы загрузить весь корпус.
*Примечание*: загрузка всего корпуса может потребовать значительного объёма оперативной памяти (несколько ГБ).
* **k (количество результатов)** – от 1 до 50.
* **Кнопка «Search»** – отправка формы.

При изменении модели или лимита корпус перезагружается и индекс перестраивается автоматически. После отправки запроса отображается до ```k``` найденных статей с заголовком, arXiv ID и сокращённым абстрактом.

## Интерпретация результата eval

При чтении результатов полезно помнить:

- `BM25` обычно является более сильным lexical baseline;
- `TF-IDF` удобно держать как нижнюю планку и sanity check;
- если `evaluated metrics` сильно выше `penalized metrics`, значит benchmark плохо покрывается выбранным срезом корпуса;
- если при `--limit` покрытие низкое, сравнение моделей уже не очень честное.

## Troubleshooting

- Если не работает автоматическая загрузка: проверь Kaggle credentials.
- Если baseline или eval не находят данные: сначала запусти `uv run arxiv-rag-prepare-data` или передай свои пути через `--data-folder`.
- Если benchmark не валидируется: сначала запусти `uv run arxiv-rag-evaluate --benchmark eval/benchmark.tsv --validate-only` и исправь строку, на которую ругается evaluator.
- Если не хватает памяти: уменьшай `--limit`, используй `benchmark_fast.tsv` и запускай по одному retriever за раз.
- Если хочешь честный full eval по всему корпусу, а RAM не хватает: нужен отдельный рефакторинг под потоковую или дисковую индексацию.

## Дополнительные файлы

- [evaluate_models.py](evaluate_models.py) - основной eval entrypoint.
- [arxiv_rag/baseline_cli.py](arxiv_rag/baseline_cli.py) - CLI ручного retrieval-запуска.
- [arxiv_rag/evaluation/evaluator.py](arxiv_rag/evaluation/evaluator.py) - контракт evaluator и расчёт метрик.
- [eval/BENCHMARK_FORMAT.md](eval/BENCHMARK_FORMAT.md) - краткая спецификация benchmark-файла.
