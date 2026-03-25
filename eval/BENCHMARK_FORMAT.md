# Формат benchmark-файла

Benchmark-файлы — это UTF-8 TSV-файлы, которые редактируются в текстовом редакторе, например VS Code.

## Обязательные столбцы

- `query`: произвольный поисковый запрос
- `relevant_ids`: JSON-массив идентификаторов arXiv, например `["2101.01532", "1706.03762"]`

## Необязательные столбцы

- `relevance_tier`: ручная метка релевантности (`high`, `medium`, `low`)
- любые дополнительные столбцы допускаются для собственных заметок, но текущий evaluator их игнорирует

## Правила

- Используйте символ табуляции в качестве разделителя.
- Сохраняйте файлы в кодировке UTF-8. UTF-8 с BOM тоже принимается.
- Используйте реальные arXiv-идентификаторы из столбца `id` обработанного датасета.
- Поддерживаются как новые идентификаторы (например `2101.01532`), так и старые (например `cond-mat/0601050`).
- Префиксы вида `arXiv:` принимаются и нормализуются автоматически.
- Пустые строки игнорируются.
- Частично заполненные строки отклоняются при валидации.

## Пример

```text
query	relevant_ids	relevance_tier
bayesian data assimilation for epidemic evolution	["2101.01532"]	high
metallic grains and interacting electrons	["cond-mat/0101029"]	medium
```

## Рекомендуемые команды

Только валидация benchmark:

```bash
uv run arxiv-rag-evaluate --benchmark eval/benchmark.tsv --validate-only
```

Быстрый smoke-тест:

```bash
uv run arxiv-rag-evaluate --model all --limit 50000 --benchmark eval/benchmark.tsv --k 20
```

Полная оценка:

```bash
uv run arxiv-rag-evaluate --model all --benchmark eval/benchmark.tsv --k 20
```

## Советы по редактированию

- Предпочитайте VS Code вместо Excel или Google Sheets — табличные редакторы могут изменить кавычки.
- При вставке строк из браузера или LLM сохраняйте `relevant_ids` как JSON-список в одной ячейке.
- Если benchmark не проходит валидацию, сначала запустите команду `--validate-only`, чтобы получить ошибку с указанием строки.
