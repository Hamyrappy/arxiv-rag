import pandas as pd
import json
import numpy as np
from datetime import datetime
import os
import shutil
import re
import glob
from tqdm import tqdm


def data_converter(input_path, output_path, chunksize=100000, final_output=None):
    """
    Обрабатывает файл arxiv-metadata-oai-snapshot.json, выполняет очистку,
    извлечение признаков и глобальное удаление дубликатов по названию и авторам.
    При указании final_output объединяет все part-файлы в один Parquet с исправлением id.

    Parameters
    ----------
    input_path : str
        Путь к входному JSON-файлу (arxiv-metadata-oai-snapshot.json).
    output_path : str
        Путь к папке, куда будут сохранены файлы финального датасета (part_*.parquet).
    chunksize : int, optional
        Количество строк в одном чанке (по умолчанию 100000).
    final_output : str, optional
        Если указан, то после обработки все part-файлы объединяются в один файл
        по этому пути, а id при необходимости форматируются с ведущими нулями.
        Затем part-файлы удаляются.
    """
    # Создаём выходную папку и временную папку для чанков
    os.makedirs(output_path, exist_ok=True)
    temp_dir = os.path.join(output_path, 'temp_chunks')
    os.makedirs(temp_dir, exist_ok=True)

    # --- Вспомогательные функции ---
    def extract_first_version_date(versions):
        """Извлекает дату первой версии из поля 'versions'."""
        if versions is None:
            return None
        if isinstance(versions, float) and np.isnan(versions):
            return None
        # Если versions список (уже распарсено)
        if isinstance(versions, list):
            if len(versions) == 0:
                return None
            first = versions[0]
            if isinstance(first, dict) and 'created' in first:
                date_str = first['created']
                for fmt in ('%a, %d %b %Y %H:%M:%S %Z', '%Y-%m-%d'):
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        return dt.strftime('%Y-%m-%d')
                    except ValueError:
                        continue
            return None
        # Если versions строка
        if isinstance(versions, str):
            if versions.strip() in ('', '[]', '{}'):
                return None
            try:
                versions_clean = versions.replace("'", '"')
                versions_list = json.loads(versions_clean)
                if isinstance(versions_list, list) and len(versions_list) > 0:
                    first = versions_list[0]
                    if isinstance(first, dict) and 'created' in first:
                        date_str = first['created']
                        for fmt in ('%a, %d %b %Y %H:%M:%S %Z', '%Y-%m-%d'):
                            try:
                                dt = datetime.strptime(date_str, fmt)
                                return dt.strftime('%Y-%m-%d')
                            except ValueError:
                                continue
            except Exception:
                pass
        return None

    def clean_update_date(ud):
        """Приводит update_date к единому формату YYYY-MM-DD."""
        if pd.isna(ud):
            return None
        try:
            return pd.to_datetime(ud).strftime('%Y-%m-%d')
        except:
            return None

    def split_categories(cat_str):
        """Разбивает строку категорий на список."""
        if pd.isna(cat_str):
            return []
        return str(cat_str).split()

    def clean_title(title):
        """Очищает заголовок от LaTeX и приводит к нижнему регистру."""
        if pd.isna(title):
            return ''
        title = re.sub(r'\$.*?\$', '', title)
        title = re.sub(r'\\[a-zA-Z]+', '', title)
        title = re.sub(r'\s+', ' ', title).strip().lower()
        return title

    def authors_to_set(authors_parsed):
        """Преобразует список авторов в множество кортежей (фамилия, имя)."""
        if isinstance(authors_parsed, list):
            return set(tuple(a[:2]) for a in authors_parsed if len(a) >= 2)
        else:
            return set()

    def process_duplicate_group(entries):
        """
        entries: список кортежей (global_index, authors_set, date_sort)
        Возвращает список global_index, которые нужно удалить (оставить одну с max date).
        """
        n = len(entries)
        if n <= 1:
            return []
        indices = [e[0] for e in entries]
        authors_sets = [e[1] for e in entries]
        dates = [e[2] for e in entries]

        # DSU для кластеризации по общим авторам
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for i in range(n):
            for j in range(i+1, n):
                if authors_sets[i] & authors_sets[j]:
                    union(i, j)

        clusters = {}
        for i in range(n):
            root = find(i)
            clusters.setdefault(root, []).append(i)

        to_drop = []
        for members in clusters.values():
            if len(members) == 1:
                continue
            best_local = max(members, key=lambda i: dates[i])
            for i in members:
                if i != best_local:
                    to_drop.append(indices[i])
        return to_drop

    # ------------------ Первый проход: обработка чанков и сбор информации для дедупликации ----------
    reader = pd.read_json(input_path, lines=True, chunksize=chunksize, dtype={'id': str})
    dup_map = {}          # ключ: title_cleaned -> список (global_index, authors_set, date_sort)
    global_index = 0
    chunk_num = 0

    for chunk in tqdm(reader, desc="Processing chunks", unit="chunk"):
        df = chunk.copy()

        # Приводим id к строке сразу после чтения
        df['id'] = df['id'].astype(str)

        # 1. Удаление ненужных столбцов
        cols_to_drop = ['submitter', 'authors', 'comments']
        existing_cols = [c for c in cols_to_drop if c in df.columns]
        if existing_cols:
            df.drop(columns=existing_cols, inplace=True)

        # 2. Бинаризация journal-ref, doi, report-no, license
        binary_cols = ['journal-ref', 'doi', 'report-no', 'license']
        for col in binary_cols:
            if col in df.columns:
                df[f'{col}_present'] = df[col].apply(lambda x: 0 if pd.isna(x) or x == '' else 1)
                df.drop(columns=[col], inplace=True)

        # 3. Обработка дат
        if 'versions' in df.columns:
            df['first_version_date'] = df['versions'].apply(extract_first_version_date)
            df.drop(columns=['versions'], inplace=True)
        if 'update_date' in df.columns:
            df['last_update_date'] = df['update_date'].apply(clean_update_date)
            df.drop(columns=['update_date'], inplace=True)

        # 4. Преобразование categories
        if 'categories' in df.columns:
            df['categories_list'] = df['categories'].apply(split_categories)
            df.drop(columns=['categories'], inplace=True)

        # 5. Подготовка к дедупликации
        df['title_cleaned'] = df['title'].apply(clean_title)
        df['authors_set'] = df['authors_parsed'].apply(authors_to_set)
        df['date_sort'] = pd.to_datetime(df['last_update_date'])
        df['global_index'] = range(global_index, global_index + len(df))

        # Сохраняем чанк во временный файл
        chunk_path = os.path.join(temp_dir, f'chunk_{chunk_num:04d}.parquet')
        df.to_parquet(chunk_path, index=False)

        # Сбор информации для дедупликации
        for _, row in df.iterrows():
            title_cleaned = row['title_cleaned']
            if title_cleaned:
                if title_cleaned not in dup_map:
                    dup_map[title_cleaned] = []
                dup_map[title_cleaned].append(
                    (row['global_index'], row['authors_set'], row['date_sort'])
                )

        global_index += len(df)
        chunk_num += 1
        tqdm.write(f"Обработан чанк {chunk_num}, всего записей: {global_index}")

    # ------------------ Второй этап: глобальная дедупликация ------------------
    tqdm.write("Сбор информации о дубликатах завершён. Выполняется кластеризация...")
    indices_to_drop = set()
    total_groups = len(dup_map)
    for title, entries in tqdm(dup_map.items(), total=total_groups, desc="Deduplicating", unit="group"):
        to_drop = process_duplicate_group(entries)
        indices_to_drop.update(to_drop)

    tqdm.write(f"Найдено дублирующихся записей: {len(indices_to_drop)}")

    # ------------------ Третий проход: запись финального датасета без дубликатов ------------------
    chunk_files = sorted([f for f in os.listdir(temp_dir) if f.startswith('chunk_')])
    part_num = 0
    for chunk_file in tqdm(chunk_files, desc="Writing final dataset", unit="file"):
        chunk_path = os.path.join(temp_dir, chunk_file)
        df_chunk = pd.read_parquet(chunk_path)

        mask = ~df_chunk['global_index'].isin(indices_to_drop)
        df_filtered = df_chunk.loc[mask].copy()
        if df_filtered.empty:
            continue

        # Приведение id к строке (гарантия)
        df_filtered['id'] = df_filtered['id'].astype(str)

        # Удаляем временные столбцы
        cols_to_remove = ['authors_set', 'date_sort', 'global_index', 'title_cleaned']
        df_filtered.drop(columns=[c for c in cols_to_remove if c in df_filtered.columns], inplace=True)

        # Сохраняем как часть финального датасета
        part_path = os.path.join(output_path, f'part_{part_num:04d}.parquet')
        df_filtered.to_parquet(part_path, index=False)
        part_num += 1

    # Удаляем временную папку
    shutil.rmtree(temp_dir)
    tqdm.write(f"Готово! Финальный датасет сохранён в папку {output_path} в виде {part_num} файлов.")
    tqdm.write(f"Всего записей после дедупликации: {global_index - len(indices_to_drop)}")

    # ------------------ Финальное объединение (если запрошено) ------------------
    if final_output is not None:
        tqdm.write("Объединение part-файлов в один датасет...")
        df_list = []
        part_files = sorted([f for f in os.listdir(output_path) if f.startswith('part_')])
        for pf in tqdm(part_files, desc="Merging part files", unit="file"):
            df_part = pd.read_parquet(os.path.join(output_path, pf))
            # Если id оказался числовым (маловероятно, но на всякий случай) – форматируем
            if df_part['id'].dtype != 'object':
                df_part['id'] = df_part['id'].apply(
                    lambda x: f"{float(x):07.4f}" if isinstance(x, (int, float)) else str(x)
                )
            df_list.append(df_part)

        df_final = pd.concat(df_list, ignore_index=True)
        df_final.to_parquet(final_output, index=False)
        tqdm.write(f"Объединённый датасет сохранён в {final_output}")

        # Удаляем part-файлы (опционально)
        for pf in tqdm(part_files, desc="Removing part files", unit="file"):
            os.remove(os.path.join(output_path, pf))
        tqdm.write("Временные part-файлы удалены.")



def load_arxiv_data(
    data_folder,
    limit=None,
    categories=None,
    columns=None,
    shuffle=False,
    random_state=None,
    must_include_ids=None,
):
    """
    Загружает данные из обработанного датасета arXiv, возвращая единый DataFrame.

    Parameters
    ----------
    data_folder : str
        Путь к папке, содержащей part-файлы Parquet.
    limit : int, optional
        Максимальное количество статей для загрузки.
    categories : list of str, optional
        Список категорий (например, ['cs.AI', 'physics']).
        Загружаются статьи, у которых хотя бы одна категория из списка.
    columns : list of str, optional
        Список колонок для загрузки (по умолчанию все).
    shuffle : bool, default False
        Перемешать ли итоговую выборку перед применением limit.
    random_state : int, optional
        Seed для воспроизводимости перемешивания.
    must_include_ids : set of str, optional
        Идентификаторы статей, которые всегда должны попасть в выборку
        (даже если не вмещаются в limit). Для поиска таких статей
        просматриваются все parquet-файлы; лимит заполняется остатком
        обычных строк.

    Returns
    -------
    pandas.DataFrame
        Запрошенные данные.
    """
    # Получаем список всех part-файлов и сортируем
    file_pattern = os.path.join(data_folder, 'part_*.parquet')
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise ValueError(f"В папке {data_folder} не найдены файлы part_*.parquet")

    # Определяем, нужна ли фильтрация по категориям
    need_categories = categories is not None
    categories_set = set(categories) if categories else None

    must_include_set = set(must_include_ids) if must_include_ids else set()

    # Если нужна фильтрация, а колонка categories_list не запрошена, добавим её временно
    effective_columns = columns
    if need_categories and columns is not None and 'categories_list' not in columns:
        effective_columns = list(effective_columns) + ['categories_list']

    # Если нужно искать must-include статьи, а колонка id не запрошена, добавим её временно
    _need_temp_id = bool(must_include_set) and columns is not None and 'id' not in columns
    if _need_temp_id:
        effective_columns = list(effective_columns) + ['id']

    must_accumulated = []   # строки с must-include ID
    regular_accumulated = []  # обычные строки
    must_found: set = set()
    regular_count = 0

    for file in tqdm(files, desc="Loading parquet files", unit="file"):
        # Читаем файл
        df_chunk = pd.read_parquet(file)

        # Выбираем только те колонки, которые есть в файле и нужны
        if effective_columns is not None:
            cols_present = [c for c in effective_columns if c in df_chunk.columns]
            df_chunk = df_chunk[cols_present]

        # Фильтрация по категориям
        if categories_set:
            if 'categories_list' not in df_chunk.columns:
                raise ValueError(f"В файле {file} отсутствует колонка 'categories_list'")
            mask = df_chunk['categories_list'].apply(lambda cats: bool(set(cats) & categories_set))
            df_chunk = df_chunk[mask]

        # Если после фильтрации чанк пуст, пропускаем
        if df_chunk.empty:
            continue

        # Разделяем чанк на must-include и обычные строки
        if must_include_set and 'id' in df_chunk.columns:
            remaining_must = must_include_set - must_found
            must_mask = df_chunk['id'].astype(str).isin(remaining_must)
            must_chunk = df_chunk[must_mask]
            regular_chunk = df_chunk[~must_mask]

            if not must_chunk.empty:
                must_accumulated.append(must_chunk)
                must_found.update(must_chunk['id'].astype(str).tolist())
        else:
            must_chunk = pd.DataFrame()
            regular_chunk = df_chunk

        # Накапливаем обычные строки до лимита
        if limit is None or regular_count < limit:
            if limit is not None:
                remaining = limit - regular_count
                if len(regular_chunk) > remaining:
                    regular_chunk = regular_chunk.iloc[:remaining]
            if not regular_chunk.empty:
                regular_accumulated.append(regular_chunk)
                regular_count += len(regular_chunk)

        # Ранний выход: все must-include найдены И обычных строк достаточно
        all_must_found = (must_include_set <= must_found)
        if limit is not None and regular_count >= limit and all_must_found:
            break

    # Собираем итоговый DataFrame
    must_df = pd.concat(must_accumulated, ignore_index=True) if must_accumulated else pd.DataFrame()
    regular_df = pd.concat(regular_accumulated, ignore_index=True) if regular_accumulated else pd.DataFrame()

    # Если must-include статей много — урезаем обычные строки, чтобы total ≤ limit
    if limit is not None and not must_df.empty:
        regular_to_keep = max(0, limit - len(must_df))
        regular_df = regular_df.head(regular_to_keep)

    frames = [f for f in [must_df, regular_df] if not f.empty]
    if not frames:
        if columns is not None:
            return pd.DataFrame(columns=columns)
        else:
            return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Перемешивание
    if shuffle:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Удаляем временные колонки, если они не были запрошены
    if _need_temp_id and 'id' in df.columns:
        df = df.drop(columns=['id'])
    if need_categories and columns is not None and 'categories_list' not in columns:
        if 'categories_list' in df.columns:
            df = df.drop(columns=['categories_list'])

    return df