import pandas as pd
from tqdm import tqdm

from arxiv_rag.dataset import data_converter, load_arxiv_data

for _ in tqdm([1], desc="Converting arXiv metadata", unit="run"):
    data_converter('C:/Users/denisrtyhb/Desktop/ML_project/arxiv-metadata-oai-snapshot.json', 'output')

# Настройка отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 50)

for _ in tqdm([1], desc="Loading data (limit=1000)", unit="run"):
    df = load_arxiv_data(
        data_folder='output',
        limit=1000,
    )

print(df.head(10))
print(df.shape)

df = None

# Загрузить первые 1000 статей из категории 'cs.CL'
for _ in tqdm([1], desc="Loading data (cs.CL, limit=1000)", unit="run"):
    df = load_arxiv_data(
        data_folder='output',
        categories=['cs.CL'],
        limit=1000,
        columns=['id', 'title'],
    )

print(df.head(10))
print(df.shape)

df = None
# Загрузить все статьи, перемешать и взять случайные 5000
for _ in tqdm([1], desc="Loading data (shuffled, limit=5000)", unit="run"):
    df = load_arxiv_data(
        data_folder='output',
        limit=5000,
        shuffle=True,
        random_state=42,
    )
print(df.head(10))
print(df.shape)