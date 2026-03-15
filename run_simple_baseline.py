"""Simple RAG baseline: load data, fit model, run retrieval for example queries."""

from arxiv_rag.dataset import load_arxiv_data
from arxiv_rag.models import TfidfRAG, BM25RAG


# Load dataset
DATA_FOLDER = "output"
LIMIT = 2000  # use a subset for a quick baseline

df = load_arxiv_data(
    data_folder=DATA_FOLDER,
    limit=LIMIT,
    columns=["id", "title", "abstract"],
)
# Build document texts for indexing (title + abstract)
texts = df["abstract"].tolist()
print(f"Loaded {len(df)} documents")

# Load model
model = BM25RAG()

# Call fit method
model.fit(texts)

# Example queries and retrieval
EXAMPLE_QUERIES = [
    "transformer attention mechanism",
    "reinforcement learning for robotics",
    "graph neural networks",
]

for query in EXAMPLE_QUERIES:
    indices = model.topk(query, k=5) or []
    print(f"\nQuery: {query}")
    for i, idx in enumerate(indices, 1):
        row = df.iloc[idx]
        title = (row["title"] or "")[:80]
        print(f"  {i}. [{idx}] {title}...")
