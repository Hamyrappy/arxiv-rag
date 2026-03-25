import sys
import os
from flask import Flask, render_template_string, request

sys.path.insert(0, os.path.dirname(__file__))

from arxiv_rag.dataset.dataloader import load_arxiv_data
from arxiv_rag.models import BM25RAG, TfidfRAG

try:
    from arxiv_rag.models.dense import DenseRetriever, Specter2Retriever, BGERetriever
    from arxiv_rag.models.hybrid import HybridRetriever
    from arxiv_rag.models.cross_encoder import CrossEncoderReranker
    DENSE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Dense models not available ({e})")
    DENSE_AVAILABLE = False

app = Flask(__name__)

DATA_FOLDER = "data/processed"
MODEL_NAME = "bm25"
LIMIT = 50000

corpus = None
texts = None
retriever = None

def load_corpus_and_build_index(model_name, limit):
    global corpus, texts, retriever
    print(f"Loading data with limit={limit}...")
    must_include_ids = None
    try:
        import pandas as pd
        if os.path.exists("eval/benchmark_fast.tsv"):
            bench = pd.read_csv("eval/benchmark_fast.tsv", sep="\t")
            must_include_ids = set(bench["Relevant_Doc_ID"].astype(str).tolist())
    except Exception:
        pass
    df = load_arxiv_data(data_folder=DATA_FOLDER, limit=limit, must_include_ids=must_include_ids)
    if df.empty:
        raise RuntimeError("No data loaded. Check DATA_FOLDER and data preparation.")
    corpus = df.to_dict(orient="records")
    texts = [f"{doc.get('title','')} {doc.get('abstract','')}".strip() for doc in corpus]
    print(f"Loaded {len(corpus)} documents.")

    if model_name == "bm25":
        ret = BM25RAG()
    elif model_name == "tfidf":
        ret = TfidfRAG()
    elif model_name == "dense" and DENSE_AVAILABLE:
        ret = DenseRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
    elif model_name == "specter2" and DENSE_AVAILABLE:
        ret = Specter2Retriever()
    elif model_name == "bge" and DENSE_AVAILABLE:
        ret = BGERetriever()
    elif model_name == "hybrid" and DENSE_AVAILABLE:
        ret = HybridRetriever(BM25RAG(), BGERetriever(), fusion="weighted")
    elif model_name == "cross_encoder" and DENSE_AVAILABLE:
        ret = CrossEncoderReranker(HybridRetriever(BM25RAG(), BGERetriever(), fusion="weighted"), top_n=100)
    else:
        raise ValueError(f"Unknown model: {model_name} or dense models not available")

    print(f"Building index for {model_name}...")
    ret.fit(texts)
    retriever = ret
    print("Ready.")

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>arXiv RAG Demo</title>
    <style>
        body { font-family: sans-serif; margin: 2em; }
        .result { margin-bottom: 1em; border-bottom: 1px solid #ccc; padding-bottom: 0.5em; }
        .title { font-weight: bold; }
        .id { color: #666; font-size: 0.9em; }
        .abstract { margin-top: 0.3em; font-size: 0.95em; }
        form { margin-bottom: 2em; }
        input[type=text] { width: 400px; padding: 0.5em; }
        select, input[type=number], input[type=submit], .limit-input { padding: 0.5em; margin-left: 0.5em; }
        .limit-input { display: inline-flex; align-items: center; gap: 0.5em; }
        .info { font-size: 0.8em; color: #666; margin-top: 0.5em; }
    </style>
    <script>
        function toggleLimitInput() {
            const allCheckbox = document.getElementById('limit_all');
            const limitNumber = document.getElementById('limit_number');
            limitNumber.disabled = allCheckbox.checked;
        }
    </script>
</head>
<body>
    <h1>arXiv RAG Demo</h1>
    <form method="post">
        <input type="text" name="query" placeholder="Enter your query..." value="{{ query }}" required>
        <select name="model">
            <option value="bm25" {% if model == 'bm25' %}selected{% endif %}>BM25</option>
            <option value="tfidf" {% if model == 'tfidf' %}selected{% endif %}>TF-IDF</option>
            {% if DENSE_AVAILABLE %}
            <option value="dense" {% if model == 'dense' %}selected{% endif %}>Dense (MiniLM)</option>
            <option value="specter2" {% if model == 'specter2' %}selected{% endif %}>Specter 2</option>
            <option value="bge" {% if model == 'bge' %}selected{% endif %}>BGE</option>
            <option value="hybrid" {% if model == 'hybrid' %}selected{% endif %}>Hybrid (BM25 + BGE)</option>
            <option value="cross_encoder" {% if model == 'cross_encoder' %}selected{% endif %}>Cross-Encoder (MS-MARCO)</option>
            {% endif %}
        </select>

        <div class="limit-input">
            <label>Limit (docs):</label>
            <input type="number" name="limit_number" id="limit_number" value="{{ limit_number }}" min="1000" step="1000" {% if limit_all %}disabled{% endif %}>
            <label>
                <input type="checkbox" name="limit_all" id="limit_all" {% if limit_all %}checked{% endif %} onchange="toggleLimitInput()"> All
            </label>
        </div>

        <label>k:</label>
        <input type="number" name="k" value="{{ k }}" min="1" max="50">
        <input type="submit" value="Search">
    </form>
    <div class="info">
        Current corpus size: {{ corpus_size }} documents
        {% if not limit_all %} (limit: {{ limit_number }}){% else %} (all documents){% endif %}
    </div>

    {% if results %}
        <h2>Top-{{ k }} results for "{{ query }}" ({{ model }})</h2>
        {% for doc in results %}
            <div class="result">
                <div class="title">{{ doc.title }}</div>
                <div class="id">arXiv ID: {{ doc.id }}</div>
                <div class="abstract">{{ doc.abstract[:500] }}{% if doc.abstract|length > 500 %}...{% endif %}</div>
            </div>
        {% endfor %}
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    global retriever, corpus, MODEL_NAME, LIMIT
    query = ""
    k = 10
    model = MODEL_NAME
    limit_number = 50000
    limit_all = False
    results = []

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        k = int(request.form.get("k", 10))
        model = request.form.get("model", MODEL_NAME)
        limit_all = request.form.get("limit_all") == "on"
        if limit_all:
            new_limit = None
            limit_number = 50000
        else:
            try:
                new_limit = int(request.form.get("limit_number", 50000))
                limit_number = new_limit
            except ValueError:
                new_limit = 50000
                limit_number = 50000

        if model != MODEL_NAME or new_limit != LIMIT:
            MODEL_NAME = model
            LIMIT = new_limit
            load_corpus_and_build_index(MODEL_NAME, LIMIT)

        if query and retriever:
            idxs = retriever.topk(query, k)
            results = [corpus[i] for i in idxs if i < len(corpus)]
    else:
        if LIMIT is None:
            limit_all = True
            limit_number = 50000
        else:
            limit_all = False
            limit_number = LIMIT

    return render_template_string(
        HTML_TEMPLATE,
        query=query,
        k=k,
        model=MODEL_NAME,
        limit_number=limit_number,
        limit_all=limit_all,
        results=results,
        corpus_size=len(corpus) if corpus else 0,
        DENSE_AVAILABLE=DENSE_AVAILABLE
    )

if __name__ == "__main__":
    load_corpus_and_build_index(MODEL_NAME, LIMIT)
    app.run(debug=True, host="0.0.0.0", port=5000)