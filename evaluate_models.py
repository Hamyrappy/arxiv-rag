"""Evaluation script: run the existing BM25RAG and TfidfRAG baselines on a
5-query arXiv mini-benchmark and print Recall@k, MRR and nDCG metrics.

Usage
-----
    python evaluate_models.py

No external data files are required – the script uses a self-contained
mini-corpus that pairs each benchmark query with a relevant document.
"""

from arxiv_rag.models import BM25RAG, TfidfRAG
from arxiv_rag.evaluation import Evaluator

# ---------------------------------------------------------------------------
# Mini-benchmark: 5 realistic arXiv queries with ground-truth relevant doc IDs
# ---------------------------------------------------------------------------
BENCHMARK = [
    {
        "query": "Applications of cross-attention mechanisms in medical image segmentation",
        "relevant_ids": ["doc_001"],
    },
    {
        "query": "Quantum error correction codes for topological quantum computers",
        "relevant_ids": ["doc_002"],
    },
    {
        "query": "Sample efficient reinforcement learning using offline datasets",
        "relevant_ids": ["doc_003"],
    },
    {
        "query": (
            "Trade-offs between exact and approximate nearest neighbor search "
            "in high dimensions"
        ),
        "relevant_ids": ["doc_004"],
    },
    {
        "query": "Constraints on dark matter from cosmic microwave background radiation",
        "relevant_ids": ["doc_005"],
    },
]

# ---------------------------------------------------------------------------
# Mini-corpus: the 5 relevant documents (title + abstract-like text) plus
# 10 distractor documents so retrieval is non-trivial.
# ---------------------------------------------------------------------------
DOC_IDS = [
    # Relevant documents
    "doc_001",
    "doc_002",
    "doc_003",
    "doc_004",
    "doc_005",
    # Distractor documents
    "doc_006",
    "doc_007",
    "doc_008",
    "doc_009",
    "doc_010",
    "doc_011",
    "doc_012",
    "doc_013",
    "doc_014",
    "doc_015",
]

TEXTS = [
    # doc_001 – relevant to query 1
    (
        "Cross-attention mechanisms for medical image segmentation. "
        "We propose a transformer-based architecture that applies cross-attention "
        "between encoder features and decoder queries to improve segmentation accuracy "
        "in CT and MRI scans.  Experiments on benchmark medical datasets demonstrate "
        "state-of-the-art performance."
    ),
    # doc_002 – relevant to query 2
    (
        "Topological quantum error correction with surface codes. "
        "We present a family of quantum error correcting codes tailored for "
        "topological quantum computers.  Our construction leverages anyonic "
        "braiding operations to achieve fault-tolerant logical qubits robust to "
        "local noise."
    ),
    # doc_003 – relevant to query 3
    (
        "Offline reinforcement learning with limited data. "
        "We study sample-efficient policy learning from fixed offline datasets "
        "without environment interaction.  Our method combines conservative "
        "Q-learning with data augmentation to extract robust policies from small "
        "batches of logged experience."
    ),
    # doc_004 – relevant to query 4
    (
        "Approximate nearest neighbor search in high-dimensional spaces. "
        "We analyze the accuracy-speed trade-off between exact and approximate "
        "k-nearest-neighbor algorithms including HNSW, IVF-PQ and LSH across "
        "various embedding dimensions and dataset sizes."
    ),
    # doc_005 – relevant to query 5
    (
        "Dark matter constraints from the cosmic microwave background. "
        "Using Planck CMB power spectra we derive tight constraints on dark matter "
        "annihilation cross-sections.  Our bounds are consistent with the standard "
        "ΛCDM cosmological model and rule out light WIMPs below 10 GeV."
    ),
    # Distractors
    (
        "Federated learning for privacy-preserving NLP. "
        "We investigate communication-efficient federated fine-tuning of large "
        "language models while preserving user privacy via differential privacy."
    ),
    (
        "Graph neural networks for molecular property prediction. "
        "We introduce an equivariant message-passing architecture that encodes "
        "3-D molecular geometry for accurate property regression."
    ),
    (
        "Neural architecture search with evolutionary algorithms. "
        "Our method evolves neural network topologies to maximise validation "
        "accuracy under a FLOPs budget, outperforming random search baselines."
    ),
    (
        "Self-supervised speech representation learning. "
        "We pre-train a convolutional encoder on unlabelled audio using contrastive "
        "objectives and achieve competitive ASR with only one hour of labelled data."
    ),
    (
        "Continual learning without catastrophic forgetting. "
        "We propose a neuroscience-inspired replay mechanism that selectively "
        "rehearses past experiences to maintain performance on previous tasks while "
        "learning new ones."
    ),
    (
        "Vision-language models for zero-shot image classification. "
        "CLIP-style contrastive training on image-text pairs yields strong "
        "zero-shot transfer to unseen visual categories."
    ),
    (
        "Efficient transformers: a survey. "
        "We review sparse attention, linear attention and low-rank approximations "
        "that reduce the quadratic complexity of self-attention."
    ),
    (
        "Causal discovery from observational data. "
        "We compare constraint-based and score-based algorithms for recovering "
        "causal graphs from high-dimensional observational datasets."
    ),
    (
        "Robotic manipulation with diffusion policies. "
        "Denoising diffusion models trained on robot trajectories produce smooth, "
        "multi-modal action distributions for dexterous manipulation tasks."
    ),
    (
        "Protein structure prediction using language models. "
        "Large-scale pre-training on protein sequences enables accurate 3-D "
        "structure prediction competitive with AlphaFold on standard benchmarks."
    ),
]

assert len(DOC_IDS) == len(TEXTS), "DOC_IDS and TEXTS must have the same length."


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def run_evaluation(model_name: str, retriever, k: int = 10) -> None:
    evaluator = Evaluator(retriever=retriever, doc_ids=DOC_IDS, texts=TEXTS)
    results = evaluator.evaluate(BENCHMARK, k=k)

    print(f"\n{'=' * 60}")
    print(f"Model : {model_name}")
    print(f"{'=' * 60}")
    print(f"  Recall@{k} : {results[f'recall@{k}']:.4f}")
    print(f"  MRR       : {results['mrr']:.4f}")
    print(f"  nDCG@{k}  : {results[f'ndcg@{k}']:.4f}")
    print()
    print("  Per-query breakdown:")
    for item in results["per_query"]:
        rr_val = item["rr"]
        hit = "✓" if rr_val > 0 else "✗"
        print(
            f"    [{hit}] RR={rr_val:.2f}  "
            f"Recall@{k}={item[f'recall@{k}']:.2f}  "
            f"nDCG@{k}={item[f'ndcg@{k}']:.2f}  "
            f"| {item['query'][:65]}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    K = 10

    print("arXiv RAG – Mini-Benchmark Evaluation")
    print(f"Corpus size : {len(DOC_IDS)} documents")
    print(f"Queries     : {len(BENCHMARK)}")
    print(f"Cut-off k   : {K}")

    run_evaluation("BM25RAG", BM25RAG(), k=K)
    run_evaluation("TfidfRAG", TfidfRAG(), k=K)
