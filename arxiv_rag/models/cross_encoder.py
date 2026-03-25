import numpy as np

class CrossEncoderReranker:
    """
    A re-ranker that takes a base retriever and applies a Cross-Encoder
    to score and re-sort the top_n candidate documents.
    """

    def __init__(self, base_retriever, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=100):
        self.base_retriever = base_retriever
        self.model_name = model_name
        self.top_n = top_n
        self.texts = []
        self._model = None

    def fit(self, texts: list[str]):
        """Fits the base retriever and stores the texts for cross-encoder pairs."""
        # Store a clean version of texts (replace [SEP] with space for standard cross-encoders)
        self.texts = [t.replace(" [SEP] ", " ") for t in texts]
        self.base_retriever.fit(texts)
        return self

    def topk(self, query: str, k: int) -> list[int]:
        if not self.texts:
            raise RuntimeError("Call fit() before topk()")

        # 1. First-stage retrieval using base (cheap) retriever
        candidate_indices = self.base_retriever.topk(query, self.top_n)
        
        if not candidate_indices:
            return []

        # Lazy load model
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading Cross-Encoder {self.model_name} on {device}...")
            self._model = CrossEncoder(self.model_name, device=device)

        # 2. Build Query-Document pairs
        pairs = [[query, self.texts[idx]] for idx in candidate_indices]

        # 3. Second-stage scoring
        scores = self._model.predict(pairs)

        # 4. Re-rank (sort descending)
        sorted_order = np.argsort(scores)[::-1]

        # Select top k
        final_k = min(k, len(candidate_indices))
        best_indices = [candidate_indices[i] for i in sorted_order[:final_k]]

        return best_indices
