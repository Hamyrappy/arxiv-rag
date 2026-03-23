"""Dense retrieval using sentence-transformers and FAISS."""

import hashlib
from pathlib import Path
from typing import Iterable, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


CACHE_DIR = Path("data/embeddings_cache")


def _default_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class DenseRetriever:
    """Dense retriever: encode texts with a sentence-transformer, search with FAISS.

    Supports on-disk embedding cache so repeated runs on the same corpus
    skip the expensive encoding step.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    query_prompt : str or None
        Prefix prepended to queries at search time (e.g. BGE instruction).
        Not applied to corpus documents during fit().
    batch_size : int
        Encoding batch size.
    device : str or None
        torch device; auto-detects CUDA if None.
    cache_dir : Path or None
        Directory for embedding cache files.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        query_prompt: Optional[str] = None,
        batch_size: int = 256,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.query_prompt = query_prompt
        self.batch_size = batch_size
        self.device = device or _default_device()
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self._model: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.IndexFlatIP] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _cache_path(self, corpus_hash: str) -> Path:
        safe_name = self.model_name.replace("/", "__")
        return self.cache_dir / f"{safe_name}_{corpus_hash}.npy"

    @staticmethod
    def _corpus_hash(texts: list[str]) -> str:
        h = hashlib.sha256()
        h.update(str(len(texts)).encode())
        for t in texts[:50]:
            h.update(t[:200].encode("utf-8", errors="replace"))
        if len(texts) > 50:
            for t in texts[-10:]:
                h.update(t[:200].encode("utf-8", errors="replace"))
        return h.hexdigest()[:16]

    def _encode_corpus(self, texts: list[str]) -> np.ndarray:
        corpus_hash = self._corpus_hash(texts)
        cache_path = self._cache_path(corpus_hash)

        if cache_path.exists():
            embeddings = np.load(str(cache_path))
            if embeddings.shape[0] == len(texts):
                print(f"  Loaded cached embeddings from {cache_path}")
                return embeddings

        print(f"  Encoding {len(texts)} documents with {self.model_name} on {self.device}...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), embeddings)
        print(f"  Cached embeddings to {cache_path}")
        return embeddings

    def _format_text(self, text: str, is_query: bool = False) -> str:
        """Format text (e.g. resolve special separator injected by dataloader)."""
        return text.replace(" [SEP] ", ". ")

    def fit(self, texts: Iterable[str]) -> "DenseRetriever":
        texts_list = [self._format_text(t, is_query=False) for t in texts]
        embeddings = self._encode_corpus(texts_list)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        return self

    def topk(self, query: str, k: int) -> list[int]:
        if self._index is None:
            raise RuntimeError("Call fit() before topk()")
        q = self._format_text(query, is_query=True)
        q = self.query_prompt + q if self.query_prompt else q
        q_emb = self.model.encode(
            [q], normalize_embeddings=True
        ).astype(np.float32)
        n = min(k, self._index.ntotal)
        _, indices = self._index.search(q_emb, n)
        return indices[0].tolist()


class Specter1Retriever(DenseRetriever):
    """SPECTER (v1) — trained on scientific documents (S2ORC).
    
    Standard sentence-transformers/allenai-specter, natively compatible.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("batch_size", 64)
        super().__init__(model_name="sentence-transformers/allenai-specter", **kwargs)

    def _format_text(self, text: str, is_query: bool = False) -> str:
        sep = self.model.tokenizer.sep_token
        if " [SEP] " in text:
            return text.replace(" [SEP] ", sep)
        if is_query:
            return text + sep
        return text


class Specter2Retriever(DenseRetriever):
    """SPECTER-2 with native adapter injection (no `adapters` library).

    Loads allenai/specter2_base and manually injects bottleneck adapter
    weights downloaded from allenai/specter2_adhoc_query (queries) and
    allenai/specter2_proximity (documents).
    """

    _BASE = "allenai/specter2_base"
    _ADAPTER_QUERY = "allenai/specter2_adhoc_query"
    _ADAPTER_DOC = "allenai/specter2_proximity"

    def __init__(self, **kwargs):
        kwargs.setdefault("batch_size", 64)
        super().__init__(model_name=self._BASE, **kwargs)
        self._bert = None
        self._tokenizer = None
        self._query_adapters = None  # nn.ModuleList
        self._doc_adapters = None

    # ---- adapter module ------------------------------------------------
    @staticmethod
    def _make_adapters(state_dict, tag: str, device: str):
        """Build a list of 12 BottleneckAdapter modules from a state_dict."""
        import torch, torch.nn as nn

        class BottleneckAdapter(nn.Module):
            def __init__(self, down_w, down_b, up_w, up_b):
                super().__init__()
                self.down = nn.Linear(down_w.shape[1], down_w.shape[0])
                self.down.weight = nn.Parameter(down_w)
                self.down.bias = nn.Parameter(down_b)
                self.up = nn.Linear(up_w.shape[1], up_w.shape[0])
                self.up.weight = nn.Parameter(up_w)
                self.up.bias = nn.Parameter(up_b)

            def forward(self, x):
                return x + self.up(torch.relu(self.down(x)))

        adapters = nn.ModuleList()
        for i in range(12):
            prefix = f"bert.encoder.layer.{i}.output.adapters.{tag}."
            adapters.append(BottleneckAdapter(
                state_dict[prefix + "adapter_down.0.weight"],
                state_dict[prefix + "adapter_down.0.bias"],
                state_dict[prefix + "adapter_up.weight"],
                state_dict[prefix + "adapter_up.bias"],
            ))
        return adapters.to(device).eval()

    # ---- lazy init ------------------------------------------------------
    def _load_model(self):
        if self._bert is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer
        from huggingface_hub import hf_hub_download

        self._tokenizer = AutoTokenizer.from_pretrained(self._BASE)
        self._bert = AutoModel.from_pretrained(self._BASE).to(self.device).eval()

        q_path = hf_hub_download(self._ADAPTER_QUERY, "pytorch_adapter.bin")
        d_path = hf_hub_download(self._ADAPTER_DOC, "pytorch_adapter.bin")
        q_sd = torch.load(q_path, map_location=self.device, weights_only=True)
        d_sd = torch.load(d_path, map_location=self.device, weights_only=True)
        self._query_adapters = self._make_adapters(q_sd, "[QRY]", self.device)
        self._doc_adapters = self._make_adapters(d_sd, "[PRX]", self.device)
        self._hooks = []

    def _install_hooks(self, adapters):
        """Register hooks on BertOutput.dense to inject adapters at the correct position.

        Original adapter placement: after FF dense projection, before residual + LayerNorm.
        """
        self._remove_hooks()
        for idx, layer in enumerate(self._bert.encoder.layer):
            adapter = adapters[idx]
            def make_hook(a):
                def hook(module, inp, out):
                    return a(out)
                return hook
            h = layer.output.dense.register_forward_hook(make_hook(adapter))
            self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ---- encoding -------------------------------------------------------
    def _encode(self, texts: list[str], adapters) -> np.ndarray:
        import torch
        from tqdm import tqdm

        self._install_hooks(adapters)
        all_embs = []
        try:
            for i in tqdm(range(0, len(texts), self.batch_size),
                          desc="Encoding", disable=len(texts) < self.batch_size):
                batch = texts[i:i + self.batch_size]
                inputs = self._tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=512, return_tensors="pt",
                    return_token_type_ids=False,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = self._bert(**inputs)
                    cls = out.last_hidden_state[:, 0]
                    all_embs.append(cls.cpu().numpy())
        finally:
            self._remove_hooks()
        return np.vstack(all_embs)

    def _format_text(self, text: str, is_query: bool = False) -> str:
        self._load_model()
        sep = self._tokenizer.sep_token
        # Official format: title + sep_token + abstract (no spaces around sep)
        if " [SEP] " in text:
            return text.replace(" [SEP] ", sep)
        return text

    def _encode_corpus(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        corpus_hash = self._corpus_hash(texts)
        cache_path = self._cache_path(corpus_hash + "_s2v2")
        if cache_path.exists():
            emb = np.load(str(cache_path))
            if emb.shape[0] == len(texts):
                print(f"  Loaded cached embeddings from {cache_path}")
                return emb
        print(f"  Encoding {len(texts)} docs with {self._BASE} + proximity adapter on {self.device}...")
        emb = self._encode(texts, self._doc_adapters)
        emb = np.ascontiguousarray(emb, dtype=np.float32)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), emb)
        print(f"  Cached embeddings to {cache_path}")
        return emb

    def fit(self, texts):
        """Override to use L2 index (official Specter2 uses euclidean distance)."""
        self._load_model()
        texts_list = [self._format_text(t, is_query=False) for t in texts]
        embeddings = self._encode_corpus(texts_list)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dim)
        self._index.add(embeddings)
        return self

    def topk(self, query: str, k: int) -> list[int]:
        if self._index is None:
            raise RuntimeError("Call fit() before topk()")
        self._load_model()
        q = self._format_text(query, is_query=True)
        q_emb = self._encode([q], self._query_adapters).astype(np.float32)
        n = min(k, self._index.ntotal)
        _, indices = self._index.search(q_emb, n)
        return indices[0].tolist()


class BGERetriever(DenseRetriever):
    """BGE-small-en-v1.5 — retrieval-optimized model with query instruction prefix."""

    def __init__(self, **kwargs):
        super().__init__(
            model_name="BAAI/bge-small-en-v1.5",
            query_prompt="Represent this sentence for searching relevant passages: ",
            **kwargs,
        )


class MiniLMRetriever(DenseRetriever):
    """all-MiniLM-L6-v2 — fast general-purpose embedding model."""

    def __init__(self, **kwargs):
        super().__init__(
            model_name="sentence-transformers/all-MiniLM-L6-v2", **kwargs
        )
