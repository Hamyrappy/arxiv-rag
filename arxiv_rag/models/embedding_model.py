import numpy as np
from typing import Iterable

class RAGModel:
    def __init__(self):
        ...

    def fit(self, texts: Iterable[str]) -> None:
        ...

    def search(self, text: str, k: int) -> list[int]:
        ...
    
    def embed_text(self, text: str) -> np.ndarray:
        ...

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        ...