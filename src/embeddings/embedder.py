"""Sentence embedding generation."""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

import config


class Embedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str], batch_size: int = config.EMBEDDING_BATCH_SIZE) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors.astype(np.float32))


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row so inner product == cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # guard degenerate all-zero rows
    return np.ascontiguousarray((vectors / norms).astype(np.float32))
