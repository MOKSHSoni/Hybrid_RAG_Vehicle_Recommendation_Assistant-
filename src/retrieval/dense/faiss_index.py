"""Dense retrieval via a FAISS IndexFlatIP.

Normalized vectors + inner product = exact cosine similarity. No
training step needed (flat index).
"""

from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors.astype(np.float32)))
    return index


def search(index: faiss.Index, query_vector: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
    query = np.ascontiguousarray(query_vector.reshape(1, -1).astype(np.float32))
    scores, indices = index.search(query, top_k)
    return list(zip(indices[0].tolist(), scores[0].tolist()))


def save_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))
