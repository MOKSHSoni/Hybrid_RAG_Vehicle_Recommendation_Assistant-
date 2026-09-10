"""BM25 sparse retrieval over the chunk corpus."""

import pickle
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Simple lowercase alphanumeric tokenizer.

    No stemming/stopword removal for Phase 1 -- fine at 300-document
    scale; a documented future improvement if the corpus grows.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, bm25: BM25Okapi, chunk_ids: List[str]):
        self._bm25 = bm25
        self._chunk_ids = chunk_ids

    @classmethod
    def build(cls, texts: List[str], chunk_ids: List[str]) -> "BM25Index":
        tokenized = [tokenize(t) for t in texts]
        return cls(BM25Okapi(tokenized), chunk_ids)

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self._chunk_ids[i], float(scores[i])) for i in top_indices]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunk_ids": self._chunk_ids}, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["bm25"], data["chunk_ids"])
