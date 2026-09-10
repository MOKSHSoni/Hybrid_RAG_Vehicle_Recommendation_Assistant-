"""Shared result type returned by every retrieval backend (dense, BM25,
and later hybrid/HyDE) so downstream phases (merge/dedup, reranking,
generation) can consume them uniformly regardless of source.
"""

from dataclasses import dataclass

from src.chunking.models import Chunk


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int
    method: str  # "dense" | "bm25" | "hybrid" | "hyde" (later phases)
