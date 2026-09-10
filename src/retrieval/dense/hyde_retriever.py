"""HyDE-augmented dense retrieval: embeds a synthetic hypothetical
description (see src/query/hyde.py) instead of the raw query, then reuses
DenseRetriever's real FAISS search unchanged. Only real Chunks come back
-- the hypothetical text itself is discarded after producing the search
vector and is never part of the result.
"""

from dataclasses import replace
from typing import List, Optional

import config
from src.query.hyde import generate_hypothetical_description
from src.retrieval.dense.retriever import DenseRetriever
from src.retrieval.models import RetrievalResult


class HydeRetriever:
    def __init__(self, dense_retriever: DenseRetriever):
        self._dense_retriever = dense_retriever

    def retrieve(
        self,
        query: str,
        top_k: int = config.DEFAULT_TOP_K,
        precomputed_hypothetical: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """`precomputed_hypothetical` lets a caller that already generated
        the description (e.g. to display it) pass it through instead of
        triggering a second, redundant LLM call for the same query."""
        hypothetical = (
            precomputed_hypothetical if precomputed_hypothetical is not None else generate_hypothetical_description(query)
        )
        search_text = hypothetical if hypothetical else query  # graceful fallback: plain dense retrieval
        results = self._dense_retriever.retrieve(search_text, top_k)
        return [replace(r, method="hyde") for r in results]
