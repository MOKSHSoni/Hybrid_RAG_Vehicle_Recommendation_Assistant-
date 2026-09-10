"""Phase 9: cross-encoder reranking.

Fast Retrieval -> Top-N Candidates -> Cross-Encoder -> Accurate Relevance
Scores -> Final Top-K. Example concrete flow: 150 vehicles -> metadata
filtering -> BM25+Dense -> Top 30 -> Cross-Encoder -> Top 5.

GLOBAL RULE: the cross-encoder must never score the full vehicle set --
only a pre-filtered/retrieved candidate set. Enforced here, not just left
to caller discipline: if more than config.RERANK_CANDIDATE_TOP_N
candidates are passed in, only the top N (by their existing
aggregate_score, i.e. whatever Phase 7/8 already ranked them by) are ever
handed to the model.

Uses the Sentence Transformers CrossEncoder API directly (locked decision
-- not an sklearn substitute). Scores the FULL vehicle text (both
product_overview and features chunks via ChunkStore.get_by_doc_id, not
just whichever single chunk the upstream retrieval stage happened to
surface) against the query, for a more complete relevance judgment than
scoring one chunk alone would give.
"""

from dataclasses import replace
from typing import List

from sentence_transformers import CrossEncoder

import config
from src.retrieval.chunk_store import ChunkStore
from src.retrieval.merge import MergedResult

_CHUNK_TYPE_ORDER = ["product_overview", "features"]


class CrossEncoderReranker:
    def __init__(self, chunk_store: ChunkStore, model_name: str = config.CROSS_ENCODER_MODEL_NAME):
        self._chunk_store = chunk_store
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[MergedResult],
        top_k: int = config.RERANK_TOP_K,
    ) -> List[MergedResult]:
        if not candidates:
            return []

        # Enforce the global rule: never score more than RERANK_CANDIDATE_TOP_N,
        # regardless of how many candidates the caller passed in.
        bounded = sorted(candidates, key=lambda m: m.aggregate_score, reverse=True)[: config.RERANK_CANDIDATE_TOP_N]

        pairs = [(query, self._vehicle_text(m.vehicle_id)) for m in bounded]
        scores = self._model.predict(pairs)

        rescored = [replace(m, aggregate_score=float(score)) for m, score in zip(bounded, scores)]
        rescored.sort(key=lambda m: m.aggregate_score, reverse=True)
        return rescored[:top_k]

    def _vehicle_text(self, vehicle_id: str) -> str:
        chunks = self._chunk_store.get_by_doc_id(vehicle_id)
        by_type = {c.chunk_type: c.text for c in chunks}
        return " ".join(by_type[t] for t in _CHUNK_TYPE_ORDER if t in by_type)
