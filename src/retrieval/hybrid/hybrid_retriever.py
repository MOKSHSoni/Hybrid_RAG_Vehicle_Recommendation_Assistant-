"""Phase 7: hybrid retrieval.

Candidate Set (Phase 6, optional) -> [BM25, Dense] -> Score Normalization
-> Weighted Fusion.

Both backends are asked to rank the ENTIRE corpus (cheap at 300 chunks),
then results are restricted to the candidate set (if given) before
normalizing and fusing -- this is what "search within the Phase 6
candidate set" actually means at this scale: correct (BM25's IDF stays
computed over the true full corpus, not skewed by rebuilding on a tiny
subset; dense similarity has no corpus-dependent term to skew) and far
simpler than maintaining a second, subset-specific index per query.

Score normalization is min-max per query (not global): BM25 scores are
unbounded, dense scores are ~[-1, 1] cosine similarities -- min-max maps
each backend's OWN result distribution for THIS query onto a comparable
[0, 1] scale before the weighted sum.
"""

from typing import Dict, List, Optional, Set

import config
from src.retrieval.bm25.retriever import BM25Retriever
from src.retrieval.chunk_store import ChunkStore
from src.retrieval.dense.retriever import DenseRetriever
from src.retrieval.models import RetrievalResult


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        chunk_store: ChunkStore,
        alpha: float = config.HYBRID_FUSION_ALPHA,
    ):
        self._dense = dense_retriever
        self._bm25 = bm25_retriever
        self._chunk_store = chunk_store
        self._alpha = alpha

    def retrieve(
        self,
        query: str,
        top_k: int = config.DEFAULT_TOP_K,
        candidate_chunk_ids: Optional[Set[str]] = None,
        alpha: Optional[float] = None,
    ) -> List[RetrievalResult]:
        alpha = alpha if alpha is not None else self._alpha
        corpus_size = len(self._chunk_store)

        dense_results = self._dense.retrieve(query, top_k=corpus_size)
        bm25_results = self._bm25.retrieve(query, top_k=corpus_size)

        if candidate_chunk_ids is not None:
            dense_results = [r for r in dense_results if r.chunk.chunk_id in candidate_chunk_ids]
            bm25_results = [r for r in bm25_results if r.chunk.chunk_id in candidate_chunk_ids]

        dense_norm = _min_max_normalize({r.chunk.chunk_id: r.score for r in dense_results})
        bm25_norm = _min_max_normalize({r.chunk.chunk_id: r.score for r in bm25_results})

        all_ids = set(dense_norm) | set(bm25_norm)
        fused = {cid: alpha * bm25_norm.get(cid, 0.0) + (1 - alpha) * dense_norm.get(cid, 0.0) for cid in all_ids}

        ranked_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:top_k]
        return [
            RetrievalResult(chunk=self._chunk_store.get_by_chunk_id(cid), score=fused[cid], rank=rank, method="hybrid")
            for rank, cid in enumerate(ranked_ids, start=1)
        ]


def _min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {k: 0.0 for k in scores}  # no signal (e.g. BM25 all-zero) -- constant offset, doesn't skew ranking
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}
