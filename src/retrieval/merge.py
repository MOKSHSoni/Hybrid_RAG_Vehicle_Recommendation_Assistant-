"""Phase 8: multi-query result merging & deduplication.

By this point there can be several result sets in flight for one user
turn: the (transformed) original query Q0, expanded queries Q1-Q3
(Phase 4), and optionally a HyDE pass (Phase 5) -- these overlap heavily,
and each vehicle has 2 chunks, so a vehicle can show up many times over.

Multiple Query Results -> Merge -> Chunk-level Dedup -> Vehicle-level
Dedup -> Evidence/Score Aggregation.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Set

import config
from src.chunking.models import Chunk
from src.retrieval.models import RetrievalResult


@dataclass
class MergedResult:
    vehicle_id: str
    best_chunk: Chunk  # highest-scoring chunk -- the representative "evidence" text
    aggregate_score: float
    supporting_chunks: List[Chunk] = field(default_factory=list)  # all unique chunks that matched, across queries
    match_count: int = 0  # total (query, chunk) hits across every query variant -- match-strength signal
    methods: Set[str] = field(default_factory=set)  # which retrieval methods contributed (dense/bm25/hybrid/hyde)


def merge_and_deduplicate(result_lists: List[List[RetrievalResult]]) -> List[MergedResult]:
    """Chunk-level dedup first (the same chunk_id retrieved by more than
    one query -> one entry, keeping its best score), then vehicle-level
    dedup (a vehicle's product_overview and features chunks collapse
    into one MergedResult), with an evidence-based score boost for
    vehicles confirmed by more matches.
    """
    chunk_best: Dict[str, RetrievalResult] = {}
    chunk_hit_count: Dict[str, int] = defaultdict(int)
    chunk_methods: Dict[str, Set[str]] = defaultdict(set)

    for results in result_lists:
        for r in results:
            cid = r.chunk.chunk_id
            chunk_hit_count[cid] += 1
            chunk_methods[cid].add(r.method)
            if cid not in chunk_best or r.score > chunk_best[cid].score:
                chunk_best[cid] = r

    by_vehicle: Dict[str, List[RetrievalResult]] = defaultdict(list)
    for cid, r in chunk_best.items():
        by_vehicle[r.chunk.doc_id].append(r)

    merged = []
    for vehicle_id, chunk_results in by_vehicle.items():
        chunk_results.sort(key=lambda r: r.score, reverse=True)
        best = chunk_results[0]
        total_hits = sum(chunk_hit_count[r.chunk.chunk_id] for r in chunk_results)
        methods = set().union(*(chunk_methods[r.chunk.chunk_id] for r in chunk_results))

        aggregate_score = best.score + config.DEDUP_EVIDENCE_BOOST * (total_hits - 1)

        merged.append(
            MergedResult(
                vehicle_id=vehicle_id,
                best_chunk=best.chunk,
                aggregate_score=aggregate_score,
                supporting_chunks=[r.chunk for r in chunk_results],
                match_count=total_hits,
                methods=methods,
            )
        )

    merged.sort(key=lambda m: m.aggregate_score, reverse=True)
    return merged


def retrieve_multi_query(
    retrieve_fn: Callable[[str, int], List[RetrievalResult]],
    queries: List[str],
    top_k_per_query: int = config.DEFAULT_TOP_K,
) -> List[MergedResult]:
    """Convenience wrapper: run the same retriever across several query
    variants (Q0/transformed + Q1-Q3, optionally + a HyDE pass appended
    by the caller) and merge/dedup the results. `retrieve_fn` is any
    retriever's .retrieve method (Dense/BM25/Hybrid/HyDE all share the
    (query, top_k) -> List[RetrievalResult] signature)."""
    result_lists = [retrieve_fn(q, top_k_per_query) for q in queries]
    return merge_and_deduplicate(result_lists)
