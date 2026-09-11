"""Phase 12: standard retrieval evaluation metrics.

Plain-Python implementations (binary relevance) -- no extra dependency
needed for metrics this standard. Every function takes a ranked list of
retrieved vehicle identifiers (names, in this project) and a set of
identifiers considered relevant (ground truth from the eval query set).
"""

import math
from typing import List, Sequence, Set, Tuple


def precision_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: Set[str]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Binary-relevance NDCG@K."""
    top_k = retrieved[:k]
    dcg = sum((1.0 if item in relevant else 0.0) / math.log2(i + 2) for i, item in enumerate(top_k))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def retrieval_success_rate(results: List[Tuple[Sequence[str], Set[str]]]) -> float:
    """Fraction of queries where at least one relevant vehicle appears
    anywhere in that query's retrieved list. `results` is a list of
    (retrieved, relevant) pairs, one per evaluated query."""
    if not results:
        return 0.0
    successes = sum(1 for retrieved, relevant in results if any(item in relevant for item in retrieved))
    return successes / len(results)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
