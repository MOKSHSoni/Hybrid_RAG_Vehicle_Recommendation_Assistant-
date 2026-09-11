"""Phase 12: retrieval evaluation across configurations.

BM25 alone / Dense alone / Hybrid / Hybrid+Expansion / Hybrid+Transformation
/ Hybrid+HyDE / Hybrid+Cross-Encoder, each scored with Precision@K,
Recall@K, MRR, NDCG@K, and Retrieval Success Rate against the eval query
set's ground truth -- plus per-query latency.

Ground truth is vehicle NAMES (not chunk_ids), since that's what the eval
set records and what a human reviewing results actually judges against;
each configuration's raw per-chunk results are deduplicated to unique
vehicle names, in rank order, before scoring.

Queries whose expected_relevant_vehicles is empty (deliberately, e.g. the
"no_result" category) or contains a "needs_verification" placeholder are
excluded from these precision/recall/NDCG averages -- there's no usable
ground truth to score against yet. See evaluation/queries_v1.json's
per-entry notes for exactly which queries this affects; this is a known,
transparent limitation of a v1 DRAFT eval set, not something this module
should silently paper over.
"""

import time
from dataclasses import dataclass
from typing import Callable, List

import config
from src.evaluation.dataset import EvalQuery
from src.evaluation.metrics import mean, ndcg_at_k, precision_at_k, reciprocal_rank, recall_at_k, retrieval_success_rate

RetrieveFn = Callable[[EvalQuery], List[str]]


@dataclass
class ConfigEvalResult:
    config_name: str
    queries_evaluated: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    success_rate: float
    avg_latency_ms: float


def has_usable_ground_truth(query: EvalQuery) -> bool:
    if not query.expected_relevant_vehicles:
        return False
    return not any("needs_verification" in v for v in query.expected_relevant_vehicles)


def dedupe_names_preserve_order(names: List[str]) -> List[str]:
    seen = set()
    result = []
    for name in names:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def evaluate_configuration(
    config_name: str,
    retrieve_fn: RetrieveFn,
    queries: List[EvalQuery],
    k: int = config.RERANK_TOP_K,
) -> ConfigEvalResult:
    usable = [q for q in queries if has_usable_ground_truth(q)]

    precisions, recalls, rrs, ndcgs, latencies_ms = [], [], [], [], []
    success_pairs = []

    for q in usable:
        relevant = set(q.expected_relevant_vehicles)

        t0 = time.time()
        retrieved = dedupe_names_preserve_order(retrieve_fn(q))
        latencies_ms.append((time.time() - t0) * 1000)

        precisions.append(precision_at_k(retrieved, relevant, k))
        recalls.append(recall_at_k(retrieved, relevant, k))
        rrs.append(reciprocal_rank(retrieved, relevant))
        ndcgs.append(ndcg_at_k(retrieved, relevant, k))
        success_pairs.append((retrieved, relevant))

    return ConfigEvalResult(
        config_name=config_name,
        queries_evaluated=len(usable),
        precision_at_k=mean(precisions),
        recall_at_k=mean(recalls),
        mrr=mean(rrs),
        ndcg_at_k=mean(ndcgs),
        success_rate=retrieval_success_rate(success_pairs),
        avg_latency_ms=mean(latencies_ms),
    )


# ---------------------------------------------------------------------------
# RetrieveFn builders -- one per configuration under comparison
# ---------------------------------------------------------------------------


def bm25_only_fn(bm25_retriever, top_k: int) -> RetrieveFn:
    def fn(q: EvalQuery) -> List[str]:
        results = bm25_retriever.retrieve(q.query, top_k=top_k)
        return [r.chunk.metadata["name"] for r in results]

    return fn


def dense_only_fn(dense_retriever, top_k: int) -> RetrieveFn:
    def fn(q: EvalQuery) -> List[str]:
        results = dense_retriever.retrieve(q.query, top_k=top_k)
        return [r.chunk.metadata["name"] for r in results]

    return fn


def hybrid_fn(hybrid_retriever, top_k: int, alpha=None) -> RetrieveFn:
    def fn(q: EvalQuery) -> List[str]:
        results = hybrid_retriever.retrieve(q.query, top_k=top_k, alpha=alpha)
        return [r.chunk.metadata["name"] for r in results]

    return fn


def hybrid_plus_expansion_fn(hybrid_retriever, top_k: int) -> RetrieveFn:
    from src.query.expansion import expand_query
    from src.retrieval.merge import merge_and_deduplicate

    def fn(q: EvalQuery) -> List[str]:
        expanded = expand_query(q.query)
        all_queries = [q.query] + expanded
        result_lists = [hybrid_retriever.retrieve(variant, top_k=top_k) for variant in all_queries]
        merged = merge_and_deduplicate(result_lists)
        return [m.best_chunk.metadata["name"] for m in merged]

    return fn


def hybrid_plus_transformation_fn(hybrid_retriever, top_k: int) -> RetrieveFn:
    from src.query.transformation import transform_query

    def fn(q: EvalQuery) -> List[str]:
        transformed = transform_query(q.query)
        results = hybrid_retriever.retrieve(transformed, top_k=top_k)
        return [r.chunk.metadata["name"] for r in results]

    return fn


def hybrid_plus_hyde_fn(hybrid_retriever, hyde_retriever, top_k: int) -> RetrieveFn:
    from src.retrieval.merge import merge_and_deduplicate

    def fn(q: EvalQuery) -> List[str]:
        hybrid_results = hybrid_retriever.retrieve(q.query, top_k=top_k)
        hyde_results = hyde_retriever.retrieve(q.query, top_k=top_k)
        merged = merge_and_deduplicate([hybrid_results, hyde_results])
        return [m.best_chunk.metadata["name"] for m in merged]

    return fn


def hybrid_plus_cross_encoder_fn(hybrid_retriever, reranker, candidate_top_n: int, final_top_k: int) -> RetrieveFn:
    from src.retrieval.merge import merge_and_deduplicate

    def fn(q: EvalQuery) -> List[str]:
        raw = hybrid_retriever.retrieve(q.query, top_k=candidate_top_n)
        merged = merge_and_deduplicate([raw])
        reranked = reranker.rerank(q.query, merged, top_k=final_top_k)
        return [m.best_chunk.metadata["name"] for m in reranked]

    return fn
