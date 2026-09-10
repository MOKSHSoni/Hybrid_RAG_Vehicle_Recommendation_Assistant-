"""Phase 10: constraint relaxation & fallback.

Three explicit retrieval modes:

Mode A -- Exact: Query -> Filters -> Retrieval -> Reranking -> Results.
All (populated) constraints satisfied.

Mode B -- Relaxed: triggered when Mode A returns nothing. Constraints are
dropped one at a time, in config.CONSTRAINT_RELAXATION_ORDER (soft/lowest
priority first; seating_capacity, the one hard constraint, is relaxed
last). Price fields are the exception to "drop outright": they widen
through real dataset price-quantile boundaries (config.PRICE_RELAXATION_*)
a step at a time before finally being dropped, since a price ceiling has a
natural notion of "a bit further" that a categorical field (brand,
body_type, ...) doesn't.

Mode C -- Semantic fallback: triggered when constraint extraction found
nothing to filter on in the first place, OR relaxation exhausts every
constraint with zero results. Pure semantic retrieval, no filtering.

GLOBAL RULE: fallback (and relaxed) results must be explicitly labeled as
such by the caller (see RetrievalOutcome.mode) and must never be presented
as satisfying constraints that were never verified -- this module never
claims a Mode B/C result satisfies the original constraints; it reports
exactly what was relaxed/dropped so Phase 11's generation prompt can say
so honestly.
"""

from dataclasses import dataclass, field, replace
from typing import List, Optional

import numpy as np

import config
from src.query.metadata_filter import candidate_chunk_ids, unique_vehicles
from src.query.models import Constraints
from src.reranking.cross_encoder import CrossEncoderReranker
from src.retrieval.chunk_store import ChunkStore
from src.retrieval.hybrid.hybrid_retriever import HybridRetriever
from src.retrieval.merge import MergedResult, merge_and_deduplicate

MODE_EXACT = "exact"
MODE_RELAXED = "relaxed"
MODE_FALLBACK = "fallback"


@dataclass
class RelaxationStep:
    field: str
    description: str  # human-readable, e.g. "price_max_lakhs widened to Rs 25.0 Lakh"


@dataclass
class RetrievalOutcome:
    mode: str  # "exact" | "relaxed" | "fallback"
    results: List[MergedResult]
    original_constraints: Constraints
    relaxation_steps: List[RelaxationStep] = field(default_factory=list)  # every step TRIED, in order (not just the winner)


def retrieve_with_relaxation(
    query: str,
    constraints: Constraints,
    chunk_store: ChunkStore,
    hybrid_retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
    all_prices: Optional[List[float]] = None,
) -> RetrievalOutcome:
    if constraints.is_empty():
        results = _search(query, None, chunk_store, hybrid_retriever, reranker)
        return RetrievalOutcome(mode=MODE_FALLBACK, results=results, original_constraints=constraints)

    # Mode A: Exact.
    results = _search(query, constraints, chunk_store, hybrid_retriever, reranker)
    if results:
        return RetrievalOutcome(mode=MODE_EXACT, results=results, original_constraints=constraints)

    if all_prices is None:
        all_prices = [v["price_lakhs"] for v in unique_vehicles(chunk_store) if v.get("price_lakhs") is not None]

    # Mode B: Relaxed, one field at a time (price fields step through
    # several widenings before being dropped -- see _relaxation_steps_for_field).
    working = constraints
    steps_tried: List[RelaxationStep] = []
    for field_name in config.CONSTRAINT_RELAXATION_ORDER:
        if field_name not in working.populated_fields():
            continue
        for trial, description in _relaxation_steps_for_field(working, field_name, all_prices):
            steps_tried.append(RelaxationStep(field=field_name, description=description))
            results = _search(query, trial, chunk_store, hybrid_retriever, reranker)
            working = trial  # carry the relaxed state forward regardless of outcome
            if results:
                return RetrievalOutcome(
                    mode=MODE_RELAXED, results=results, original_constraints=constraints, relaxation_steps=steps_tried
                )

    # Mode C: Semantic fallback -- relaxation exhausted every constraint, still nothing.
    results = _search(query, None, chunk_store, hybrid_retriever, reranker)
    return RetrievalOutcome(mode=MODE_FALLBACK, results=results, original_constraints=constraints, relaxation_steps=steps_tried)


def _search(
    query: str,
    constraints: Optional[Constraints],
    chunk_store: ChunkStore,
    hybrid_retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
) -> List[MergedResult]:
    candidate_ids = candidate_chunk_ids(chunk_store, constraints) if constraints is not None else None
    raw = hybrid_retriever.retrieve(query, top_k=config.RERANK_CANDIDATE_TOP_N, candidate_chunk_ids=candidate_ids)
    merged = merge_and_deduplicate([raw])
    if not merged:
        return []
    return reranker.rerank(query, merged, top_k=config.RERANK_TOP_K)


def _relaxation_steps_for_field(constraints: Constraints, field_name: str, all_prices: List[float]):
    """Yields (relaxed_constraints, description) pairs for one field, from
    least to most relaxed. Categorical fields yield exactly one step (drop
    outright); price fields yield up to PRICE_RELAXATION_MAX_STEPS
    quantile-widened steps before a final full-drop step."""
    if field_name == "price_max_lakhs" and constraints.price_max_lakhs is not None:
        current = constraints.price_max_lakhs
        for step in range(1, config.PRICE_RELAXATION_MAX_STEPS + 1):
            new_max = _next_price_boundary_up(current, all_prices, step)
            if new_max is None:
                break
            yield replace(constraints, price_max_lakhs=new_max), f"price_max_lakhs widened to Rs {new_max:.2f} Lakh"
        yield replace(constraints, price_max_lakhs=None), "price_max_lakhs removed (widening exhausted)"

    elif field_name == "price_min_lakhs" and constraints.price_min_lakhs is not None:
        current = constraints.price_min_lakhs
        for step in range(1, config.PRICE_RELAXATION_MAX_STEPS + 1):
            new_min = _next_price_boundary_down(current, all_prices, step)
            if new_min is None:
                break
            yield replace(constraints, price_min_lakhs=new_min), f"price_min_lakhs lowered to Rs {new_min:.2f} Lakh"
        yield replace(constraints, price_min_lakhs=None), "price_min_lakhs removed (widening exhausted)"

    else:
        current_value = getattr(constraints, field_name, None)
        if current_value:  # non-None, non-empty-list
            empty_value = [] if field_name == "fuel_types" else None
            yield replace(constraints, **{field_name: empty_value}), f"{field_name} removed"


def _price_quantile_boundaries(prices: List[float]) -> List[float]:
    if not prices:
        return []
    levels = np.arange(config.PRICE_RELAXATION_QUANTILE_STEP, 1.0 + 1e-9, config.PRICE_RELAXATION_QUANTILE_STEP)
    levels = np.clip(levels, 0.0, 1.0)
    return sorted({float(np.quantile(prices, q)) for q in levels})


def _next_price_boundary_up(current_max: float, prices: List[float], step: int) -> Optional[float]:
    candidates = [b for b in _price_quantile_boundaries(prices) if b > current_max]
    if not candidates:
        return None
    return candidates[min(step - 1, len(candidates) - 1)]


def _next_price_boundary_down(current_min: float, prices: List[float], step: int) -> Optional[float]:
    candidates = sorted((b for b in _price_quantile_boundaries(prices) if b < current_min), reverse=True)
    if not candidates:
        return None
    return candidates[min(step - 1, len(candidates) - 1)]
