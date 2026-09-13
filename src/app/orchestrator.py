"""Phase 13: ties Phases 3-11 into one callable for the Streamlit app.

Default per-turn pipeline: context resolution -> constraint extraction ->
query transformation -> retrieve_with_relaxation (Phase 7 hybrid + Phase 9
cross-encoder + Phase 10 relaxation modes, already independently tested)
-> Phase 11 generation. Query expansion (Phase 4) and HyDE (Phase 5) are
NOT wired into this default path -- deliberately: HyDE is explicitly
documented as experimental in the project spec ("not wired into the
default retrieval path... until Phase 12 evaluation" -- see hyde.py), and
folding expansion into every relaxation attempt would multiply LLM calls
per turn without a corresponding evaluated benefit yet. Both remain fully
available: `include_expansion_hyde_debug=True` runs them informationally
(shown in the UI's debug panel) without changing the actual answer, and
demo_phase4.py/demo_phase5.py/demo_phase12.py exercise them directly.
"""

import time
from dataclasses import dataclass
from typing import FrozenSet, List, Optional

from src.app.logging_utils import PipelineLog
from src.pipeline import KnowledgeBase, build_knowledge_base
from src.query.expansion import expand_query
from src.query.hyde import generate_hypothetical_description
from src.query.models import ConversationTurn
from src.query.regex_extraction import known_brands_from_chunk_store
from src.query.scope import (
    OFF_TOPIC_MESSAGE,
    build_vehicle_vocabulary,
    is_off_topic,
    unknown_brand,
    unknown_brand_message,
)
from src.query.transformation import transform_query
from src.query.understanding import understand_query
from src.rag.generation import generate_answer
from src.reranking.cross_encoder import CrossEncoderReranker
from src.retrieval.bm25.retriever import BM25Retriever
from src.retrieval.dense.hyde_retriever import HydeRetriever
from src.retrieval.dense.retriever import DenseRetriever
from src.retrieval.hybrid.hybrid_retriever import HybridRetriever
from src.retrieval.modes import MODE_OUT_OF_SCOPE, RetrievalOutcome, retrieve_with_relaxation


@dataclass
class Pipeline:
    """The built knowledge base + retrievers, held once per app session --
    rebuilding per turn would reload the embedding/cross-encoder models."""

    kb: KnowledgeBase
    dense: DenseRetriever
    bm25: BM25Retriever
    hybrid: HybridRetriever
    hyde: HydeRetriever
    reranker: CrossEncoderReranker
    known_brands: List[str]
    # Built lazily from the knowledge base on first use (see
    # _scope_answer), so callers constructing a Pipeline directly need not
    # supply it.
    vehicle_vocabulary: Optional[FrozenSet[str]] = None

    @classmethod
    def build(cls) -> "Pipeline":
        kb = build_knowledge_base(save=False)
        dense = DenseRetriever(kb.faiss_index, kb.chunk_store, kb.embedder)
        bm25 = BM25Retriever(kb.bm25_index, kb.chunk_store)
        hybrid = HybridRetriever(dense, bm25, kb.chunk_store)
        hyde = HydeRetriever(dense)
        reranker = CrossEncoderReranker(kb.chunk_store)
        known_brands = known_brands_from_chunk_store(kb.chunk_store)
        return cls(kb=kb, dense=dense, bm25=bm25, hybrid=hybrid, hyde=hyde, reranker=reranker, known_brands=known_brands)


@dataclass
class TurnResult:
    answer: str
    outcome: RetrievalOutcome
    log: PipelineLog
    expansion_queries: Optional[List[str]] = None
    hyde_description: Optional[str] = None


def answer_query(
    pipeline: Pipeline,
    query: str,
    history: List[ConversationTurn],
    include_expansion_hyde_debug: bool = False,
    generate: bool = True,
) -> TurnResult:
    """Set generate=False to run everything except the final generation
    stage -- used by the Streamlit UI, which streams generation itself so
    the user sees prose as it arrives rather than waiting on the slowest
    stage in silence. The returned TurnResult then carries an empty
    `answer`, and the caller is responsible for producing one."""
    log = PipelineLog()

    t0 = time.time()
    understanding = understand_query(query, history, pipeline.known_brands)
    log.add_timing("query_understanding", (time.time() - t0) * 1000)
    log.extraction_method = understanding.extraction_method
    log.standalone_query = understanding.standalone_query

    # Refuse before retrieving, not after. Semantic fallback always returns
    # *something*, so an out-of-scope request would otherwise come back as
    # a list of confident-looking but unrelated recommendations. Checked on
    # the standalone query, so a follow-up like "what about cheaper ones"
    # is judged on its history-resolved meaning rather than its bare words.
    refusal = _scope_answer(pipeline, understanding)
    if refusal is not None:
        log.mode = MODE_OUT_OF_SCOPE
        return TurnResult(
            answer=refusal,
            outcome=RetrievalOutcome(
                mode=MODE_OUT_OF_SCOPE, results=[], original_constraints=understanding.constraints
            ),
            log=log,
        )

    t0 = time.time()
    transformed = transform_query(understanding.standalone_query)
    log.add_timing("transformation", (time.time() - t0) * 1000)
    log.transformed_query = transformed

    constraints = understanding.constraints
    log.constraints_summary = {
        "brand": constraints.brand,
        "fuel_types": constraints.fuel_types,
        "transmission": constraints.transmission,
        "seating_capacity": constraints.seating_capacity,
        "body_type": constraints.body_type,
        "price_max_lakhs": constraints.price_max_lakhs,
        "price_min_lakhs": constraints.price_min_lakhs,
        "numeric_ranges": constraints.numeric_ranges,
        "superlative_field": constraints.superlative_field,
        "superlative_direction": constraints.superlative_direction,
    }

    t0 = time.time()
    outcome = retrieve_with_relaxation(transformed, constraints, pipeline.kb.chunk_store, pipeline.hybrid, pipeline.reranker)
    log.add_timing("retrieval+relaxation", (time.time() - t0) * 1000)
    log.mode = outcome.mode
    log.relaxation_steps = [step.description for step in outcome.relaxation_steps]
    log.candidate_count = len(outcome.results)
    log.final_chunk_ids = [m.best_chunk.chunk_id for m in outcome.results]

    answer = ""
    if generate:
        t0 = time.time()
        answer = generate_answer(query, outcome, pipeline.kb.chunk_store)
        log.add_timing("generation", (time.time() - t0) * 1000)

    expansion_queries = None
    hyde_description = None
    if include_expansion_hyde_debug:
        t0 = time.time()
        expansion_queries = expand_query(transformed)
        log.add_timing("expansion (debug-only, not used in the answer)", (time.time() - t0) * 1000)

        t0 = time.time()
        hyde_description = generate_hypothetical_description(transformed)
        log.add_timing("hyde (debug-only, not used in the answer)", (time.time() - t0) * 1000)

    return TurnResult(
        answer=answer,
        outcome=outcome,
        log=log,
        expansion_queries=expansion_queries,
        hyde_description=hyde_description,
    )


def _scope_answer(pipeline: Pipeline, understanding) -> Optional[str]:
    """A refusal message if the request is outside the catalogue, else None."""
    brand = unknown_brand(understanding.constraints, pipeline.known_brands)
    if brand is not None:
        return unknown_brand_message(brand, pipeline.known_brands)
    if pipeline.vehicle_vocabulary is None:
        pipeline.vehicle_vocabulary = build_vehicle_vocabulary(pipeline.kb.chunk_store, pipeline.known_brands)
    if is_off_topic(understanding.standalone_query, understanding.constraints, pipeline.vehicle_vocabulary):
        return OFF_TOPIC_MESSAGE
    return None
