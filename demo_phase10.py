"""Phase 10 deliverable: constraint relaxation & fallback.

Demonstrates all three modes with real, deliberately-chosen scenarios:
  Mode A (Exact)    -- a satisfiable constraint combination.
  Mode B (Relaxed)  -- brand=Ferrari + seating_capacity=7 (every Ferrari in
                       the dataset is a 2-seater, so this only succeeds
                       once brand is relaxed away).
  Mode C (Fallback) -- an empty Constraints object (nothing to filter on),
                       pure semantic retrieval.
"""

from src.pipeline import build_knowledge_base
from src.query.models import Constraints
from src.reranking.cross_encoder import CrossEncoderReranker
from src.retrieval.bm25.retriever import BM25Retriever
from src.retrieval.dense.retriever import DenseRetriever
from src.retrieval.hybrid.hybrid_retriever import HybridRetriever
from src.retrieval.modes import retrieve_with_relaxation


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def show_outcome(query: str, constraints: Constraints, outcome) -> None:
    print(f'\nQuery: "{query}"')
    print(f"Constraints: {constraints.populated_fields()}")
    print(f"MODE: {outcome.mode.upper()}")
    if outcome.relaxation_steps:
        print("Relaxation steps tried:")
        for step in outcome.relaxation_steps:
            print(f"  - [{step.field}] {step.description}")
    print(f"Results ({len(outcome.results)}):")
    for m in outcome.results[:5]:
        print(f"  - {m.best_chunk.metadata['name']}  (score={m.aggregate_score:.4f})")


def main() -> None:
    banner("PHASE 10 DEMO: Constraint Relaxation & Fallback")

    print("\nBuilding knowledge base...")
    kb = build_knowledge_base(save=False)
    dense = DenseRetriever(kb.faiss_index, kb.chunk_store, kb.embedder)
    bm25 = BM25Retriever(kb.bm25_index, kb.chunk_store)
    hybrid = HybridRetriever(dense, bm25, kb.chunk_store)
    reranker = CrossEncoderReranker(kb.chunk_store)

    banner("Mode A: Exact (satisfiable constraints)")
    constraints_a = Constraints(brand="Toyota", body_type="SUV")
    outcome_a = retrieve_with_relaxation("SUV", constraints_a, kb.chunk_store, hybrid, reranker)
    show_outcome("SUV", constraints_a, outcome_a)
    assert outcome_a.mode == "exact"

    banner("Mode B: Relaxed (Ferrari + 7 seats -- no Ferrari in the dataset seats 7)")
    constraints_b = Constraints(brand="Ferrari", seating_capacity=7)
    outcome_b = retrieve_with_relaxation("sports car", constraints_b, kb.chunk_store, hybrid, reranker)
    show_outcome("sports car", constraints_b, outcome_b)
    assert outcome_b.mode == "relaxed"

    banner("Mode B: Relaxed with price widening (unrealistically low budget for an SUV)")
    constraints_b2 = Constraints(body_type="SUV", price_max_lakhs=2.0)  # cheapest SUV is ~6.8 Lakh
    outcome_b2 = retrieve_with_relaxation("SUV", constraints_b2, kb.chunk_store, hybrid, reranker)
    show_outcome("SUV", constraints_b2, outcome_b2)

    banner("Mode C: Semantic fallback (no constraints extracted at all)")
    constraints_c = Constraints()
    outcome_c = retrieve_with_relaxation("something fun to drive on weekends", constraints_c, kb.chunk_store, hybrid, reranker)
    show_outcome("something fun to drive on weekends", constraints_c, outcome_c)
    assert outcome_c.mode == "fallback"

    banner("Summary")
    print("Mode A never touches relaxation. Mode B reports exactly which constraints")
    print("were relaxed/widened -- Phase 11's generation prompt will use this to avoid")
    print("ever claiming a relaxed/fallback result satisfies an unverified constraint.")


if __name__ == "__main__":
    main()
