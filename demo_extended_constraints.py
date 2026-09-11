"""Demo: extended numeric constraints, superlative ranking, and result
comparison -- an extension beyond the original 14-phase spec (which only
reserved "Phase 14" for optional Langfuse tracing).

Exercises:
  1. A range constraint on a field that previously had no filtering at
     all (boot space), extracted via regex and enforced by the existing
     metadata filter.
  2. A superlative query ("cheapest Lamborghini"), answered by a direct
     metadata sort rather than semantic retrieval -- verified against
     real prices confirmed during Phase 10 development (Urus Rs 310L <
     Huracan Rs 322L).
  3. A multi-result query rendering the new LLM-independent comparison
     table.
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.pipeline import build_knowledge_base
from src.query.models import Constraints
from src.query.regex_extraction import extract_regex_constraints, known_brands_from_chunk_store
from src.rag.comparison import build_comparison_rows
from src.rag.context_builder import build_context_block
from src.reranking.cross_encoder import CrossEncoderReranker
from src.retrieval.bm25.retriever import BM25Retriever
from src.retrieval.dense.retriever import DenseRetriever
from src.retrieval.hybrid.hybrid_retriever import HybridRetriever
from src.retrieval.modes import retrieve_with_relaxation


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    banner("DEMO: Extended Numeric Constraints, Superlative Ranking & Comparison")

    print("\nBuilding knowledge base...")
    kb = build_knowledge_base(save=False)
    known_brands = known_brands_from_chunk_store(kb.chunk_store)
    dense = DenseRetriever(kb.faiss_index, kb.chunk_store, kb.embedder)
    bm25 = BM25Retriever(kb.bm25_index, kb.chunk_store)
    hybrid = HybridRetriever(dense, bm25, kb.chunk_store)
    reranker = CrossEncoderReranker(kb.chunk_store)

    banner("1. Numeric range constraint: 'SUV with boot space over 400 liters'")
    query1 = "SUV with boot space over 400 liters"
    constraints1 = extract_regex_constraints(query1, known_brands)
    print(f"\nExtracted constraints: body_type={constraints1.body_type}, numeric_ranges={constraints1.numeric_ranges}")
    outcome1 = retrieve_with_relaxation(query1, constraints1, kb.chunk_store, hybrid, reranker)
    print(f"Mode: {outcome1.mode}")
    for m in outcome1.results:
        print(f"  - {m.best_chunk.metadata['name']}  boot_space_l={m.best_chunk.metadata.get('boot_space_l')}")

    banner("2. Superlative: 'cheapest Lamborghini'")
    query2 = "cheapest Lamborghini"
    constraints2 = extract_regex_constraints(query2, known_brands)
    print(f"\nExtracted: brand={constraints2.brand}, superlative_field={constraints2.superlative_field}, "
          f"direction={constraints2.superlative_direction}")
    outcome2 = retrieve_with_relaxation(query2, constraints2, kb.chunk_store, hybrid, reranker)
    print(f"Mode: {outcome2.mode}")
    for m in outcome2.results:
        print(f"  - {m.best_chunk.metadata['name']}  price_lakhs={m.best_chunk.metadata.get('price_lakhs')}")
    assert outcome2.mode == "superlative"
    assert outcome2.results[0].best_chunk.metadata["name"] == "Lamborghini Urus"
    print("Verified: Urus (cheaper) correctly ranked before Huracan -- exact sort, not relevance guess.")

    banner("3. Comparison table for a multi-result query")
    query3 = "SUV under 20 lakh"
    constraints3 = extract_regex_constraints(query3, known_brands)
    outcome3 = retrieve_with_relaxation(query3, constraints3, kb.chunk_store, hybrid, reranker)
    print(f"\nMode: {outcome3.mode}, results: {len(outcome3.results)}")
    rows = build_comparison_rows(outcome3)
    if rows:
        print("\nComparison rows (also what streamlit_app.py's Compare expander renders):")
        for row in rows:
            print(f"  {row}")
    context = build_context_block(outcome3, kb.chunk_store)
    print(f"\nContext block includes 'COMPARISON TABLE': {'COMPARISON TABLE' in context}")

    banner("Summary")
    print("Extended numeric fields (top speed, boot space, ground clearance, mileage,")
    print("engine size) now participate in real filtering, not just semantic guessing.")
    print("Superlatives bypass retrieval entirely for the ranking decision -- an exact")
    print("metadata sort, verified against known real prices. The comparison table is")
    print("built straight from metadata, so its numbers can never be garbled by generation.")


if __name__ == "__main__":
    main()
