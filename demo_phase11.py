"""Phase 11 deliverable: final RAG generation.

Top-K Actual Vehicle Chunks -> Context Builder -> Grounded Qwen3 Prompt ->
Qwen3/Ollama -> Final Answer. Runs the full pipeline (Phase 6 filtering ->
Phase 7 hybrid -> Phase 9 reranking -> Phase 10 relaxation -> Phase 11
generation) end to end for an Exact and a Relaxed scenario, plus proves
the graceful-error path when Ollama is unreachable.
"""

import sys
from unittest.mock import patch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.pipeline import build_knowledge_base
from src.query.models import Constraints
from src.query.ollama_client import OllamaError
from src.rag.generation import generate_answer
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


def run_case(title: str, query: str, constraints: Constraints, chunk_store, hybrid, reranker) -> None:
    banner(title)
    outcome = retrieve_with_relaxation(query, constraints, chunk_store, hybrid, reranker)
    print(f'\nQuery: "{query}"')
    print(f"Constraints: {constraints.populated_fields()}")
    print(f"Mode: {outcome.mode.upper()}")
    if outcome.relaxation_steps:
        for step in outcome.relaxation_steps:
            print(f"  relaxed: [{step.field}] {step.description}")

    answer = generate_answer(query, outcome, chunk_store)
    print("\nGenerated answer:")
    print(answer)


def main() -> None:
    banner("PHASE 11 DEMO: Final RAG Generation")

    print("\nBuilding knowledge base...")
    kb = build_knowledge_base(save=False)
    dense = DenseRetriever(kb.faiss_index, kb.chunk_store, kb.embedder)
    bm25 = BM25Retriever(kb.bm25_index, kb.chunk_store)
    hybrid = HybridRetriever(dense, bm25, kb.chunk_store)
    reranker = CrossEncoderReranker(kb.chunk_store)

    run_case(
        "Case 1: Exact match",
        "affordable SUV under 15 lakh",
        Constraints(body_type="SUV", price_max_lakhs=15.0),
        kb.chunk_store,
        hybrid,
        reranker,
    )

    run_case(
        "Case 2: Relaxed (Ferrari + 7 seats -- no Ferrari seats 7)",
        "a Ferrari that fits the whole family",
        Constraints(brand="Ferrari", seating_capacity=7),
        kb.chunk_store,
        hybrid,
        reranker,
    )

    banner("Case 3: graceful error when Ollama is unreachable")
    outcome = retrieve_with_relaxation(
        "affordable SUV", Constraints(body_type="SUV", price_max_lakhs=15.0), kb.chunk_store, hybrid, reranker
    )
    with patch("src.rag.generation.chat_text", side_effect=OllamaError("simulated: connection refused")):
        answer = generate_answer("affordable SUV", outcome, kb.chunk_store)
    print("\nGenerated answer (Ollama simulated down):")
    print(answer)

    banner("Summary")
    print("Every answer is grounded in retrieved chunk text, states its mode explicitly,")
    print("and a simulated Ollama outage produces a graceful message (with the raw")
    print("results still surfaced) instead of crashing.")


if __name__ == "__main__":
    main()
