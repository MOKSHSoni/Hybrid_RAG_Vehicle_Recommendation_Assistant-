"""Phase 7 deliverable: hybrid retrieval.

Candidate Set (Phase 6) -> [BM25, Dense] -> Score Normalization ->
Weighted Fusion. Shows fusion at the default alpha, a weight sweep
(10/90 .. 50/50 BM25/Dense -- Phase 12 will pick the winner against real
metrics, not intuition), and fusion restricted to a Phase 6 candidate set.
"""

import config
from src.pipeline import build_knowledge_base
from src.query.metadata_filter import candidate_chunk_ids
from src.query.models import Constraints
from src.retrieval.bm25.retriever import BM25Retriever
from src.retrieval.dense.retriever import DenseRetriever
from src.retrieval.hybrid.hybrid_retriever import HybridRetriever


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_result(r) -> None:
    print(f"  #{r.rank}  score={r.score:.4f}  [{r.chunk.chunk_type}]  {r.chunk.metadata.get('name', '?')}")


def main() -> None:
    banner("PHASE 7 DEMO: Hybrid Retrieval (Score Normalization + Weighted Fusion)")

    print("\nBuilding knowledge base...")
    kb = build_knowledge_base(save=False)
    dense = DenseRetriever(kb.faiss_index, kb.chunk_store, kb.embedder)
    bm25 = BM25Retriever(kb.bm25_index, kb.chunk_store)
    hybrid = HybridRetriever(dense, bm25, kb.chunk_store)

    query = "affordable 7 seater SUV with good mileage"
    banner(f'Query: "{query}" -- hybrid at default alpha ({config.HYBRID_FUSION_ALPHA})')
    for r in hybrid.retrieve(query, top_k=config.DEFAULT_TOP_K):
        print_result(r)

    banner("Weight sweep (BM25/Dense) -- Phase 12 picks the winner against real metrics")
    for bm25_weight in config.HYBRID_FUSION_SWEEP_ALPHAS:
        results = hybrid.retrieve(query, top_k=3, alpha=bm25_weight)
        names = [r.chunk.metadata["name"] for r in results]
        print(f"  {int(bm25_weight*100):2d}/{int((1-bm25_weight)*100):2d}  top3: {names}")

    banner("Hybrid restricted to a Phase 6 candidate set (brand=Porsche)")
    constraints = Constraints(brand="Porsche")
    ids = candidate_chunk_ids(kb.chunk_store, constraints)
    print(f"\nCandidate set: {len(ids)} chunks (Porsche only)")
    query2 = "affordable 7 seater SUV with good mileage"
    print(f'Query: "{query2}" (note: no Porsche is a cheap 7-seater -- this proves restriction,')
    print("not relevance, since results are still forced to be Porsche-only)")
    for r in hybrid.retrieve(query2, top_k=5, candidate_chunk_ids=ids):
        print_result(r)
        assert r.chunk.metadata["brand"] == "Porsche"

    banner("Summary")
    print("Fusion is min-max normalized per query, then alpha-weighted. The default")
    print(f"alpha ({config.HYBRID_FUSION_ALPHA}) is a starting point, NOT assumed optimal --")
    print("Phase 12 sweeps config.HYBRID_FUSION_SWEEP_ALPHAS against the annotated eval set.")


if __name__ == "__main__":
    main()
