"""Phase 2 deliverable: Dense and BM25 retrieval baselines, evaluated
independently through the common DenseRetriever / BM25Retriever
interface that later phases (hybrid fusion, dedup, reranking) build on.
"""

import config
from src.pipeline import build_knowledge_base
from src.retrieval.bm25.retriever import BM25Retriever
from src.retrieval.dense.retriever import DenseRetriever

QUERIES = [
    "affordable 7 seater SUV with good mileage",
    "Porsche Cayenne price and top speed",
    "electric car with automatic transmission",
    "luxury convertible sports car",
    "budget hatchback under 10 lakh",
]


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    banner("PHASE 2 DEMO: Basic Retrieval Baseline (Dense vs BM25, independently)")

    print("\nLoading knowledge base (rebuilding from CSV; use demo_phase1.py's saved")
    print("index if you want to load instead of rebuild)...")
    kb = build_knowledge_base(save=False)

    dense = DenseRetriever(kb.faiss_index, kb.chunk_store, kb.embedder)
    bm25 = BM25Retriever(kb.bm25_index, kb.chunk_store)

    for query in QUERIES:
        banner(f'Query: "{query}"')

        print(f"\n[Dense] top {config.DEFAULT_TOP_K}:")
        for r in dense.retrieve(query, top_k=config.DEFAULT_TOP_K):
            _print_result(r)

        print(f"\n[BM25] top {config.DEFAULT_TOP_K}:")
        for r in bm25.retrieve(query, top_k=config.DEFAULT_TOP_K):
            _print_result(r)

    banner("Summary")
    print("Both retrievers ran independently through the same RetrievalResult")
    print("interface. This baseline is what Phase 7 (hybrid fusion) and Phase 12")
    print("(formal evaluation) will be compared against.")


def _print_result(result) -> None:
    name = result.chunk.metadata.get("name", "?")
    print(f"  #{result.rank}  score={result.score:.4f}  [{result.chunk.chunk_type}]  {name}")


if __name__ == "__main__":
    main()
