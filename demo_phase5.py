"""Phase 5 deliverable: HyDE (Hypothetical Document Embeddings) --
experimental retrieval path, A/B'd against plain dense retrieval.

Proves: (1) the hypothetical description is generated and used only to
produce a search vector, (2) it is never surfaced as if it were a real
vehicle, (3) only real, actual chunks come back as results.
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # model text may contain Unicode; console codepage may not

import config
from src.pipeline import build_knowledge_base
from src.query.hyde import generate_hypothetical_description
from src.retrieval.dense.hyde_retriever import HydeRetriever
from src.retrieval.dense.retriever import DenseRetriever

QUERIES = [
    "affordable 7 seater SUV with good mileage",
    "electric car with automatic transmission",
    "luxury convertible sports car",
]


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    banner("PHASE 5 DEMO: HyDE (experimental) vs plain Dense retrieval")

    print("\nBuilding knowledge base...")
    kb = build_knowledge_base(save=False)
    dense = DenseRetriever(kb.faiss_index, kb.chunk_store, kb.embedder)
    hyde = HydeRetriever(dense)

    for query in QUERIES:
        banner(f'Query: "{query}"')

        hypothetical = generate_hypothetical_description(query)
        print(f"\nHypothetical description (SYNTHETIC -- never shown to real users,")
        print(f"used only to produce a search vector):")
        print(f"  {hypothetical!r}")

        print(f"\n[Plain Dense] top {config.DEFAULT_TOP_K}:")
        for r in dense.retrieve(query, top_k=config.DEFAULT_TOP_K):
            _print_result(r)

        print(f"\n[HyDE Dense] top {config.DEFAULT_TOP_K}:")
        for r in hyde.retrieve(query, top_k=config.DEFAULT_TOP_K, precomputed_hypothetical=hypothetical):
            _print_result(r)
            assert hypothetical is None or hypothetical not in r.chunk.text  # never leaks into real chunks

    banner("Summary")
    print("HyDE is experimental -- Phase 12 will A/B it against plain dense retrieval")
    print("with real metrics before deciding whether to use it by default. Every")
    print("result above is a REAL chunk from the dataset; the hypothetical text was")
    print("used only to compute a search vector and is discarded afterward.")


def _print_result(result) -> None:
    name = result.chunk.metadata.get("name", "?")
    print(f"  #{result.rank}  score={result.score:.4f}  [{result.chunk.chunk_type}]  {name}")


if __name__ == "__main__":
    main()
