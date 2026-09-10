"""Phase 1 deliverable: prove the knowledge base works end to end.

Query in, raw chunks out -- no reranking, no query understanding, no LLM.
That intelligence comes in later phases; this only proves the chunks,
embeddings, FAISS index, and BM25 index are correct and retrievable.
"""

from collections import Counter

import config
from src.embeddings.embedder import Embedder, normalize_embeddings
from src.pipeline import build_knowledge_base
from src.retrieval.dense.faiss_index import search as faiss_search


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    banner("PHASE 1 DEMO: Data Ingestion & Knowledge Base")

    print("\nBuilding knowledge base (ingest -> enrich -> chunk -> embed -> index)...")
    kb = build_knowledge_base()
    enriched = kb.enriched_documents
    chunk_store = kb.chunk_store

    # --- Step 2: enrichment sanity stats ---
    banner("Enrichment sanity stats")
    body_types = Counter(d.metadata["body_type"] for d in enriched)
    brands = Counter(d.metadata["brand"] for d in enriched)
    ev_count = sum(1 for d in enriched if d.metadata["is_electric"])
    missing_seating = sum(1 for d in enriched if d.metadata["seating_capacity"] is None)

    print(f"Vehicles enriched: {len(enriched)}")
    print(f"Body type distribution: {dict(body_types)}")
    print(f"Distinct brands: {len(brands)}")
    print(f"Electric vehicles: {ev_count} (expected 6)")
    print(f"Vehicles with unknown seating capacity: {missing_seating} (expected 2)")

    # --- Step 3: example chunks ---
    banner("Example chunks (Porsche Macan)")
    macan_chunks = [c for c in chunk_store.all_chunks() if c.doc_id == "cars_cleaned_0"]
    for c in macan_chunks:
        print(f"\n[{c.chunk_type}] chunk_id={c.chunk_id}")
        print(c.text)

    # --- Step 4: embedding shape ---
    banner("Embeddings")
    print(f"Total chunks embedded: {len(chunk_store)}")
    print(f"FAISS index size (ntotal): {kb.faiss_index.ntotal}")

    # --- Steps 5-7 already done inside build_knowledge_base(); confirm files ---
    banner("Saved index files")
    for path in (config.CHUNK_STORE_PATH, config.FAISS_INDEX_PATH, config.BM25_INDEX_PATH):
        size_kb = path.stat().st_size / 1024 if path.exists() else 0
        print(f"  {path}  ({size_kb:.1f} KB)")

    # --- Step 8: sample queries against both backends ---
    banner("Sample queries: Dense (FAISS) vs Sparse (BM25)")
    embedder = Embedder()
    queries = [
        "affordable 7 seater SUV with good mileage",
        "Porsche Cayenne price and top speed",
        "electric car with automatic transmission",
    ]

    for query in queries:
        print(f'\n--- Query: "{query}" ---')

        query_vector = normalize_embeddings(embedder.embed_texts([query]))[0]
        dense_results = faiss_search(kb.faiss_index, query_vector, config.DEFAULT_TOP_K)

        print(f"\n[Dense / FAISS] top {config.DEFAULT_TOP_K}:")
        for rank, (vec_idx, score) in enumerate(dense_results, start=1):
            chunk = chunk_store.get_by_vector_index(vec_idx)
            _print_result(rank, score, chunk)

        bm25_results = kb.bm25_index.search(query, config.DEFAULT_TOP_K)
        print(f"\n[Sparse / BM25] top {config.DEFAULT_TOP_K}:")
        for rank, (chunk_id, score) in enumerate(bm25_results, start=1):
            chunk = chunk_store.get_by_chunk_id(chunk_id)
            _print_result(rank, score, chunk)

    # --- Step 9: closing summary ---
    banner("Summary")
    print(f"Vehicles: {len(enriched)}")
    print(f"Chunks: {len(chunk_store)}")
    print("Knowledge base built successfully. No reranking, query understanding,")
    print("or LLM involved yet -- that intelligence is layered on in later phases.")


def _print_result(rank: int, score: float, chunk) -> None:
    name = chunk.metadata.get("name", "?")
    snippet = chunk.text[:100] + ("..." if len(chunk.text) > 100 else "")
    print(f"  #{rank}  score={score:.4f}  [{chunk.chunk_type}]  {name}  ({chunk.chunk_id})")
    print(f"       {snippet}")


if __name__ == "__main__":
    main()
