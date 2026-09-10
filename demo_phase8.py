"""Phase 8 deliverable: multi-query result merging & deduplication.

Ties together Phase 4 (transformation + expansion -> Q0..Q3) and Phase 7
(hybrid retrieval) to produce the genuinely overlapping multi-query
scenario Phase 8 exists to handle, then merges/dedups down to one ranked
entry per vehicle.
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config
from src.pipeline import build_knowledge_base
from src.query.expansion import expand_query
from src.query.transformation import transform_query
from src.retrieval.bm25.retriever import BM25Retriever
from src.retrieval.dense.retriever import DenseRetriever
from src.retrieval.hybrid.hybrid_retriever import HybridRetriever
from src.retrieval.merge import merge_and_deduplicate


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    banner("PHASE 8 DEMO: Multi-Query Result Merging & Deduplication")

    print("\nBuilding knowledge base...")
    kb = build_knowledge_base(save=False)
    dense = DenseRetriever(kb.faiss_index, kb.chunk_store, kb.embedder)
    bm25 = BM25Retriever(kb.bm25_index, kb.chunk_store)
    hybrid = HybridRetriever(dense, bm25, kb.chunk_store)

    original_query = "I want an affordable 7 seater SUV with good mileage please"

    banner(f'Original query: "{original_query}"')
    q0 = transform_query(original_query)
    expanded = expand_query(q0)
    all_queries = [q0] + expanded
    print(f"\nQ0 (transformed): {q0!r}")
    for i, q in enumerate(expanded, start=1):
        print(f"Q{i} (expanded):   {q!r}")

    print(f"\nRunning hybrid retrieval for all {len(all_queries)} query variants (top 10 each)...")
    result_lists = [hybrid.retrieve(q, top_k=10) for q in all_queries]
    total_raw_hits = sum(len(r) for r in result_lists)
    print(f"Total raw (query, chunk) hits before dedup: {total_raw_hits}")

    merged = merge_and_deduplicate(result_lists)
    print(f"After chunk-level + vehicle-level dedup: {len(merged)} unique vehicles")

    banner("Top merged results (evidence-boosted ranking)")
    for m in merged[:8]:
        methods = ",".join(sorted(m.methods))
        print(
            f"  {m.best_chunk.metadata['name']:30s}  aggregate_score={m.aggregate_score:.4f}  "
            f"match_count={m.match_count}  methods=[{methods}]  chunks={len(m.supporting_chunks)}"
        )

    banner("Summary")
    print(f"{len(all_queries)} query variants x up to 10 results each = up to {total_raw_hits} raw hits,")
    print(f"collapsed to {len(merged)} unique vehicles. Vehicles confirmed by more query")
    print("variants and/or both their chunks rank higher (match_count + evidence boost),")
    print("and no vehicle is ever shown twice just because multiple queries found it.")


if __name__ == "__main__":
    main()
