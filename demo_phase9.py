"""Phase 9 deliverable: cross-encoder reranking.

150 vehicles -> metadata filtering (Phase 6, skipped here for a
broad query) -> BM25+Dense hybrid (Phase 7) -> merge/dedup (Phase 8) ->
Top 30 candidates -> Cross-Encoder -> Top 5. Shows the reranked order vs.
the pre-rerank (hybrid+evidence) order side by side, since the whole
point of this phase is that the cross-encoder can reorder things.
"""

import time

import config
from src.pipeline import build_knowledge_base
from src.reranking.cross_encoder import CrossEncoderReranker
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
    banner("PHASE 9 DEMO: Cross-Encoder Reranking")

    print("\nBuilding knowledge base and loading the cross-encoder"
          f" ({config.CROSS_ENCODER_MODEL_NAME})...")
    kb = build_knowledge_base(save=False)
    dense = DenseRetriever(kb.faiss_index, kb.chunk_store, kb.embedder)
    bm25 = BM25Retriever(kb.bm25_index, kb.chunk_store)
    hybrid = HybridRetriever(dense, bm25, kb.chunk_store)
    reranker = CrossEncoderReranker(kb.chunk_store)

    query = "affordable 7 seater SUV with good mileage"
    banner(f'Query: "{query}"')

    candidates = merge_and_deduplicate([hybrid.retrieve(query, top_k=config.RERANK_CANDIDATE_TOP_N)])
    print(f"\nCandidates fed into the cross-encoder: {len(candidates)}"
          f" (config.RERANK_CANDIDATE_TOP_N = {config.RERANK_CANDIDATE_TOP_N}, never the full 150)")

    print(f"\nPre-rerank order (hybrid + evidence aggregate_score), top {config.RERANK_TOP_K}:")
    for m in candidates[: config.RERANK_TOP_K]:
        print(f"  {m.best_chunk.metadata['name']:30s}  score={m.aggregate_score:.4f}")

    t0 = time.time()
    reranked = reranker.rerank(query, candidates, top_k=config.RERANK_TOP_K)
    elapsed = time.time() - t0

    print(f"\nPost-rerank order (cross-encoder relevance score), top {config.RERANK_TOP_K}"
          f"  [{elapsed:.2f}s for {min(len(candidates), config.RERANK_CANDIDATE_TOP_N)} pairs]:")
    for m in reranked:
        print(f"  {m.best_chunk.metadata['name']:30s}  score={m.aggregate_score:.4f}")

    pre_order = [m.vehicle_id for m in candidates[: config.RERANK_TOP_K]]
    post_order = [m.vehicle_id for m in reranked]
    banner("Summary")
    print(f"Reordered: {pre_order != post_order}")
    print(f"Latency: {elapsed:.2f}s for {min(len(candidates), config.RERANK_CANDIDATE_TOP_N)} (query, vehicle) pairs")
    print("on CPU -- Phase 12 will benchmark this checkpoint's latency/quality formally")
    print("and confirm whether it's suitable for this retrieval task, per the project spec.")


if __name__ == "__main__":
    main()
