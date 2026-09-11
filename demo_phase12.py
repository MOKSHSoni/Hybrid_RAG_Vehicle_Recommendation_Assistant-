"""Phase 12 deliverable: evaluation.

Runs the real pipeline against evaluation/queries_v2.json (config.EVAL_QUERIES_PATH
-- a DRAFT eval set, see its _meta.description) and reports:
  1. Query understanding accuracy (extracted vs expected constraints, per field).
  2. Retrieval quality across configurations: BM25 / Dense / Hybrid /
     Hybrid+Expansion / Hybrid+Transformation / Hybrid+HyDE / Hybrid+Cross-Encoder.
  3. Hybrid fusion weight sweep (config.HYBRID_FUSION_SWEEP_ALPHAS).
  4. Cross-encoder latency (real numbers, not a claim).

Only queries with usable (non-placeholder) expected_relevant_vehicles
contribute to the retrieval metrics -- see retrieval_eval.py's docstring.
Numbers here are only as good as the DRAFT ground truth; treat this run
as a demonstration that the evaluation machinery works end to end, not
as final, human-validated results.
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import time

import config
from src.evaluation.dataset import load_eval_queries
from src.evaluation.query_understanding_eval import evaluate_query_understanding
from src.evaluation.retrieval_eval import (
    bm25_only_fn,
    dense_only_fn,
    evaluate_configuration,
    has_usable_ground_truth,
    hybrid_fn,
    hybrid_plus_cross_encoder_fn,
    hybrid_plus_expansion_fn,
    hybrid_plus_hyde_fn,
    hybrid_plus_transformation_fn,
)
from src.pipeline import build_knowledge_base
from src.query.regex_extraction import known_brands_from_chunk_store
from src.reranking.cross_encoder import CrossEncoderReranker
from src.retrieval.bm25.retriever import BM25Retriever
from src.retrieval.dense.retriever import DenseRetriever
from src.retrieval.dense.hyde_retriever import HydeRetriever
from src.retrieval.hybrid.hybrid_retriever import HybridRetriever


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_config_result(result) -> None:
    print(
        f"  {result.config_name:28s}  P@K={result.precision_at_k:.3f}  R@K={result.recall_at_k:.3f}  "
        f"MRR={result.mrr:.3f}  NDCG@K={result.ndcg_at_k:.3f}  SuccessRate={result.success_rate:.3f}  "
        f"latency={result.avg_latency_ms:.0f}ms  (n={result.queries_evaluated})"
    )


def main() -> None:
    banner("PHASE 12 DEMO: Evaluation")

    queries = load_eval_queries()
    usable = [q for q in queries if has_usable_ground_truth(q)]
    print(f"\nLoaded {len(queries)} eval queries ({config.EVAL_QUERIES_PATH.name}, a DRAFT -- see _meta).")
    print(f"{len(usable)} have usable (non-placeholder) ground truth for retrieval metrics.")

    print("\nBuilding knowledge base...")
    kb = build_knowledge_base(save=False)
    known_brands = known_brands_from_chunk_store(kb.chunk_store)
    dense = DenseRetriever(kb.faiss_index, kb.chunk_store, kb.embedder)
    bm25 = BM25Retriever(kb.bm25_index, kb.chunk_store)
    hybrid = HybridRetriever(dense, bm25, kb.chunk_store)
    hyde = HydeRetriever(dense)
    reranker = CrossEncoderReranker(kb.chunk_store)

    banner("1. Query Understanding: Expected vs Extracted Constraints")
    field_accuracy = evaluate_query_understanding(queries, known_brands)
    for field, accuracy in sorted(field_accuracy.items()):
        print(f"  {field:20s} {accuracy:.1%}")

    banner("2. Retrieval quality across configurations (Top-5)")
    top_k = config.RERANK_TOP_K
    configurations = [
        ("BM25 alone", bm25_only_fn(bm25, top_k)),
        ("Dense alone", dense_only_fn(dense, top_k)),
        ("Hybrid", hybrid_fn(hybrid, top_k)),
        ("Hybrid + Expansion", hybrid_plus_expansion_fn(hybrid, top_k)),
        ("Hybrid + Transformation", hybrid_plus_transformation_fn(hybrid, top_k)),
        ("Hybrid + HyDE", hybrid_plus_hyde_fn(hybrid, hyde, top_k)),
        ("Hybrid + Cross-Encoder", hybrid_plus_cross_encoder_fn(hybrid, reranker, config.RERANK_CANDIDATE_TOP_N, top_k)),
    ]
    for name, fn in configurations:
        result = evaluate_configuration(name, fn, queries, k=top_k)
        print_config_result(result)

    banner("3. Hybrid fusion weight sweep (BM25/Dense)")
    for alpha in config.HYBRID_FUSION_SWEEP_ALPHAS:
        fn = hybrid_fn(hybrid, top_k, alpha=alpha)
        result = evaluate_configuration(f"alpha={alpha}", fn, queries, k=top_k)
        print(
            f"  {int(alpha*100):3d}/{int((1-alpha)*100):3d} BM25/Dense  P@K={result.precision_at_k:.3f}  "
            f"MRR={result.mrr:.3f}  NDCG@K={result.ndcg_at_k:.3f}"
        )

    banner("4. Cross-encoder checkpoint latency (real measurement)")
    sample = usable[0] if usable else queries[0]
    raw = hybrid.retrieve(sample.query, top_k=config.RERANK_CANDIDATE_TOP_N)
    from src.retrieval.merge import merge_and_deduplicate

    merged = merge_and_deduplicate([raw])
    t0 = time.time()
    reranker.rerank(sample.query, merged, top_k=top_k)
    elapsed = time.time() - t0
    print(f"\n  {config.CROSS_ENCODER_MODEL_NAME}: {elapsed:.2f}s for {len(merged)} (query, vehicle) pairs on CPU")

    banner("Summary")
    print("This run proves the evaluation machinery (metrics, config comparisons, weight")
    print("sweep) works end to end against real components. The DRAFT ground truth in")
    print(f"{config.EVAL_QUERIES_PATH.name} needs human review before these specific numbers")
    print("should be quoted as final results -- see the file's _meta and per-query notes")
    print("for exactly which entries are placeholders.")


if __name__ == "__main__":
    main()
