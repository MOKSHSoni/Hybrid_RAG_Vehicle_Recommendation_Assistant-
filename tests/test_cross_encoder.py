import pytest

import config
from src.reranking.cross_encoder import CrossEncoderReranker
from src.retrieval.merge import MergedResult


@pytest.fixture(scope="module")
def reranker(knowledge_base):
    return CrossEncoderReranker(knowledge_base.chunk_store)


def _merged(vehicle_id: str, name: str, score: float) -> MergedResult:
    chunk = type("FakeChunk", (), {})()  # placeholder; real tests use chunk_store lookups via vehicle_id
    return MergedResult(
        vehicle_id=vehicle_id,
        best_chunk=chunk,
        aggregate_score=score,
        supporting_chunks=[],
        match_count=1,
        methods={"hybrid"},
    )


def test_vehicle_text_combines_both_chunk_types(reranker):
    text = reranker._vehicle_text("cars_cleaned_0")  # Porsche Macan
    assert "Porsche Macan" in text
    assert "SUV" in text  # from product_overview
    assert "Peak power" in text  # from features


def test_rerank_empty_candidates_returns_empty(reranker):
    assert reranker.rerank("anything", []) == []


def test_rerank_enforces_candidate_cap(monkeypatch, reranker, knowledge_base):
    # Build more "candidates" than RERANK_CANDIDATE_TOP_N to prove the
    # cross-encoder is never handed the full set.
    all_vehicle_ids = list({c.doc_id for c in knowledge_base.chunk_store.all_chunks()})
    assert len(all_vehicle_ids) == 150
    candidates = [_merged(vid, vid, score=float(i)) for i, vid in enumerate(all_vehicle_ids)]

    seen_pair_counts = []
    real_predict = reranker._model.predict

    def spying_predict(pairs, *args, **kwargs):
        seen_pair_counts.append(len(pairs))
        return real_predict(pairs, *args, **kwargs)

    monkeypatch.setattr(reranker._model, "predict", spying_predict)
    reranker.rerank("affordable SUV", candidates, top_k=5)

    assert seen_pair_counts == [config.RERANK_CANDIDATE_TOP_N]  # never scored more than the cap


def test_rerank_real_query_ranks_exact_match_first(reranker, knowledge_base):
    from src.retrieval.bm25.retriever import BM25Retriever
    from src.retrieval.dense.retriever import DenseRetriever
    from src.retrieval.hybrid.hybrid_retriever import HybridRetriever
    from src.retrieval.merge import merge_and_deduplicate

    dense = DenseRetriever(knowledge_base.faiss_index, knowledge_base.chunk_store, knowledge_base.embedder)
    bm25 = BM25Retriever(knowledge_base.bm25_index, knowledge_base.chunk_store)
    hybrid = HybridRetriever(dense, bm25, knowledge_base.chunk_store)

    query = "Porsche Cayenne price and top speed"
    merged = merge_and_deduplicate([hybrid.retrieve(query, top_k=30)])

    results = reranker.rerank(query, merged, top_k=5)

    assert len(results) <= 5
    assert "Cayenne" in results[0].best_chunk.metadata["name"]  # Cayenne or Cayenne Coupe, both legitimate top matches
