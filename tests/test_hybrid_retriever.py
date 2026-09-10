import config
from src.retrieval.bm25.retriever import BM25Retriever
from src.retrieval.dense.retriever import DenseRetriever
from src.retrieval.hybrid.hybrid_retriever import HybridRetriever, _min_max_normalize
from src.query.metadata_filter import candidate_chunk_ids


def test_min_max_normalize_basic():
    result = _min_max_normalize({"a": 10.0, "b": 20.0, "c": 15.0})
    assert result["a"] == 0.0
    assert result["b"] == 1.0
    assert result["c"] == 0.5


def test_min_max_normalize_all_equal_returns_zero_not_nan():
    result = _min_max_normalize({"a": 5.0, "b": 5.0})
    assert result == {"a": 0.0, "b": 0.0}


def test_min_max_normalize_empty():
    assert _min_max_normalize({}) == {}


def _build_hybrid(knowledge_base) -> HybridRetriever:
    dense = DenseRetriever(knowledge_base.faiss_index, knowledge_base.chunk_store, knowledge_base.embedder)
    bm25 = BM25Retriever(knowledge_base.bm25_index, knowledge_base.chunk_store)
    return HybridRetriever(dense, bm25, knowledge_base.chunk_store)


def test_hybrid_retrieve_returns_ranked_results(knowledge_base):
    hybrid = _build_hybrid(knowledge_base)
    results = hybrid.retrieve("Porsche Cayenne price and top speed", top_k=5)

    assert len(results) == 5
    assert all(r.method == "hybrid" for r in results)
    assert [r.rank for r in results] == [1, 2, 3, 4, 5]
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert "Cayenne" in results[0].chunk.metadata["name"]


def test_hybrid_alpha_1_favors_bm25_ranking(knowledge_base):
    hybrid = _build_hybrid(knowledge_base)
    bm25 = BM25Retriever(knowledge_base.bm25_index, knowledge_base.chunk_store)

    hybrid_results = hybrid.retrieve("Porsche Cayenne price and top speed", top_k=5, alpha=1.0)
    bm25_results = bm25.retrieve("Porsche Cayenne price and top speed", top_k=5)

    assert [r.chunk.chunk_id for r in hybrid_results] == [r.chunk.chunk_id for r in bm25_results]


def test_hybrid_alpha_0_favors_dense_ranking(knowledge_base):
    hybrid = _build_hybrid(knowledge_base)
    dense = DenseRetriever(knowledge_base.faiss_index, knowledge_base.chunk_store, knowledge_base.embedder)

    hybrid_results = hybrid.retrieve("Porsche Cayenne price and top speed", top_k=5, alpha=0.0)
    dense_results = dense.retrieve("Porsche Cayenne price and top speed", top_k=5)

    assert [r.chunk.chunk_id for r in hybrid_results] == [r.chunk.chunk_id for r in dense_results]


def test_hybrid_retrieve_respects_candidate_restriction(knowledge_base):
    from src.query.models import Constraints

    hybrid = _build_hybrid(knowledge_base)
    porsche_ids = candidate_chunk_ids(knowledge_base.chunk_store, Constraints(brand="Porsche"))

    results = hybrid.retrieve("affordable SUV", top_k=20, candidate_chunk_ids=porsche_ids)

    assert len(results) == len(porsche_ids)  # only 12 Porsche chunks exist, all returned since top_k=20
    assert all(r.chunk.metadata["brand"] == "Porsche" for r in results)


def test_hybrid_default_alpha_from_config(knowledge_base):
    hybrid = _build_hybrid(knowledge_base)
    assert hybrid._alpha == config.HYBRID_FUSION_ALPHA
