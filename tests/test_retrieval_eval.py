from src.evaluation.dataset import EvalQuery
from src.evaluation.retrieval_eval import (
    ConfigEvalResult,
    bm25_only_fn,
    dedupe_names_preserve_order,
    evaluate_configuration,
    has_usable_ground_truth,
)
from src.retrieval.bm25.retriever import BM25Retriever


def _query(expected_relevant_vehicles) -> EvalQuery:
    return EvalQuery(
        id="test",
        category="test",
        query="Porsche Cayenne price and top speed",
        history=[],
        expected_constraints={},
        expected_relevant_vehicles=expected_relevant_vehicles,
        expected_mode="exact",
    )


def test_dedupe_names_preserve_order():
    assert dedupe_names_preserve_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_has_usable_ground_truth():
    assert has_usable_ground_truth(_query(["Porsche Cayenne"])) is True
    assert has_usable_ground_truth(_query([])) is False
    assert has_usable_ground_truth(_query(["needs_verification: something"])) is False


def test_evaluate_configuration_real_bm25(knowledge_base):
    bm25 = BM25Retriever(knowledge_base.bm25_index, knowledge_base.chunk_store)
    fn = bm25_only_fn(bm25, top_k=5)
    queries = [_query(["Porsche Cayenne", "Porsche Cayenne Coupe"])]

    result = evaluate_configuration("bm25", fn, queries, k=5)

    assert isinstance(result, ConfigEvalResult)
    assert result.queries_evaluated == 1
    assert result.precision_at_k > 0  # Cayenne should be found by BM25 for this exact-entity query
    assert result.avg_latency_ms >= 0


def test_evaluate_configuration_skips_queries_without_ground_truth(knowledge_base):
    bm25 = BM25Retriever(knowledge_base.bm25_index, knowledge_base.chunk_store)
    fn = bm25_only_fn(bm25, top_k=5)
    queries = [_query([]), _query(["needs_verification: x"])]

    result = evaluate_configuration("bm25", fn, queries, k=5)

    assert result.queries_evaluated == 0
    assert result.precision_at_k == 0.0
