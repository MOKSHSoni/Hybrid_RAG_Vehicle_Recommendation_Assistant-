from src.chunking.models import Chunk
from src.retrieval.bm25.retriever import BM25Retriever
from src.retrieval.dense.retriever import DenseRetriever
from src.retrieval.merge import merge_and_deduplicate, retrieve_multi_query
from src.retrieval.models import RetrievalResult


def _chunk(chunk_id, doc_id, chunk_type="product_overview") -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, chunk_type=chunk_type, text=f"text for {chunk_id}", metadata={})


def _result(chunk, score, rank=1, method="dense") -> RetrievalResult:
    return RetrievalResult(chunk=chunk, score=score, rank=rank, method=method)


def test_chunk_level_dedup_keeps_best_score():
    overview = _chunk("v1::product_overview", "v1")
    list_a = [_result(overview, 0.5, method="dense")]
    list_b = [_result(overview, 0.9, method="bm25")]  # same chunk, higher score elsewhere

    merged = merge_and_deduplicate([list_a, list_b])

    assert len(merged) == 1
    assert merged[0].best_chunk.chunk_id == "v1::product_overview"
    assert merged[0].best_chunk == overview
    # best_chunk score should reflect the higher of the two (0.9), captured via aggregate_score
    assert merged[0].aggregate_score >= 0.9


def test_vehicle_level_dedup_collapses_both_chunks():
    overview = _chunk("v1::product_overview", "v1", "product_overview")
    features = _chunk("v1::features", "v1", "features")
    results = [_result(overview, 0.8), _result(features, 0.6)]

    merged = merge_and_deduplicate([results])

    assert len(merged) == 1  # one vehicle, not two entries
    assert merged[0].vehicle_id == "v1"
    assert merged[0].best_chunk.chunk_id == "v1::product_overview"  # higher score wins
    assert {c.chunk_id for c in merged[0].supporting_chunks} == {"v1::product_overview", "v1::features"}


def test_evidence_boost_favors_more_confirmed_vehicle():
    v1_overview = _chunk("v1::product_overview", "v1")
    v2_overview = _chunk("v2::product_overview", "v2")

    # Both vehicles' best single score is identical (0.7), but v1 is confirmed
    # by two separate query result lists and v2 by only one.
    list_a = [_result(v1_overview, 0.7), _result(v2_overview, 0.7)]
    list_b = [_result(v1_overview, 0.65)]  # v1 hit again (weaker score) in a second query

    merged = merge_and_deduplicate([list_a, list_b])
    by_vehicle = {m.vehicle_id: m for m in merged}

    assert by_vehicle["v1"].match_count == 2
    assert by_vehicle["v2"].match_count == 1
    assert by_vehicle["v1"].aggregate_score > by_vehicle["v2"].aggregate_score


def test_methods_aggregated_across_hits():
    chunk = _chunk("v1::product_overview", "v1")
    list_a = [_result(chunk, 0.7, method="dense")]
    list_b = [_result(chunk, 0.9, method="bm25")]

    merged = merge_and_deduplicate([list_a, list_b])
    assert merged[0].methods == {"dense", "bm25"}


def test_merged_results_sorted_by_aggregate_score_descending():
    v1 = _chunk("v1::product_overview", "v1")
    v2 = _chunk("v2::product_overview", "v2")
    v3 = _chunk("v3::product_overview", "v3")
    results = [_result(v1, 0.3), _result(v2, 0.9), _result(v3, 0.6)]

    merged = merge_and_deduplicate([results])

    assert [m.vehicle_id for m in merged] == ["v2", "v3", "v1"]


def test_merge_empty_input():
    assert merge_and_deduplicate([]) == []
    assert merge_and_deduplicate([[], []]) == []


def test_retrieve_multi_query_calls_retriever_per_query_and_merges():
    calls = []

    def fake_retrieve(query, top_k):
        calls.append(query)
        return [_result(_chunk(f"{query}::product_overview", query), 1.0)]

    merged = retrieve_multi_query(fake_retrieve, ["Q0", "Q1", "Q2"], top_k_per_query=3)

    assert calls == ["Q0", "Q1", "Q2"]
    assert len(merged) == 3  # three distinct vehicles, no overlap in this fake scenario


def test_real_multi_query_hybrid_merge(knowledge_base):
    dense = DenseRetriever(knowledge_base.faiss_index, knowledge_base.chunk_store, knowledge_base.embedder)
    bm25 = BM25Retriever(knowledge_base.bm25_index, knowledge_base.chunk_store)

    # Simulate Q0 (original) + a couple of "expanded" query variants that
    # should overlap heavily on the same vehicle.
    queries = [
        "Porsche Cayenne price and top speed",
        "Porsche Cayenne engine specifications",
        "luxury Porsche SUV",
    ]
    result_lists = [dense.retrieve(q, top_k=10) for q in queries] + [bm25.retrieve(q, top_k=10) for q in queries]
    merged = merge_and_deduplicate(result_lists)

    # No vehicle should appear more than once in the deduplicated list.
    vehicle_ids = [m.vehicle_id for m in merged]
    assert len(vehicle_ids) == len(set(vehicle_ids))

    top = merged[0]
    assert "Cayenne" in top.best_chunk.metadata["name"]
    assert top.match_count > 1  # confirmed by multiple query variants/chunks
