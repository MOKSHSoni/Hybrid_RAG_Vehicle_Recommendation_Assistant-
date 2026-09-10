from unittest.mock import patch

import pytest

from src.query.models import Constraints
from src.reranking.cross_encoder import CrossEncoderReranker
from src.retrieval.bm25.retriever import BM25Retriever
from src.retrieval.dense.retriever import DenseRetriever
from src.retrieval.hybrid.hybrid_retriever import HybridRetriever
from src.retrieval.modes import (
    MODE_EXACT,
    MODE_FALLBACK,
    MODE_RELAXED,
    _next_price_boundary_down,
    _next_price_boundary_up,
    _price_quantile_boundaries,
    _relaxation_steps_for_field,
    retrieve_with_relaxation,
)

PRICES = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0, 100.0, 200.0]


# ---------------------------------------------------------------------------
# Pure logic: price quantile stepping (no infra needed)
# ---------------------------------------------------------------------------


def test_price_quantile_boundaries_nonempty_and_sorted():
    boundaries = _price_quantile_boundaries(PRICES)
    assert boundaries == sorted(boundaries)
    assert boundaries[-1] == max(PRICES)  # the 100th percentile is the real max, not an arbitrary bump


def test_price_quantile_boundaries_empty_input():
    assert _price_quantile_boundaries([]) == []


def test_next_price_boundary_up_moves_through_real_quantiles():
    step1 = _next_price_boundary_up(12.0, PRICES, step=1)
    assert step1 is not None
    assert step1 > 12.0
    assert step1 in _price_quantile_boundaries(PRICES)  # a REAL quantile, not current * 1.1 or similar


def test_next_price_boundary_up_returns_none_past_real_max():
    assert _next_price_boundary_up(1000.0, PRICES, step=1) is None  # nothing left to widen to


def test_next_price_boundary_down_moves_through_real_quantiles():
    step1 = _next_price_boundary_down(50.0, PRICES, step=1)
    assert step1 is not None
    assert step1 < 50.0
    assert step1 in _price_quantile_boundaries(PRICES)


def test_next_price_boundary_down_returns_none_below_real_min():
    assert _next_price_boundary_down(0.01, PRICES, step=1) is None


# ---------------------------------------------------------------------------
# Relaxation step generation (no infra needed)
# ---------------------------------------------------------------------------


def test_relaxation_steps_for_categorical_field_yields_one_drop_step():
    constraints = Constraints(brand="Ferrari")
    steps = list(_relaxation_steps_for_field(constraints, "brand", PRICES))
    assert len(steps) == 1
    trial, description = steps[0]
    assert trial.brand is None
    assert "brand" in description


def test_relaxation_steps_for_fuel_types_drops_to_empty_list_not_none():
    constraints = Constraints(fuel_types=["Diesel"])
    steps = list(_relaxation_steps_for_field(constraints, "fuel_types", PRICES))
    assert steps[0][0].fuel_types == []


def test_relaxation_steps_for_price_max_widens_before_dropping():
    constraints = Constraints(price_max_lakhs=12.0)
    steps = list(_relaxation_steps_for_field(constraints, "price_max_lakhs", PRICES))

    assert len(steps) > 1  # multiple widening steps, not a single drop
    widened_values = [trial.price_max_lakhs for trial, _ in steps[:-1]]
    assert widened_values == sorted(widened_values)  # monotonically widening
    assert steps[-1][0].price_max_lakhs is None  # final step fully drops it


def test_relaxation_steps_for_unpopulated_field_yields_nothing():
    constraints = Constraints()  # brand not set
    assert list(_relaxation_steps_for_field(constraints, "brand", PRICES)) == []


# ---------------------------------------------------------------------------
# Full orchestration with mocked _search (controls exactly when results appear)
# ---------------------------------------------------------------------------


def test_mode_a_exact_when_first_search_succeeds():
    fake_results = ["fake_merged_result"]
    with patch("src.retrieval.modes._search", return_value=fake_results) as mock_search:
        outcome = retrieve_with_relaxation(
            "query", Constraints(brand="Toyota"), chunk_store=None, hybrid_retriever=None, reranker=None
        )
    assert outcome.mode == MODE_EXACT
    assert outcome.results == fake_results
    assert outcome.relaxation_steps == []
    assert mock_search.call_count == 1


def test_mode_c_fallback_when_constraints_start_empty():
    with patch("src.retrieval.modes._search", return_value=["fake"]) as mock_search:
        outcome = retrieve_with_relaxation(
            "query", Constraints(), chunk_store=None, hybrid_retriever=None, reranker=None
        )
    assert outcome.mode == MODE_FALLBACK
    mock_search.assert_called_once_with("query", None, None, None, None)


def test_mode_b_relaxed_succeeds_after_dropping_one_field(knowledge_base):
    # First call (exact) fails, second call (after relaxing the first
    # field in CONSTRAINT_RELAXATION_ORDER that's populated) succeeds.
    call_results = [[], ["fake_result"]]
    with patch("src.retrieval.modes._search", side_effect=call_results) as mock_search:
        outcome = retrieve_with_relaxation(
            "query",
            Constraints(brand="Ferrari", seating_capacity=7),
            chunk_store=knowledge_base.chunk_store,
            hybrid_retriever=None,
            reranker=None,
        )
    assert outcome.mode == MODE_RELAXED
    assert outcome.results == ["fake_result"]
    assert len(outcome.relaxation_steps) == 1
    assert outcome.relaxation_steps[0].field == "brand"  # brand precedes seating_capacity in the relaxation order
    assert mock_search.call_count == 2


def test_mode_c_fallback_when_relaxation_exhausted(knowledge_base):
    with patch("src.retrieval.modes._search", return_value=[]) as mock_search:  # every attempt fails, always
        outcome = retrieve_with_relaxation(
            "query",
            Constraints(brand="Ferrari", seating_capacity=7),
            chunk_store=knowledge_base.chunk_store,
            hybrid_retriever=None,
            reranker=None,
        )
    assert outcome.mode == MODE_FALLBACK
    assert outcome.results == []
    assert len(outcome.relaxation_steps) == 2  # brand dropped, then seating_capacity dropped
    assert mock_search.call_count == 4  # exact + relax(brand) + relax(seating_capacity) + final fallback search


# ---------------------------------------------------------------------------
# Real end-to-end integration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_components(knowledge_base):
    dense = DenseRetriever(knowledge_base.faiss_index, knowledge_base.chunk_store, knowledge_base.embedder)
    bm25 = BM25Retriever(knowledge_base.bm25_index, knowledge_base.chunk_store)
    hybrid = HybridRetriever(dense, bm25, knowledge_base.chunk_store)
    reranker = CrossEncoderReranker(knowledge_base.chunk_store)
    return knowledge_base.chunk_store, hybrid, reranker


def test_real_mode_a_exact_satisfiable_constraints(pipeline_components):
    chunk_store, hybrid, reranker = pipeline_components
    outcome = retrieve_with_relaxation(
        "SUV", Constraints(brand="Toyota", body_type="SUV"), chunk_store, hybrid, reranker
    )
    assert outcome.mode == MODE_EXACT
    assert len(outcome.results) > 0


def test_real_mode_b_relaxed_ferrari_seven_seats(pipeline_components):
    # Real fact confirmed during Phase 1 data exploration: every Ferrari in
    # the dataset is a 2-seater, so brand=Ferrari + seating_capacity=7 has
    # zero exact matches but plenty once brand is relaxed away.
    chunk_store, hybrid, reranker = pipeline_components
    outcome = retrieve_with_relaxation(
        "sports car", Constraints(brand="Ferrari", seating_capacity=7), chunk_store, hybrid, reranker
    )
    assert outcome.mode == MODE_RELAXED
    assert len(outcome.results) > 0
    assert any(s.field == "brand" for s in outcome.relaxation_steps)
