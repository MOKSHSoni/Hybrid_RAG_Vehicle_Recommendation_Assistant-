"""Tests for the extended-numeric-constraints and superlative-ranking
additions: regex extraction, LLM schema, metadata filtering, and the
Phase 10 relaxation/superlative code paths.
"""

from unittest.mock import patch

import pytest
from conftest import requires_ollama

import json

from src.query.llm_extraction import (
    CORE_JSON_SCHEMA,
    JSON_SCHEMA,
    _to_constraints,
    _validate_schema,
    extract_constraints_via_llm,
    needs_extended_extraction,
)
from src.query.metadata_filter import matches_constraints
from src.query.models import Constraints
from src.query.regex_extraction import (
    extract_numeric_range_constraints,
    extract_regex_constraints,
    extract_superlative,
)
from src.query.understanding import understand_query
from src.reranking.cross_encoder import CrossEncoderReranker
from src.retrieval.bm25.retriever import BM25Retriever
from src.retrieval.dense.retriever import DenseRetriever
from src.retrieval.hybrid.hybrid_retriever import HybridRetriever
from src.retrieval.modes import (
    MODE_EXACT,
    MODE_SUPERLATIVE,
    _relaxation_steps_for_field,
    _search_superlative,
    retrieve_with_relaxation,
)

KNOWN_BRANDS = ["Porsche", "BMW", "Lamborghini", "Ferrari"]


# ---------------------------------------------------------------------------
# Regex: numeric range extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,field,expected_min,expected_max",
    [
        ("SUV with top speed above 200", "top_speed_kmph", 200.0, None),
        ("boot space over 400 liters", "boot_space_l", 400.0, None),
        ("ground clearance above 180mm", "ground_clearance_mm", 180.0, None),
        ("mileage above 20 kmpl", "mileage_max_kmpl", 20.0, None),
        ("engine displacement under 1500cc", "engine_max_cc", None, 1500.0),
        ("boot space between 300 and 500", "boot_space_l", 300.0, 500.0),
        ("no numeric preferences here", "top_speed_kmph", None, None),
    ],
)
def test_extract_numeric_range_constraints(text, field, expected_min, expected_max):
    result = extract_numeric_range_constraints(text)
    if expected_min is None and expected_max is None:
        assert field not in result
    else:
        assert result[field] == (expected_min, expected_max)


def test_extract_numeric_range_constraints_multiple_metrics_in_one_query():
    result = extract_numeric_range_constraints("top speed above 200 and boot space over 400")
    assert result["top_speed_kmph"] == (200.0, None)
    assert result["boot_space_l"] == (400.0, None)


def test_price_extraction_unaffected_by_numeric_range_additions():
    # extract_price_constraints must stay exactly as before -- price and
    # the new metrics are independent extraction paths.
    from src.query.regex_extraction import extract_price_constraints

    assert extract_price_constraints("SUV under 15 lakh") == (None, 15.0)


# ---------------------------------------------------------------------------
# Regex: superlative detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected_field,expected_direction",
    [
        ("cheapest Lamborghini", "price_lakhs", "asc"),
        ("most expensive SUV", "price_lakhs", "desc"),
        ("fastest car", "top_speed_kmph", "desc"),
        ("biggest boot space", "boot_space_l", "desc"),
        ("best mileage sedan", "mileage_max_kmpl", "desc"),
        ("highest ground clearance SUV", "ground_clearance_mm", "desc"),
        ("most powerful engine", "engine_max_cc", "desc"),
        ("affordable SUV under 15 lakh", None, None),
    ],
)
def test_extract_superlative(text, expected_field, expected_direction):
    assert extract_superlative(text) == (expected_field, expected_direction)


def test_extract_regex_constraints_includes_numeric_ranges_and_superlative():
    result = extract_regex_constraints("cheapest SUV with boot space over 400", KNOWN_BRANDS)
    assert result.superlative_field == "price_lakhs"
    assert result.superlative_direction == "asc"
    assert result.numeric_ranges["boot_space_l"] == (400.0, None)


# ---------------------------------------------------------------------------
# LLM schema: validation + _to_constraints for the new fields
# ---------------------------------------------------------------------------

_VALID_PAYLOAD = {
    "brand": None,
    "fuel_types": [],
    "transmission": None,
    "seating_capacity": None,
    "body_type": None,
    "price_max_lakhs": None,
    "price_min_lakhs": None,
    "top_speed_kmph_min": 200.0,
    "top_speed_kmph_max": None,
    "boot_space_l_min": None,
    "boot_space_l_max": None,
    "ground_clearance_mm_min": None,
    "ground_clearance_mm_max": None,
    "mileage_kmpl_min": None,
    "mileage_kmpl_max": None,
    "engine_cc_min": None,
    "engine_cc_max": None,
    "superlative_field": None,
    "superlative_direction": None,
}


def test_validate_schema_accepts_new_fields():
    assert _validate_schema(_VALID_PAYLOAD) is True


def test_validate_schema_rejects_missing_new_field():
    bad = dict(_VALID_PAYLOAD)
    del bad["top_speed_kmph_min"]
    assert _validate_schema(bad) is False


def test_to_constraints_builds_numeric_ranges_dict():
    constraints = _to_constraints(_VALID_PAYLOAD)
    assert constraints.numeric_ranges == {"top_speed_kmph": (200.0, None)}


def test_to_constraints_superlative_valid():
    payload = dict(_VALID_PAYLOAD, superlative_field="price_lakhs", superlative_direction="asc")
    constraints = _to_constraints(payload)
    assert constraints.superlative_field == "price_lakhs"
    assert constraints.superlative_direction == "asc"


def test_to_constraints_rejects_hallucinated_superlative_field():
    payload = dict(_VALID_PAYLOAD, superlative_field="not_a_real_field", superlative_direction="asc")
    constraints = _to_constraints(payload)
    assert constraints.superlative_field is None
    assert constraints.superlative_direction is None


def test_to_constraints_rejects_hallucinated_superlative_direction():
    payload = dict(_VALID_PAYLOAD, superlative_field="price_lakhs", superlative_direction="sideways")
    constraints = _to_constraints(payload)
    assert constraints.superlative_field is None
    assert constraints.superlative_direction is None


# ---------------------------------------------------------------------------
# Metadata filtering: numeric_ranges
# ---------------------------------------------------------------------------


def test_matches_constraints_numeric_range_in_bounds():
    v = {"top_speed_kmph": 250.0}
    c = Constraints(numeric_ranges={"top_speed_kmph": (200.0, None)})
    assert matches_constraints(v, c) is True


def test_matches_constraints_numeric_range_out_of_bounds():
    v = {"top_speed_kmph": 150.0}
    c = Constraints(numeric_ranges={"top_speed_kmph": (200.0, None)})
    assert matches_constraints(v, c) is False


def test_matches_constraints_numeric_range_missing_data_excluded():
    v = {"top_speed_kmph": None}
    c = Constraints(numeric_ranges={"top_speed_kmph": (200.0, None)})
    assert matches_constraints(v, c) is False


def test_matches_constraints_numeric_range_max_bound():
    v = {"engine_max_cc": 1200.0}
    c = Constraints(numeric_ranges={"engine_max_cc": (None, 1500.0)})
    assert matches_constraints(v, c) is True
    v2 = {"engine_max_cc": 2000.0}
    assert matches_constraints(v2, c) is False


# ---------------------------------------------------------------------------
# Relaxation: numeric_ranges drop in one step
# ---------------------------------------------------------------------------


def test_relaxation_steps_for_numeric_range_drops_in_one_step():
    constraints = Constraints(numeric_ranges={"boot_space_l": (400.0, None)})
    steps = list(_relaxation_steps_for_field(constraints, "boot_space_l", all_prices=[]))
    assert len(steps) == 1
    trial, description = steps[0]
    assert "boot_space_l" not in trial.numeric_ranges
    assert "boot_space_l" in description


# ---------------------------------------------------------------------------
# Superlative: direct metadata sort (real data)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_components(knowledge_base):
    dense = DenseRetriever(knowledge_base.faiss_index, knowledge_base.chunk_store, knowledge_base.embedder)
    bm25 = BM25Retriever(knowledge_base.bm25_index, knowledge_base.chunk_store)
    hybrid = HybridRetriever(dense, bm25, knowledge_base.chunk_store)
    reranker = CrossEncoderReranker(knowledge_base.chunk_store)
    return knowledge_base.chunk_store, hybrid, reranker


def test_search_superlative_cheapest_lamborghini(pipeline_components):
    chunk_store, _, _ = pipeline_components
    constraints = Constraints(brand="Lamborghini", superlative_field="price_lakhs", superlative_direction="asc")
    results = _search_superlative(constraints, chunk_store)

    # Verified real prices from Phase 10 development: Urus Rs 310L, Huracan Rs 322L.
    assert len(results) == 2
    assert results[0].best_chunk.metadata["name"] == "Lamborghini Urus"
    assert results[1].best_chunk.metadata["name"] == "Lamborghini Huracan"
    assert list(results[0].methods) == ["metadata_sort"]


def test_retrieve_with_relaxation_superlative_mode(pipeline_components):
    chunk_store, hybrid, reranker = pipeline_components
    constraints = Constraints(brand="Lamborghini", superlative_field="price_lakhs", superlative_direction="asc")
    outcome = retrieve_with_relaxation("cheapest Lamborghini", constraints, chunk_store, hybrid, reranker)

    assert outcome.mode == MODE_SUPERLATIVE
    assert outcome.results[0].best_chunk.metadata["name"] == "Lamborghini Urus"


def test_retrieve_with_relaxation_superlative_falls_through_when_nothing_eligible(pipeline_components):
    chunk_store, hybrid, reranker = pipeline_components
    # No vehicle is both brand=Toyota AND has an electric+manual combo priced
    # as a "superlative" on a field that would exclude everything -- use an
    # impossible brand/field combo to force the fallthrough path.
    constraints = Constraints(
        brand="NoSuchBrand", superlative_field="price_lakhs", superlative_direction="asc"
    )
    outcome = retrieve_with_relaxation("cheapest NoSuchBrand car", constraints, chunk_store, hybrid, reranker)

    assert outcome.mode != MODE_SUPERLATIVE  # fell through to the normal Exact/Relaxed/Fallback flow


@requires_ollama
def test_understand_query_extracts_superlative_real(pipeline_components):
    _, _, _ = pipeline_components
    result = understand_query("what is the cheapest SUV", history=[], known_brands=KNOWN_BRANDS)
    assert result.constraints.superlative_field == "price_lakhs"
    assert result.constraints.superlative_direction == "asc"
    assert result.constraints.body_type == "SUV"


@requires_ollama
def test_understand_query_affordable_with_explicit_budget_is_not_superlative(pipeline_components):
    # Regression test: "affordable" + an explicit price ceiling was
    # initially (incorrectly) interpreted by the LLM as "find the single
    # cheapest," discovered via a real test run. The user wants options
    # within budget, not just the one cheapest vehicle -- price_max_lakhs
    # should be set and superlative_field should stay null.
    _, _, _ = pipeline_components
    result = understand_query("affordable SUV under 15 lakh", history=[], known_brands=KNOWN_BRANDS)
    assert result.constraints.price_max_lakhs == pytest.approx(15.0, abs=1.0)
    assert result.constraints.superlative_field is None
    assert result.constraints.superlative_direction is None


# ---------------------------------------------------------------------------
# Conditional schema escalation (latency optimisation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected",
    [
        ("affordable SUV under 15 lakh", False),
        ("electric car with automatic transmission", False),
        ("7 seater diesel SUV under 20 lakh", False),
        ("tell me about the Tata Nexon EV", False),
        ("cheapest Lamborghini", True),
        ("how fast does it go", True),  # phrasing regex can't parse -- must still reach the LLM
        ("roomiest boot", True),
        ("engine under 2 litres", True),
        ("best mileage sedan", True),
        ("smallest engine", True),
    ],
)
def test_needs_extended_extraction_routing(query, expected):
    assert needs_extended_extraction(query) is expected


def test_core_schema_is_the_original_seven_fields():
    assert len(CORE_JSON_SCHEMA["required"]) == 7
    assert len(JSON_SCHEMA["required"]) == 19
    # Derived from the full schema, so the two can't drift apart.
    for field in CORE_JSON_SCHEMA["required"]:
        assert CORE_JSON_SCHEMA["properties"][field] == JSON_SCHEMA["properties"][field]


def test_extraction_sends_core_schema_for_plain_query():
    payload = json.dumps({
        "brand": None, "fuel_types": [], "transmission": None, "seating_capacity": None,
        "body_type": "SUV", "price_max_lakhs": 15.0, "price_min_lakhs": None,
    })
    with patch("src.query.llm_extraction.chat", return_value=payload) as mock_chat:
        constraints = extract_constraints_via_llm("affordable SUV under 15 lakh")
    assert len(mock_chat.call_args.kwargs["format"]["required"]) == 7
    assert constraints.body_type == "SUV"
    assert constraints.numeric_ranges == {}  # absent keys must not crash _to_constraints
    assert constraints.superlative_field is None


def test_extraction_sends_full_schema_for_superlative_query():
    payload = json.dumps({k: None for k in JSON_SCHEMA["required"]} | {
        "fuel_types": [], "superlative_field": "price_lakhs", "superlative_direction": "asc",
    })
    with patch("src.query.llm_extraction.chat", return_value=payload) as mock_chat:
        constraints = extract_constraints_via_llm("cheapest Lamborghini")
    assert len(mock_chat.call_args.kwargs["format"]["required"]) == 19
    assert constraints.superlative_field == "price_lakhs"
