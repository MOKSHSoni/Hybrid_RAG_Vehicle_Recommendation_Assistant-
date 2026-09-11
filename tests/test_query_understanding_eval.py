from conftest import requires_ollama

from src.evaluation.dataset import EvalQuery
from src.evaluation.query_understanding_eval import compare_constraints, evaluate_query_understanding
from src.query.models import Constraints


def test_compare_constraints_only_checks_specified_fields():
    extracted = Constraints(brand="Toyota", body_type="Sedan")  # body_type not expected -- should be ignored
    expected = {"brand": "Toyota"}
    result = compare_constraints(extracted, expected)
    assert result == {"brand": True}


def test_compare_constraints_detects_mismatch():
    extracted = Constraints(brand="Honda")
    expected = {"brand": "Toyota"}
    assert compare_constraints(extracted, expected) == {"brand": False}


def test_compare_constraints_fuel_types_set_comparison():
    extracted = Constraints(fuel_types=["Diesel", "Petrol"])
    expected = {"fuel_types": ["Petrol", "Diesel"]}  # different order
    assert compare_constraints(extracted, expected) == {"fuel_types": True}


@requires_ollama
def test_evaluate_query_understanding_real(knowledge_base):
    from src.query.regex_extraction import known_brands_from_chunk_store

    known_brands = known_brands_from_chunk_store(knowledge_base.chunk_store)
    queries = [
        EvalQuery(
            id="t1",
            category="test",
            query="automatic diesel SUV under 20 lakh with 7 seats",
            history=[],
            expected_constraints={
                "body_type": "SUV",
                "transmission": "Automatic",
                "fuel_types": ["Diesel"],
                "seating_capacity": 7,
                "price_max_lakhs": 20.0,
            },
            expected_relevant_vehicles=[],
            expected_mode="exact",
        )
    ]
    accuracy = evaluate_query_understanding(queries, known_brands)
    assert "body_type" in accuracy
    assert accuracy["body_type"] == 1.0  # this exact phrasing was validated during Phase 3 development
