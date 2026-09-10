from src.query.metadata_filter import (
    candidate_chunk_ids,
    candidate_chunks,
    filter_vehicles,
    matches_constraints,
    unique_vehicles,
)
from src.query.models import Constraints


def test_unique_vehicles_count(knowledge_base):
    vehicles = unique_vehicles(knowledge_base.chunk_store)
    assert len(vehicles) == 150


def test_matches_constraints_brand():
    v = {"brand": "Porsche"}
    assert matches_constraints(v, Constraints(brand="Porsche")) is True
    assert matches_constraints(v, Constraints(brand="BMW")) is False


def test_matches_constraints_body_type():
    v = {"body_type": "SUV"}
    assert matches_constraints(v, Constraints(body_type="SUV")) is True
    assert matches_constraints(v, Constraints(body_type="Sedan")) is False


def test_matches_constraints_fuel_types_any_overlap():
    v = {"fuel_types": ["Petrol", "Diesel"]}
    assert matches_constraints(v, Constraints(fuel_types=["Diesel"])) is True
    assert matches_constraints(v, Constraints(fuel_types=["Electric"])) is False
    assert matches_constraints(v, Constraints(fuel_types=["Electric", "Diesel"])) is True


def test_matches_constraints_transmission():
    v = {"has_automatic": True, "has_manual": False}
    assert matches_constraints(v, Constraints(transmission="Automatic")) is True
    assert matches_constraints(v, Constraints(transmission="Manual")) is False


def test_matches_constraints_seating_at_least_semantics():
    v = {"seating_capacity": 7}
    assert matches_constraints(v, Constraints(seating_capacity=7)) is True
    assert matches_constraints(v, Constraints(seating_capacity=5)) is True  # 7-seater satisfies "at least 5"
    assert matches_constraints(v, Constraints(seating_capacity=8)) is False


def test_matches_constraints_seating_missing_never_matches():
    v = {"seating_capacity": None}
    assert matches_constraints(v, Constraints(seating_capacity=5)) is False


def test_matches_constraints_price_range():
    v = {"price_lakhs": 15.0}
    assert matches_constraints(v, Constraints(price_max_lakhs=20.0)) is True
    assert matches_constraints(v, Constraints(price_max_lakhs=10.0)) is False
    assert matches_constraints(v, Constraints(price_min_lakhs=10.0)) is True
    assert matches_constraints(v, Constraints(price_min_lakhs=20.0)) is False


def test_matches_constraints_price_missing_never_matches():
    v = {"price_lakhs": None}
    assert matches_constraints(v, Constraints(price_max_lakhs=20.0)) is False


def test_matches_constraints_empty_constraints_always_true():
    assert matches_constraints({}, Constraints()) is True


def test_matches_constraints_restricted_fields():
    v = {"brand": "BMW", "body_type": "Sedan"}
    constraints = Constraints(brand="Porsche", body_type="Sedan")
    # Only checking body_type -- the mismatched brand should be ignored.
    assert matches_constraints(v, constraints, fields=["body_type"]) is True


def test_filter_vehicles_real_dataset_porsche(knowledge_base):
    vehicles = unique_vehicles(knowledge_base.chunk_store)
    result = filter_vehicles(vehicles, Constraints(brand="Porsche"))
    assert len(result) == 6  # confirmed brand count from Phase 1 data exploration
    assert all(v["brand"] == "Porsche" for v in result)


def test_filter_vehicles_real_dataset_electric(knowledge_base):
    vehicles = unique_vehicles(knowledge_base.chunk_store)
    result = filter_vehicles(vehicles, Constraints(fuel_types=["Electric"]))
    assert len(result) == 6  # confirmed EV count from Phase 1


def test_candidate_chunk_ids_includes_both_chunks_per_vehicle(knowledge_base):
    ids = candidate_chunk_ids(knowledge_base.chunk_store, Constraints(brand="Porsche"))
    assert len(ids) == 12  # 6 Porsche vehicles x 2 chunks each
    assert "cars_cleaned_0::product_overview" in ids
    assert "cars_cleaned_0::features" in ids


def test_candidate_chunks_returns_chunk_objects(knowledge_base):
    chunks = candidate_chunks(knowledge_base.chunk_store, Constraints(brand="Porsche"))
    assert len(chunks) == 12
    assert all(c.metadata["brand"] == "Porsche" for c in chunks)


def test_candidate_chunk_ids_no_constraints_returns_everything(knowledge_base):
    ids = candidate_chunk_ids(knowledge_base.chunk_store, Constraints())
    assert len(ids) == 300
