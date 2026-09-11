from src.rag.comparison import build_comparison_rows
from src.retrieval.merge import MergedResult
from src.retrieval.modes import RetrievalOutcome


def _merged(name, metadata_extra=None) -> MergedResult:
    metadata = {"name": name, "price_lakhs": 20.0, "seating_capacity": 5, "body_type": "SUV", "fuel_types": ["Petrol"]}
    metadata.update(metadata_extra or {})
    chunk = type("FakeChunk", (), {"metadata": metadata})()
    return MergedResult(vehicle_id=name, best_chunk=chunk, aggregate_score=1.0)


def test_build_comparison_rows_base_columns_always_present():
    outcome = RetrievalOutcome(mode="exact", results=[_merged("A"), _merged("B")], original_constraints=None)
    rows = build_comparison_rows(outcome)
    assert len(rows) == 2
    assert set(rows[0].keys()) == {"name", "price_lakhs", "seating_capacity", "body_type", "fuel_types"}


def test_build_comparison_rows_includes_extended_column_when_present():
    outcome = RetrievalOutcome(
        mode="exact",
        results=[_merged("A", {"top_speed_kmph": 200.0}), _merged("B")],
        original_constraints=None,
    )
    rows = build_comparison_rows(outcome)
    assert "top_speed_kmph" in rows[0]
    assert rows[0]["top_speed_kmph"] == 200.0
    assert rows[1]["top_speed_kmph"] == "N/A"  # missing for B, shown not silently dropped


def test_build_comparison_rows_drops_all_none_extended_columns():
    outcome = RetrievalOutcome(mode="exact", results=[_merged("A"), _merged("B")], original_constraints=None)
    rows = build_comparison_rows(outcome)
    assert "top_speed_kmph" not in rows[0]  # neither result has it -- column omitted entirely


def test_build_comparison_rows_formats_fuel_types_as_string():
    outcome = RetrievalOutcome(
        mode="exact", results=[_merged("A", {"fuel_types": ["Petrol", "Diesel"]})], original_constraints=None
    )
    rows = build_comparison_rows(outcome)
    assert rows[0]["fuel_types"] == "Petrol, Diesel"
