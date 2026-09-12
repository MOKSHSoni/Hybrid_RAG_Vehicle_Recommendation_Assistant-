from src.rag.comparison import build_comparison_rows, format_comparison_for_prompt
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
    # None, NOT the string "N/A" -- see test_rows_are_arrow_serializable below.
    assert rows[1]["top_speed_kmph"] is None


def test_rows_are_arrow_serializable():
    """Regression: numeric columns used to hold the string "N/A" for missing
    values, so a column mixed floats and strings and st.dataframe() raised
    ArrowInvalid ("Could not convert 'N/A' with type str: tried to convert
    to double"). Streamlit serializes via PyArrow, so exercise that directly.
    """
    import pandas as pd
    import pyarrow as pa

    outcome = RetrievalOutcome(
        mode="exact",
        results=[_merged("A", {"boot_space_l": 447.0}), _merged("B", {"boot_space_l": None})],
        original_constraints=None,
    )
    rows = build_comparison_rows(outcome)
    table = pa.Table.from_pandas(pd.DataFrame(rows))  # must not raise
    # The column must stay numeric so the UI table can sort by it.
    assert pa.types.is_floating(table.schema.field("boot_space_l").type)


def test_no_numeric_column_holds_a_string():
    outcome = RetrievalOutcome(
        mode="exact",
        results=[_merged("A", {"boot_space_l": 447.0}), _merged("B")],
        original_constraints=None,
    )
    numeric_cols = {"price_lakhs", "seating_capacity", "top_speed_kmph", "boot_space_l",
                    "ground_clearance_mm", "mileage_max_kmpl", "engine_max_cc"}
    for row in build_comparison_rows(outcome):
        for col, value in row.items():
            if col in numeric_cols:
                assert value is None or isinstance(value, (int, float)), f"{col}={value!r}"


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


def test_prompt_table_renders_missing_values_as_not_recorded():
    # The text table spells missing values out even though the rows carry
    # None. It used to render "N/A", which the model read as zero: given a
    # Mahindra Thar whose boot space is absent from the dataset, it wrote
    # "no boot space". Missing data must never become a claim of absence.
    rendered = format_comparison_for_prompt(
        [{"name": "A", "boot_space_l": 447.0}, {"name": "B", "boot_space_l": None}]
    )
    assert "not recorded" in rendered
    assert "N/A" not in rendered
    assert "447.0" in rendered


def test_prompt_table_does_not_mark_a_tied_extreme():
    # Regression: min()/max() return the FIRST row holding the extreme, so a
    # tie was tagged as an outright winner -- two 5-seaters and one 4-seater
    # got the first 5-seater labelled "most seats", and the model repeated it
    # as fact. Only a uniquely-held extreme may be marked.
    rendered = format_comparison_for_prompt(
        [
            {"name": "A", "seating_capacity": 5},
            {"name": "B", "seating_capacity": 5},
            {"name": "C", "seating_capacity": 4},
        ]
    )
    assert "most seats" not in rendered  # 5 is tied, so nobody wins it
    assert "fewest seats" in rendered  # 4 is uniquely lowest


def test_prompt_table_marks_highest_and_lowest():
    # The model kept mis-comparing numbers (calling a 515L boot "less
    # practical" than 480L), so the extremes are precomputed for it.
    rendered = format_comparison_for_prompt([
        {"name": "A", "price_lakhs": 42.6, "boot_space_l": 480.0},
        {"name": "B", "price_lakhs": 138.0, "boot_space_l": 515.0},
        {"name": "C", "price_lakhs": 133.0, "boot_space_l": 440.0},
    ])
    assert "42.6 (cheapest)" in rendered
    assert "138.0 (most expensive)" in rendered
    assert "515.0 (largest boot)" in rendered
    assert "440.0 (smallest boot)" in rendered


def test_prompt_table_skips_annotation_when_all_equal():
    rendered = format_comparison_for_prompt(
        [{"name": "A", "price_lakhs": 50.0}, {"name": "B", "price_lakhs": 50.0}]
    )
    assert "cheapest" not in rendered and "most expensive" not in rendered


def test_prompt_table_ignores_non_numeric_and_missing_for_extremes():
    rendered = format_comparison_for_prompt(
        [{"name": "A", "boot_space_l": 400.0}, {"name": "B", "boot_space_l": None}]
    )
    # Only one comparable value -> nothing to call largest/smallest.
    assert "largest boot" not in rendered
