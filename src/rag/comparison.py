"""Builds a plain, LLM-independent comparison table from a Phase 10
RetrievalOutcome's results -- pure metadata reads, so the numbers can
never be garbled by generation. Used by both the generation context
block (src/rag/context_builder.py) and the Streamlit UI's Compare
expander (streamlit_app.py) so both surfaces show identical, accurate
data.
"""

from typing import Any, Dict, List

from src.retrieval.modes import RetrievalOutcome

_BASE_COLUMNS = ["name", "price_lakhs", "seating_capacity", "body_type", "fuel_types"]
_EXTENDED_COLUMNS = ["top_speed_kmph", "boot_space_l", "ground_clearance_mm", "mileage_max_kmpl", "engine_max_cc"]


def build_comparison_rows(outcome: RetrievalOutcome, chunk_store=None) -> List[Dict[str, Any]]:
    """One row per result. Base columns (name/price/seats/body/fuel)
    always included; extended numeric columns included only if at least
    one result has a non-None value for them, to avoid an all-empty column.

    Missing values stay None rather than becoming the string "N/A": these
    rows feed st.dataframe(), and a numeric column holding a mix of floats
    and "N/A" strings makes PyArrow (Streamlit's serializer) raise
    ArrowInvalid. Keeping them None also preserves the column's numeric
    dtype, so the UI table stays sortable by price/boot space. Rendering
    missing values is each display layer's job -- see context_builder's
    _format_table() for the text version.

    `chunk_store` is unused (metadata already lives on each result's
    best_chunk) -- kept as a parameter for a uniform call signature
    alongside context_builder's other helpers.
    """
    all_metadata = [merged.best_chunk.metadata for merged in outcome.results]

    extended_columns = [c for c in _EXTENDED_COLUMNS if any(m.get(c) is not None for m in all_metadata)]
    columns = _BASE_COLUMNS + extended_columns

    rows = []
    for metadata in all_metadata:
        row: Dict[str, Any] = {}
        for col in columns:
            value = metadata.get(col)
            if col == "fuel_types":
                row[col] = ", ".join(value) if value else None
            else:
                row[col] = value
        rows.append(row)
    return rows


# Which extreme is worth naming, per column: (label for min, label for max).
# Precomputing these means the generation model reads the comparison instead
# of deriving it -- it cannot get arithmetic wrong if it never does any.
_SUPERLATIVE_LABELS = {
    "price_lakhs": ("cheapest", "most expensive"),
    "boot_space_l": ("smallest boot", "largest boot"),
    "mileage_max_kmpl": ("worst mileage", "best mileage"),
    "top_speed_kmph": ("slowest", "fastest"),
    "seating_capacity": ("fewest seats", "most seats"),
    "ground_clearance_mm": ("lowest clearance", "highest clearance"),
    "engine_max_cc": ("smallest engine", "largest engine"),
}


_MISSING_VALUE_TEXT = "not recorded"


def format_comparison_for_prompt(rows: List[Dict[str, Any]]) -> str:
    """Markdown table for the generation context, with each column's
    highest/lowest value tagged inline.

    Distinct from the raw rows handed to st.dataframe(): those must stay
    numeric and untagged so the UI table sorts correctly. This version is
    for the LLM, which was observed both mis-attributing figures between
    cars and calling a 515L boot "less practical" than a 480L one.
    """
    if not rows:
        return ""

    columns = list(rows[0].keys())
    annotations: Dict[str, Dict[int, str]] = {}
    for col in columns:
        if col not in _SUPERLATIVE_LABELS:
            continue
        values = [(i, r[col]) for i, r in enumerate(rows) if isinstance(r.get(col), (int, float))]
        if len(values) < 2:
            continue
        low_label, high_label = _SUPERLATIVE_LABELS[col]
        low_value = min(v for _, v in values)
        high_value = max(v for _, v in values)
        if low_value == high_value:
            continue  # all equal -- no extreme worth naming

        # Only tag an extreme that exactly one vehicle holds. Taking min()/max()
        # directly returns the FIRST row at that value, so a tie was being
        # tagged as an outright winner: with two 5-seaters and one 4-seater,
        # the first 5-seater got labelled "most seats" and the model duly
        # wrote "it offers the most seats (5)" about a car that merely ties.
        # A tag this code emits is one the prompt instructs the model to
        # trust, so an inaccurate tag becomes an inaccurate answer.
        marks: Dict[int, str] = {}
        lows = [i for i, v in values if v == low_value]
        highs = [i for i, v in values if v == high_value]
        if len(lows) == 1:
            marks[lows[0]] = low_label
        if len(highs) == 1:
            marks[highs[0]] = high_label
        if marks:
            annotations[col] = marks

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for i, row in enumerate(rows):
        cells = []
        for col in columns:
            value = row.get(col)
            # "not recorded", not "N/A": the model read N/A as a value of
            # zero and wrote "no boot space" about a vehicle whose boot
            # simply isn't in the dataset. Missing data must not become a
            # claim of absence. (The st.dataframe rows keep None -- this
            # wording is for the prompt only.)
            text = _MISSING_VALUE_TEXT if value is None else str(value)
            tag = annotations.get(col, {}).get(i)
            cells.append(f"{text} ({tag})" if tag else text)
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *body])
