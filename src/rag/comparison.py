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
    one result has a non-None value for them, to avoid an all-N/A column.
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
                row[col] = ", ".join(value) if value else "N/A"
            else:
                row[col] = value if value is not None else "N/A"
        rows.append(row)
    return rows
