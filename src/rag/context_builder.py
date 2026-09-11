"""Phase 11: context builder.

Formats a Phase 10 RetrievalOutcome into a grounded, structured text block
for the generation prompt -- every fact in it comes directly from the
dataset (both chunks per vehicle, fetched fresh via ChunkStore.get_by_doc_id
so the LLM always sees the complete picture regardless of which single
chunk the upstream retrieval stage happened to surface). The mode
(Exact/Relaxed/Fallback/Superlative) and every relaxation step actually
tried are included explicitly, since the generation prompt must never
claim a relaxed/fallback result satisfies a constraint that was never
verified, and a superlative result must never be conflated with relevance
ranking (it's a direct metadata sort, not a search).
"""

from src.rag.comparison import build_comparison_rows
from src.retrieval.chunk_store import ChunkStore
from src.retrieval.modes import RetrievalOutcome

_CHUNK_TYPE_ORDER = ["product_overview", "features"]


def build_context_block(outcome: RetrievalOutcome, chunk_store: ChunkStore) -> str:
    lines = [f"RETRIEVAL MODE: {outcome.mode.upper()}"]

    if outcome.mode == "relaxed":
        lines.append(
            "Some requested constraints were relaxed to find these results -- they do NOT "
            "all satisfy the original request exactly:"
        )
        for step in outcome.relaxation_steps:
            lines.append(f"  - {step.description}")
    elif outcome.mode == "fallback":
        lines.append(
            "No structured constraints (price, seating, brand, fuel, transmission, body type) "
            "could be verified for these results -- they are general semantic matches only."
        )
    elif outcome.mode == "superlative":
        field = outcome.original_constraints.superlative_field
        direction = "highest" if outcome.original_constraints.superlative_direction == "desc" else "lowest"
        lines.append(
            f"These results are ranked by a DIRECT SORT on the real '{field}' value ({direction} first), "
            "not by search relevance -- this is an exact ranking, not a semantic guess."
        )

    lines.append("")
    lines.append(f"RETRIEVED VEHICLES ({len(outcome.results)}), most relevant first:")

    for i, merged in enumerate(outcome.results, start=1):
        name = merged.best_chunk.metadata.get("name", "Unknown vehicle")
        lines.append(f"\n[Vehicle {i}] {name}")
        chunks = chunk_store.get_by_doc_id(merged.vehicle_id)
        by_type = {c.chunk_type: c.text for c in chunks}
        for chunk_type in _CHUNK_TYPE_ORDER:
            if chunk_type in by_type:
                lines.append(f"  {by_type[chunk_type]}")

    if len(outcome.results) > 1:
        rows = build_comparison_rows(outcome, chunk_store)
        if rows:
            lines.append("")
            lines.append("COMPARISON TABLE (exact metadata values, not LLM-derived):")
            lines.append(_format_table(rows))

    return "\n".join(lines)


def _format_table(rows) -> str:
    # build_comparison_rows() leaves missing values as None (so the UI's
    # dataframe keeps numeric dtypes); rendering them as "N/A" is this
    # text layer's job.
    columns = list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join("N/A" if row.get(c) is None else str(row.get(c)) for c in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])
