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

from src.rag.comparison import build_comparison_rows, format_comparison_for_prompt
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

    price_cap = getattr(outcome.original_constraints, "price_max_lakhs", None)
    if price_cap is not None:
        # Every row's price_lakhs IS its starting price (verified: price_lakhs
        # == Price_Min_Lakhs for all 150 rows), so the budget filter matches
        # the entry variant, not the whole range. 10 vehicles have a base
        # under 15 lakh and a top variant above it -- the Tata Safari spans
        # Rs 14.69-21.45 Lakh. Saying "the Safari at Rs 14.69 Lakh fits your
        # budget" is then true only of the cheapest trim, and stating the
        # basis is the difference between an accurate answer and a
        # misleading one. It also resolves a contradiction the model was
        # visibly deliberating over ("...which is over 15 lakh, so it
        # doesn't meet the requirement. Wait, the...").
        lines.append(
            f"PRICE BASIS: every vehicle below starts UNDER Rs {price_cap} Lakh -- that is what "
            f"the budget was matched on, and none of them is 'over budget'. Where a vehicle's "
            f"range runs past Rs {price_cap} Lakh, the entry variant still fits and only the "
            f"higher trims do not; phrase that as 'fits at its starting price, though higher "
            f"trims go beyond it'."
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
            lines.append(
                "COMPARISON TABLE (exact values, with each column's extreme already marked -- "
                "use those tags rather than comparing the numbers yourself):"
            )
            lines.append(format_comparison_for_prompt(rows))

    return "\n".join(lines)
