"""Phase 3 deliverable: query understanding & context resolution.

Demonstrates: standalone constraint extraction via Qwen3 (validated,
never trusted raw), multi-turn follow-up resolution, hard vs soft
constraint classification, and the regex-only fallback path used when
Ollama is unreachable or returns invalid output.
"""

import sys
from unittest.mock import patch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # model text may contain Unicode; console codepage may not

import config
from src.pipeline import build_knowledge_base
from src.query.models import ConversationTurn
from src.query.ollama_client import OllamaError
from src.query.regex_extraction import extract_regex_constraints, known_brands_from_chunk_store
from src.query.understanding import understand_query


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_result(result) -> None:
    c = result.constraints
    print(f"  original query:    {result.original_query!r}")
    print(f"  standalone query:  {result.standalone_query!r}")
    print(f"  extraction method: {result.extraction_method}"
          + (f"  (error: {result.extraction_error})" if result.extraction_error else ""))
    print(f"  constraints:       brand={c.brand} fuel_types={c.fuel_types} transmission={c.transmission}")
    print(f"                     seating={c.seating_capacity} body_type={c.body_type}")
    print(f"                     price_lakhs=[{c.price_min_lakhs}, {c.price_max_lakhs}]")
    print(f"  hard constraints:  {c.hard_fields()}")
    print(f"  soft constraints:  {c.soft_fields()}")


def main() -> None:
    banner("PHASE 3 DEMO: Query Understanding & Context")

    print("\nBuilding knowledge base (for the known-brands list used by regex extraction)...")
    kb = build_knowledge_base(save=False)
    known_brands = known_brands_from_chunk_store(kb.chunk_store)

    banner("Single-turn constraint extraction (Qwen3, schema-validated)")
    for query in [
        "automatic diesel SUV under 20 lakh with 7 seats",
        "I want an affordable electric hatchback",
        "show me Porsche sports cars",
    ]:
        print(f'\n--- "{query}" ---')
        result = understand_query(query, history=[], known_brands=known_brands)
        print_result(result)

    banner("Multi-turn follow-up resolution")
    history = [
        ConversationTurn(role="user", content="Show me SUVs under 20 lakh"),
        ConversationTurn(role="assistant", content="Here are a few SUVs under 20 lakh: Nexon, Venue, Sonet."),
    ]
    follow_up = "what about diesel options instead"
    print(f'\nHistory: user asked for "SUVs under 20 lakh", assistant replied with some options.')
    print(f'Follow-up: "{follow_up}"')
    result = understand_query(follow_up, history=history, known_brands=known_brands)
    print_result(result)

    banner("Regex-only fallback (simulated Ollama outage)")
    print("\nForcing every Ollama call to fail, to prove the fallback path works")
    print("without retrying against a dead endpoint:")
    with patch("src.query.llm_extraction.chat", side_effect=OllamaError("simulated: connection refused")):
        result = understand_query(
            "automatic diesel SUV under 20 lakh with 7 seats", history=[], known_brands=known_brands
        )
    print_result(result)

    banner("Summary")
    print("Constraints are always Python-validated before use -- the LLM never")
    print("directly controls filtering. Phase 6 will consume these Constraints")
    print("objects; Phase 10 will use hard/soft classification for relaxation.")


if __name__ == "__main__":
    main()
