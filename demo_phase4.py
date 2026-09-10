"""Phase 4 deliverable: query transformation & expansion.

Shows the standalone query (Phase 3) being rewritten into a
retrieval-optimized form, then expanded into 3 genuinely different
angles (Q1/Q2/Q3) that, combined with the original/transformed query
(Q0), give later phases 4 queries to retrieve against.
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # model text may contain Unicode; console codepage may not

from src.query.expansion import expand_query
from src.query.transformation import transform_query

QUERIES = [
    "I would really really love it if you could show me an affordable SUV please, something with good mileage",
    "I'm looking for a family car, maybe 7 seats, nothing too expensive",
    "show me something sporty and fast, budget's not really a concern",
]


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    banner("PHASE 4 DEMO: Query Transformation & Expansion")

    for query in QUERIES:
        banner(f'Original (Q0 candidate): "{query}"')

        transformed = transform_query(query)
        print(f"\nTransformed (optimized standalone query):\n  {transformed!r}")

        expanded = expand_query(transformed)
        print(f"\nExpanded queries ({len(expanded)} generated):")
        for i, q in enumerate(expanded, start=1):
            print(f"  Q{i}: {q!r}")

        total = 1 + len(expanded)
        print(f"\nTotal queries feeding retrieval: {total} (Q0={transformed!r}" + (f" + Q1..Q{len(expanded)})" if expanded else ", expansion unavailable)"))

    banner("Summary")
    print("Transformation strips conversational filler; expansion produces distinct")
    print("angles, not paraphrases. If Ollama is unavailable, transform_query falls")
    print("back to the input unchanged and expand_query returns [] -- retrieval still")
    print("works with just Q0, per the graceful-degradation design.")


if __name__ == "__main__":
    main()
