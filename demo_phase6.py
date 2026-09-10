"""Phase 6 deliverable: metadata filtering.

Query -> Constraints (Phase 3) -> Metadata Filtering -> Candidate Set.
Plain Python/Pandas-style filtering over vehicle metadata -- no LLM
involved in the filtering decision itself, only in producing the
Constraints object upstream (Phase 3).
"""

from src.pipeline import build_knowledge_base
from src.query.metadata_filter import candidate_chunk_ids, filter_vehicles, unique_vehicles
from src.query.models import Constraints


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def show_filter(title: str, vehicles, constraints: Constraints) -> None:
    print(f"\n{title}")
    matches = filter_vehicles(vehicles, constraints)
    print(f"  constraints: {constraints.populated_fields()}")
    print(f"  matches: {len(matches)} / {len(vehicles)} vehicles")
    for v in matches[:5]:
        print(f"    - {v['name']} (brand={v['brand']}, body_type={v['body_type']}, "
              f"price={v['price_lakhs']}, seats={v['seating_capacity']}, fuel={v['fuel_types']})")
    if len(matches) > 5:
        print(f"    ... and {len(matches) - 5} more")


def main() -> None:
    banner("PHASE 6 DEMO: Metadata Filtering")

    print("\nBuilding knowledge base...")
    kb = build_knowledge_base(save=False)
    vehicles = unique_vehicles(kb.chunk_store)
    print(f"Total vehicles: {len(vehicles)}")

    show_filter("Filter: brand=Porsche", vehicles, Constraints(brand="Porsche"))
    show_filter("Filter: fuel_types=[Electric]", vehicles, Constraints(fuel_types=["Electric"]))
    show_filter(
        "Filter: body_type=SUV, transmission=Automatic, price_max_lakhs=20",
        vehicles,
        Constraints(body_type="SUV", transmission="Automatic", price_max_lakhs=20.0),
    )
    show_filter(
        "Filter: seating_capacity=7 (at-least semantics: a 7/8-seater also matches)",
        vehicles,
        Constraints(seating_capacity=7),
    )
    show_filter(
        "Filter: brand=Ferrari AND seating_capacity=5 (no Ferrari seats 5 -- exercises the empty-result path)",
        vehicles,
        Constraints(brand="Ferrari", seating_capacity=5),
    )

    banner("Candidate chunk_ids feeding Phase 7 (hybrid retrieval)")
    constraints = Constraints(body_type="SUV", price_max_lakhs=20.0)
    ids = candidate_chunk_ids(kb.chunk_store, constraints)
    print(f"\nConstraints: {constraints.populated_fields()}")
    print(f"Candidate chunk_ids: {len(ids)} (both chunks per matching vehicle)")
    print("Phase 7 will run BM25/Dense retrieval scoring restricted to exactly this set,")
    print("rather than the full 300-chunk corpus.")

    banner("Summary")
    print("Filtering is vehicle-level (both chunks of a matching vehicle are candidates),")
    print("missing metadata never guesses a match (safe default: excluded), and this never")
    print("touches an LLM -- Constraints objects are already Python-validated by Phase 3.")


if __name__ == "__main__":
    main()
