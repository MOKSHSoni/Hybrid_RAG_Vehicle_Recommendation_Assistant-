"""Phase 6: metadata filtering.

Query -> Constraints (Phase 3) -> Validated -> Metadata Filtering -> Candidate Set.

Plain Python over vehicle metadata dicts -- appropriate at 150 rows, no
need for a heavier filtering engine. Filtering is inherently
VEHICLE-level (a vehicle either matches or it doesn't); both of a
matching vehicle's chunks are included in the candidate set.

Semantics, matching the metadata schema fixed in Phase 1:
- brand: exact match (extraction already canonicalizes to a known brand).
- body_type: exact match.
- fuel_types: ANY overlap between requested and the vehicle's (multi-valued).
- transmission: "Automatic" requires has_automatic; "Manual" requires has_manual.
- seating_capacity: vehicle must seat AT LEAST the requested number (a
  bigger vehicle still satisfies "7 seater" -- deliberate, documented choice;
  missing seating data can never confirm a match, so it's excluded, not
  guessed).
- price_max/min_lakhs: vehicle's price_lakhs must fall within range;
  missing price can never confirm a match, so it's excluded, not guessed.
"""

from typing import Any, Dict, List, Optional, Set

from src.chunking.models import Chunk
from src.query.models import Constraints
from src.retrieval.chunk_store import ChunkStore


def unique_vehicles(chunk_store: ChunkStore) -> List[Dict[str, Any]]:
    """One metadata dict per vehicle (both of a vehicle's chunks carry
    identical metadata, so any one of them is representative)."""
    seen: Dict[str, Dict[str, Any]] = {}
    for chunk in chunk_store.all_chunks():
        seen.setdefault(chunk.doc_id, chunk.metadata)
    return list(seen.values())


def matches_constraints(
    vehicle_metadata: Dict[str, Any],
    constraints: Constraints,
    fields: Optional[List[str]] = None,
) -> bool:
    """Whether one vehicle satisfies the given constraints.

    `fields` restricts the check to a subset of constraint fields (used
    by Phase 10's relaxation to re-check against a reduced constraint
    set without needing a second Constraints object). Defaults to every
    populated field on `constraints`.
    """
    fields_to_check = fields if fields is not None else constraints.populated_fields()

    for field in fields_to_check:
        if field == "brand":
            if constraints.brand and vehicle_metadata.get("brand") != constraints.brand:
                return False

        elif field == "body_type":
            if constraints.body_type and vehicle_metadata.get("body_type") != constraints.body_type:
                return False

        elif field == "fuel_types":
            if constraints.fuel_types:
                vehicle_fuels = set(vehicle_metadata.get("fuel_types") or [])
                if not vehicle_fuels.intersection(constraints.fuel_types):
                    return False

        elif field == "transmission":
            if constraints.transmission == "Automatic" and not vehicle_metadata.get("has_automatic"):
                return False
            if constraints.transmission == "Manual" and not vehicle_metadata.get("has_manual"):
                return False

        elif field == "seating_capacity":
            if constraints.seating_capacity is not None:
                vehicle_seating = vehicle_metadata.get("seating_capacity")
                if vehicle_seating is None or vehicle_seating < constraints.seating_capacity:
                    return False

        elif field == "price_max_lakhs":
            if constraints.price_max_lakhs is not None:
                price = vehicle_metadata.get("price_lakhs")
                if price is None or price > constraints.price_max_lakhs:
                    return False

        elif field == "price_min_lakhs":
            if constraints.price_min_lakhs is not None:
                price = vehicle_metadata.get("price_lakhs")
                if price is None or price < constraints.price_min_lakhs:
                    return False

    return True


def filter_vehicles(
    vehicles: List[Dict[str, Any]],
    constraints: Constraints,
    fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    return [v for v in vehicles if matches_constraints(v, constraints, fields)]


def candidate_chunk_ids(
    chunk_store: ChunkStore,
    constraints: Constraints,
    fields: Optional[List[str]] = None,
) -> Set[str]:
    """The Phase 6 deliverable: the candidate set, as chunk_ids (both
    chunks of every matching vehicle) for Phase 7's constrained retrieval
    to search within."""
    matching_vehicle_ids = {
        v["vehicle_id"] for v in filter_vehicles(unique_vehicles(chunk_store), constraints, fields)
    }
    return {c.chunk_id for c in chunk_store.all_chunks() if c.doc_id in matching_vehicle_ids}


def candidate_chunks(
    chunk_store: ChunkStore,
    constraints: Constraints,
    fields: Optional[List[str]] = None,
) -> List[Chunk]:
    ids = candidate_chunk_ids(chunk_store, constraints, fields)
    return [c for c in chunk_store.all_chunks() if c.chunk_id in ids]
