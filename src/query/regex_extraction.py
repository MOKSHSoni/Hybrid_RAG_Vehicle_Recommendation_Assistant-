"""Pure regex/keyword constraint extraction -- no LLM involved.

Serves two roles: (1) the fallback used whenever the LLM path fails or
returns invalid output, (2) fast, fully deterministic extraction for
constraints (especially price/seating) that regex handles reliably.
"""

import re
from typing import List, Optional, Tuple

from src.query.models import Constraints
from src.retrieval.chunk_store import ChunkStore

_NUMBER = r"(\d+(?:\.\d+)?)"
# Longer/more specific alternatives first for readability; trailing \b plus
# backtracking would resolve ordering correctly either way, but this also
# avoids relying on that subtlety. "l" alone covers the common "20L" shorthand.
_UNIT = r"(lakhs?|lacs?|crores?|cr|l)\b"

_FUEL_SYNONYMS = {
    "electric": "Electric",
    "ev": "Electric",
    "petrol": "Petrol",
    "diesel": "Diesel",
    "hybrid": "Hybrid",
    "cng": "CNG",
}

_BODY_TYPE_SYNONYMS = {
    "suv": "SUV",
    "sedan": "Sedan",
    "saloon": "Sedan",
    "hatchback": "Hatchback",
    "hatch": "Hatchback",
    "mpv": "MPV",
    "minivan": "MPV",
    "coupe": "Coupe",
    "convertible": "Convertible",
    "cabriolet": "Convertible",
    "cabrio": "Convertible",
    "roadster": "Convertible",
}


def known_brands_from_chunk_store(chunk_store: ChunkStore) -> List[str]:
    brands = {c.metadata.get("brand") for c in chunk_store.all_chunks()}
    return sorted(b for b in brands if b)


def extract_regex_constraints(query: str, known_brands: List[str]) -> Constraints:
    price_min, price_max = extract_price_constraints(query)
    return Constraints(
        brand=extract_brand_constraint(query, known_brands),
        fuel_types=extract_fuel_constraints(query),
        transmission=extract_transmission_constraint(query),
        seating_capacity=extract_seating_constraint(query),
        body_type=extract_body_type_constraint(query),
        price_max_lakhs=price_max,
        price_min_lakhs=price_min,
    )


def extract_price_constraints(text: str) -> Tuple[Optional[float], Optional[float]]:
    """Normalizes price mentions (Lakh/Crore, ranges, ceilings, floors)
    into (price_min_lakhs, price_max_lakhs)."""
    text_lower = text.lower()

    range_pattern = rf"(?:between\s+)?{_NUMBER}\s*(?:-|to|and)\s*{_NUMBER}\s*{_UNIT}"
    m = re.search(range_pattern, text_lower)
    if m:
        low = _to_lakhs(float(m.group(1)), m.group(3))
        high = _to_lakhs(float(m.group(2)), m.group(3))
        return min(low, high), max(low, high)

    max_pattern = rf"(?:under|below|less than|up ?to|within|max(?:imum)?(?:\s+budget\s+of)?)\s+{_NUMBER}\s*{_UNIT}"
    m = re.search(max_pattern, text_lower)
    if m:
        return None, _to_lakhs(float(m.group(1)), m.group(2))

    min_pattern = rf"(?:above|over|more than|min(?:imum)?|at least)\s+{_NUMBER}\s*{_UNIT}"
    m = re.search(min_pattern, text_lower)
    if m:
        return _to_lakhs(float(m.group(1)), m.group(2)), None

    # Loose mention, e.g. "around 20 lakh", "20 lakh budget", or bare "20 lakh"
    # -- treated as a soft ceiling.
    loose_pattern = rf"{_NUMBER}\s*{_UNIT}"
    m = re.search(loose_pattern, text_lower)
    if m:
        return None, _to_lakhs(float(m.group(1)), m.group(2))

    return None, None


def extract_seating_constraint(text: str) -> Optional[int]:
    text_lower = text.lower()
    m = re.search(r"(\d+)[\s-]*seat", text_lower)
    if m:
        return int(m.group(1))
    m = re.search(r"seat(?:s|ing)?\s*(?:for|of)?\s*(\d+)", text_lower)
    if m:
        return int(m.group(1))
    return None


def extract_fuel_constraints(text: str) -> List[str]:
    text_lower = text.lower()
    found = []
    for keyword, canonical in _FUEL_SYNONYMS.items():
        if re.search(rf"\b{keyword}\b", text_lower) and canonical not in found:
            found.append(canonical)
    return found


def extract_transmission_constraint(text: str) -> Optional[str]:
    text_lower = text.lower()
    if re.search(r"\bautomatic\b|\bauto\b", text_lower):
        return "Automatic"
    if re.search(r"\bmanual\b", text_lower):
        return "Manual"
    return None


def extract_body_type_constraint(text: str) -> Optional[str]:
    text_lower = text.lower()
    for keyword, canonical in _BODY_TYPE_SYNONYMS.items():
        if re.search(rf"\b{keyword}\b", text_lower):
            return canonical
    return None


def extract_brand_constraint(text: str, known_brands: List[str]) -> Optional[str]:
    text_lower = text.lower()
    for brand in known_brands:
        if re.search(rf"\b{re.escape(brand.lower())}\b", text_lower):
            return brand
    return None


def _to_lakhs(value: float, unit: str) -> float:
    return value * 100 if unit.lower().startswith(("cr",)) else value
