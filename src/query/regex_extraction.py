"""Pure regex/keyword constraint extraction -- no LLM involved.

Serves two roles: (1) the fallback used whenever the LLM path fails or
returns invalid output, (2) fast, fully deterministic extraction for
constraints (especially price/seating) that regex handles reliably.
"""

import re
from typing import Dict, List, Optional, Tuple

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

# Canonical metadata field name -> trigger phrases. Units are optional in
# the regex below ("top speed above 200" needs no unit word).
_NUMERIC_METRIC_PHRASES: Dict[str, List[str]] = {
    "top_speed_kmph": ["top speed", "topspeed", "max speed", "maximum speed"],
    "boot_space_l": ["boot space", "boot capacity", "cargo space", "luggage space", "trunk space"],
    "ground_clearance_mm": ["ground clearance"],
    "mileage_max_kmpl": ["mileage", "fuel efficiency", "fuel economy"],
    "engine_max_cc": ["engine size", "engine displacement", "displacement", "engine capacity"],
}

# (pattern, metadata field, sort direction) -- checked in order, first match wins.
_SUPERLATIVE_PATTERNS: List[Tuple[str, str, str]] = [
    (r"\bcheapest\b|\blowest price\b|\bleast expensive\b", "price_lakhs", "asc"),
    (r"\bmost expensive\b|\bhighest price\b", "price_lakhs", "desc"),
    (r"\bfastest\b|\bhighest top speed\b|\bmax(?:imum)? top speed\b", "top_speed_kmph", "desc"),
    (r"\bbiggest boot\b|\blargest boot\b|\bmost boot space\b|\bmost cargo space\b", "boot_space_l", "desc"),
    (r"\bbest mileage\b|\bmost fuel efficient\b|\bhighest mileage\b", "mileage_max_kmpl", "desc"),
    (r"\bhighest ground clearance\b", "ground_clearance_mm", "desc"),
    (r"\bmost powerful\b|\bbiggest engine\b|\blargest engine\b", "engine_max_cc", "desc"),
]


def known_brands_from_chunk_store(chunk_store: ChunkStore) -> List[str]:
    brands = {c.metadata.get("brand") for c in chunk_store.all_chunks()}
    return sorted(b for b in brands if b)


def extract_regex_constraints(query: str, known_brands: List[str]) -> Constraints:
    price_min, price_max = extract_price_constraints(query)
    superlative_field, superlative_direction = extract_superlative(query)
    return Constraints(
        brand=extract_brand_constraint(query, known_brands),
        fuel_types=extract_fuel_constraints(query),
        transmission=extract_transmission_constraint(query),
        seating_capacity=extract_seating_constraint(query),
        body_type=extract_body_type_constraint(query),
        price_max_lakhs=price_max,
        price_min_lakhs=price_min,
        numeric_ranges=extract_numeric_range_constraints(query),
        superlative_field=superlative_field,
        superlative_direction=superlative_direction,
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


def extract_numeric_range_constraints(text: str) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Range constraints on the extended numeric metrics (top speed, boot
    space, ground clearance, mileage, engine size), keyed by canonical
    metadata field name. Independent of extract_price_constraints -- price
    keeps its own Lakh/Crore-aware implementation unchanged."""
    text_lower = text.lower()
    result: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for field_name, phrases in _NUMERIC_METRIC_PHRASES.items():
        min_v, max_v = _extract_metric_range(text_lower, phrases)
        if min_v is not None or max_v is not None:
            result[field_name] = (min_v, max_v)
    return result


def _extract_metric_range(text_lower: str, phrases: List[str]) -> Tuple[Optional[float], Optional[float]]:
    """Generic over/under/between number extraction scoped near one of the
    given trigger phrases. Units are optional -- "top speed above 200"
    needs no unit word, unlike price."""
    phrase_pattern = "|".join(re.escape(p) for p in phrases)

    range_pattern = rf"(?:{phrase_pattern}).{{0,20}}?{_NUMBER}\s*(?:-|to|and)\s*{_NUMBER}"
    m = re.search(range_pattern, text_lower)
    if m:
        low, high = float(m.group(1)), float(m.group(2))
        return min(low, high), max(low, high)

    at_least_pattern = rf"(?:{phrase_pattern}).{{0,20}}?(?:above|over|more than|at least|min(?:imum)?)\s+{_NUMBER}"
    m = re.search(at_least_pattern, text_lower)
    if m:
        return float(m.group(1)), None

    at_most_pattern = rf"(?:{phrase_pattern}).{{0,20}}?(?:under|below|less than|up ?to|max(?:imum)?)\s+{_NUMBER}"
    m = re.search(at_most_pattern, text_lower)
    if m:
        return None, float(m.group(1))

    return None, None


def extract_superlative(text: str) -> Tuple[Optional[str], Optional[str]]:
    text_lower = text.lower()
    for pattern, field_name, direction in _SUPERLATIVE_PATTERNS:
        if re.search(pattern, text_lower):
            return field_name, direction
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
