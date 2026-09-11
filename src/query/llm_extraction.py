"""Qwen3-based constraint extraction.

The model's JSON is validated against a plain-Python schema check (no
jsonschema dependency needed for a schema this small) before it is ever
trusted -- per the project's global rule, the LLM never directly controls
filtering. Invalid JSON/schema is retried up to config.EXTRACTION_RETRY_COUNT
times; a failed/unreachable Ollama call is NOT retried here (propagates
immediately so the caller can fall back to regex without hitting a dead
endpoint again).
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import config
from src.query.models import Constraints
from src.query.ollama_client import chat

# Canonical metadata field name -> (schema key prefix). Kept in one place so
# the schema, validation, and _to_constraints stay in sync.
_NUMERIC_RANGE_FIELDS = {
    "top_speed_kmph": "top_speed_kmph",
    "boot_space_l": "boot_space_l",
    "ground_clearance_mm": "ground_clearance_mm",
    "mileage_max_kmpl": "mileage_kmpl",
    "engine_max_cc": "engine_cc",
}
_SUPERLATIVE_FIELDS = set(_NUMERIC_RANGE_FIELDS.keys()) | {"price_lakhs"}

JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "brand": {"type": ["string", "null"]},
        "fuel_types": {"type": "array", "items": {"type": "string"}},
        "transmission": {"type": ["string", "null"]},
        "seating_capacity": {"type": ["integer", "null"]},
        "body_type": {"type": ["string", "null"]},
        "price_max_lakhs": {"type": ["number", "null"]},
        "price_min_lakhs": {"type": ["number", "null"]},
        "top_speed_kmph_min": {"type": ["number", "null"]},
        "top_speed_kmph_max": {"type": ["number", "null"]},
        "boot_space_l_min": {"type": ["number", "null"]},
        "boot_space_l_max": {"type": ["number", "null"]},
        "ground_clearance_mm_min": {"type": ["number", "null"]},
        "ground_clearance_mm_max": {"type": ["number", "null"]},
        "mileage_kmpl_min": {"type": ["number", "null"]},
        "mileage_kmpl_max": {"type": ["number", "null"]},
        "engine_cc_min": {"type": ["number", "null"]},
        "engine_cc_max": {"type": ["number", "null"]},
        "superlative_field": {"type": ["string", "null"]},
        "superlative_direction": {"type": ["string", "null"]},
    },
    "required": [
        "brand",
        "fuel_types",
        "transmission",
        "seating_capacity",
        "body_type",
        "price_max_lakhs",
        "price_min_lakhs",
        "top_speed_kmph_min",
        "top_speed_kmph_max",
        "boot_space_l_min",
        "boot_space_l_max",
        "ground_clearance_mm_min",
        "ground_clearance_mm_max",
        "mileage_kmpl_min",
        "mileage_kmpl_max",
        "engine_cc_min",
        "engine_cc_max",
        "superlative_field",
        "superlative_direction",
    ],
}

_NUMERIC_RANGE_KEYS = [k for k in JSON_SCHEMA["required"] if k.endswith(("_min", "_max"))]

# The original 7-field schema, used for queries that mention no numeric spec
# and no superlative -- see needs_extended_extraction() for the latency
# rationale. Deliberately derived from JSON_SCHEMA rather than duplicated, so
# the two can't drift apart.
_CORE_FIELDS = [
    "brand",
    "fuel_types",
    "transmission",
    "seating_capacity",
    "body_type",
    "price_max_lakhs",
    "price_min_lakhs",
]
CORE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {k: JSON_SCHEMA["properties"][k] for k in _CORE_FIELDS},
    "required": list(_CORE_FIELDS),
}

# Intentionally broad: a false positive only costs latency, a false negative
# denies the LLM a phrasing regex can't parse. Covers metric vocabulary plus
# superlative morphology ("-est", most/least/best/worst).
_EXTENDED_TRIGGER_RE = re.compile(
    r"\b("
    r"speed|fast|quick|kmph|km/h|mph"
    r"|boot|trunk|cargo|luggage"
    r"|clearance"
    r"|mileage|millage|economy|economical|efficien\w*|kmpl"
    r"|engine|cc|litre|liter|displacement|power|powerful"
    # Explicit superlatives only: a "\w+est" catch-all matched ordinary
    # words like "suggest"/"request"/"latest" and escalated needlessly.
    r"|cheapest|fastest|quickest|biggest|largest|smallest|highest|lowest"
    r"|roomiest|priciest|costliest|longest|shortest|widest|tallest"
    r"|most|least|best|worst"
    r")\b",
    re.IGNORECASE,
)

_SYSTEM_PROMPT = f"""You extract structured vehicle search constraints from a user's query.
Respond with ONLY a JSON object matching this schema -- no explanation, no markdown fences:
{{
  "brand": string or null,
  "fuel_types": array of strings (e.g. ["Petrol"], ["Diesel","Petrol"], or [] if unspecified),
  "transmission": string or null,
  "seating_capacity": integer or null,
  "body_type": string or null,
  "price_max_lakhs": number or null,
  "price_min_lakhs": number or null,
  "top_speed_kmph_min": number or null, "top_speed_kmph_max": number or null,
  "boot_space_l_min": number or null, "boot_space_l_max": number or null,
  "ground_clearance_mm_min": number or null, "ground_clearance_mm_max": number or null,
  "mileage_kmpl_min": number or null, "mileage_kmpl_max": number or null,
  "engine_cc_min": number or null, "engine_cc_max": number or null,
  "superlative_field": string or null,
  "superlative_direction": "asc" or "desc" or null
}}
Only set a field when the query actually states or clearly implies it; otherwise use null ([] for fuel_types).
Valid body_type values: {", ".join(config.VALID_BODY_TYPES)}
Valid fuel_types values: {", ".join(config.VALID_FUEL_TYPES)}
Valid transmission values: {", ".join(config.VALID_TRANSMISSIONS)}
price_max_lakhs / price_min_lakhs are in Lakhs INR (1 Crore = 100 Lakhs).
top_speed_kmph is in km/h, boot_space_l in liters, ground_clearance_mm in millimeters, mileage_kmpl
in km/l, engine_cc in cc. Use *_min for "at least X"/"above X" and *_max for "under X"/"below X".
superlative_field is set ONLY for requests unambiguously asking for the single most extreme match --
explicit superlative words like "cheapest", "fastest", "best mileage", "biggest boot space", "highest
ground clearance", "most powerful" -- one of: {", ".join(sorted(_SUPERLATIVE_FIELDS))}.
superlative_direction is "asc" for cheapest/lowest-style requests, "desc" for fastest/highest/most/
biggest-style requests. Leave both null if the query is not asking for an extreme -- in particular,
general price sentiment words like "affordable", "budget-friendly", or "cheap" WITHOUT an explicit
superlative ("cheapest", "the most affordable") describe a price RANGE, not a request for the single
cheapest option: set price_max_lakhs instead (if a number is given or implied) and leave
superlative_field null. Example: "affordable SUV under 15 lakh" -> price_max_lakhs=15,
superlative_field=null (NOT price_lakhs/asc) -- the user wants options within budget, not just the
one cheapest vehicle.

CRITICAL -- NEVER FILL IN A FIELD THE QUERY DID NOT MENTION.
Most queries mention only one or two things. Emitting a value for anything else is an ERROR, even
if the value looks harmless or like a sensible default. A full range (0 to some maximum) is NOT a
default -- it is wrong. Use null.
Example: "suggest cars of bmw" -> "brand": "BMW", "fuel_types": [], EVERY other field null. Filling
in seating_capacity, body_type, price or any spec range there would be wrong: the query said
nothing about them."""


_CORE_SYSTEM_PROMPT = f"""You extract structured vehicle search constraints from a user's query.
Respond with ONLY a JSON object matching this schema -- no explanation, no markdown fences:
{{
  "brand": string or null,
  "fuel_types": array of strings (e.g. ["Petrol"], ["Diesel","Petrol"], or [] if unspecified),
  "transmission": string or null,
  "seating_capacity": integer or null,
  "body_type": string or null,
  "price_max_lakhs": number or null,
  "price_min_lakhs": number or null
}}
Only set a field when the query actually states or clearly implies it; otherwise use null ([] for fuel_types).
Valid body_type values: {", ".join(config.VALID_BODY_TYPES)}
Valid fuel_types values: {", ".join(config.VALID_FUEL_TYPES)}
Valid transmission values: {", ".join(config.VALID_TRANSMISSIONS)}
price_max_lakhs / price_min_lakhs are in Lakhs INR (1 Crore = 100 Lakhs).
Note: "affordable"/"budget-friendly"/"cheap" describe a price RANGE -- set price_max_lakhs if a
number is given or implied, and do not treat them as a request for the single cheapest vehicle.

CRITICAL -- NEVER FILL IN A FIELD THE QUERY DID NOT MENTION.
Most queries mention only one or two things. Emitting a value for anything else is an ERROR, even
if the value looks harmless or like a sensible default. A full range (0 to some maximum) is NOT a
default -- it is wrong. Use null.
Example: "suggest cars of bmw" -> "brand": "BMW", "fuel_types": [], EVERY other field null. Filling
in seating_capacity, body_type, price or any spec range there would be wrong: the query said
nothing about them."""

def needs_extended_extraction(query: str) -> bool:
    """Whether a query is worth paying the full 19-field schema for.

    Grammar-constrained decoding must emit every `required` key, so
    latency scales with field count -- measured on the target hardware at
    5.3s for the 7-field core schema vs 14.2s for the full 19-field one
    (2.7x, matching the 19/7 field ratio almost exactly). Most queries
    never mention a numeric spec or a superlative, so they shouldn't pay
    for 12 keys that come back null.

    The net here is deliberately WIDER than regex_extraction's precise
    phrase tables: this only decides which schema to send, and a false
    positive merely costs latency, whereas a false negative would deny
    the LLM the chance to parse a phrasing regex can't handle (e.g. "how
    fast does it go", "roomiest boot"). When in doubt, escalate.
    """
    return bool(_EXTENDED_TRIGGER_RE.search(query))


def extract_constraints_via_llm(standalone_query: str) -> Constraints:
    """May raise src.query.ollama_client.OllamaError (propagated, not
    retried) or ValueError (after exhausting retries on invalid output)."""
    extended = needs_extended_extraction(standalone_query)
    schema = JSON_SCHEMA if extended else CORE_JSON_SCHEMA
    system_prompt = _SYSTEM_PROMPT if extended else _CORE_SYSTEM_PROMPT

    last_error = None
    for _ in range(config.EXTRACTION_RETRY_COUNT + 1):
        raw = chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": standalone_query},
            ],
            format=schema,
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            last_error = f"invalid JSON from model: {e}; raw={raw!r}"
            continue
        if not _validate_schema(data, required=schema["required"]):
            last_error = f"schema validation failed: {data!r}"
            continue
        return _to_constraints(data)

    raise ValueError(last_error)


def _validate_schema(data: Any, required: Optional[List[str]] = None) -> bool:
    if not isinstance(data, dict):
        return False
    required = required if required is not None else JSON_SCHEMA["required"]
    if not all(key in data for key in required):
        return False

    if data["brand"] is not None and not isinstance(data["brand"], str):
        return False
    if not isinstance(data["fuel_types"], list) or not all(isinstance(x, str) for x in data["fuel_types"]):
        return False
    if data["transmission"] is not None and not isinstance(data["transmission"], str):
        return False
    if data["seating_capacity"] is not None and not isinstance(data["seating_capacity"], int):
        return False
    if data["body_type"] is not None and not isinstance(data["body_type"], str):
        return False
    # .get() from here down: these keys are required by the full schema but
    # legitimately absent from a CORE_JSON_SCHEMA response.
    for key in ("price_max_lakhs", "price_min_lakhs", *_NUMERIC_RANGE_KEYS):
        value = data.get(key)
        if value is not None and not isinstance(value, (int, float)):
            return False
    if data.get("superlative_field") is not None and not isinstance(data.get("superlative_field"), str):
        return False
    if data.get("superlative_direction") is not None and not isinstance(data.get("superlative_direction"), str):
        return False
    return True


def _to_constraints(data: Dict[str, Any]) -> Constraints:
    # .get() throughout, not [] -- a CORE_JSON_SCHEMA response legitimately
    # omits the 12 extended keys.
    transmission = data.get("transmission")
    if transmission not in config.VALID_TRANSMISSIONS:
        transmission = None  # model hallucinated a value outside the enum -- treat as unset, don't guess

    body_type = data.get("body_type")
    if body_type not in config.VALID_BODY_TYPES:
        body_type = None

    fuel_types = [f for f in (data.get("fuel_types") or []) if f in config.VALID_FUEL_TYPES]

    # A model asked to emit every field tends to fill it with "sensible
    # defaults" -- a 0 floor, every fuel type, a range spanning the whole
    # domain. Those constrain nothing, but they DO push the pipeline into
    # needless relaxation, so they're stripped here rather than trusted.
    # (The prompt discourages this too; this is the deterministic backstop,
    # per the project rule that the LLM never directly controls filtering.)
    if set(fuel_types) >= set(config.VALID_FUEL_TYPES):
        fuel_types = []  # "every fuel type" is the absence of a fuel preference

    price_min = data.get("price_min_lakhs")
    if price_min == 0:
        price_min = None  # a zero floor excludes nothing

    numeric_ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for canonical_field, schema_prefix in _NUMERIC_RANGE_FIELDS.items():
        min_v, max_v = data.get(f"{schema_prefix}_min"), data.get(f"{schema_prefix}_max")
        if min_v == 0:
            min_v = None  # these metrics are all non-negative, so a 0 floor is a no-op
        if min_v is not None or max_v is not None:
            numeric_ranges[canonical_field] = (min_v, max_v)

    superlative_field = data.get("superlative_field")
    superlative_direction = data.get("superlative_direction")
    if superlative_field not in _SUPERLATIVE_FIELDS or superlative_direction not in ("asc", "desc"):
        superlative_field, superlative_direction = None, None  # hallucinated/malformed -- don't guess

    return Constraints(
        brand=data.get("brand"),
        fuel_types=fuel_types,
        transmission=transmission,
        seating_capacity=data.get("seating_capacity"),
        body_type=body_type,
        price_max_lakhs=data.get("price_max_lakhs"),
        price_min_lakhs=price_min,
        numeric_ranges=numeric_ranges,
        superlative_field=superlative_field,
        superlative_direction=superlative_direction,
    )
