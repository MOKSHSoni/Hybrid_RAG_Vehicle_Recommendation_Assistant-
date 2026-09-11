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
from typing import Any, Dict, Optional, Tuple

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
one cheapest vehicle."""


def extract_constraints_via_llm(standalone_query: str) -> Constraints:
    """May raise src.query.ollama_client.OllamaError (propagated, not
    retried) or ValueError (after exhausting retries on invalid output)."""
    last_error = None
    for _ in range(config.EXTRACTION_RETRY_COUNT + 1):
        raw = chat(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": standalone_query},
            ],
            format=JSON_SCHEMA,
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            last_error = f"invalid JSON from model: {e}; raw={raw!r}"
            continue
        if not _validate_schema(data):
            last_error = f"schema validation failed: {data!r}"
            continue
        return _to_constraints(data)

    raise ValueError(last_error)


def _validate_schema(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    required = JSON_SCHEMA["required"]
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
    for key in ("price_max_lakhs", "price_min_lakhs", *_NUMERIC_RANGE_KEYS):
        value = data[key]
        if value is not None and not isinstance(value, (int, float)):
            return False
    if data["superlative_field"] is not None and not isinstance(data["superlative_field"], str):
        return False
    if data["superlative_direction"] is not None and not isinstance(data["superlative_direction"], str):
        return False
    return True


def _to_constraints(data: Dict[str, Any]) -> Constraints:
    transmission = data["transmission"]
    if transmission not in config.VALID_TRANSMISSIONS:
        transmission = None  # model hallucinated a value outside the enum -- treat as unset, don't guess

    body_type = data["body_type"]
    if body_type not in config.VALID_BODY_TYPES:
        body_type = None

    fuel_types = [f for f in data["fuel_types"] if f in config.VALID_FUEL_TYPES]

    numeric_ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for canonical_field, schema_prefix in _NUMERIC_RANGE_FIELDS.items():
        min_v, max_v = data.get(f"{schema_prefix}_min"), data.get(f"{schema_prefix}_max")
        if min_v is not None or max_v is not None:
            numeric_ranges[canonical_field] = (min_v, max_v)

    superlative_field = data["superlative_field"]
    superlative_direction = data["superlative_direction"]
    if superlative_field not in _SUPERLATIVE_FIELDS or superlative_direction not in ("asc", "desc"):
        superlative_field, superlative_direction = None, None  # hallucinated/malformed -- don't guess

    return Constraints(
        brand=data["brand"],
        fuel_types=fuel_types,
        transmission=transmission,
        seating_capacity=data["seating_capacity"],
        body_type=body_type,
        price_max_lakhs=data["price_max_lakhs"],
        price_min_lakhs=data["price_min_lakhs"],
        numeric_ranges=numeric_ranges,
        superlative_field=superlative_field,
        superlative_direction=superlative_direction,
    )
