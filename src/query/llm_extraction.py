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
from typing import Any, Dict

import config
from src.query.models import Constraints
from src.query.ollama_client import chat

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
    },
    "required": [
        "brand",
        "fuel_types",
        "transmission",
        "seating_capacity",
        "body_type",
        "price_max_lakhs",
        "price_min_lakhs",
    ],
}

_SYSTEM_PROMPT = f"""You extract structured vehicle search constraints from a user's query.
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
price_max_lakhs / price_min_lakhs are in Lakhs INR (1 Crore = 100 Lakhs)."""


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
    for key in ("price_max_lakhs", "price_min_lakhs"):
        value = data[key]
        if value is not None and not isinstance(value, (int, float)):
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

    return Constraints(
        brand=data["brand"],
        fuel_types=fuel_types,
        transmission=transmission,
        seating_capacity=data["seating_capacity"],
        body_type=body_type,
        price_max_lakhs=data["price_max_lakhs"],
        price_min_lakhs=data["price_min_lakhs"],
    )
