"""Data models shared across query understanding (Phase 3+)."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import config


@dataclass
class ConversationTurn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class Constraints:
    """Structured, PYTHON-VALIDATED search constraints.

    Never populated by trusting raw LLM output directly -- extraction
    functions validate/coerce before constructing this. This is the only
    representation Phase 6 (metadata filtering) is allowed to read.
    """

    brand: Optional[str] = None
    fuel_types: List[str] = field(default_factory=list)
    transmission: Optional[str] = None  # "Automatic" | "Manual" | None
    seating_capacity: Optional[int] = None
    body_type: Optional[str] = None
    price_max_lakhs: Optional[float] = None
    price_min_lakhs: Optional[float] = None

    # Extended numeric range constraints, keyed by canonical metadata field
    # name (e.g. "top_speed_kmph", "boot_space_l") so metadata_filter needs
    # no translation table. mileage/engine use the vehicle's best-case
    # (max) variant as the single representative value -- same pattern as
    # price_lakhs (not price_min/max_lakhs) being the value compared
    # against above. Value is (min_threshold, max_threshold), either
    # optionally None.
    numeric_ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = field(default_factory=dict)

    # Superlative request ("cheapest", "fastest", "highest ground
    # clearance"): field is a metadata field name, direction is "asc" or
    # "desc". Drives a distinct direct-metadata-sort code path (Phase 10's
    # retrieve_with_relaxation), not a filter check -- deliberately NOT
    # included in populated_fields()/CONSTRAINT_RELAXATION_ORDER.
    superlative_field: Optional[str] = None
    superlative_direction: Optional[str] = None  # "asc" | "desc" | None

    def populated_fields(self) -> List[str]:
        fields = []
        for name in ("brand", "transmission", "seating_capacity", "body_type", "price_max_lakhs", "price_min_lakhs"):
            if getattr(self, name) is not None:
                fields.append(name)
        if self.fuel_types:
            fields.append("fuel_types")
        fields.extend(self.numeric_ranges.keys())
        return fields

    def hard_fields(self) -> List[str]:
        return [f for f in self.populated_fields() if f in config.HARD_CONSTRAINT_FIELDS]

    def soft_fields(self) -> List[str]:
        return [f for f in self.populated_fields() if f not in config.HARD_CONSTRAINT_FIELDS]

    def is_empty(self) -> bool:
        return not self.populated_fields()


@dataclass
class QueryUnderstandingResult:
    original_query: str
    standalone_query: str  # context-resolved, follow-up-independent query text
    constraints: Constraints
    extraction_method: str  # "llm" | "regex_fallback"
    extraction_error: Optional[str] = None  # set when the LLM path failed and we fell back
