"""Output representation of the enrichment layer."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class EnrichedDocument:
    """A NormalizedDocument plus domain-specific structured metadata.

    `metadata` is a plain dict in this generic model -- keeping
    BaseEnrichment domain-agnostic -- while a concrete implementation
    (e.g. VehicleEnrichment) defines and documents the concrete keys it
    populates.
    """

    doc_id: str
    raw_fields: Dict[str, Any]
    source_metadata: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
