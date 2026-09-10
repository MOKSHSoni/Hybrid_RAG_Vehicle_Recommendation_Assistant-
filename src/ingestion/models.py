"""Generic, domain-agnostic document representation.

Nothing in this module (or the rest of src/ingestion/) may reference
vehicle-specific concepts. It must work identically for a CSV of cars,
real estate listings, or any other structured source.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class NormalizedDocument:
    """One source row, cleaned but not yet domain-enriched.

    raw_fields keys are the ORIGINAL column headers, verbatim (including
    spaces/units, e.g. "FUEL TYPE", "Length (mm)"). The generic layer
    cleans values, never renames columns -- renaming would require the
    generic layer to know which columns matter, which is domain knowledge
    that belongs in the enrichment layer instead.
    """

    doc_id: str
    raw_fields: Dict[str, Any]
    source_metadata: Dict[str, Any] = field(default_factory=dict)
