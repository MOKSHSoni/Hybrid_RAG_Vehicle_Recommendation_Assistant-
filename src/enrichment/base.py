"""The enrichment interface.

Deliberately lightweight -- NOT a plugin/registry system. Adding a new
domain later (e.g. ProductEnrichment, RealEstateEnrichment) means writing
a new BaseEnrichment subclass; nothing here needs to change.
"""

from abc import ABC, abstractmethod
from typing import List

from src.enrichment.models import EnrichedDocument
from src.ingestion.models import NormalizedDocument


class BaseEnrichment(ABC):
    """Domain-specific enrichment interface.

    Implementations must inspect a NormalizedDocument's raw_fields AT
    RUNTIME (never assume a fixed column set) and return an
    EnrichedDocument with derived, structured metadata.
    """

    @abstractmethod
    def enrich(self, document: NormalizedDocument) -> EnrichedDocument:
        ...

    def enrich_batch(self, documents: List[NormalizedDocument]) -> List[EnrichedDocument]:
        """Default batch implementation; override if batch-vectorization ever helps."""
        return [self.enrich(d) for d in documents]
