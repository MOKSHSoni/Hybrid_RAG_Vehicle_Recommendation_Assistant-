"""The Chunk representation shared by every retrieval backend."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Chunk:
    chunk_id: str  # f"{doc_id}::{chunk_type}", + "::part{i}" suffix if ever split
    doc_id: str  # = vehicle_id
    chunk_type: str  # "product_overview" | "features"
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector_index: Optional[int] = None  # assigned later by the pipeline
