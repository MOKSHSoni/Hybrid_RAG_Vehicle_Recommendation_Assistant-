"""Phase 13: basic structured logging.

Captures which mode fired, which retrieval path was used, candidate
counts, and per-stage timings for one conversational turn -- feeds the
Streamlit debug panel now and is the foundation Phase 14 (optional
Langfuse) can build on rather than starting tracing from scratch.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class StageTiming:
    stage: str
    duration_ms: float


@dataclass
class PipelineLog:
    mode: str = ""
    extraction_method: str = ""
    standalone_query: str = ""
    transformed_query: str = ""
    constraints_summary: Dict[str, Any] = field(default_factory=dict)
    relaxation_steps: List[str] = field(default_factory=list)
    candidate_count: int = 0
    final_chunk_ids: List[str] = field(default_factory=list)
    retrieval_method: str = "hybrid"
    stage_timings: List[StageTiming] = field(default_factory=list)

    def add_timing(self, stage: str, duration_ms: float) -> None:
        self.stage_timings.append(StageTiming(stage, duration_ms))

    def total_ms(self) -> float:
        return sum(t.duration_ms for t in self.stage_timings)
